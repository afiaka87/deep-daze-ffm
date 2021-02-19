import time
from base64 import b64encode

import ipywidgets as ipy
import numpy as np
import torch
from IPython.core.display import display


def get_mgrid(sideX, sideY):
    tensors = [np.linspace(-1, 1, num=sideY), np.linspace(-1, 1, num=sideX)]
    mgrid = np.stack(np.meshgrid(*tensors), axis=-1)
    mgrid = mgrid.reshape(-1, 2)  # dim 2
    return mgrid


def slice_imgs(imgs, count, transform=None, uniform=False):
    def map(x, a, b):
        return x * (b - a) + a

    rnd_size = torch.rand(count)
    rnd_offx = torch.rand(count)
    rnd_offy = torch.rand(count)

    sz = [img.shape[2:] for img in imgs]
    sz_min = [np.min(s) for s in sz]
    if uniform is True:
        upsize = [[2 * s[0], 2 * s[1]] for s in list(sz)]
        imgs = [pad_up_to(imgs[i], upsize[i], type='centr') for i in range(len(imgs))]

    sliced = []
    for i, img in enumerate(imgs):
        cuts = []
        for c in range(count):
            csize = map(rnd_size[c], 0.5 * sz_min[i], 0.98 * sz_min[i]).int()
            if uniform is True:
                offsetx = map(rnd_offx[c], sz[i][1] - csize, 2 * sz[i][1] - csize).int()
                offsety = map(rnd_offy[c], sz[i][0] - csize, 2 * sz[i][0] - csize).int()
            else:
                offsetx = map(rnd_offx[c], 0, sz[i][1] - csize).int()
                offsety = map(rnd_offy[c], 0, sz[i][0] - csize).int()
            cut = img[:, :, offsety:offsety + csize, offsetx:offsetx + csize]
            cut = torch.nn.functional.interpolate(cut, (224, 224), mode='bilinear')
            if transform is not None:
                cut = transform(cut)
            cuts.append(cut)
        sliced.append(torch.cat(cuts, 0))
    return sliced


def makevid(seq_dir, size=None):
    out_sequence = seq_dir + '/%03d.jpg'
    out_video = seq_dir + '.mp4'
    # !ffmpeg -y -v warning -i $out_sequence $out_video
    data_url = "data:video/mp4;base64," + b64encode(open(out_video, 'rb').read()).decode()
    wh = '' if size is None else 'width=%d height=%d' % (size, size)
    return """<video %s controls><source src="%s" type="video/mp4"></video>""" % (wh, data_url)


def tile_pad(xt, padding):
    h, w = xt.shape[-2:]
    left, right, top, bottom = padding

    def tile(x, minx, maxx):
        rng = maxx - minx
        mod = np.remainder(x - minx, rng)
        out = mod + minx
        return np.array(out, dtype=x.dtype)

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)
    x_pad = tile(x_idx, -0.5, w - 0.5)
    y_pad = tile(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return xt[..., yy, xx]


def pad_up_to(x, size, type='centr'):
    sh = x.shape[2:][::-1]
    if list(x.shape[2:]) == list(size): return x
    padding = []
    for i, s in enumerate(size[::-1]):
        if 'side' in type.lower():
            padding = padding + [0, s - sh[i]]
        else:  # centr
            p0 = (s - sh[i]) // 2
            p1 = s - sh[i] - p0
            padding = padding + [p0, p1]
    y = tile_pad(x, padding)
    return y


def time_days(sec):
    return '%dd %d:%02d:%02d' % (sec / 86400, (sec / 3600) % 24, (sec / 60) % 60, sec % 60)


def time_hrs(sec):
    return '%d:%02d:%02d' % (sec / 3600, (sec / 60) % 60, sec % 60)


def shortime(sec):
    if sec < 60:
        time_short = '%d' % (sec)
    elif sec < 3600:
        time_short = '%d:%02d' % ((sec / 60) % 60, sec % 60)
    elif sec < 86400:
        time_short = time_hrs(sec)
    else:
        time_short = time_days(sec)
    return time_short


class ProgressBar(object):
    def __init__(self, task_num=10):
        self.pbar = ipy.IntProgress(min=0, max=task_num,
                                    bar_style='')  # (value=0, min=0, max=max, step=1, description=description, bar_style='')
        self.labl = ipy.Label()
        display(ipy.HBox([self.pbar, self.labl]))
        self.task_num = task_num
        self.completed = 0
        self.start()

    def start(self, task_num=None):
        if task_num is not None:
            self.task_num = task_num
        if self.task_num > 0:
            self.labl.value = '0/{}'.format(self.task_num)
        else:
            self.labl.value = 'completed: 0, elapsed: 0s'
        self.start_time = time.time()

    def upd(self, *p, **kw):
        self.completed += 1
        elapsed = time.time() - self.start_time + 0.0000000000001
        fps = self.completed / elapsed if elapsed > 0 else 0
        if self.task_num > 0:
            finaltime = time.asctime(time.localtime(self.start_time + self.task_num * elapsed / float(self.completed)))
            fin = ' end %s' % finaltime[11:16]
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            self.labl.value = '{}/{}, rate {:.3g}s, time {}s, left {}s, {}'.format(self.completed, self.task_num,
                                                                                   1. / fps, shortime(elapsed),
                                                                                   shortime(eta), fin)
        else:
            self.labl.value = 'completed {}, time {}s, {:.1f} steps/s'.format(self.completed, int(elapsed + 0.5), fps)
        self.pbar.value += 1
        if self.completed == self.task_num: self.pbar.bar_style = 'success'
        return
        # return self.completed


def fourierfm(xy, map=256, fourier_scale=4, mapping_type='gauss'):
    def input_mapping(x, B):  # feature mappings
        x_proj = (2. * np.pi * x) @ B
        y = np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)
        print(' mapping input:', x.shape, 'output', y.shape)
        return y

    if mapping_type == 'gauss':  # Gaussian Fourier feature mappings
        B = np.random.randn(2, map)
        B *= fourier_scale  # scale Gauss
    else:  # basic
        B = np.eye(2).T

    xy = input_mapping(xy, B)
    return xy