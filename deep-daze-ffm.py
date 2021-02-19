from googletrans import Translator

import clip
import os
import imageio
import numpy as np
from skimage import exposure

import torch
import torchvision

from IPython.display import Image, display, clear_output

from siren import Siren
from util import get_mgrid, slice_imgs, ProgressBar, fourierfm


def displ(img, fname=None):
    img = np.array(img)[:, :, :]
    img = np.transpose(img, (1, 2, 0))
    img = exposure.equalize_adapthist(np.clip(img, -1., 1.))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if fname is not None:
        imageio.imsave(fname, np.array(img))
        imageio.imsave('result.jpg', np.array(img))


def checkin(num):
    with torch.no_grad():
        img = model(mesh_grid).cpu().numpy()[0]
    displ(img, os.path.join(temp_dir, '%03d.jpg' % num))


def train(iteration):
    img_out = model(mesh_grid)
    imgs_sliced = slice_imgs([img_out], samples, norm_in, uniform)
    loss = 0
    out_enc = perceptor.encode_image(imgs_sliced[-1])
    loss += -100 * torch.cosine_similarity(txt_encode, out_enc, dim=-1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iteration % save_freq == 0:
        checkin(iteration // save_freq)


work_dir = '_out'
temp_dir = os.path.join(work_dir, 'ttt')
os.makedirs(temp_dir, exist_ok=True)
clear_output()
translator = Translator()

text = "hollow knight"  # @param {type:"string"}
translate = False  # @param {type:"boolean"}
# @markdown or
upload_image = True  # @param {type:"boolean"}

if translate:
    text = translator.translate(text, dest='en').text

side_x = 512  # @param {type:"integer"}
side_y = 512  # @param {type:"integer"}
uniform = False  # @param {type:"boolean"}
sync_cut = True  # @param {type:"boolean"}
steps = 750  # @param {type:"integer"}
save_freq = 250  # @param {type:"integer"}
learning_rate = .00001  # @param {type:"number"}
samples = 120  # @param {type:"integer"}
siren_layers = 32  # @param {type:"integer"}
use_fourier_feat_map = True  # @param {type:"boolean"}
fourier_maps = 256  # @param {type:"integer"}
fourier_scale = 4  # @param {type:"number"}
out_name = text.replace(' ', '_')

mesh_grid = get_mgrid(side_y, side_x)  # [262144,2]
if use_fourier_feat_map:
    mesh_grid = fourierfm(mesh_grid, fourier_maps, fourier_scale)
mesh_grid = torch.from_numpy(mesh_grid.astype(np.float32)).cuda()
model = Siren(mesh_grid.shape[-1], 256, siren_layers, 3, side_x, side_y).cuda()
norm_in = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
text_tokens = clip.tokenize(text)

perceptor, preprocess = clip.load('ViT-B/32')
txt_encode = perceptor.encode_text(text_tokens.cuda()).detach().clone()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

progress_bar = ProgressBar(steps)
for i in range(steps):
    train(i)
    _ = progress_bar.update()
