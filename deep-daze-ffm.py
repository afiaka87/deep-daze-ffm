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
        img = model(mgrid).cpu().numpy()[0]
    displ(img, os.path.join(tempdir, '%03d.jpg' % num))


def train(iteration, img_encoding):
    img_out = model(mgrid)
    imgs_sliced = slice_imgs([img_out], samples, norm_in, uniform)
    loss = 0
    if vit:
        vit_out_enc = vit_perceptor.encode_image(imgs_sliced[-1])
        loss += -100 * torch.cosine_similarity(vit_txt_encode, vit_out_enc, dim=-1).mean()
    if rn50:
        rn50_out_enc = rn50_perceptor.encode_image(imgs_sliced[-1])
        loss += -100 * torch.cosine_similarity(rn50_txt_encode, rn50_out_enc, dim=-1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iteration % save_freq == 0:
        checkin(iteration // save_freq)


workdir = '_out'
tempdir = os.path.join(workdir, 'ttt')
os.makedirs(tempdir, exist_ok=True)
clear_output()
translator = Translator()


text = "hollow knight"  # @param {type:"string"}
translate = False  # @param {type:"boolean"}
# @markdown or
upload_image = True  # @param {type:"boolean"}

if translate:
    text = translator.translate(text, dest='en').text

sideX = 256 # @param {type:"integer"}
sideY = 256  # @param {type:"integer"}
uniform = False  # @param {type:"boolean"}
sync_cut = True  # @param {type:"boolean"}
# @markdown > Training
steps = 500  # @param {type:"integer"}
save_freq = 1  # @param {type:"integer"}
learning_rate = .0001  # @param {type:"number"}
samples = 60  # @param {type:"integer"}
# @markdown > Network
siren_layers = 24  # @param {type:"integer"}
use_fourier_feat_map = True  # @param {type:"boolean"}
fourier_maps = 128  # @param {type:"integer"}
fourier_scale = 4  # @param {type:"number"}
# @markdown > Misc
audio_notification = False  # @param {type:"boolean"}
vit = True
rn50 = True
out_name = text.replace(' ', '_')

mgrid = get_mgrid(sideY, sideX)  # [262144,2]
if use_fourier_feat_map:
    mgrid = fourierfm(mgrid, fourier_maps, fourier_scale)
mgrid = torch.from_numpy(mgrid.astype(np.float32)).cuda()
model = Siren(mgrid.shape[-1], 256, siren_layers, 3).cuda()
norm_in = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
text_tokens = clip.tokenize(text)

if vit:
    vit_perceptor, vit_preprocess = clip.load('ViT-B/32')
    vit_txt_encode = vit_perceptor.encode_text(text_tokens.cuda()).detach().clone()
if rn50:
    rn50_perceptor, rn50_preprocess = clip.load('RN50')
    rn50_txt_encode = rn50_perceptor.encode_text(text_tokens.cuda()).detach().clone()

optimizer = torch.optim.Adam(model.parameters(), learning_rate)

pbar = ProgressBar(steps)
for i in range(steps):
    train(i)
    _ = pbar.upd()