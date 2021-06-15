# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 09:50:20 2021

@author: User
"""
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from util.data_vis import plot_img_and_mask
from util.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )
        #print(probs.squeeze()[1,:,:].shape)
        mask_all = []
        for i in range(10):
            probs_i = tf(probs.squeeze()[i,:,:].cpu())
            full_mask = probs_i.squeeze().cpu().numpy()
            mask_all.append(full_mask)
            #mask_z = np.argmax(full_mask,1)
            #print(np.max(full_mask))
        mask_z = np.array(mask_all)
        mask = np.argmax(mask_z,0)
    return mask



def get_output_filenames(input_path,out_path):
    in_files = input_path
    out_files = []

    if not out_path:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(out_path):
        print(len(in_files),len(out_path))
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = out_path

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    root = 'data/test_set'
    out_root = 'test1'
    input_path = list(sorted(os.listdir(os.path.join(root))))
    #input_path = ['image.png','1111']
    out_files = input_path
    #out_path = [out_path]
    model_path = 'checkpoints/CP_epoch8.pth'
    scale = 0.5
    mask_threshold = 0.3
    in_files = input_path
    #out_files = get_output_filenames(input_path,out_path)

    net = UNet(n_channels=3, n_classes=10)

    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(root+'/'+fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=scale,
                           out_threshold=mask_threshold,
                           device=device)


        out_fn = out_files[i]
        aaa = np.uint8(mask)
        mask_out = np.array([aaa,aaa,aaa]).transpose((1, 2, 0))
        mask_PIL = Image.fromarray(mask_out)
        mask_PIL.save(out_root+'/'+out_files[i])
        print(i)
        # result = mask_to_image(mask)
        # result.save(out_files[i])

        # logging.info("Mask saved to {}".format(out_files[i]))


        # logging.info("Visualizing results for image {}, close to continue ...".format(fn))
        # plot_img_and_mask(img, mask)