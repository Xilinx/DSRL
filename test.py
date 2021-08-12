# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import glob
import os
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import functional as F
from torch import nn
from tqdm import tqdm
import numpy as np
import time
import cv2
from torch.nn import functional as FF

def relabel(img):
    """
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    """
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    
    return img

def data_transform(img, im_size):
    
    img = img.resize(im_size, Image.BILINEAR)
    img_tensor = F.to_tensor(img)  # convert to tensor 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = F.normalize(img_tensor, mean, std)  # normalize the tensor
    return tensor


def predict(args, model, image_list, device):
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    im_size = tuple(args.im_size)
    model.eval()
    for i, imgName in enumerate(image_list):
        print('Process Batch Id {}/{}'.format(i, len(image_list)))
        img = Image.open(imgName).convert('RGB')
        w, h = img.size
        img = data_transform(img, im_size)
        img = img.unsqueeze(0)  # add a batch dimension
        img = img.to(device)
        seg_out = model(img)
        
        name = imgName.split('/')[-1]
        img_extn = imgName.split('.')[-1]

        # get semantic segmnattaion results
        seg_out = seg_out.squeeze(0)  
        seg_out = seg_out.max(0)[1].byte()  
        seg_out = seg_out.to(device='cpu').numpy()

        if args.dataset == 'city':
            if args.split=='test':
                #for test set, change from Train IDs to label IDs
                seg_out = relabel(seg_out)
            elif args.split=='val':
                seg_out = seg_out

        seg_out = Image.fromarray(seg_out)
        # resize to original size
        seg_out = seg_out.resize((w, h), Image.NEAREST)
        # save the prediction 
        img_extn = imgName.split('.')[-1]
        name = '{}/{}'.format(args.savedir, name.replace(img_extn, 'png'))
        seg_out.save(name)
        

def main(args):
    
    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'

    #===================== load datasets=========================
    # read all the images, e.g., here we only provide cityscapes
    if args.dataset == 'city':
        image_path = os.path.join(args.data_path, "leftImg8bit", args.split, "*", "*.png")
        image_list = glob.glob(image_path)
        seg_classes = 20 # 19 valid classes + ignore class
    else:
        print('{} dataset not yet supported'.format(args.dataset))

    if len(image_list) == 0:
        print('No files in directory: {}'.format(image_path))
        exit(-1)

    print('# of images for testing: {}'.format(len(image_list)))

    #===================== load model ============================
    from model.segmentation_DSRL.espnetv2_dsrl import espnetv2_seg
    args.classes = seg_classes
    model = espnetv2_seg(args)
    model = model.to(device=device)

    if args.ckpt_file:
        model_dict = model.state_dict()
        weight_dict = torch.load(args.ckpt_file, map_location=torch.device('cpu'))
        overlap_dict = {k: v for k, v in weight_dict.items() if k in model_dict}
        model.load_state_dict(overlap_dict)
        print('Weight loaded successfully')
    else:
        print('weight file does not exist or not specified. Please check: {}', format(args.ckpt_file))
        exit(-1)

    #===================== Begin Testing ============================
    predict(args, model, image_list, device=device)


if __name__ == '__main__':

    parser = ArgumentParser()
    # mdoel details
    parser.add_argument('--cross-os', type=float, default=2.0, help='Factor by which feature for cross')
    parser.add_argument('--model', default="espnetv2", help='Model name')
    parser.add_argument('--ckpt-file', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--data-path', default="", help='Data directory')
    parser.add_argument('--dataset', default='city', help='Dataset name')
    parser.add_argument('--savedir', default="result", help='save prediction directory')
    # input details
    parser.add_argument('--im-size', type=int, nargs="+", default=[512, 256], help='Image size for testing (W x H)')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='data split')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')

    args = parser.parse_args()
    
    main(args)

