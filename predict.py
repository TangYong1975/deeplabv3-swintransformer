from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, default='datasets/data/CameraBase/VOCdevkit/VOC2012/JPEGImages', 
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of training set')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default = 'deeplabv3plus_ghostnet_v2_1_0',
                        choices=['deeplabv3_resnet18', 'deeplabv3plus_resnet18',
                                 'deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet_v2', 'deeplabv3plus_mobilenet_v2',
                                 'deeplabv3_mobilenet_v3_small', 'deeplabv3plus_mobilenet_v3_small',
                                 'deeplabv3_mobilenet_v3_large', 'deeplabv3plus_mobilenet_v3_large',
                                 'deeplabv3_berniwal_swintransformer_swin_t', 'deeplabv3plus_berniwal_swintransformer_swin_t'
                                 'deeplabv3_berniwal_swintransformer_swin_s', 'deeplabv3plus_berniwal_swintransformer_swin_s'
                                 'deeplabv3_berniwal_swintransformer_swin_b', 'deeplabv3plus_berniwal_swintransformer_swin_b'
                                 'deeplabv3_berniwal_swintransformer_swin_l', 'deeplabv3plus_berniwal_swintransformer_swin_l'
                                 'deeplabv3_microsoft_swintransformer_swin_t', 'deeplabv3plus_microsoft_swintransformer_swin_t'
                                 'deeplabv3_microsoft_swintransformer_swin_s', 'deeplabv3plus_microsoft_swintransformer_swin_s'
                                 'deeplabv3_microsoft_swintransformer_swin_b', 'deeplabv3plus_microsoft_swintransformer_swin_b'
                                 'deeplabv3_microsoft_swintransformer_swin_l', 'deeplabv3plus_microsoft_swintransformer_swin_l'
                                 'deeplabv3_hrnetv2_32', 'deeplabv3plus_hrnetv2_32',
                                 'deeplabv3_hrnetv2_48', 'deeplabv3plus_hrnetv2_48',
                                 'deeplabv3_xception', 'deeplabv3plus_xception',
                                 'deeplabv3_regnet_y_400mf', 'deeplabv3plus_regnet_y_400mf',
                                 'deeplabv3_regnet_y_8gf', 'deeplabv3plus_regnet_y_8gf',
                                 'deeplabv3_regnet_y_32gf', 'deeplabv3plus_regnet_y_32gf',
                                 'deeplabv3_vgg11_bn', 'deeplabv3plus_vgg11_bn',
                                 'deeplabv3_vgg16_bn', 'deeplabv3plus_vgg16_bn',
                                 'deeplabv3_vgg19_bn', 'deeplabv3plus_vgg19_bn',
                                 'deeplabv3_shufflenet_v2_x0_5', 'deeplabv3plus_shufflenet_v2_x0_5',
                                 'deeplabv3_shufflenet_v2_x1_0', 'deeplabv3plus_shufflenet_v2_x1_0',
                                 'deeplabv3_ghostnet_v2_1_0', 'deeplabv3plus_ghostnet_v2_1_0',
                                 'deeplabv3_ghostnet_v2_1_3', 'deeplabv3plus_ghostnet_v2_1_3',
                                 'deeplabv3_ghostnet_v2_1_6', 'deeplabv3plus_ghostnet_v2_1_6'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default="datasets/data/CameraBase/VOCdevkit/VOC2012/Output",
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=448) # swin-transformer, 7*x, for example, 448=7*64

    
    parser.add_argument("--ckpt", default='checkpoints/best_deeplabv3plus_regnet_y_32gf_voc_os16.pth', type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    
    if opts.dataset.lower() == 'voc':
        if opts.num_classes == None:
            opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.'+ext), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model
    model_map = {        
            'deeplabv3_resnet18': network.deeplabv3_resnet18,
            'deeplabv3plus_resnet18': network.deeplabv3plus_resnet18,
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet_v2': network.deeplabv3_mobilenet_v2,
            'deeplabv3plus_mobilenet_v2': network.deeplabv3plus_mobilenet_v2,
            'deeplabv3_mobilenet_v3_small': network.deeplabv3_mobilenet_v3_small,
            'deeplabv3plus_mobilenet_v3_small': network.deeplabv3plus_mobilenet_v3_small,
            'deeplabv3_mobilenet_v3_large': network.deeplabv3_mobilenet_v3_large,
            'deeplabv3plus_mobilenet_v3_large': network.deeplabv3plus_mobilenet_v3_large,
            'deeplabv3_berniwal_swintransformer_swin_t': network.deeplabv3_berniwal_swintransformer_swin_t,
            'deeplabv3_berniwal_swintransformer_swin_s': network.deeplabv3_berniwal_swintransformer_swin_s,
            'deeplabv3_berniwal_swintransformer_swin_b': network.deeplabv3_berniwal_swintransformer_swin_b,
            'deeplabv3_berniwal_swintransformer_swin_l': network.deeplabv3_berniwal_swintransformer_swin_l,        
            'deeplabv3plus_berniwal_swintransformer_swin_t': network.deeplabv3plus_berniwal_swintransformer_swin_t,
            'deeplabv3plus_berniwal_swintransformer_swin_s': network.deeplabv3plus_berniwal_swintransformer_swin_s,
            'deeplabv3plus_berniwal_swintransformer_swin_b': network.deeplabv3plus_berniwal_swintransformer_swin_b,
            'deeplabv3plus_berniwal_swintransformer_swin_l': network.deeplabv3plus_berniwal_swintransformer_swin_l,       
            'deeplabv3_microsoft_swintransformer_swin_t': network.deeplabv3_microsoft_swintransformer_swin_t,
            'deeplabv3_microsoft_swintransformer_swin_s': network.deeplabv3_microsoft_swintransformer_swin_s,
            'deeplabv3_microsoft_swintransformer_swin_b': network.deeplabv3_microsoft_swintransformer_swin_b,
            'deeplabv3_microsoft_swintransformer_swin_l': network.deeplabv3_microsoft_swintransformer_swin_l,
            'deeplabv3plus_microsoft_swintransformer_swin_t': network.deeplabv3plus_microsoft_swintransformer_swin_t,
            'deeplabv3plus_microsoft_swintransformer_swin_s': network.deeplabv3plus_microsoft_swintransformer_swin_s,
            'deeplabv3plus_microsoft_swintransformer_swin_b': network.deeplabv3plus_microsoft_swintransformer_swin_b,
            'deeplabv3plus_microsoft_swintransformer_swin_l': network.deeplabv3plus_microsoft_swintransformer_swin_l,
            'deeplabv3_hrnetv2_32': network.deeplabv3_hrnetv2_32,
            'deeplabv3plus_hrnetv2_32': network.deeplabv3plus_hrnetv2_32,
            'deeplabv3_hrnetv2_48': network.deeplabv3_hrnetv2_48,
            'deeplabv3plus_hrnetv2_48': network.deeplabv3plus_hrnetv2_48,
            'deeplabv3_xception': network.deeplabv3_xception,
            'deeplabv3plus_xception': network.deeplabv3plus_xception,
            'deeplabv3_regnet_y_400mf': network.deeplabv3_regnet_y_400mf,
            'deeplabv3plus_regnet_y_400mf': network.deeplabv3plus_regnet_y_400mf,
            'deeplabv3_regnet_y_8gf': network.deeplabv3_regnet_y_8gf,
            'deeplabv3plus_regnet_y_8gf': network.deeplabv3plus_regnet_y_8gf,
            'deeplabv3_regnet_y_32gf': network.deeplabv3_regnet_y_32gf,
            'deeplabv3plus_regnet_y_32gf': network.deeplabv3plus_regnet_y_32gf,
            'deeplabv3_vgg11_bn': network.deeplabv3_vgg11_bn,
            'deeplabv3plus_vgg11_bn': network.deeplabv3plus_vgg11_bn,
            'deeplabv3_vgg16_bn': network.deeplabv3_vgg16_bn,
            'deeplabv3plus_vgg16_bn': network.deeplabv3plus_vgg16_bn,
            'deeplabv3_vgg19_bn': network.deeplabv3_vgg19_bn,
            'deeplabv3plus_vgg19_bn': network.deeplabv3plus_vgg19_bn,
            'deeplabv3_shufflenet_v2_x0_5': network.deeplabv3_shufflenet_v2_x0_5,
            'deeplabv3plus_shufflenet_v2_x0_5': network.deeplabv3plus_shufflenet_v2_x0_5,
            'deeplabv3_shufflenet_v2_x1_0': network.deeplabv3_shufflenet_v2_x1_0,
            'deeplabv3plus_shufflenet_v2_x1_0': network.deeplabv3plus_shufflenet_v2_x1_0,
            'deeplabv3_ghostnet_v2_1_0': network.deeplabv3_ghostnet_v2_1_0,
            'deeplabv3plus_ghostnet_v2_1_0': network.deeplabv3plus_ghostnet_v2_1_0,
            'deeplabv3_ghostnet_v2_1_3': network.deeplabv3_ghostnet_v2_1_3,
            'deeplabv3plus_ghostnet_v2_1_3': network.deeplabv3plus_ghostnet_v2_1_3,
            'deeplabv3_ghostnet_v2_1_6': network.deeplabv3_ghostnet_v2_1_6,
            'deeplabv3plus_ghostnet_v2_1_6': network.deeplabv3plus_ghostnet_v2_1_6
            }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = transforms.Compose([
                transforms.Resize(opts.crop_size),
                transforms.CenterCrop(opts.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((opts.crop_size, opts.crop_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    transform_resize = transforms.Resize((opts.crop_size, opts.crop_size))

    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            img_name = os.path.basename(img_path).split('.')[0]
            img = Image.open(img_path).convert('RGB')
            img2 = transform(img).unsqueeze(0) # To tensor of NCHW
            img2 = img2.to(device)
            
            pred = model(img2).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)

            img2 = transform_resize(img)
            colorized_preds = Image.blend(img2, colorized_preds, 0.5)

            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name+'.png'))

if __name__ == '__main__':
    main()
