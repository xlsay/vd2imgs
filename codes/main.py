# coding=utf-8
import argparse
import os
import cv2
from vd2imgs import multiprocess_extract


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='dir_of_your_vds')
    parser.add_argument('--save-dir', type=str, required=True, help='dir_of_your_imgs_want_to_save_in') 
    parser.add_argument('--vd-type', '--vd', nargs='+', type=str, required=True, help='video types:--vd *.mp4, or --vd *.mp4 *.wmv') 
    parser.add_argument('--n-work', type=int, default=2, help='number of extracting process') 
    parser.add_argument('--ssim-threshhold', type=float, default=0.75, help='ssim threshhold') 
    params = parser.parse_args()
    return params


if __name__ == '__main__':
    params = parse_params()
    multiprocess_extract(params.source, params.save_dir, params.vd_type, \
                        params.ssim_threshhold, params.n_work)