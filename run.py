#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Renwu Gao
@Contact: re.gao@szu.edu.cn
@File: main.py
@Time: 2021/03/04 10:39 PM
"""

from __future__ import print_function
import os
os.environ['OMP_NUM_THREADS'] = '8'
import sys
sys.path.append(os.getcwd())
sys.path.append('/home/xiaox/studio/usr/lib')
import argparse
import warnings
warnings.simplefilter('ignore',category=UserWarning)
import importlib
# import json


def recognize(argv,args):
    module = importlib.import_module('models.rec.{}'.format(argv[0]))
    model = module.get()
    parser,_ = model.get_argument_parser()
    margs = parser.parse_args(argv[1:])
    # if args.config is not None:
    #     if not os.path.exists(args.config):
    #         config = json.loads(args.config)
    #     else:
    #         with open(args.config) as fp:
    #             config = json.load(fp,args.config)
    #     update_args(config,args,margs)
    margs.func(args,margs)


def segment(argv,args):
    module = importlib.import_module('models.seg.{}'.format(argv[0]))
    model = module.get()
    parser,_ = model.get_argument_parser()
    margs = parser.parse_args(argv[1:])
    # if args.config is not None:
    #     if not os.path.exists(args.config):
    #         config = json.loads(args.config)
    #     else:
    #         with open(args.config) as fp:
    #             config = json.load(fp,args.config)
    #     update_args(config,args,margs)
    margs.func(args,margs)


def get_argument_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Analysis')
    parser.add_argument('--batch-size', type=int, default=32, metavar='batch-size',help='size of batch')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--num-points', type=int, default=1024, help='number of points for jittor.dataset.Dataset')
    parser.add_argument('--num-workers',type=int,default=8,help='number of workers for jittor.dataset.Dataset')
    parser.add_argument('--ipo', type=int, default=1,help='iterations per optimization')
    parser.add_argument('--weights-path', type=str, default='', metavar='N',help='pretrained model path')
    parser.add_argument('--gpus',type=str,default='0')
    parser.add_argument('--loss',type=str,default='cce')
    parser.add_argument('--dataset-root',type=str,default=None,required=True)
    parser.add_argument('--config',type=str,default=None)

    subparser = parser.add_subparsers()

    subparser_rec = subparser.add_parser('rec')
    subparser_rec.set_defaults(func=lambda argv,args:recognize(argv,args))

    subparser_seg = subparser.add_parser('seg')
    subparser_seg.set_defaults(func=lambda argv,args:segment(argv,args))

    return parser


if __name__ == "__main__":
    parser = get_argument_parser()
    args,remains = parser.parse_known_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    from sigma.nn.jittor.utils import set_seed
    set_seed(args.seed)
    import jittor as jt
    jt.flags.use_cuda = 1 # run in CUDA mode
    args.func(remains,args)
