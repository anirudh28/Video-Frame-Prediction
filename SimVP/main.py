import argparse
from exp import Exp
from segformer_predict import *

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./outputs', type=str)
    parser.add_argument('--ex_name', default='simvp', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/moving_objects/')
    parser.add_argument('--dataname', default='moving_objects')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--is_train',default=True, type=bool)
    parser.add_argument('--predict',default=False, type=bool)
    parser.add_argument('--model_path', default='./outputs/simvp/checkpoint.pth', type=str)
    parser.add_argument('--model2_path', default='./outputs/simvp/segformer.pt', type=str)

    # model parameters
    parser.add_argument('--in_shape', default=[11, 3, 160, 240], type=int,nargs='*')  
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=512, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    if not config['predict']:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.train(args)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  end training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    else:
        exp.config['is_train'] = False
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start prediction <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.predict(args)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  end prediction <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
