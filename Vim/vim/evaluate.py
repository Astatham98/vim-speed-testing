import argparse
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from datasets import build_dataset
from engine import evaluate
from contextlib import suppress
import utils
import models_mamba

def load_dataload(dataset_val, sampler_val, batch_size, num_workers, pin_mem):
    data_loader_val = torch.utils.data.DataLoader(
    dataset_val, sampler=sampler_val,
    batch_size=int(1.5 * batch_size * 20),
    num_workers=num_workers,
    pin_memory=pin_mem,
    drop_last=False
    )
    return data_loader_val

def evaluate_model(model, data_loader_val, dataset_val, device, amp_autocast):
    test_stats = evaluate(data_loader_val, model, device, amp_autocast)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    # test_stats = evaluate(data_loader_val, model_ema.ema, device, amp_autocast)
    # print(f"Accuracy of the ema network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

def load_model_from_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    return model.to(device)
    # TODO add ema loading here also

def get_args():
    parser = argparse.ArgumentParser('VIM training and evaluation script', add_help=False)
    parser.add_argument('--data-path', default='../imagenet-val', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IMNET', type=str, choices=['CIFAR', 'IMNET', 'INAT'],
                        help='dataset name')
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)  
    parser.add_argument('--amp', action='store_true', default=False, help='Use torch AMP for mixed precision training') 
    parser.add_argument('--ckpt', default='vim_t_midclstok_76p1acc.pth', help='resume from checkpoint | tiny by default')
    parser.add_argument('--model', default='vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2', type=str, metavar='MODEL')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help='evaluation crop ratio') # Might change this to just be 1.0 for evaluation, but for now keep it as is
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    dataset_val, nb_classes = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = load_dataload(dataset_val, sampler_val, args.batch_size, args.num_workers, args.pin_mem)

    model = create_model(
        args.model,
        pretrained=True,
        num_classes=nb_classes,
        ckpt_path=args.ckpt,
        img_size=args.input_size
    )
    
    model.to(device)
    
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    
    # model = load_model_from_checkpoint(model.copy(), args.ckpt, device)

    start = time.time()
    evaluate_model(model, data_loader_val, dataset_val, device, amp_autocast)
    end = time.time() - start
    print(f"Evaluation time took: {end:.2f} seconds, for images of size {args.input_size}x{args.input_size} and batch size {args.batch_size}.")