from timm import create_model
import torch
import time
import numpy as np
import argparse
import models_mamba


def throughput(name="vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2", 
               bs=16,
               resolution=224,
               ckpt_path=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    model = create_model(
    name,
    pretrained=False,
    ckpt_path=ckpt_path,
    img_size=resolution
)
    
    input_data = form_data(bs=bs, resolution=resolution)
    do_throughoput_eval(model, input_data)

@torch.no_grad()
def do_throughoput_eval(model, input_data):
    model.eval()
    model.cuda()
    
    #Warmup
    runs=30
    with torch.cuda.amp.autocast():
        for ii in range(runs):
            with torch.no_grad():
                output = model(input_data)

    timer = []
    start_time = time.time()
    runs=50
    with torch.cuda.amp.autocast(True):

        for ii in range(runs):
            start_time_loc = time.time()
            with torch.no_grad():
                output = model(input_data)

            timer.append(time.time()-start_time_loc)
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"Throughput {input_data.shape[0] * 1.0 / ((end_time - start_time) / runs)}")
    print(f"Throughput Med {int(input_data.shape[0] * 1.0 / ((np.median(timer))))}")

    
def form_data(bs=16, resolution=224):
    input_data = torch.randn((bs, 3, resolution, resolution), device='cuda').cuda()
    return input_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name", type=str,
                        default="vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2")
    parser.add_argument("--ckpt", help="checkpoint path",type=str,
                        default='vim_t_midclstok_76p1acc.pth')
    parser.add_argument("--resolution", help="model resolution",type=int,
                        default=224)
    parser.add_argument("--bs", help="batch size",type=int,
                        default=16)
    args = parser.parse_args()
    throughput(name=args.model, bs=args.bs, resolution=args.resolution, ckpt_path=args.ckpt)