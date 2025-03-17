import os
import time
from tqdm import tqdm
import argparse
import subprocess
import torch

from einops import rearrange

from vit_models.t2t_vit import t2t_vit_t_24
from imagenet import ImagenetDataModule
import timm

MODELPATH="please-add-your-model-path"
DATASETPATH="please-add-your-dataset-path"

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int, default=1)
    parser.add_argument("--attn_method", type=str, default='full')
    parser.add_argument("--fp16",action="store_true")
    args = parser.parse_args()

    name = f"{args.attn_method}"
    print(f"attn_method name: {name}")

    model = t2t_vit_t_24(**{
        'drop_rate': 0.0, 'drop_path_rate': 0.1, 'img_size': 224,
        't2tattn1_cfg' : {'name': args.attn_method, 'sample_size': 98, 'bucket_size': 32},
        't2tattn2_cfg' : {'name': args.attn_method, 'sample_size': 24, 'bucket_size': 32},
    })

    model.load_state_dict(torch.load(os.path.join(MODELPATH, "82.6_T2T_ViTt_24.pth.tar")), strict=True)

    kwargs = {
        'data_dir': DATASETPATH,
        'shuffle': True, 'batch_size': 128, 'batch_size_eval': 256, 
        'num_workers': 8, 'pin_memory': True, 'image_size': 224
    }
    datamodule = ImagenetDataModule(**kwargs)
    datamodule.prepare_data()
    datamodule.setup()
    datamodule.dataset_val.transform = timm.data.create_transform(input_size=224, interpolation='bicubic', crop_pct=0.9)

    device = 'cuda'
    if args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"dtype: {dtype}")
    model.eval()
    model = model.to(device=device, dtype=dtype)
    tic = time.time()
    cnt = 0
    corrects = 0
    pbar = tqdm(datamodule.val_dataloader())
    for images, labels in pbar:
        images = images.to(device=device, dtype=dtype)
        out = model(images)

        corrects += (out.detach().cpu().argmax(-1) == labels).sum().item()
        cnt += len(labels)
        pbar.set_description(f"{corrects} / {cnt}")

    toc = time.time() - tic
    accuracy = float(corrects) / cnt
    print(f"corrects: {corrects}, cnt: {cnt}")
    res_str = f"[{name:<10}] dtype: {dtype}, time: {toc:.4f}, accuracy: {accuracy:.8f}, corrects: {corrects}, seed: {args.seed}\n"
    print(res_str)

if __name__ == "__main__":
    main()
