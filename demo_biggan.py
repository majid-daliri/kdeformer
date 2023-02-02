import argparse
import os
import torch
import time
import numpy as np
from tqdm import tqdm
from biggan_models.model import BigGAN
from biggan_models.model_performer import PerformerBigGAN
from biggan_models.model_reformer import ReformerBigGAN
from biggan_models.model_kdeformer import KDEformerBigGAN
from biggan_models.utils import one_hot_from_int, truncated_noise_sample, save_as_images


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str, default='biggan-deep-512')
    parser.add_argument("--num_classes",type=int, default=1000)
    parser.add_argument("--num_outputs",type=int, default=-1)
    parser.add_argument("--data_per_class",type=int, default=1)
    parser.add_argument("--seed",type=int, default=123)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--attention",type=str, default='exact', choices=['exact', 'kdeformer', 'performer', 'reformer', 'sblocal'])
    parser.add_argument("--truncation",type=float, default=0.4)
    parser.add_argument("--no_store",action='store_true')    
    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()

    if args.attention == 'sblocal':
        try:
            from src.models.attention.sblocal_attention import SBLocalAttention
            from biggan_models.model_sblocal import SBlocalBigGAN
        except:
            print("ScatterBrain is not installed.")
            quit(-1)

    for args_name, args_value in args.__dict__.items():
        print(f"{args_name}: {args_value}")

    model_name = args.model_name
    num_classes = args.num_classes
    data_per_class = args.data_per_class
    batch_size = args.batch_size
    attention = args.attention
    truncation = args.truncation

    # Load pre-trained model tokenizer (vocabulary)
    if attention == 'exact':
        model = BigGAN.from_pretrained(model_name)
    elif attention == 'kdeformer':
        model = KDEformerBigGAN.from_pretrained(model_name)
    elif attention == 'performer':
        model = PerformerBigGAN.from_pretrained(model_name)
    elif attention == 'reformer':
        model = ReformerBigGAN.from_pretrained(model_name)
    elif attention == 'sblocal':
        model = SBlocalBigGAN.from_pretrained(model_name)
    else:
        raise NotImplementedError("Invalid attention option")

    print(model.__class__)

    # Prepare a input
    if args.data_per_class > 0:
        labels = np.repeat(np.arange(num_classes), data_per_class).tolist()
    elif args.num_outputs > 0:
        labels = np.random.randint(num_classes, size=(args.num_outputs,))

    class_vector = one_hot_from_int(labels, batch_size=len(labels))
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=len(labels), seed=args.seed)

    # All in tensors
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    if torch.cuda.is_available():
        # If you have a GPU, put everything on cuda
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')
        model = model.to('cuda')

    tic = time.time()
    model.eval()
    output_all = []
    num_batches = len(labels) // batch_size + 1
    for idx in tqdm(range(num_batches)):
        batch_idx = list(range(idx * batch_size, min(len(labels), (idx+1) * batch_size)))
        if len(batch_idx) == 0:
            continue
        # res_all.append(batch_idx)

        n_vec = noise_vector[batch_idx]
        c_vec = class_vector[batch_idx]

        # Generate an image
        output = model(n_vec, c_vec, truncation)
        output = output.to('cpu')

        output_all.append(output)

    time_generation = time.time() - tic

    output_all = torch.cat(output_all)
    print(f"output_all.shape: {output_all.shape}")
    print(f"generation time : {time_generation:.4f} sec")

    if not args.no_store:
        output_path = f"./generations/{model_name.replace('-','_')}/{attention}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tic = time.time()
        print("saving images....")
        save_as_images(output_all, output_path + "/img")
        print(f"done. ({time.time() - tic:.4f} sec)")


if __name__ == "__main__":
    main()