import time
import torch
import sys
from KDEformer import KDEformer
from Performer import PerformerAttention
from Reformer import ReformerAttention

try:
    from src.models.attention.sblocal_attention import SBLocalAttention
    SB_INSTALLED = True
except:
    print("ScatterBrain is not installed.")
    SB_INSTALLED = False

from functools import partial


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_glove(seq_len=8192, seed=1):

    data = open("glove.twitter.27B.100d.txt", "r").read().split("\n")
    batch_size, head_size = 8, 1
    max_seq_len = 16384
    data_all = []
    for i in range(batch_size * head_size * max_seq_len * 3):
        data_i = [float(_x) for _x in data[i].split()[1:]]
        if len(data_i) != 100:
            data_i = [float(_x) for _x in data[i].split()]
        data_all.append(data_i)
    data_all = torch.tensor(data_all).double()

    query = data_all[:batch_size*head_size*max_seq_len]
    query = query.reshape(batch_size, max_seq_len, head_size, -1)
    value = data_all[2*batch_size*head_size*max_seq_len:]
    value = value.reshape(batch_size, max_seq_len, head_size, -1)
    del data, data_all

    query = query[:,:seq_len,:,:]
    value = value[:,:seq_len,:,:]

    normalizer = 10**0.25
    query /= normalizer
    key = query.clone()

    return query, key, value


class ExactAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        qk = torch.einsum("bhnd,bhmd->bhnm", query.transpose(1,2), key.transpose(1,2))
        attn = torch.softmax(qk, dim=-1)
        return torch.einsum('bhnm,bhmd->bhnd', attn, value.transpose(1,2))


@torch.no_grad()
def run(seq_len=8192, device='cpu', seed=1, num_feats=256):

    query, key, value = get_glove(seq_len)
    query, key, value = map(lambda _x: _x.to(device).double(), (query, key, value))
    
    tic0 = time.time()
    set_random_seed(seed)

    batch_size = query.shape[0]
    head_size = query.shape[2]
    
    exact_attn = ExactAttention()
    inputs = (query, key, value)

    # 1. Exact attention
    method = 'Exact'
    tic = time.time()
    out_exact = exact_attn(query, key, value)
    tim_exact = time.time() - tic
    
    print(f"[TIM] computing exact attention           | {tim_exact:<12.8f} sec  | ")

    out_exact = out_exact.to('cpu')
    err_normalizer = torch.linalg.norm(out_exact, ord=2, dim=(2, 3))
    def compute_rel_err(A):
        return torch.mean(torch.linalg.norm(A - out_exact, ord=2, dim=(2,3)) / err_normalizer)

    # 2. Reformer attention
    method = 'Reformer'
    if query.shape[1] != value.shape[1]:
        print(f"Reformer is not available for this input.")
        tim_rfm, err_rfm, reformer_flops, reformer_mems = -1, -1, -1, -1
    else:
        reformer = ReformerAttention(bucket_size=num_feats//2).to(device)
        reformer_call = partial(reformer, qk=query, k=None, v=value)
        out_rfm, _ = reformer_call()
        tim_rfm = time.time() - tic

        out_rfm = out_rfm.unsqueeze(1).to('cpu')
        err_rfm = compute_rel_err(out_rfm).item()
        print(f"{method:<16}      | {num_feats} | {err_rfm:.10f} | {tim_rfm:<12.8f} sec  | ")
        del out_rfm


    # 3. Performer attetion
    method = 'Performer'
    performer = PerformerAttention(num_feats=num_feats).to(device)
    performer_call = partial(performer, query=query, key=key, value=value)

    tic = time.time()
    out_pfm = performer_call()
    tim_pfm = time.time() - tic

    out_pfm = out_pfm.to('cpu')
    err_pfm = compute_rel_err(out_pfm).item()
    print(f"{method:<16}      | {num_feats} | {err_pfm:.10f} | {tim_pfm:<12.8f} sec  | ")
    del out_pfm


    # 4. ScatterBrain (sblocal)
    if SB_INSTALLED:
        method = 'ScatterBrain'
        local_context = num_feats // 2
        tic = time.time()
        sblocal_attn = SBLocalAttention(local_context=local_context, dim_heads=query.shape[-1], nb_features=num_feats, softmax_temp=1.).to(device)
        sblocal_call = partial(sblocal_attn, query=query, key=key, value=value)
        out_sbn, _ = sblocal_call()
        tim_sbn = time.time() - tic

        out_sbn = out_sbn.transpose(1,2)
        out_sbn = out_sbn.to('cpu')
        err_sbn = compute_rel_err(out_sbn).item()
        print(f"{method:<16}      | {local_context + num_feats} | {err_sbn:.10f} | {tim_sbn:<12.8f} sec  | ")
        del out_sbn

    # 5. Our KDEformer
    method = 'KDEformer'
    sample_size = num_feats
    mask_size = batch_size * head_size * sample_size * query.shape[1]
    kde_attn = KDEformer(num_projs=7, Bucket_size=num_feats//2 , sample_size=num_feats, mask_size=mask_size).to(device)
    kde_call = partial(kde_attn, query=query, key=key, value=value)

    tic = time.time()
    out_our = kde_call()
    tim_our = time.time() - tic

    out_our = out_our.to('cpu')
    err_our = compute_rel_err(out_our).item()
    del out_our
    print(f"{method:<16}      | {int(1.5 * num_feats)} | {err_our:.10f} | {tim_our:<12.8f} sec  | ")
    print()

    


def varying_num_feats(
        seq_len=8192,
        num_iters=1,
        num_feats_all=[64,  128, 256,  512, 1024, 2048, 4096],
        calc_flops=False,
        calc_memory=False,
        debug=False
    ):
    it = 0
    tic00 = time.time()
    for seed in range(num_iters):
        print(f"iter: {it} | ", end='')
        print(f" seed: {seed}")
        results = {}
        for bs in num_feats_all:
            print(f"seq_len: {seq_len}, bucket_size: {bs}, seed: {seed}")
            # results[bs] = main(seq_len=seq_len, bucket_size=bs, seed=seed)
            results[bs] = run(seq_len=seq_len, seed=seed, num_feats=bs, query_key_same=True)
        print(f"elapsed time: {time.time() - tic00:.4f} sec")
        print(f"[iter {it} is done.]")
        it += 1

if __name__ == "__main__":
    run()
    # varying_num_feats()