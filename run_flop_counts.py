import time
import numpy as np
import sys
import torch
# from src.models.attention.performer_attention import PerformerAttention
# from src.models.attention.reformer_attention import ReformerAttention

from Performer import PerformerAttention
from Reformer import ReformerAttention
from KDEformer import KDEformer

try:
    from src.models.attention.sblocal_attention import SBLocalAttention
    from biggan_models.model_sblocal import SBlocalBigGAN
except:
    print("ScatterBrain is not installed.")
    quit(-1)


from fvcore.nn import FlopCountAnalysis, flop_count_table
from typing import Any, Callable, List
from numbers import Number
from functools import partial


class ExactAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        qk = torch.einsum("bhnd,bhmd->bhnm", query.transpose(1,2), key.transpose(1,2))
        attn = torch.softmax(qk, dim=-1)
        return torch.einsum('bhnm,bhmd->bhnd', attn, value.transpose(1,2))



# Transcribed by https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/jit_handles.py
def local_product_flop_jit(inputs: List[Any], outputs: List[Any], local_context:int) -> Number:
    """
    Count flops for local_dot_product
    """
    batch_size, head_size, seq_len, embed_dim = inputs[0].type().sizes()
    flops = batch_size * head_size * seq_len * embed_dim * local_context
    return flops


def local_weighted_average_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for local_weighted_average
    """
    batch_size, head_size, seq_len, embed_dim = inputs[0].type().sizes()
    _, _, _, embed_dim2 = inputs[1].type().sizes()
    flops = batch_size * head_size * seq_len * embed_dim * embed_dim2
    return flops


def linalg_qr_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for linalg_qr
    """
    # For m-by-n matrices, the QR decomposition requires 2(m-n/3)*n**2 flops.
    # See https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/09-num-lin-alg-scribed.pdf
    batch_size, m, n = inputs[0].type().sizes()
    flops = int(2*(m - n/3.)*n**2) * batch_size
    return flops


def get_flops(model, inputs, local_context=-1):
    flops = FlopCountAnalysis(model, inputs)
    # Ignore some basic operations
    manual_counter = {  
        "aten::div": None,
        "aten::mul": None,
        "aten::neg": None,
        "aten::add": None,
        "aten::sub": None,
        "aten::exp": None,
        "aten::sum": None,
        "aten::pow": None,
        "aten::logsumexp": None,
        "aten::lt": None,
        "aten::lt": None,
        'aten::le': None,
        "aten::ge": None,
        "aten::argmax": None,
        "aten::randn": None,
        "aten::linalg_qr": linalg_qr_jit,
        "aten::linalg_norm": None
    }

    if local_context > 0:
        manual_counter["prim::PythonOp.LocalDotProduct"] = partial(local_product_flop_jit, local_context=local_context)
        manual_counter["prim::PythonOp.LocalWeightedAverage"] = local_weighted_average_jit

    flops.set_op_handle(**manual_counter)
    return flops.total()


def get_sblocal_flops(local_context, nb_features, N, D, batch_size, M=None, E=None):
    M = M or N  # if M is None then M = N
    E = E or D

    sblocal_cflops = 0
    sblocal_cflops += (N + M) * D * (nb_features//2) # random projection
    sblocal_cflops += N
    sblocal_cflops += N * nb_features  # linear_attention_normalization
    sblocal_cflops += (N + M) * E * nb_features  # linear_attention
    sblocal_cflops += N*D*local_context + N*nb_features*local_context  # local_dot_product
    # sblocal_cflops += (N*local_context) -> we skip this
    sblocal_cflops += N * local_context * D
    sblocal_cflops *= batch_size
    sblocal_cflops += (4/3)*D**3 # for QR decomposition (not counted)
    return sblocal_cflops


def get_kde_flops(num_projs, sample_size, bucket_size, N, D, batch_size, M=None, E=None):
    M = M or N  # if M is None then M = N
    E = E or D

    # 1. For hashing
    robust_cflops = N*D*num_projs*2
    
    # 2. A_sparse
    num_blocks = N // bucket_size
    robust_cflops += num_blocks * (bucket_size**2 * D) * 2
    robust_cflops += (E**2) * M

    # 3. A_res
    robust_cflops += (N * D * sample_size)*2 # Q @ V_pi = A_pi, A_pi @ V_pi^T
    robust_cflops += N * sample_size # masking
    return robust_cflops * batch_size


class BatchReformer(torch.nn.Module):

    def __init__(self, bucket_size, n_hashes):
        super().__init__()
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes

    def forward(self, query, key, value):
        att = []
        for i in range(4):
            # reformer = ReformerAttention(softmax_temp=1., bucket_size=64, n_hashes=1)
            att_i = ReformerAttention(bucket_size=self.bucket_size, n_hashes=self.n_hashes).forward(
                k=None,
                qk=query[:,1024*i:1024*(i+1),:,:],
                v=value
            )[0]
            att.append(att_i)
        return torch.cat(att, 1).transpose(1,2)



class BatchSBN(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        att = []
        for i in range(4):
            # reformer = ReformerAttention(softmax_temp=1., bucket_size=64, n_hashes=1)
            att_i = SBLocalAttention(local_context=32, dim_heads=query.shape[-1], nb_features=128, softmax_temp=1.)(
                query=query[:,1024*i:1024*(i+1),:,:],
                key=key,
                value=value
            )[0]
            att.append(att_i)
        return torch.cat(att, 1).transpose(1,2)


@torch.no_grad()
def flops_biggan_only_attention(num_iters=10):
    from tqdm import tqdm
    from demo_cuda import seed_cpu_cuda, get_qkv, ExactAttention
    
    flops_all = {
        'exact':[],
        'performer':[],
        'reformer':[],
        'sblocal':[],
        'kde':[]
    }
    
    for i in tqdm(range(num_iters)):
        query, key, value = get_qkv('biggan', -1, seed=i) # [8, 4096, 1, 64],[8, 1024, 1, 64], [8, 1024, 1, 256]
        query, key, value = map(lambda _x: _x.double(), (query, key, value))
        
        exact_attn = ExactAttention()
        exact_flops = get_flops(exact_attn, (query, key, value))

        performer = PerformerAttention(num_feats=128)
        performer_flops = get_flops(performer, (query, key, value))

        reformer = BatchReformer(bucket_size=64, n_hashes=8)
        reformer_flops = get_flops(reformer, (query, None, value))

        sblocal = BatchSBN()
        sblocal_flops = get_flops(sblocal, (query, key, value), local_context=32)

        kde = KDEformer(num_projs=7, Bucket_size=64, sample_size=128)
        kde_flops = get_flops(kde, (query, key, value))

        flops_all['exact'].append(exact_flops)
        flops_all['performer'].append(performer_flops)
        flops_all['reformer'].append(reformer_flops)
        flops_all['sblocal'].append(sblocal_flops)
        flops_all['kde'].append(kde_flops)

        print("="*20 + f"iter {i} " + "="*20)
        print(f"{'Exact':<16} flops : {exact_flops:.5g} ({exact_flops/1e9:.3f} GFLOPS)")
        print(f"{'Performer':<16} flops: {performer_flops:.5g}  ({performer_flops/1e9:.3f} GFLOPS) | {exact_flops/performer_flops}")
        print(f"{'Reformer':<16} flops: {reformer_flops:.5g} ({reformer_flops/1e9:.3f} GFLOPS) | {exact_flops/reformer_flops}")
        print(f"{'ScatterBrain':<16} flops: {sblocal_flops:.5g} ({sblocal_flops/1e9:.3f} GFLOPS) | {exact_flops/sblocal_flops}")
        print(f"{'KDEformer':<16} flops: {kde_flops:.5g} ({kde_flops/1e9:.3f} GFLOPS) | {exact_flops/kde_flops}")
        print("="*200 + '\n')

    exact_flops = np.mean(flops_all['exact'])
    performer_flops = np.mean(flops_all['performer'])
    reformer_flops = np.mean(flops_all['reformer'])
    sblocal_flops = np.mean(flops_all['sblocal'])
    kde_flops = np.mean(flops_all['kde'])

    res_str = f"{'Exact':<16} flops : {exact_flops:.5g} ({exact_flops/1e9:.3f} GFLOPS)\n"+\
        f"{'Performer':<16} flops: {performer_flops:.5g}  ({performer_flops/1e9:.3f} GFLOPS) | {exact_flops/performer_flops}\n"+\
        f"{'Reformer':<16} flops: {reformer_flops:.5g} ({reformer_flops/1e9:.3f} GFLOPS) | {exact_flops/reformer_flops}\n"+\
        f"{'ScatterBrain':<16} flops: {sblocal_flops:.5g} ({sblocal_flops/1e9:.3f} GFLOPS) | {exact_flops/sblocal_flops}\n"+\
        f"{'KDEformer':<16} flops: {kde_flops:.5g} ({kde_flops/1e9:.3f} GFLOPS) | {exact_flops/kde_flops}\n"+\
        f"Date: {time.strftime('%y%m%d%H%M')}"

    print("\nTotal\n")
    print(res_str)
    

@torch.no_grad()
def flops_t2tvit_only_attention(num_iters=10):
    from tqdm import tqdm
    from demo_cuda import seed_cpu_cuda, get_qkv, ExactAttention
    
    flops_all = {
        'exact':[],
        'performer':[],
        'reformer':[],
        'sblocal':[],
        'kde':[]
    }
    
    for i in tqdm(range(num_iters)):
        seed_cpu_cuda(i)

        seq_len = 3136
        query = torch.randn(128, seq_len, 1, 64)
        key = torch.randn(128, seq_len, 1, 64)
        value = torch.randn(128, seq_len, 1, 64)

        query, key, value = map(lambda _x: _x.double(), (query, key, value))
        
        exact_attn = ExactAttention()
        exact_flops = get_flops(exact_attn, (query, key, value))

        performer = PerformerAttention(num_feats=49)
        performer_flops = get_flops(performer, (query, key, value))

        reformer = ReformerAttention(bucket_size=49, n_hashes=2)
        reformer_flops = get_flops(reformer, (query, None, value))

        lc = 49
        nf = 48
        sblocal = SBLocalAttention(local_context=lc, dim_heads=64, nb_features=nf, softmax_temp=1.)
        sblocal_flops = get_flops(sblocal, (query, key, value), local_context=lc)

        sz = 98
        bs = 32
        kde = KDEformer(num_projs=7, Bucket_size=bs, sample_size=sz)
        kde_flops = get_flops(kde, (query, key, value))

        print(f"sblocal_flops: {sblocal_flops/1e9}, kde_flops: {kde_flops/1e9}" )

        flops_all['exact'].append(exact_flops)
        flops_all['performer'].append(performer_flops)
        flops_all['reformer'].append(reformer_flops)
        flops_all['sblocal'].append(sblocal_flops)
        flops_all['kde'].append(kde_flops)

        print("="*20 + f"iter {i} " + "="*20)
        print(f"{'Exact':<16} flops : {exact_flops:.5g} ({exact_flops/1e9:.3f} GFLOPS)")
        print(f"{'Performer':<16} flops: {performer_flops:.5g}  ({performer_flops/1e9:.3f} GFLOPS) | {exact_flops/performer_flops}")
        print(f"{'Reformer':<16} flops: {reformer_flops:.5g} ({reformer_flops/1e9:.3f} GFLOPS) | {exact_flops/reformer_flops}")
        print(f"{'ScatterBrain':<16} flops: {sblocal_flops:.5g} ({sblocal_flops/1e9:.3f} GFLOPS) | {exact_flops/sblocal_flops}")
        print(f"{'KDEformer':<16} flops: {kde_flops:.5g} ({kde_flops/1e9:.3f} GFLOPS) | {exact_flops/kde_flops}")
        print("="*200 + '\n')

    exact_flops = np.mean(flops_all['exact'])
    performer_flops = np.mean(flops_all['performer'])
    reformer_flops = np.mean(flops_all['reformer'])
    sblocal_flops = np.mean(flops_all['sblocal'])
    kde_flops = np.mean(flops_all['kde'])

    res_str = f"{'Exact':<16} flops : {exact_flops:.5g} ({exact_flops/1e9:.3f} GFLOPS)\n"+\
        f"{'Performer':<16} flops: {performer_flops:.5g}  ({performer_flops/1e9:.3f} GFLOPS) | {exact_flops/performer_flops}\n"+\
        f"{'Reformer':<16} flops: {reformer_flops:.5g} ({reformer_flops/1e9:.3f} GFLOPS) | {exact_flops/reformer_flops}\n"+\
        f"{'ScatterBrain':<16} flops: {sblocal_flops:.5g} ({sblocal_flops/1e9:.3f} GFLOPS) | {exact_flops/sblocal_flops}\n"+\
        f"{'KDEformer':<16} flops: {kde_flops:.5g} ({kde_flops/1e9:.3f} GFLOPS) | {exact_flops/kde_flops}\n"+\
        f"Date: {time.strftime('%y%m%d%H%M')}"

    print("\nTotal\n")
    print(res_str)


if __name__ == "__main__":
    flops_t2tvit_only_attention()
