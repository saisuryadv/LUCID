# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
from einops import reduce

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.lucid_path.cumprod_householder_bwd import chunk_cumprod_householder_bwd_fn
from fla.ops.lucid_path.cumprod_householder_fwd import chunk_cumprod_householder_fwd_fn
from fla.ops.lucid_path.intra_chunk_preprocess_bwd import intra_chunk_preprocess_bwd_fn
from fla.ops.lucid_path.intra_chunk_preprocess_bwd_prepare import intra_chunk_preprocess_bwd_prepare_fn
from fla.ops.lucid_path.parallel_path_bwd_inter_dkv import parallel_path_bwd_dkv_fn
from fla.ops.lucid_path.parallel_path_bwd_inter_dqh import parallel_path_bwd_dq_fn
from fla.ops.lucid_path.parallel_path_bwd_intra import parallel_path_bwd_intra_chunk_fn
from fla.ops.lucid_path.prepare_k_cache import prepare_k_cache_fn
from fla.ops.lucid_path.transform_q import transform_q_fwd_fn
from fla.ops.lucid_path.triangular_solve_fwd import triangular_solve_fwd_fn
# from fla.ops.lucid_path.triangular_solve_bwd_dv_recompute import triangular_solve_bwd_dv_fn_recompute
from fla.ops.lucid_path.triangular_solve_bwd_dv_reuse import triangular_solve_bwd_dv_fn_reuse
from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.solve_tril import solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class ParallelLUCIDPathAttentionFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, beta, g, scale, cu_seqlens, use_cache=False):
        # Gating (commented out for now)
        # g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, output_dtype=torch.float32) if g is not None else None
        g_cumsum = None

        # Block size = 64
        BT = 64

        # Compute T_inv = (I + diag(beta) * stril(w @ w.T))^{-1}
        A = chunk_scaled_dot_kkt_fwd(
            k=w,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            output_dtype=torch.float32
        )
        A = solve_tril(
            A=A,
            cu_seqlens=cu_seqlens,
            output_dtype=w.dtype
        )

        # Triangular solve forward: (I + L) O = V
        q_new, k_new, w2, o = triangular_solve_fwd_fn(
            q=q,
            k=k,
            v=v,
            w=w,
            beta=beta,
            # g_cumsum=g_cumsum,
            A=A,
            scale=scale,
            BT=BT,
            cu_seqlens=cu_seqlens,
        )

        # Cache for decoding
        k_cache = prepare_k_cache_fn(k=k_new, w1=w, w2=w2, cu_seqlens=cu_seqlens, BS=BT, use_cache=use_cache)

        # Save for backward
        # ctx.save_for_backward(q, k, v, w, g_cumsum, o, beta, A)
        ctx.save_for_backward(q, k, v, w, g_cumsum, o, beta, A, q_new, k_new, w2)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens

        return o, k_cache

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dk_new):
    # def backward(ctx, do, dk_new):
        # q, k, v, w, g_cumsum, o, beta, A = ctx.saved_tensors
        q, k, v, w, g_cumsum, o, beta, A, q_new, k_new, w2 = ctx.saved_tensors
        BT = 64
        BS = BT
        S = 512
        scale = ctx.scale
        cu_seqlens = ctx.cu_seqlens

        # dv = triangular_solve_bwd_dv_fn_recompute(
        #     q=q,
        #     k=k,
        #     w=w,
        #     beta=beta,
        #     A=A,
        #     do=do,
        #     scale=scale,
        #     BT=BT,
        #     cu_seqlens=cu_seqlens,
        # )

        dv = triangular_solve_bwd_dv_fn_reuse(
            q=q,
            k=k,
            w=w,
            beta=beta,
            A=A,
            do=do,
            q_new=q_new, k_new=k_new, w2=w2, 
            scale=scale,
            BT=BT,
            cu_seqlens=cu_seqlens,
        )

        q_new, k_new, h, dA_local = intra_chunk_preprocess_bwd_prepare_fn(
            q=q,
            k=k,
            o=o,
            w=w,
            beta=beta,
            A=A,
            dv=dv,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            return_h=False,
        )

        k_new_large, hc_suffix, hc_whole = chunk_cumprod_householder_fwd_fn(
            k=k_new, w1=w, w2=h, S=S, BT=BS, cu_seqlens=cu_seqlens
        )

        q_new_large = transform_q_fwd_fn(q=q_new, w1=w, w2=h, cu_seqlens=cu_seqlens, BT=BT, BS=BS, S=S)

        dk = parallel_path_bwd_dkv_fn(
            q=q_new_large, k=k_new_large, o=o, dv=dv,
            hc_whole=hc_whole, scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            S=S, BT=BT, BS=BS
        )
        dq, dhc_whole = parallel_path_bwd_dq_fn(
            q=q_new_large, k=k_new_large, o=o, dv=dv,
            hc_whole=hc_whole, scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            S=S, BT=BT, BS=BS
        )
        dw1, dw2, dk = chunk_cumprod_householder_bwd_fn(
            w1=w, w2=h,
            k=k_new, dk=dk, hc_suffix=hc_suffix, dhc_whole=dhc_whole,
            cu_seqlens=cu_seqlens, S=S, BT=BS
        )
        dq, dk, dw1, dw2 = parallel_path_bwd_intra_chunk_fn(
            q=q_new, k=k_new, o=o, w1=w, w2=h,
            dq=dq, dk=dk, dv=dv, dw1=dw1, dw2=dw2,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            S=S, BT=BS
        )
        dq, dk, dbeta, dw = intra_chunk_preprocess_bwd_fn(
            q=q, k=k, w=w, beta=beta,
            dq=dq, dk=dk, dA_local=dA_local, dw1=dw1, dw2=dw2,
            A=A, cu_seqlens=cu_seqlens
        )

        G = q.shape[-2] // k.shape[-2]
        if G > 1:
            assert dk.dtype == dv.dtype == dw.dtype == dbeta.dtype == torch.float32, 'reduction requires float32'
            dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
            dv = reduce(dv, 'b t (h g) k -> b t h k', g=G, reduction='sum')
            dw = reduce(dw, 'b t (h g) k -> b t h k', g=G, reduction='sum')
            dbeta = reduce(dbeta, 'b t (h g) -> b t h', g=G, reduction='sum')
        # if dg_cumsum is not None:
        #     dg_cumsum = chunk_global_cumsum(dg_cumsum, cu_seqlens=cu_seqlens, reverse=True)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dw.to(w.dtype),
                dbeta.to(beta.dtype),
                dg_cumsum.to(g_cumsum.dtype) if g_cumsum is not None else None,
                None, None, None, None)


@torch.compiler.disable
def parallel_lucid_path_attention(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    q: Optional[torch.Tensor] = None,
    g: Optional[torch.Tensor] = None,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_cache: bool = False
) -> torch.Tensor:
    r"""
    LUCID Path Attention with triangular solve.

    Solves (I + L) O = V where L = stril(exp(A * scale - scale))
    and A is the path attention matrix.

    Args:
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, T, H, V]`
        w (torch.Tensor):
            weights of shape `[B, T, H, K]`
        beta (torch.Tensor):
            beta of shape `[B, T, H]`
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`. If None, defaults to k.
        g (torch.Tensor):
            g of shape `[B, T, HQ]` (currently disabled)
        scale (float):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        use_cache (bool):
            Whether to transform and cache the key values for decoding. Default: `False`.
    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, V]`
    """
    # Default q to k if not provided
    if q is None:
        q = k

    if scale is None:
        scale = k.shape[-1]**-0.5
    
    assert q.shape[-1] in [16, 32, 64, 128], "only support head_dim in [16, 32, 64, 128] for now. Stay tuned!"
    assert v.shape[-1] in [16, 32, 64, 128], "only support head_dim in [16, 32, 64, 128] for now. Stay tuned!"
    assert q.shape[-1] == k.shape[-1], 'q, k should have the same head_dim.'
    assert k.shape == w.shape, 'k, w should have the same shape.'
    assert beta.shape[:3] == k.shape[:3], 'beta should have the same number of heads as k'
    assert g is None, 'g is not supported'

    if g is not None:
        assert g.shape[:3] == q.shape[:3], 'g should have the same number of heads as q'
    assert q.shape[-2] % k.shape[-2] == 0, 'the number of query heads should be divisible by the number of key heads'
    o, k_cache = ParallelLUCIDPathAttentionFunction.apply(q, k, v, w, beta, g, scale, cu_seqlens, use_cache)
    return o, k_cache
