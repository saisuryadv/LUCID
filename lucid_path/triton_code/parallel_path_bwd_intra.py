import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices


@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
    # 'USE_GATE': lambda args: args['g_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_path_bwd_intra_chunk_kernel(
    q, k, o, # g_cumsum,
    w1, w2,
    dq, dq_new, dk, dv, dw1, dw2,  # dg_cumsum,
    offsets, indices,
    T, scale,
    G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,  BV: tl.constexpr,
    BT: tl.constexpr, S: tl.constexpr,
    IS_VARLEN: tl.constexpr,  # USE_GATE: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

    # offset calculations
    k += (bos * H + i_h) * K  # GQA when H!=HQ
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K

    q += (bos * HQ + i_hq) * K
    o += (bos * HQ + i_hq) * V   # o for dA computation
    dq += (bos * HQ + i_hq) * K
    dq_new += (bos * HQ + i_hq) * K
    dk += (bos * HQ + i_hq) * K
    dv += (bos * HQ + i_hq) * V  # dv is input (computed externally)
    dw1 += (bos * HQ + i_hq) * K
    dw2 += (bos * HQ + i_hq) * K
    # if USE_GATE:
    #     g_cumsum += (bos * HQ + i_hq)
    #     dg_cumsum += (bos * HQ + i_hq)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    p_dq = tl.make_block_ptr(dq, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))

    # if USE_GATE:
    #     p_gq_cumsum = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    #     b_gq_cumsum = tl.load(p_gq_cumsum, boundary_check=(0, ))
    #     b_dgq = tl.zeros([BT, ], dtype=tl.float32)
    # else:
    #     b_dgq = None

    curr_start = (tl.floor(i_t * BT / S).to(tl.int32) * S).to(tl.int32)

    for offset in range(curr_start, i_t * BT, BT):
        mask = offset + tl.arange(0, BT) < T
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (offset, 0), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q_tmp = tl.zeros([BT, BK], dtype=tl.float32)
        b_q_tmp += b_q
        for i_t_small in range(i_t * BT - BT, offset, -BT):
            p_w1 = tl.make_block_ptr(w1, (T, K), (H*K, 1), (i_t_small, 0), (BT, BK), (1, 0))
            b_w1 = tl.load(p_w1, boundary_check=(0, 1))
            p_w2 = tl.make_block_ptr(w2, (T, K), (H*K, 1), (i_t_small, 0), (BT, BK), (1, 0))
            b_w2 = tl.load(p_w2, boundary_check=(0, 1))
            b_A_tmp = tl.dot(b_q_tmp.to(b_w1.dtype), tl.trans(b_w1))
            b_q_tmp -= tl.dot(b_A_tmp.to(b_w1.dtype), b_w2)
        b_q2 = b_q_tmp.to(b_k.dtype)

        # Compute A = Q @ K.T
        b_A = tl.dot(b_q2, tl.trans(b_k))

        # if USE_GATE:
        #     p_gk_cumsum = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (offset, ), (BT, ), (0, ))
        #     b_gk_cumsum = tl.load(p_gk_cumsum, boundary_check=(0, ))
        #     b_A = b_A + b_gq_cumsum[:, None] - b_gk_cumsum[None, :]
        #     b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))  # avoid nan

        # Load dv (at query positions) and o (at key positions) for dA computation
        # dA[query, key] = -scale * dV[query] @ O[key].T * L[query, key]
        p_dv = tl.make_block_ptr(dv, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
        b_dv = tl.load(p_dv, boundary_check=(0, 1))  # [BT, V] at query positions
        p_o = tl.make_block_ptr(o, (V, T), (1, HQ*V), (0, offset), (BV, BT), (0, 1))
        b_ot = tl.load(p_o, boundary_check=(0, 1))  # [V, BT] at key positions (transposed)

        # dA = -scale * (dV @ O.T) * exp(A * scale - 1/scale)
        # No masking needed (all key positions < query positions)
        # b_A is [query, key] format (from Q @ K.T), so b_L is also [query, key]
        b_L = tl.exp((b_A * scale - 1.0 / scale).to(tl.float32))
        b_dA = -scale * tl.dot(b_dv, b_ot) * b_L  # [BT, BT] = [query, key]

        # if USE_GATE:
        #     b_dgk = -tl.sum(b_dA, axis=0)
        #     tl.atomic_add(dg_cumsum + (offset + tl.arange(0, BT)) * HQ, b_dgk, mask=mask, sem='relaxed')
        #     b_dgq += tl.sum(b_dA, axis=1)

        b_dA = b_dA.to(b_k.dtype)
        b_dk = tl.dot(tl.trans(b_dA), b_q2)
        tl.atomic_add(dk + (offset + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0,
                      BK)[None, :], b_dk, mask=mask[:, None], sem='relaxed')
        p_w1 = tl.make_block_ptr(w1, (T, K), (H*K, 1), (offset, 0), (BT, BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H*K, 1), (offset, 0), (BT, BK), (1, 0))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_dA2 = tl.dot(b_dq.to(b_w2.dtype), tl.trans(b_w2)).to(b_k.dtype)
        b_A2 = tl.dot(b_q2.to(b_w1.dtype), tl.trans(b_w1)).to(b_k.dtype)
        b_dw2 = -tl.dot(tl.trans(b_A2), b_dq.to(b_k.dtype))
        tl.atomic_add(dw2 + (offset + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0,
                      BK)[None, :], b_dw2, mask=mask[:, None], sem='relaxed')
        b_dw1 = -tl.dot(tl.trans(b_dA2), b_q2.to(b_k.dtype))
        tl.atomic_add(dw1 + (offset + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0,
                      BK)[None, :], b_dw1, mask=mask[:, None], sem='relaxed')
        b_dq -= tl.dot(b_dA2, b_w1.to(b_k.dtype))
        b_dq += tl.dot(b_dA.to(b_k.dtype), b_k)

    p_dq_new = tl.make_block_ptr(dq_new, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dq_new, b_dq.to(dq_new.dtype.element_ty), boundary_check=(0, 1))
    # if USE_GATE:
    #     tl.atomic_add(dg_cumsum + (i_t * BT + tl.arange(0, BT)) * HQ, b_dgq, sem='relaxed')


def parallel_path_bwd_intra_chunk_fn(
    q, k, o, w1, w2,
    dq, dk, dv, dw1, dw2,
    scale,
    cu_seqlens,
    S, BT,
):
    """
    Backward pass for intra-large-chunk (but inter-small-block) computation.

    Args:
        q: Query tensor
        k: Key tensor
        o: Output tensor from forward pass
        w1, w2: W tensors
        dq: Gradient of Q (input, accumulated)
        dk: Gradient of K (output, accumulated via atomic_add)
        dv: Gradient of V (input, computed externally by triangular solve backward)
        dw1, dw2: Gradients of W tensors (output, accumulated via atomic_add)
        scale: Scale factor
        cu_seqlens: Cumulative sequence lengths for variable length
        S, BT: Block sizes

    Returns:
        dq_new: Updated gradient of Q
        dk, dw1, dw2: Accumulated gradients (modified in-place via atomic_add)
    """
    assert dk.dtype == dw1.dtype == dw2.dtype == torch.float32, 'atomic_add requires float32'
    B, T, HQ, K = q.shape
    assert dk.shape == dq.shape

    V = o.shape[-1]
    H = k.shape[-2]
    G = HQ // H
    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    dq_new = torch.empty_like(dq, dtype=q.dtype)
    parallel_path_bwd_intra_chunk_kernel[(NT, B*HQ)](
        q=q, k=k, o=o,  # g_cumsum=g_cumsum,
        w1=w1, w2=w2,
        dq=dq, dq_new=dq_new, dk=dk, dv=dv, dw1=dw1, dw2=dw2,  # dg_cumsum=dg_cumsum,
        offsets=cu_seqlens, indices=indices,
        T=T, S=S, BT=BT, scale=scale,
        G=G, HQ=HQ, H=H, K=K, V=V,
        BK=triton.next_power_of_2(K), BV=triton.next_power_of_2(V),
    )
    return dq_new, dk, dw1, dw2
