import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import check_shared_mem


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    # 'USE_GATE': lambda args: args['g_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_path_bwd_dq_kernel(
    q, k, o, # g_cumsum,
    hc_whole, scale,
    dq, dv, dhc_whole,  # dg_cumsum,
    cu_seqlens, indices, split_offsets,  # varlen specific
    T,
    G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr,
    S: tl.constexpr,  # aka larger chunk size
    NUM_BLOCKS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    # USE_GATE: tl.constexpr
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        boh_large = tl.load(split_offsets + i_n).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        boh_large = i_n * tl.cdiv(T, S)

    # offset calculations
    dq += (bos * HQ + i_hq) * K
    k += (bos * H + i_h) * K  # GQA when H!=HQ
    o += (bos * HQ + i_hq) * V  # o for dA computation
    dv += (bos * HQ + i_hq) * V  # dv is input (computed externally)
    hc_whole += (boh_large * H + i_h) * K * K
    dhc_whole += (boh_large * HQ + i_hq) * K * K
    # if USE_GATE:
    #     g_cumsum += (bos * HQ + i_hq)
    #     dg_cumsum += (bos * HQ + i_hq)

    # constants
    stride_h = H * K * K
    stride_hq = HQ * K * K

    # if USE_GATE:
    #     b_g_cumsum_q = tl.zeros([BT,], dtype=tl.float32)
    #     p_g_cumsum_q = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    #     b_g_cumsum_q += tl.load(p_g_cumsum_q, boundary_check=(0, ))
    #     b_dg_cumsum_q = tl.zeros([BT,], dtype=tl.float32)
    # else:
    #     b_g_cumsum_q = None
    #     b_dg_cumsum_q = None

    curr_end = (tl.floor(i_t * BT / S).to(tl.int32) * S).to(tl.int32)
    b_dq = tl.zeros([BT, K], dtype=tl.float32)

    for offset_outer in range(0, curr_end, S):
        idx_j = (offset_outer // S)
        p_q = tl.make_block_ptr(q + ((bos * NUM_BLOCKS + idx_j + 1) * HQ + i_hq) * K, (T, K),
                                (HQ*K*NUM_BLOCKS, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))

        b_dh = -tl.dot(tl.trans(b_q), b_dq.to(b_q.dtype))
        tl.atomic_add(dhc_whole + idx_j * stride_hq + tl.arange(0, K)
                      [:, None] * K + tl.arange(0, K)[None, :], b_dh, sem='relaxed')
        p_h = tl.make_block_ptr(hc_whole + idx_j * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dq = b_dq - tl.dot(b_dq.to(b_h.dtype), tl.trans(b_h))

        for offset in range(offset_outer, min(offset_outer+S, i_t*BT), BS):
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (offset, 0), (BS, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))

            # Compute A = Q @ K.T
            b_A = tl.dot(b_q, tl.trans(b_k).to(b_q.dtype))

            # if USE_GATE:
            #     p_g_cumsum_k = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (offset, ), (BS, ), (0, ))
            #     b_g_cumsum_k = tl.load(p_g_cumsum_k, boundary_check=(0, ))
            #     b_A = b_A + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]
            #     b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))  # avoid nan

            # Load dv (at query positions) and o (at key positions) for dA computation
            # dA[query, key] = -scale * dV[query] @ O[key].T * L[query, key]
            # Requires BT == BS for correct dimensions
            p_dv = tl.make_block_ptr(dv, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))  # [BT, V] at query positions
            p_o = tl.make_block_ptr(o, (V, T), (1, HQ*V), (0, offset), (BV, BS), (0, 1))
            b_ot = tl.load(p_o, boundary_check=(0, 1))  # [V, BS] at key positions (transposed)

            # dA = -scale * (dV @ O.T) * exp(A * scale - 1/scale)
            # No masking needed for inter-chunk (all key positions < query positions)
            # b_A is [query, key] format (from Q @ K.T), so b_L is also [query, key]
            b_L = tl.exp((b_A * scale - 1.0 / scale).to(tl.float32))
            b_dA = -scale * tl.dot(b_dv, b_ot) * b_L  # [BT, BS] = [query, key]

            b_dq += tl.dot(b_dA.to(b_k.dtype), b_k)
            # if USE_GATE:
            #     b_dg_cumsum_q += tl.sum(b_dA, axis=1)

    p_dq = tl.make_block_ptr(dq, (T, K), (K * HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(dq.dtype.element_ty), boundary_check=(0, 1))
    # if USE_GATE:
    #     tl.atomic_add(dg_cumsum + (i_t * BT + tl.arange(0, BT)) * HQ, b_dg_cumsum_q, sem='relaxed')


def parallel_path_bwd_dq_fn(
    q, k, o, dv,
    hc_whole, scale,
    cu_seqlens,
    S, BT, BS,
):
    """
    Backward pass for inter-chunk dq and dhc computation.

    Args:
        q: Query tensor
        k: Key tensor
        o: Output tensor from forward pass
        dv: Gradient of V (computed externally by triangular solve backward)
        hc_whole: Hidden state cache
        scale: Scale factor
        cu_seqlens: Cumulative sequence lengths for variable length
        S, BT, BS: Block sizes (BT must equal BS)

    Returns:
        dq: Gradient of Q
        dhc_whole: Gradient of hidden state cache
    """
    # BT must equal BS for correct dA computation dimensions
    assert BT == BS, f"BT ({BT}) must equal BS ({BS}) for inter-chunk backward"

    B, T, num_blocks, HQ, K = q.shape
    V = o.shape[-1]
    H = k.shape[-2]
    G = HQ // H

    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    split_offsets = prepare_chunk_offsets(cu_seqlens, S) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)

    # should be NS
    if cu_seqlens is not None:
        assert split_offsets[-1] == hc_whole.shape[0]
    dq = torch.empty(B, T, HQ, K, dtype=torch.float32, device=q.device)  # for later reduction use

    # [NS, HQ, K, K] instead of [NS, H, K, K]
    # atomic add must be initialized to 0
    dhc_whole = torch.zeros(hc_whole.shape[0], HQ, K, K, dtype=torch.float32, device=q.device)

    parallel_path_bwd_dq_kernel[(NT, B*HQ)](
        q=q, k=k, o=o,  # g_cumsum=g_cumsum,
        hc_whole=hc_whole, scale=scale,
        dq=dq, dv=dv, dhc_whole=dhc_whole,  # dg_cumsum=dg_cumsum,
        cu_seqlens=cu_seqlens, indices=indices, split_offsets=split_offsets,
        T=T, S=S, BT=BT, BS=BS,
        G=G, HQ=HQ, H=H, K=K, V=V,
        BK=triton.next_power_of_2(K), BV=triton.next_power_of_2(V),
        num_warps=8 if (BT == 128 and K == 128) else 4,
        NUM_BLOCKS=num_blocks,
        num_stages=2,
    )
    return dq, dhc_whole
