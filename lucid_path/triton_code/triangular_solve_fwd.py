# -*- coding: utf-8 -*-
# Triangular Solve Forward Pass
# Solves (I + L) O = V where L = exp(A * scale - 1 / scale + mask)
# A is the path attention matrix, mask is strictly lower triangular

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.lucid_path.mini_tril_solve import mini_triangular_solve


@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
    # 'USE_GATE': lambda args: args['g_cumsum'] is not None,  # TODO: uncomment when adding gating
})
@triton.jit(do_not_specialize=['T'])
def triangular_solve_fwd_kernel(
    # Inputs (same as intra_chunk_preprocess_fwd + parallel_path_fwd)
    q, k, v, w, beta, A,
    # g_cumsum,  # TODO: uncomment when adding gating
    # Outputs
    o, q_new, k_new, w2,
    # Params
    scale,
    offsets, indices, chunk_offsets,
    T,
    G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    # USE_GATE: tl.constexpr,  # TODO: uncomment when adding gating
):
    """
    Sequential triangular solve over block rows.
    For each block row i:
      1. Compute q_new, k_new, w2, A_diag using same logic as intra_chunk_preprocess_fwd
      2. Load V[i] as initial residual R
      3. For j < i: compute A[i,j] like parallel_path_fwd, then R -= L[i,j] @ O[j]
      4. Compute L_diag from A_diag and solve (I + L_diag) O[i] = R
      5. Store O[i], q_new[i], k_new[i], w2[i]
    """
    # Grid: (B_or_N, HQ) - one program per batch/sequence and query head, sequential over blocks within
    # For non-varlen: B_or_N = B (batch size)
    # For varlen: B_or_N = N (number of sequences), B=1 after unpad_input
    # NOTE: Unlike path_attn kernels which can parallelize across chunks, triangular_solve
    # has sequential dependencies (block i needs O[j] for j < i), so we must process
    # chunks sequentially within each sequence.
    i_n = tl.program_id(0)   # sequence/batch index
    i_hq = tl.program_id(1)  # query head index
    i_h = i_hq // G          # GQA: map query head to key/value head

    if IS_VARLEN:
        bos = tl.load(offsets + i_n).to(tl.int32)
        eos = tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    # # OLD CODE (incorrect for varlen - commented out for revert):
    # i_t, i_bh = tl.program_id(0), tl.program_id(1)
    # i_b, i_hq = i_bh // HQ, i_bh % HQ
    # i_h = i_hq // G
    # if IS_VARLEN:
    #     i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
    #     bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
    #     T = eos - bos
    # else:
    #     i_n = i_b
    #     bos, eos = i_n * T, i_n * T + T

    # Number of blocks
    NT = tl.cdiv(T, BT)

    # Same offset calculations as intra_chunk_preprocess_fwd
    A += (bos * H + i_h) * BT
    q += (bos * HQ + i_hq) * K
    q_new += (bos * HQ + i_hq) * K
    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * HQ + i_hq) * V
    beta += (bos * H + i_h)
    # if USE_GATE:
    #     g_cumsum += (bos * HQ + i_hq)

    # Helper masks (same as intra_chunk_preprocess_fwd)
    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]   # lower including diagonal: i >= j
    m_stril = o_i[:, None] > o_i[None, :]  # strictly lower: i > j

    # ==========================================================================
    # Sequential loop over block rows
    # ==========================================================================
    for i_t in range(NT):
        # ----------------------------------------------------------------------
        # Step 1: Load inputs and compute q_new, k_new, w2, A_diag
        # (Same as intra_chunk_preprocess_fwd lines 71-103)
        # ----------------------------------------------------------------------
        p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (BK, BT), (0, 1))
        p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
        p_T = tl.make_block_ptr(A, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kt = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_T = tl.load(p_T, boundary_check=(0, 1))  # T_inv matrix from solve_tril
        b_beta = tl.load(p_beta, boundary_check=(0,))

        # w_beta = w * beta[:, None]
        b_w_beta = b_w * b_beta[:, None]

        # qw = tril(q @ w.T)
        b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w)), 0).to(b_q.dtype)

        # qwT = qw @ T_inv
        b_qwT = tl.dot(b_qw, b_T).to(b_q.dtype)

        # wbk = stril(w_beta @ k.T)
        b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w_beta, b_kt), 0).to(b_q.dtype)

        # A_diag = tril(q @ k.T - qwT @ wbk) - compute in float32 for precision
        b_A_diag = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT, b_wbk), 0.0).to(tl.float32)

        # q_new = q - qwT @ w_beta (keep in original dtype for storage, but use float32 copy for loop)
        b_q_new = b_q - tl.dot(b_qwT, b_w_beta)
        p_q_new = tl.make_block_ptr(q_new, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        tl.store(p_q_new, b_q_new.to(p_q_new.dtype.element_ty), boundary_check=(0, 1))

        # k_new and w2 only computed by first query head in GQA group
        if i_hq % G == 0:
            # w2 = T_inv @ w_beta
            b_Twb = tl.dot(b_T, b_w_beta).to(b_w.dtype)
            p_w2 = tl.make_block_ptr(w2, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            tl.store(p_w2, b_Twb, boundary_check=(0, 1))

            # k_new.T = k.T - w.T @ (T_inv @ wbk)
            b_T_wbk = tl.dot(b_T, b_wbk).to(b_w.dtype)
            p_k_new = tl.make_block_ptr(k_new, (K, T), (1, K*H), (0, i_t * BT), (BK, BT), (0, 1))
            tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w), b_T_wbk)).to(p_k_new.dtype.element_ty), boundary_check=(0, 1))

        # # Load gating
        # if USE_GATE:
        #     p_g_cumsum = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        #     b_g_cumsum_q = tl.load(p_g_cumsum, boundary_check=(0,))
        #     b_A_diag = b_A_diag + (b_g_cumsum_q[:, None] - b_g_cumsum_q[None, :])
        #     b_A_diag = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A_diag, float("-inf"))

        # ----------------------------------------------------------------------
        # Step 2: Initialize residual R = V[i]
        # ----------------------------------------------------------------------
        b_R = b_v.to(tl.float32)

        # Convert q_new to float32 for the inner loop (for numerical stability)
        b_q_new_f32 = b_q_new.to(tl.float32)

        # ----------------------------------------------------------------------
        # Step 3: Off-diagonal contribution - R -= L[i,j] @ O[j] for j < i
        # Process blocks from RIGHT to LEFT (j = i-1 down to 0) to match
        # the progressive q_new transformation in parallel_path_fwd.
        # A[i,j] = q_new[i] @ k_new[j].T
        # L[i,j] = exp(A[i,j] * scale - 1 / scale)
        # After each block, update q_new: q_new -= dot(dot(q_new, w[j].T), w2[j])
        #
        # NOTE: b_A_diag (computed above with original b_q) is NOT modified here.
        # The diagonal A[i,i] uses the full formula, while off-diagonal A[i,j]
        # uses the simplified q_new @ k_new.T formula with progressive q update.
        # ----------------------------------------------------------------------
        for j_t in range(i_t - 1, -1, -1):
            # Load previously computed O[j]
            p_o_j = tl.make_block_ptr(o, (T, V), (HQ*V, 1), (j_t * BT, 0), (BT, BV), (1, 0))
            b_o_j = tl.load(p_o_j, boundary_check=(0, 1)).to(tl.float32)

            # Load k_new[j].T
            p_k_new_j = tl.make_block_ptr(k_new, (K, T), (1, K*H), (0, j_t * BT), (BK, BT), (0, 1))
            b_k_new_j_t = tl.load(p_k_new_j, boundary_check=(0, 1))

            # Load w[j].T (w1 in parallel_path_fwd) and w2[j] for q transformation
            p_w_j = tl.make_block_ptr(w, (K, T), (1, H*K), (0, j_t * BT), (BK, BT), (0, 1))
            p_w2_j = tl.make_block_ptr(w2, (T, K), (K*H, 1), (j_t * BT, 0), (BT, BK), (1, 0))
            b_w_j_t = tl.load(p_w_j, boundary_check=(0, 1))  # [K, BT]
            b_w2_j = tl.load(p_w2_j, boundary_check=(0, 1))  # [BT, K]

            # A[i,j] = q_new[i] @ k_new[j].T (float32 for precision)
            b_A_ij = tl.dot(b_q_new_f32, b_k_new_j_t.to(tl.float32)).to(tl.float32)

            # # Add gating
            # if USE_GATE:
            #     p_g_cumsum_k = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (j_t * BT,), (BT,), (0,))
            #     b_g_cumsum_k = tl.load(p_g_cumsum_k, boundary_check=(0,))
            #     b_A_ij = b_A_ij + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]

            # L[i,j] = exp(A[i,j] * scale - 1 / scale) (float32 for exp precision)
            b_L_ij = tl.exp(b_A_ij * scale - 1.0 / scale).to(tl.float32)

            # Residual update: R -= L[i,j] @ O[j] (all float32)
            b_R = b_R - tl.dot(b_L_ij, b_o_j)

            # Update q_new for next block (stay in float32)
            b_s2 = tl.dot(b_q_new_f32, b_w_j_t.to(tl.float32))
            b_q_new_f32 = b_q_new_f32 - tl.dot(b_s2, b_w2_j.to(tl.float32))

        # ----------------------------------------------------------------------
        # Step 4: Diagonal block - solve (I + L_diag) O[i] = R
        # L_diag = exp(A_diag * scale - 1 / scale) with strictly lower mask
        # NOTE: b_A_diag was computed at the start using original b_q (not b_q_new),
        # matching how intra_chunk_preprocess_fwd computes diagonal blocks.
        # ----------------------------------------------------------------------
        # L_diag with strictly lower triangular mask (float32 for exp precision)
        b_L_diag = tl.where(m_stril, tl.exp(b_A_diag * scale - 1.0 / scale), 0.0).to(tl.float32)

        # Solve (I + L_diag) O[i] = R using mini triangular solver
        b_o_i = mini_triangular_solve(b_L_diag, b_R, BT, lower=True)

        # Store O[i]
        p_o_i = tl.make_block_ptr(o, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
        tl.store(p_o_i, b_o_i.to(p_o_i.dtype.element_ty), boundary_check=(0, 1))


def triangular_solve_fwd_fn(
    q, k, v, w, beta,
    # g_cumsum,  # TODO: uncomment when adding gating
    A,  # precomputed T_inv from solve_tril
    scale,
    BT,
    cu_seqlens,
):
    """
    Forward pass for triangular solve: (I + L) O = V

    Args:
        q: queries [B, T, HQ, K]
        k: keys [B, T, H, K]
        v: values [B, T, H, V]
        w: weights [B, T, H, K]
        beta: beta [B, T, H]
        A: precomputed T_inv matrix [B, T, H, BT] from solve_tril
        scale: attention scale (1/sqrt(K))
        BT: block size (64)
        cu_seqlens: cumulative sequence lengths for varlen

    Returns:
        q_new: transformed queries [B, T, HQ, K]
        k_new: transformed keys [B, T, H, K]
        w2: transformed weights [B, T, H, K]
        o: output [B, T, HQ, V]
    """
    B, T, HQ, K = q.shape
    H = k.shape[-2]
    V = v.shape[-1]
    G = HQ // H

    # Allocate outputs
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    w2 = torch.empty_like(w)

    # Prepare indices for varlen (kept for compatibility, but not used in kernel)
    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None

    # Grid: (B_or_N, HQ) - one program per sequence/batch and query head
    # Sequential processing of chunks within each sequence (due to dependencies)
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1  # number of sequences
        grid = (N, HQ)
    else:
        grid = (B, HQ)

    # # OLD CODE (incorrect for varlen - commented out for revert):
    # NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    # grid = (NT, B * HQ)

    triangular_solve_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        w=w,
        beta=beta,
        A=A,
        # g_cumsum=g_cumsum,
        o=o,
        q_new=q_new,
        k_new=k_new,
        w2=w2,
        scale=scale,
        offsets=cu_seqlens,
        indices=indices,
        chunk_offsets=chunk_offsets,
        T=T,
        G=G,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        num_warps=4,
    )

    return q_new, k_new, w2, o
