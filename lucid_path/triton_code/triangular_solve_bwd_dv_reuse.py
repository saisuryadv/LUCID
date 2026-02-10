# -*- coding: utf-8 -*-
# Triangular Solve Backward Pass - dV computation
# Solves (I + L).T dV = dO where L = stril(exp(A * scale - 1 / scale))
# This is an upper triangular solve since L.T is strictly upper triangular.
#
# This version uses pre-computed q_new, k_new, w2 from the forward pass
# instead of recomputing them (saves compute, but requires more memory).

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.lucid_path.mini_tril_solve import mini_triangular_solve


@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def triangular_solve_bwd_dv_kernel(
    # Inputs
    q, k, w, beta, A,   # same as forward
    do,                 # gradient of output [B, T, HQ, V]
    # Pre-computed from forward (passed in, not computed here)
    q_new, k_new, w2,
    # Outputs
    dv,                 # gradient of values [B, T, HQ, V]
    # Params
    scale,
    offsets, indices, chunk_offsets,
    T,
    G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Backward pass to compute dV via upper triangular solve.

    Solves: (I + L.T) dV = dO
    Where L.T is strictly upper triangular (transpose of strictly lower L).

    This version uses pre-computed q_new, k_new, w2 from forward pass.

    Algorithm:
    - Process rows from bottom to top (i = NT-1, NT-2, ..., 0)
    - For each row i:
      1. Load accumulated R[i] (initialized as dO, updated by later rows)
      2. Compute diagonal L[i,i] and solve (I + L[i,i].T) dV[i] = R[i]
      3. Store dV[i]
      4. For k < i: update R[k] -= L[i,k].T @ dV[i]
         - L[i,k] is computed using q_new[i] @ k_new[k].T with progressive q update
    """
    # Grid: (1, B * HQ) - single program per batch/head, sequential over blocks
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    # Handle variable length sequences
    if IS_VARLEN:
        i_n = i_b
        bos = tl.load(offsets + i_n).to(tl.int32)
        eos = tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

    # Number of blocks
    NT = tl.cdiv(T, BT)

    # Offset calculations
    A += (bos * H + i_h) * BT
    q += (bos * HQ + i_hq) * K
    q_new += (bos * HQ + i_hq) * K
    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    beta += (bos * H + i_h)
    do += (bos * HQ + i_hq) * V
    dv += (bos * HQ + i_hq) * V

    # Helper masks
    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]      # lower including diagonal
    m_stril = o_i[:, None] > o_i[None, :]   # strictly lower

    # ==========================================================================
    # Pass 1: Initialize dV = dO (dV serves as accumulator R)
    # ==========================================================================
    for init_t in range(NT):
        p_do = tl.make_block_ptr(do, (T, V), (HQ*V, 1), (init_t * BT, 0), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (HQ*V, 1), (init_t * BT, 0), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        tl.store(p_dv, b_do, boundary_check=(0, 1))

    # ==========================================================================
    # Pass 2: Sequential loop over block rows (BOTTOM TO TOP)
    #
    # Key insight: The residual R[i] for row i is accumulated INCREMENTALLY
    # during previous iterations. When we process row j > i, we update:
    #   R[k] -= L[j,k].T @ dV[j] for all k < j
    #
    # By the time we reach row i, dV[i] contains:
    #   R[i] = dO[i] - sum_{j>i} L[j,i].T @ dV[j]
    # which is the correct residual for the upper triangular solve.
    #
    # So the order is: residual (from prev iterations) -> mini-solve -> update future R[k]
    # ==========================================================================
    for i_t in range(NT - 1, -1, -1):
        # ----------------------------------------------------------------------
        # Step 1: Load inputs and compute diagonal A[i,i]
        # (Same computation as forward for diagonal block)
        # ----------------------------------------------------------------------
        p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (BK, BT), (0, 1))
        p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        p_T = tl.make_block_ptr(A, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kt = tl.load(p_k, boundary_check=(0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_T = tl.load(p_T, boundary_check=(0, 1))
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

        # Load q_new for this row (pre-computed from forward pass)
        # Cast to fp32 for numerical stability in the inner loop
        p_q_new_i = tl.make_block_ptr(q_new, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        b_q_new = tl.load(p_q_new_i, boundary_check=(0, 1)).to(tl.float32)

        # ----------------------------------------------------------------------
        # Step 2: Load R[i] (accumulated in dV) and solve diagonal
        # L_diag = stril(exp(A_diag * scale - 1/scale))
        # L_diag.T is strictly upper triangular
        # Solve: (I + L_diag.T) dV[i] = R[i]
        # ----------------------------------------------------------------------
        p_dv_i = tl.make_block_ptr(dv, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
        b_R = tl.load(p_dv_i, boundary_check=(0, 1)).to(tl.float32)

        # L_diag with strictly lower triangular mask (float32 for exp precision)
        b_L_diag = tl.where(m_stril, tl.exp(b_A_diag * scale - 1.0 / scale), 0.0).to(tl.float32)

        # Transpose to get strictly upper triangular for the solve
        # (I + L_diag.T) dV = R => upper triangular solve
        b_L_diag_T = tl.trans(b_L_diag)

        # Solve using upper triangular (backward substitution)
        b_dv_i = mini_triangular_solve(b_L_diag_T, b_R, BT, lower=False)

        # Store dV[i]
        tl.store(p_dv_i, b_dv_i.to(p_dv_i.dtype.element_ty), boundary_check=(0, 1))

        # ----------------------------------------------------------------------
        # Step 3: Update R[k] for k < i
        # R[k] -= L[i,k].T @ dV[i]
        # Process k from i-1 down to 0 (right to left in row i of L)
        # L[i,k] = exp(A[i,k] * scale - 1/scale)
        # A[i,k] = q_new[i] @ k_new[k].T (with progressive q_new update)
        # ----------------------------------------------------------------------
        for k_t in range(i_t - 1, -1, -1):
            # Load k_new[k].T (pre-computed from forward pass)
            p_k_new_k = tl.make_block_ptr(k_new, (K, T), (1, K*H), (0, k_t * BT), (BK, BT), (0, 1))
            b_k_new_k_t = tl.load(p_k_new_k, boundary_check=(0, 1))

            # Load w[k].T and w2[k] for q_new transformation
            p_w_k = tl.make_block_ptr(w, (K, T), (1, H*K), (0, k_t * BT), (BK, BT), (0, 1))
            p_w2_k = tl.make_block_ptr(w2, (T, K), (K*H, 1), (k_t * BT, 0), (BT, BK), (1, 0))
            b_w_k_t = tl.load(p_w_k, boundary_check=(0, 1))
            b_w2_k = tl.load(p_w2_k, boundary_check=(0, 1))

            # A[i,k] = q_new[i] @ k_new[k].T (float32 for precision)
            b_A_ik = tl.dot(b_q_new, b_k_new_k_t.to(tl.float32)).to(tl.float32)

            # L[i,k] = exp(A[i,k] * scale - 1/scale) (float32 for exp precision)
            # Full block (no mask needed - entire block is in strictly lower region)
            b_L_ik = tl.exp(b_A_ik * scale - 1.0 / scale).to(tl.float32)

            # Update R[k] -= L[i,k].T @ dV[i] (all float32)
            p_dv_k = tl.make_block_ptr(dv, (T, V), (HQ*V, 1), (k_t * BT, 0), (BT, BV), (1, 0))
            b_R_k = tl.load(p_dv_k, boundary_check=(0, 1)).to(tl.float32)
            b_R_k = b_R_k - tl.dot(tl.trans(b_L_ik), b_dv_i)
            tl.store(p_dv_k, b_R_k.to(p_dv_k.dtype.element_ty), boundary_check=(0, 1))

            # Update q_new for next block (going left) - stay in float32
            b_s2 = tl.dot(b_q_new, b_w_k_t.to(tl.float32))
            b_q_new = b_q_new - tl.dot(b_s2, b_w2_k.to(tl.float32))


def triangular_solve_bwd_dv_fn_reuse(
    q, k, w, beta, A,
    do,
    q_new, k_new, w2,  # Pre-computed from forward pass
    scale,
    BT,
    cu_seqlens,
):
    """
    Backward pass for triangular solve: compute dV.

    Solves (I + L).T dV = dO where L = stril(exp(A * scale - 1/scale))

    This version uses pre-computed q_new, k_new, w2 from forward pass.

    Args:
        q: queries [B, T, HQ, K]
        k: keys [B, T, H, K]
        w: weights [B, T, H, K]
        beta: beta [B, T, H]
        A: precomputed T_inv matrix [B, T, H, BT]
        do: gradient of output [B, T, HQ, V]
        q_new: transformed queries from forward [B, T, HQ, K]
        k_new: transformed keys from forward [B, T, H, K]
        w2: transformed weights from forward [B, T, H, K]
        scale: attention scale
        BT: block size (64)
        cu_seqlens: cumulative sequence lengths for varlen

    Returns:
        dv: gradient of values [B, T, HQ, V]
    """
    B, T, HQ, K = q.shape
    H = k.shape[-2]
    V = do.shape[-1]
    G = HQ // H

    # Allocate output
    dv = torch.empty_like(do)

    # Prepare indices for varlen
    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None

    # Grid: (1, B * HQ) - sequential over blocks, parallel over batch/heads
    grid = (1, B * HQ)

    triangular_solve_bwd_dv_kernel[grid](
        q=q,
        k=k,
        w=w,
        beta=beta,
        A=A,
        do=do,
        q_new=q_new,
        k_new=k_new,
        w2=w2,
        dv=dv,
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

    return dv
