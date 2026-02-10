# -*- coding: utf-8 -*-
# Mini triangular solver using forward/backward substitution
# Solves (I + P) O = R where P is strictly triangular (lower or upper)

import triton
import triton.language as tl


@triton.jit
def mini_triangular_solve(
    b_P,                    # [BT, BT] strictly triangular matrix in registers
    b_R,                    # [BT, V] right-hand side in registers
    BT: tl.constexpr,       # block size
    lower: tl.constexpr,    # True for lower triangular (forward sub), False for upper (backward sub)
):
    """
    Solves (I + P) O = R using forward or backward substitution.

    For lower=True (P is strictly lower triangular):
        Forward substitution: solve row by row from top to bottom
        O[i] = R[i] - sum_{j<i} P[i,j] * O[j]

    For lower=False (P is strictly upper triangular):
        Backward substitution: solve row by row from bottom to top
        O[i] = R[i] - sum_{j>i} P[i,j] * O[j]

    Args:
        b_P: Strictly triangular matrix [BT, BT]
        b_R: Right-hand side [BT, V]
        BT: Block size (constexpr), must be <= 64
        lower: If True, P is strictly lower triangular (forward substitution)
               If False, P is strictly upper triangular (backward substitution)

    Returns:
        b_O: Solution [BT, V]
    """
    b_P = b_P.to(tl.float32)
    b_O = b_R.to(tl.float32)

    o_i = tl.arange(0, BT)

    if lower:
        # Forward substitution for lower triangular
        # Row 0: O[0] = R[0] (no dependencies, P[0,:] = 0 for strictly lower)
        # Row i: O[i] = R[i] - sum_{j<i} P[i,j] * O[j]
        for i in tl.static_range(1, BT):
            # Compute the correction for row i: sum_{j<i} P[i,j] * O[j]
            # Create mask for columns j < i
            mask = (o_i < i)
            # Get row i of P: P[i, j] for all j
            # Use elementwise selection: p_row[j] = P[i,j] where row index == i
            row_select = (o_i == i)[:, None]  # [BT, 1]
            p_row = tl.sum(tl.where(row_select, b_P, 0.0), axis=0)  # [BT]
            # Apply mask to zero out j >= i
            p_row = tl.where(mask, p_row, 0.0)
            # Compute dot product: sum_j P[i,j] * O[j,:] for j < i
            # p_row is [BT], b_O is [BT, V]
            correction = tl.sum(p_row[:, None] * b_O, axis=0)  # [V]
            # Update row i of O
            row_mask = (o_i == i)[:, None]
            b_O = tl.where(row_mask, b_O - correction[None, :], b_O)
    else:
        # Backward substitution for upper triangular
        # Row BT-1: O[BT-1] = R[BT-1] (no dependencies, P[BT-1,:] = 0 for strictly upper)
        # Row i: O[i] = R[i] - sum_{j>i} P[i,j] * O[j]
        for i in tl.static_range(BT - 2, -1, -1):
            # Compute the correction for row i: sum_{j>i} P[i,j] * O[j]
            # Create mask for columns j > i
            mask = (o_i > i)
            # Get row i of P: P[i, j] for all j
            row_select = (o_i == i)[:, None]  # [BT, 1]
            p_row = tl.sum(tl.where(row_select, b_P, 0.0), axis=0)  # [BT]
            # Apply mask to zero out j <= i
            p_row = tl.where(mask, p_row, 0.0)
            # Compute dot product: sum_j P[i,j] * O[j,:] for j > i
            correction = tl.sum(p_row[:, None] * b_O, axis=0)  # [V]
            # Update row i of O
            row_mask = (o_i == i)[:, None]
            b_O = tl.where(row_mask, b_O - correction[None, :], b_O)

    return b_O.to(b_R.dtype)


# =============================================================================
# COMMENTED OUT: Series Neumann implementation
# =============================================================================
# @triton.jit
# def mini_triangular_solve_series_neumann(
#     b_P,                    # [BT, BT] strictly triangular matrix in registers
#     b_R,                    # [BT, V] right-hand side in registers
#     BT: tl.constexpr,       # block size
# ):
#     """
#     Solves (I + P) O = R using series Neumann series.
#
#     For A = I + P where P is strictly triangular:
#     A^{-1} = I - P + P^2 - P^3 + P^4 - ... (alternating series)
#
#     This works because P^{BT} = 0 for strictly triangular matrices,
#     so the series naturally terminates.
#
#     Partial blocks are handled by the caller via boundary checks on load
#     (invalid positions filled with zeros) and appropriate loop bounds.
#
#     Args:
#         b_P: Strictly triangular matrix [BT, BT]
#         b_R: Right-hand side [BT, V]
#         BT: Block size (constexpr), must be <= 64
#
#     Returns:
#         b_O: Solution [BT, V]
#     """
#     # Series Neumann: A^{-1} = I - P + P^2 - P^3 + P^4 - P^5 + ...
#     # For BT <= 64, P^64 = 0, so we sum terms up to P^63
#
#     b_P = b_P.to(tl.float32)
#
#     # Identity matrix
#     o_i = tl.arange(0, BT)
#     b_I = (o_i[:, None] == o_i[None, :]).to(tl.float32)
#
#     # Start accumulating: A^{-1} = I - P + P^2 - P^3 + ...
#     b_Ainv = b_I.to(tl.float32)  # Start with I
#     b_Pk = b_P.to(tl.float32)    # Current power of P (starts as P^1)
#     sign = -1.0                   # Alternating sign (first term after I is -P)
#
#     # Unroll the loop for BT <= 64 (need up to P^63, but P^k = 0 for k >= BT)
#     for k in range(1, BT):
#         b_Ainv = b_Ainv + sign * b_Pk
#         b_Pk = tl.dot(b_Pk, b_P)
#         sign = -sign
#
#     # Multiply inverse by RHS
#     b_O = tl.dot(b_Ainv, b_R.to(b_Ainv.dtype))
#
#     return b_O


# =============================================================================
# COMMENTED OUT: Original product Neumann implementation
# =============================================================================
# @triton.jit
# def mini_triangular_solve_product_neumann(
#     b_P,                    # [BT, BT] strictly triangular matrix in registers
#     b_R,                    # [BT, V] right-hand side in registers
#     BT: tl.constexpr,       # block size
# ):
#     """
#     Solves (I + P) O = R using product Neumann series.
#
#     For A = I + P where P is strictly triangular:
#     A^{-1} = (I - P)(I + P^2)(I + P^4)(I + P^8)(I + P^16)(I + P^32)
#
#     This works because P^{BT} = 0 for strictly triangular matrices.
#
#     Only requires log2(BT) matrix multiplications for the inverse,
#     plus one final multiplication with b_R.
#
#     Partial blocks are handled by the caller via boundary checks on load
#     (invalid positions filled with zeros) and appropriate loop bounds.
#
#     Args:
#         b_P: Strictly triangular matrix [BT, BT]
#         b_R: Right-hand side [BT, V]
#         BT: Block size (constexpr), must be <= 64
#
#     Returns:
#         b_O: Solution [BT, V]
#     """
#     # Product Neumann: A^{-1} = (I - P)(I + P^2)(I + P^4)(I + P^8)(I + P^16)(I + P^32)
#     # For BT <= 64, P^64 = 0, so we need up to P^32
#
#     b_P = b_P.to(tl.float32)
#
#     # Start with (I - P)
#     o_i = tl.arange(0, BT)
#     b_I = (o_i[:, None] == o_i[None, :]).to(tl.float32)
#     b_Ainv = b_I - b_P
#
#     # P^2
#     b_P = tl.dot(b_P, b_P)
#     b_Ainv = tl.dot(b_Ainv, b_I + b_P)
#
#     # P^4
#     b_P = tl.dot(b_P, b_P)
#     b_Ainv = tl.dot(b_Ainv, b_I + b_P)
#
#     # P^8
#     b_P = tl.dot(b_P, b_P)
#     b_Ainv = tl.dot(b_Ainv, b_I + b_P)
#
#     # P^16
#     b_P = tl.dot(b_P, b_P)
#     b_Ainv = tl.dot(b_Ainv, b_I + b_P)
#
#     # P^32
#     b_P = tl.dot(b_P, b_P)
#     b_Ainv = tl.dot(b_Ainv, b_I + b_P)
#
#     # Multiply inverse by RHS
#     b_O = tl.dot(b_Ainv, b_R.to(b_Ainv.dtype))
#
#     return b_O
