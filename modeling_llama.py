# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
import torch.utils.checkpoint
from torch.utils.checkpoint import checkpoint
from torch import nn
from einops import rearrange
import gc
from torch.nn.attention import SDPBackend, sdpa_kernel

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import LossKwargs, auto_docstring, can_return_tuple, logging
from .configuration_llama import LlamaConfig
import sys

from flash_attn.flash_attn_interface import flash_attn_func


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_lucid(q, k, w, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    w_embed = (w * cos) + (rotate_half(w) * sin)
    return q_embed, k_embed, w_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_beta(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen) to (batch, num_attention_heads, seqlen)
    """
    batch, num_key_value_heads, slen = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(batch, num_key_value_heads, n_rep, slen)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = config.use_rope
        self.use_qknorm = getattr(config, "use_qknorm", True)

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        if self.use_qknorm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)#, elementwise_affine=False)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.rope_q = nn.Identity()
        self.rope_k = nn.Identity()
        # self.rope_v = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.use_rope:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        query_states = self.rope_q(query_states)
        key_states = self.rope_k(key_states)
        # value_states = self.rope_v(value_states)
        
        if self.use_qknorm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask if not self.training else None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Lucid_Tril_Solver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, key, value, scaling, BS=2**14):
        L = key.shape[-2]
        assert L > BS, f"Sequence length {L} must be greater than block size {BS}."
        assert L % BS == 0, f"Sequence length {L} must be divisible by block size {BS}."

        # Block wise solution of lower triangular system LV' = V
        v_prime = torch.zeros_like(value)
        exp_ii = None
        exp_ij = None
        rhs = None
        for i in range(0, L, BS):
            rhs = value[:, :, i: i+BS, :]
            for j in range(0, i, BS):
                exp_ij = torch.exp(key[:, :, i: i+BS, :] @ key[:, :, j: j+BS, :].transpose(-1, -2) * scaling - 1 / scaling)
                rhs.sub_(exp_ij @ v_prime[:, :, j: j+BS, :])
            exp_ii = torch.exp(key[:, :, i: i+BS, :] @ key[:, :, i: i+BS, :].transpose(-1, -2) * scaling - 1 / scaling)
            v_prime[:, :, i: i+BS, :] = torch.linalg.solve_triangular(torch.tril(exp_ii), rhs, upper=False)
        del exp_ii, exp_ij, rhs
        gc.collect()
        torch.cuda.empty_cache()
        
        # Save tensors needed for backward
        ctx.save_for_backward(key, v_prime)
        ctx.scaling = scaling
        ctx.BS = BS
        
        return v_prime

    @staticmethod
    def backward(ctx, grad_v_prime):
        # Retrieve saved tensors and non-tensor
        key, v_prime = ctx.saved_tensors
        scaling = ctx.scaling
        BS = ctx.BS

        L = key.shape[-2]
        
        # Block wise solution of upper triangular system (L.T)(dV) = dV'
        exp_ii = None
        exp_ji = None
        rhs = None
        grad_v = torch.zeros_like(key)
        for i in range(L - BS, -BS, -BS):
            rhs = grad_v_prime[:, :, i: i+BS, :]
            for j in range(L - BS, i, -BS):
                exp_ji = torch.exp(key[:, :, j: j+BS, :] @ key[:, :, i: i+BS, :].transpose(-1, -2) * scaling - 1 / scaling)
                rhs = rhs - (exp_ji.transpose(-1, -2) @ grad_v[:, :, j: j+BS, :])
            exp_ii = torch.exp(key[:, :, i: i+BS, :] @ key[:, :, i: i+BS, :].transpose(-1, -2) * scaling - 1 / scaling)
            grad_v[:, :, i: i+BS, :] = torch.linalg.solve_triangular(torch.triu(exp_ii), rhs, upper=True)
        
        del exp_ii, exp_ji, rhs
        gc.collect()
        torch.cuda.empty_cache()

        # Block wise MatMul to get dK = -scaling * [(dV' V'.T) * L]K + [(V' dV.T) * L.T]K
        exp_ii = None
        exp_ij = None
        exp_i_endi_end = None
        exp_j_endi_end = None
        dv_vprime_ii = None
        dv_vprime_ij = None
        vprime_dv_i_endi_end = None
        vprime_dv_i_endj_end = None
        k1 = None
        k2 = None
        grad_k = torch.zeros_like(key)
        for i in range(0, L, BS):
            exp_ii = torch.exp(key[:, :, i: i+BS, :] @ key[:, :, i: i+BS, :].transpose(-1, -2) * scaling - 1 / scaling)
            dv_vprime_ii = grad_v[:, :, i: i+BS, :] @ v_prime[:, :, i: i+BS, :].transpose(-1, -2)
            k1 = (dv_vprime_ii * torch.tril(exp_ii)) @ key[:, :, i: i+BS, :]

            i_end = L - (i + BS)
            exp_i_endi_end = torch.exp(key[:, :, i_end: i_end+BS, :] @ key[:, :, i_end: i_end+BS, :].transpose(-1, -2) * scaling - 1 / scaling)
            vprime_dv_i_endi_end = v_prime[:, :, i_end: i_end+BS, :] @ grad_v[:, :, i_end: i_end+BS, :].transpose(-1, -2)
            k2 = (vprime_dv_i_endi_end * torch.triu(exp_i_endi_end)) @ key[:, :, i_end: i_end+BS, :]

            for j in range(0, i, BS):
                exp_ij = torch.exp(key[:, :, i: i+BS, :] @ key[:, :, j: j+BS, :].transpose(-1, -2) * scaling - 1 / scaling)
                dv_vprime_ij = grad_v[:, :, i: i+BS, :] @ v_prime[:, :, j: j+BS, :].transpose(-1, -2)
                k1 = k1 + ((dv_vprime_ij * exp_ij) @ key[:, :, j: j+BS, :])

                j_end = L - (j + BS)
                exp_j_endi_end = torch.exp(key[:, :, j_end: j_end+BS, :] @ key[:, :, i_end: i_end+BS, :].transpose(-1, -2) * scaling - 1 / scaling)
                vprime_dv_i_endj_end = v_prime[:, :, i_end: i_end+BS, :] @ grad_v[:, :, j_end: j_end+BS, :].transpose(-1, -2)
                k2 = k2 + ((vprime_dv_i_endj_end * exp_j_endi_end.transpose(-1, -2)) @ key[:, :, j_end: j_end+BS, :])
            
            grad_k[:, :, i: i+BS, :] = grad_k[:, :, i: i+BS, :] + k1
            grad_k[:, :, i_end: i_end+BS, :] = grad_k[:, :, i_end: i_end+BS, :] + k2
        
        del exp_ii, exp_i_endi_end, exp_ij, exp_j_endi_end, dv_vprime_ii, dv_vprime_ij, vprime_dv_i_endi_end, vprime_dv_i_endj_end, k1, k2
        gc.collect()
        torch.cuda.empty_cache()
        
        grad_k = -(grad_k * scaling)
        
        # Return gradients in order of forward inputs
        return grad_k, grad_v, None, None


class LucidAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = config.use_rope
        self.which_rmsnorm = getattr(config, "which_rmsnorm", "exp_k_norm")

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )

        if self.which_rmsnorm == "exp_k_norm":
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False)
        else:
            raise NotImplementedError(
                f"""
                {self.which_rmsnorm} not implemented
                """
            )
        
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_beta:
            beta = 2 * torch.sigmoid(self.beta_proj(hidden_states).transpose(1, 2))

        if self.use_forget_gate:
            g = torch.nn.functional.logsigmoid(self.g_proj(hidden_states)).transpose(1, 2)
        
        cos, sin = position_embeddings
        if self.use_rope:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        L = key_states.shape[-2]
        
        if self.which_rmsnorm == "qk_norm":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        past_seq_len = 0 if past_key_value is None else past_key_value.get_seq_length(self.layer_idx)

        attention_mask = None

        if L <= 2**15 or not self.training:
            attention_mask = self.create_causal_mask(query_states, key_states, past_seq_len, attention_mask)
            assert attention_mask is not None, "Attention Mask is None in Lucid. Fix it."

        if past_seq_len == 0:
            # Training/Pre-filling
            v_prime = self.get_v_prime(
                key_states,
                value_states,
                self.scaling,
                attention_mask
            )
        else:
            # Decoding
            v_prime = self.get_new_v_prime(
                key_states,
                value_states,
                attention_mask,
                scaling=self.scaling,
                past_key_value=past_key_value
            )
        
        if past_key_value is not None and not self.training:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, v_prime = past_key_value.update(key_states, v_prime, self.layer_idx, cache_kwargs)
        
        # Making sure the shape matches during both training/pre-filling and decoding phases
        attention_mask_layer = None
        if attention_mask is not None:
            attention_mask_layer = attention_mask[:, :, -query_states.shape[-2]:, :]
            attention_mask_layer = attention_mask_layer.contiguous()
        
        attn_output, attn_weights = self.eager_lucid_attention_forward(
            query_states,
            key_states,
            v_prime,
            self.scaling,
            attention_mask_layer,
            **kwargs,
        )

        if self.config.use_token_grad_clip:
            attn_output = ClipGradientByNormStats.apply(attn_output)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def eager_lucid_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        v_prime: torch.Tensor,
        scaling: float,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B = query.shape[0]
        # dtype = query.dtype
        # query = query.to(torch.bfloat16).contiguous()
        # key = key.to(torch.bfloat16).contiguous()
        # v_prime = v_prime.to(torch.bfloat16).contiguous()
        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        if self.training or B == 1:
            O = nn.functional.scaled_dot_product_attention(
                query,
                repeat_kv(key, self.num_key_value_groups),
                repeat_kv(v_prime, self.num_key_value_groups),
                # attn_mask=attention_mask,
                dropout_p=0.0,
                scale=self.scaling,
                is_causal=True,
                enable_gqa=False
            )
        else:
            # enable_gqa=True forces full attention, which can work out for training
            # but it can't be used for inference as long sequences can be inputted
            # Maybe in future consider this block for both training and inference
            O = nn.functional.scaled_dot_product_attention(
                query,
                repeat_kv(key, self.num_key_value_groups),
                repeat_kv(v_prime, self.num_key_value_groups),
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=self.scaling,
                # is_causal=True,
                enable_gqa=False
            )
        
        O = O.transpose(1, 2).contiguous()
        # O = O.to(dtype)

        return O, None
    
    def get_v_prime(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        scaling: float,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = key.device
        dtype = key.dtype
        L = key.shape[2]
        gpu_seq_len_limit = 2**16

        if self.which_rmsnorm == "exp_k_norm":
            key = self.k_norm(key)
            if L < gpu_seq_len_limit:
                value_states = torch.einsum("bhld,bhsd->bhls", key, key)
                value_states.mul_(scaling).sub_(1 / scaling)
                value_states.masked_fill_(~attention_mask[:, :, :, : key.shape[-2]], float('-inf'))
                value_states.exp_()
                value_states = torch.linalg.solve_triangular(value_states, value, upper=False)
            else:
                value_states = Lucid_Tril_Solver.apply(key, value, scaling)
        
        return value_states
    
    def get_new_v_prime(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        past_key_value: Cache
    ) -> torch.Tensor:
        device = key.device
        dtype = key.dtype

        L_k = key.shape[-2]

        if self.which_rmsnorm == "exp_k_norm":
            key = self.k_norm(key)
            past_key = self.k_norm(past_key_value.key_cache[self.layer_idx])
            k_new_k_cache_t = torch.einsum("bhld,bhsd->bhls", key, past_key)
            k_new_k_cache_t.mul_(scaling).sub_(1 / scaling)

            k_new_k_cache_t.exp_()
            v_prime_new = value - k_new_k_cache_t @ past_key_value.value_cache[self.layer_idx]

        return v_prime_new
    
    def create_causal_mask(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        past_seq_len: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, _, query_len, _ = query.shape
        total_seq_len = query_len + past_seq_len
        q_indices = torch.arange(query_len, device=query.device)
        kv_indices = torch.arange(total_seq_len, device=query.device)
        mask_2d = kv_indices[None, :] <= (q_indices[:, None] + past_seq_len)
        final_mask = mask_2d[None, None, :, :].expand(batch_size, 1, -1, -1)    # Shape: (B, 1, L_q, L_q + L_cache)

        if attention_mask is not None:
            assert attention_mask.dtype == torch.bool, "attention_mask must be boolean"
            final_mask = final_mask & attention_mask.reshape(final_mask.shape)

        return final_mask


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if not hasattr(config, "attention_type"):
            raise ValueError("Missing 'attention_type' in config")

        attention_type = config.attention_type

        if attention_type == "lucid":
            self.self_attn = LucidAttention(config=config, layer_idx=layer_idx)
        elif attention_type == "llama":
            self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type!r}. Must be one of ['lucid', 'llama']")

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    # @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        # if causal_mask is None and self.config.attention_type != "llama":
        #     causal_mask = create_keep_mask(attention_mask, inputs_embeds)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
