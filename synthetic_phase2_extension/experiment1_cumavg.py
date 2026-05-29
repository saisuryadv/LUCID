import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import math
import os


def generate_unique_batch(batch_size, seq_len=256, vocab_size=10, device='cpu'):
    """Generate batch where each sequence has unique tokens.
    For seq_len > vocab_size, we sample with replacement after exhausting unique tokens."""
    # build on CPU (same randperm/randint RNG as original) then one transfer to device
    batch = torch.zeros(batch_size, seq_len, dtype=torch.long, device='cpu')
    for i in range(batch_size):
        if seq_len <= vocab_size:
            perm = torch.randperm(vocab_size)[:seq_len]
            batch[i] = perm
        else:
            # First fill with all unique tokens, then sample randomly for the rest
            perm = torch.randperm(vocab_size)
            batch[i, :vocab_size] = perm
            batch[i, vocab_size:] = torch.randint(0, vocab_size, (seq_len - vocab_size,))
    return batch.to(device)


def init_weights(module, d_model=4):
    """Standard transformer initialization."""
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.RMSNorm):
        pass


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def get_diagonal_offdiagonal_stats(matrix, use_abs=False):
    """Extract diagonal and off-diagonal statistics from attention matrix.
    matrix shape: (B, num_heads, L, L) - causal mask applied
    Returns dict with mean, max, min for diagonal and off-diagonal elements.
    If use_abs=True, computes statistics on absolute values.
    """
    B, H, L, _ = matrix.shape

    # Create masks for diagonal and off-diagonal (lower triangular only due to causal mask)
    diag_mask = torch.eye(L, device=matrix.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    lower_tri = torch.tril(torch.ones(L, L, device=matrix.device, dtype=torch.bool), diagonal=-1)
    offdiag_mask = lower_tri.unsqueeze(0).unsqueeze(0)

    # Extract elements
    diag_elements = matrix.masked_select(diag_mask.expand_as(matrix))
    offdiag_elements = matrix.masked_select(offdiag_mask.expand_as(matrix))

    if use_abs:
        diag_elements = diag_elements.abs()
        offdiag_elements = offdiag_elements.abs()

    stats = {}
    if diag_elements.numel() > 0:
        stats['diag_mean'] = diag_elements.mean().item()
        stats['diag_max'] = diag_elements.max().item()
        stats['diag_min'] = diag_elements.min().item()
    else:
        stats['diag_mean'] = stats['diag_max'] = stats['diag_min'] = 0.0

    if offdiag_elements.numel() > 0:
        stats['offdiag_mean'] = offdiag_elements.mean().item()
        stats['offdiag_max'] = offdiag_elements.max().item()
        stats['offdiag_min'] = offdiag_elements.min().item()
    else:
        stats['offdiag_mean'] = stats['offdiag_max'] = stats['offdiag_min'] = 0.0

    return stats


class HypothesisTransformer(nn.Module):
    def __init__(self, input_vocab_size=10, output_vocab_size=1, d_model=4, num_heads=1, seq_len=256, temp=1.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.temp = temp
        self.seq_len = seq_len

        self.token_embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_embedding = SinusoidalPositionalEncoding(d_model, seq_len)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.output_head = nn.Linear(d_model, output_vocab_size)

        # Store attention tensors for gradient tracking
        self.scores = None
        self.attn_probs = None

    def forward(self, x, return_attention=False):
        B, L = x.shape
        device = x.device

        h = self.token_embedding(x)
        # h = self.pos_embedding(h)

        q = self.q_proj(h).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.d_head) * self.temp)

        mask = torch.triu(torch.ones((L, L), device=device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # Store for gradient tracking (need to retain grad)
        self.scores = scores
        if self.training:
            self.scores.retain_grad()

        attn_probs = F.softmax(scores, dim=-1)
        self.attn_probs = attn_probs
        if self.training:
            self.attn_probs.retain_grad()

        context = torch.matmul(attn_probs, v)

        h = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.output_head(h).squeeze(-1)

        if return_attention:
            return output, scores, attn_probs
        return output


class HypothesisLucidTransformer(nn.Module):
    def __init__(self, input_vocab_size=10, output_vocab_size=1, d_model=4, num_heads=1, seq_len=256):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.seq_len = seq_len

        self.token_embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_embedding = SinusoidalPositionalEncoding(d_model, seq_len)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_rms = nn.RMSNorm(self.d_head)

        self.output_head = nn.Linear(d_model, output_vocab_size)

        # Store attention tensors for gradient tracking
        self.scores = None
        self.attn_probs = None

    def forward(self, x, return_attention=False):
        B, L = x.shape
        device = x.device

        h = self.token_embedding(x)
        # h = self.pos_embedding(h)

        q = self.q_proj(h).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k_rms = self.k_rms(k)
        kkt = torch.matmul(k_rms, k_rms.transpose(-2, -1)) / math.sqrt(self.d_head) - math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        mask = torch.triu(torch.ones((L, L), device=device), diagonal=1).bool()
        kkt = kkt.masked_fill(mask, -float('inf'))
        scores = scores.masked_fill(mask, -float('inf'))

        # Store for gradient tracking
        self.scores = scores
        if self.training:
            self.scores.retain_grad()

        kkt = torch.exp(kkt)
        v = torch.linalg.solve_triangular(kkt, v, upper=False, unitriangular=True)
        attn_probs = F.softmax(scores, dim=-1)
        self.attn_probs = attn_probs
        if self.training:
            self.attn_probs.retain_grad()

        context = torch.matmul(attn_probs, v)

        h = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.output_head(h).squeeze(-1)

        if return_attention:
            return output, scores, attn_probs
        return output


def create_history_dict():
    """Create a history dictionary with all tracking metrics."""
    return {
        # Loss and basic gradients
        "p1_loss": [], "p1_grad_norm": [], "p1_q_grad_norm": [],
        "p2_loss": [], "p2_grad_norm": [], "p2_q_grad_norm": [],
        # Attention logits (scores) statistics
        "p1_scores_diag_mean": [], "p1_scores_diag_max": [], "p1_scores_diag_min": [],
        "p1_scores_offdiag_mean": [], "p1_scores_offdiag_max": [], "p1_scores_offdiag_min": [],
        "p2_scores_diag_mean": [], "p2_scores_diag_max": [], "p2_scores_diag_min": [],
        "p2_scores_offdiag_mean": [], "p2_scores_offdiag_max": [], "p2_scores_offdiag_min": [],
        # Attention probs statistics
        "p1_probs_diag_mean": [], "p1_probs_diag_max": [], "p1_probs_diag_min": [],
        "p1_probs_offdiag_mean": [], "p1_probs_offdiag_max": [], "p1_probs_offdiag_min": [],
        "p2_probs_diag_mean": [], "p2_probs_diag_max": [], "p2_probs_diag_min": [],
        "p2_probs_offdiag_mean": [], "p2_probs_offdiag_max": [], "p2_probs_offdiag_min": [],
        # Gradients of attention logits (scores)
        "p1_scores_grad_diag_mean": [], "p1_scores_grad_diag_max": [], "p1_scores_grad_diag_min": [],
        "p1_scores_grad_offdiag_mean": [], "p1_scores_grad_offdiag_max": [], "p1_scores_grad_offdiag_min": [],
        "p2_scores_grad_diag_mean": [], "p2_scores_grad_diag_max": [], "p2_scores_grad_diag_min": [],
        "p2_scores_grad_offdiag_mean": [], "p2_scores_grad_offdiag_max": [], "p2_scores_grad_offdiag_min": [],
        # Gradients of attention probs
        "p1_probs_grad_diag_mean": [], "p1_probs_grad_diag_max": [], "p1_probs_grad_diag_min": [],
        "p1_probs_grad_offdiag_mean": [], "p1_probs_grad_offdiag_max": [], "p1_probs_grad_offdiag_min": [],
        "p2_probs_grad_diag_mean": [], "p2_probs_grad_diag_max": [], "p2_probs_grad_diag_min": [],
        "p2_probs_grad_offdiag_mean": [], "p2_probs_grad_offdiag_max": [], "p2_probs_grad_offdiag_min": [],
    }


def train_model(model, optimizer, criterion, num_steps, batch_size, seq_len, vocab_size, device, history, model_name):
    """Train model through Phase 1 and Phase 2."""
    print(f"Starting {model_name}")

    for phase in [1, 2]:
        print(f"Starting Phase {phase} ...")
        prefix = f"p{phase}_"

        for step in range(num_steps[phase - 1]):
            model.train()
            optimizer.zero_grad()

            inputs = generate_unique_batch(batch_size, seq_len=seq_len, vocab_size=vocab_size, device=device)

            if phase == 1:
                targets = inputs.clone().float()
            elif phase == 2:
                # cumulative AVERAGE = cumsum / position
                positions = torch.arange(1, seq_len + 1, device=device).float().unsqueeze(0)
                targets = torch.cumsum(inputs, dim=1).float() / positions

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Basic gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            q_grad_norm = model.q_proj.weight.grad.norm().item()

            # Get attention statistics (forward pass values)
            # Use abs for logits since they can be positive/negative
            with torch.no_grad():
                scores_stats = get_diagonal_offdiagonal_stats(model.scores, use_abs=True)
                probs_stats = get_diagonal_offdiagonal_stats(model.attn_probs, use_abs=False)  # probs are already positive

            # Get gradient statistics (always use abs for gradients)
            if model.scores.grad is not None:
                scores_grad_stats = get_diagonal_offdiagonal_stats(model.scores.grad, use_abs=True)
            else:
                scores_grad_stats = {'diag_mean': 0, 'diag_max': 0, 'diag_min': 0,
                                     'offdiag_mean': 0, 'offdiag_max': 0, 'offdiag_min': 0}

            if model.attn_probs.grad is not None:
                probs_grad_stats = get_diagonal_offdiagonal_stats(model.attn_probs.grad, use_abs=True)
            else:
                probs_grad_stats = {'diag_mean': 0, 'diag_max': 0, 'diag_min': 0,
                                    'offdiag_mean': 0, 'offdiag_max': 0, 'offdiag_min': 0}

            optimizer.step()

            # Store all metrics
            history[f"{prefix}loss"].append(loss.item())
            history[f"{prefix}grad_norm"].append(total_norm)
            history[f"{prefix}q_grad_norm"].append(q_grad_norm)

            # Attention logits stats
            history[f"{prefix}scores_diag_mean"].append(scores_stats['diag_mean'])
            history[f"{prefix}scores_diag_max"].append(scores_stats['diag_max'])
            history[f"{prefix}scores_diag_min"].append(scores_stats['diag_min'])
            history[f"{prefix}scores_offdiag_mean"].append(scores_stats['offdiag_mean'])
            history[f"{prefix}scores_offdiag_max"].append(scores_stats['offdiag_max'])
            history[f"{prefix}scores_offdiag_min"].append(scores_stats['offdiag_min'])

            # Attention probs stats
            history[f"{prefix}probs_diag_mean"].append(probs_stats['diag_mean'])
            history[f"{prefix}probs_diag_max"].append(probs_stats['diag_max'])
            history[f"{prefix}probs_diag_min"].append(probs_stats['diag_min'])
            history[f"{prefix}probs_offdiag_mean"].append(probs_stats['offdiag_mean'])
            history[f"{prefix}probs_offdiag_max"].append(probs_stats['offdiag_max'])
            history[f"{prefix}probs_offdiag_min"].append(probs_stats['offdiag_min'])

            # Scores gradient stats
            history[f"{prefix}scores_grad_diag_mean"].append(scores_grad_stats['diag_mean'])
            history[f"{prefix}scores_grad_diag_max"].append(scores_grad_stats['diag_max'])
            history[f"{prefix}scores_grad_diag_min"].append(scores_grad_stats['diag_min'])
            history[f"{prefix}scores_grad_offdiag_mean"].append(scores_grad_stats['offdiag_mean'])
            history[f"{prefix}scores_grad_offdiag_max"].append(scores_grad_stats['offdiag_max'])
            history[f"{prefix}scores_grad_offdiag_min"].append(scores_grad_stats['offdiag_min'])

            # Probs gradient stats
            history[f"{prefix}probs_grad_diag_mean"].append(probs_grad_stats['diag_mean'])
            history[f"{prefix}probs_grad_diag_max"].append(probs_grad_stats['diag_max'])
            history[f"{prefix}probs_grad_diag_min"].append(probs_grad_stats['diag_min'])
            history[f"{prefix}probs_grad_offdiag_mean"].append(probs_grad_stats['offdiag_mean'])
            history[f"{prefix}probs_grad_offdiag_max"].append(probs_grad_stats['offdiag_max'])
            history[f"{prefix}probs_grad_offdiag_min"].append(probs_grad_stats['offdiag_min'])

            if step % 100 == 0:
                print(f"\tStep {step + 1}/{num_steps[phase - 1]} | Loss: {loss.item():.4f}")
                print(f"\t  |Logits| - Diag: mean={scores_stats['diag_mean']:.4f}, max={scores_stats['diag_max']:.4f}, min={scores_stats['diag_min']:.4f}")
                print(f"\t           - OffD: mean={scores_stats['offdiag_mean']:.4f}, max={scores_stats['offdiag_max']:.4f}, min={scores_stats['offdiag_min']:.4f}")
                print(f"\t  Probs    - Diag: mean={probs_stats['diag_mean']:.4f}, max={probs_stats['diag_max']:.4f}, min={probs_stats['diag_min']:.4f}")
                print(f"\t           - OffD: mean={probs_stats['offdiag_mean']:.6f}, max={probs_stats['offdiag_max']:.6f}, min={probs_stats['offdiag_min']:.6f}")
                print(f"\t  |LogitGrad|- Diag: mean={scores_grad_stats['diag_mean']:.6f}, max={scores_grad_stats['diag_max']:.6f}, min={scores_grad_stats['diag_min']:.6f}")
                print(f"\t            - OffD: mean={scores_grad_stats['offdiag_mean']:.6f}, max={scores_grad_stats['offdiag_max']:.6f}, min={scores_grad_stats['offdiag_min']:.6f}")
                print(f"\t  |ProbGrad|- Diag: mean={probs_grad_stats['diag_mean']:.6f}, max={probs_grad_stats['diag_max']:.6f}, min={probs_grad_stats['diag_min']:.6f}")
                print(f"\t           - OffD: mean={probs_grad_stats['offdiag_mean']:.6f}, max={probs_grad_stats['offdiag_max']:.6f}, min={probs_grad_stats['offdiag_min']:.6f}")

    print(f"\n{model_name} Training Complete\n")
    return history

# ===== fast runner (reuses original defs above) =====
if __name__ == '__main__':
    import sys
    p2_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    suffix   = sys.argv[2] if len(sys.argv) > 2 else 'cumavg'
    which    = sys.argv[3] if len(sys.argv) > 3 else 'both'
    temp_sm  = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0   # softmax temperature
    num_steps = [30000, p2_steps]
    # paper architecture: d_model=256, single head, short sequence, no PE
    batch_size, seq_len, vocab_size, lr, d_model, num_heads = 64, 10, 10, 1e-4, 256, 1
    out_dir = os.path.expanduser('~/lucid-synthetic-exp')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"cumavg run: num_steps={num_steps} suffix={suffix} which={which} device={device}", flush=True)

    def reseed():
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    if which in ('both', 'softmax'):
        reseed()
        h = create_history_dict()
        m = HypothesisTransformer(input_vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, seq_len=seq_len, temp=temp_sm).to(device)
        m.apply(lambda z: init_weights(z, d_model))
        h = train_model(m, optim.Adam(m.parameters(), lr=lr), nn.MSELoss(), num_steps, batch_size, seq_len, vocab_size, device, h, "Standard Softmax")
        np.savez(f'{out_dir}/exp_full_metrics_softmax_{suffix}.npz', **{k: np.array(v) for k, v in h.items()})
        print("Saved softmax", flush=True)

    if which in ('both', 'lucid'):
        reseed()
        h = create_history_dict()
        m = HypothesisLucidTransformer(input_vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, seq_len=seq_len).to(device)
        m.apply(lambda z: init_weights(z, d_model))
        h = train_model(m, optim.Adam(m.parameters(), lr=lr), nn.MSELoss(), num_steps, batch_size, seq_len, vocab_size, device, h, "LUCID")
        np.savez(f'{out_dir}/exp_full_metrics_lucid_{suffix}.npz', **{k: np.array(v) for k, v in h.items()})
        print("Saved lucid", flush=True)
    print("Done!", flush=True)
