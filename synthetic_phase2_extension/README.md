# Synthetic two-phase experiment — Phase 2 extended to 30K

Reproduces the LUCID-vs-Softmax two-phase synthetic result and extends Phase 2
from 10K to 30K steps. Figure: [`assets/p1p2_horizontal_ext30k.png`](../assets/p1p2_horizontal_ext30k.png).

- **Phase 1 (30K steps):** identity task → forms a deep attention sink (`probs_diag → 0.9998`, `scores_diag → 67`).
- **Phase 2 (30K steps):** cumulative-average task (`cumsum / position`). Softmax stays
  stuck (loss ≈ 1.2–1.3) because it cannot escape the sink; LUCID recovers to ≈ 0.

## Config (important)
`d_model=256, num_heads=1, seq_len=10, vocab_size=10, temp=1.0, lr=1e-4, batch=64,
seed=42`, **no positional encoding**, MSE loss. The deep sink only forms with a
**single head** at **seq_len=10**; larger head counts dilute it and longer sequences
let the model solve identity by same-token averaging (no sink).

## Run
```bash
# args: <phase2_steps> <output_suffix> <model: softmax|lucid|both> <softmax_temp>
python experiment1_cumavg.py 30000 ext30k softmax 1.0
python experiment1_cumavg.py 30000 ext30k lucid   1.0
# writes exp_full_metrics_{softmax,lucid}_ext30k.npz, then:
python plot_paper_repro.py ext30k p1p2_horizontal_ext30k
```
