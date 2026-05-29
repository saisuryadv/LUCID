"""Plot the reproduced run in the paper's p1p2_horizontal style.
Reads exp_full_metrics_{lucid,softmax}_paper_repro.npz (d_model=256, num_heads=1,
seq_len=10, cumulative-average Phase 2)."""
import numpy as np
import matplotlib.pyplot as plt

def smooth(y, window=500):
    pad = window // 2
    yp = np.pad(y, pad, mode='edge')
    return np.convolve(yp, np.ones(window)/window, mode='same')[pad:pad+len(y)]

import sys, os
base = os.environ.get('LUCID_OUT', '.')
suffix = sys.argv[1] if len(sys.argv) > 1 else 'paper_repro'
out_name = sys.argv[2] if len(sys.argv) > 2 else 'p1p2_horizontal_repro'
lucid = np.load(f'{base}/exp_full_metrics_lucid_{suffix}.npz')
softmax = np.load(f'{base}/exp_full_metrics_softmax_{suffix}.npz')

plt.rcParams.update({'font.size': 18, 'axes.labelsize': 20, 'xtick.labelsize': 16,
                     'ytick.labelsize': 16, 'legend.fontsize': 16})
lucid_color, softmax_color = '#2E86AB', '#E94F37'
p1 = len(lucid['p1_loss'])  # 30000

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Phase 1 + Phase 2', fontsize=24)

# Loss
ax1 = axes[0]
ll = smooth(np.concatenate([lucid['p1_loss'], lucid['p2_loss']]))
sl = smooth(np.concatenate([softmax['p1_loss'], softmax['p2_loss']]))
steps = np.arange(len(ll)) / 1000
ax1.plot(steps, ll, label='Lucid', color=lucid_color, lw=2.5)
ax1.plot(steps, sl, label='Softmax', color=softmax_color, lw=2.5)
ax1.axvline(x=p1/1000, color='gray', ls='--', alpha=0.7, label='Phase 1→2')
ax1.set_xlabel('Steps (x1000)'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 2)

# Offdiagonal Jacobian ratio (log)
ax2 = axes[1]
def ratio(d, ph):
    return np.abs(d[f'{ph}_scores_grad_offdiag_mean']) / (np.abs(d[f'{ph}_probs_grad_offdiag_mean']) + 1e-10)
lr = smooth(np.concatenate([ratio(lucid, 'p1'), ratio(lucid, 'p2')]))
sr = smooth(np.concatenate([ratio(softmax, 'p1'), ratio(softmax, 'p2')]))
ax2.plot(steps, lr, label='Lucid', color=lucid_color, lw=2.5)
ax2.plot(steps, sr, label='Softmax', color=softmax_color, lw=2.5)
ax2.axvline(x=p1/1000, color='gray', ls='--', alpha=0.7, label='Phase 1→2')
ax2.set_xlabel('Steps (x1000)'); ax2.set_ylabel('Offdiagonal Jacobian')
ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_yscale('log')

plt.tight_layout(); plt.subplots_adjust(top=0.9)
plt.savefig(f'{base}/{out_name}.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{base}/{out_name}.pdf', bbox_inches='tight')
print(f'Saved {out_name}.png / .pdf')
