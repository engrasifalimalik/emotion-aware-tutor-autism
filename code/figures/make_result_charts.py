"""Result charts for Chapter 5. All values are copied from Tables 5.1-5.8 (experiment outputs)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures') + os.sep
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .3, 'axes.axisbelow': True})
DOM = ['#2b5d8a', '#e08a2c', '#b33a3a']          # adult, child, autistic
BB = ['#2b5d8a', '#2e7d3a', '#9a7400', '#6a3d8f']  # backbones
def save(f, n):
    f.tight_layout(); f.savefig(OUT + n + '.png', dpi=200, facecolor='white'); plt.close(f)

# 5.1 accuracy & macro-F1 by backbone and domain
bk = ['MobileNetV3-L', 'EfficientNet-B0', 'ResNet-50', 'ViT-S/16']
acc = np.array([[.446, .354, .205], [.444, .316, .163], [.428, .318, .151], [.560, .408, .239]])
f1 = np.array([[.397, .298, .137], [.412, .263, .129], [.400, .258, .117], [.504, .326, .124]])
f, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
x = np.arange(4); w = .26
for ax, M, t in [(axs[0], acc, 'Accuracy'), (axs[1], f1, 'Macro-F1')]:
    for j, d in enumerate(['Adult', 'Child', 'Autistic']):
        b = ax.bar(x + (j - 1) * w, M[:, j], w, color=DOM[j], label=d)
        ax.bar_label(b, fmt='%.2f', fontsize=7.5, padding=1)
    ax.set_xticks(x, bk); ax.set_title(t, fontweight='bold'); ax.set_ylim(0, .62)
axs[0].axhline(1 / 7, ls='--', c='grey', lw=1); axs[0].text(-0.45, 1 / 7 + .008, 'chance (7-class)', ha='left', fontsize=8, color='grey')
axs[0].set_ylabel('Score'); axs[1].legend(frameon=False, title='Test domain')
save(f, 'Figure5_1_C1_Domain_Gap')

# 5.2 per-class recall heatmap (ViT-S)
cls = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
R = np.array([[.45, .43, .33, .80, .43, .70, .54], [.33, .05, .12, .50, .19, .58, .63],
              [.50, np.nan, .05, .84, .01, .02, .02]])
f, ax = plt.subplots(figsize=(8.5, 3)); ax.grid(False)
im = ax.imshow(np.ma.masked_invalid(R), cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i in range(3):
    for j in range(7):
        v = R[i, j]; ax.text(j, i, '–' if np.isnan(v) else f'{v:.2f}', ha='center', va='center',
                              color='white' if (not np.isnan(v) and v > .6) else 'black', fontsize=10)
ax.set_xticks(range(7), cls); ax.set_yticks(range(3), ['Adult', 'Child', 'Autistic'])
f.colorbar(im, ax=ax, label='Recall', fraction=.03)
save(f, 'Figure5_2_C1_PerClass_Recall')

# 5.3 label-efficiency curve (frozen probes)
fr = [0, 10, 25, 50, 100]
T = {'MobileNetV3': [.137, .245, .351, .438, .562], 'EfficientNet-B0': [.180, .279, .357, .430, .525],
     'ResNet-50': [.146, .244, .296, .358, .479], 'ViT-S': [.111, .343, .442, .536, .658]}
coral = {'MobileNetV3': .157, 'EfficientNet-B0': .163, 'ResNet-50': .129, 'ViT-S': .113}
f, ax = plt.subplots(figsize=(7.5, 4.3))
for (k, v), c in zip(T.items(), BB):
    ax.plot(fr, v, 'o-', color=c, label=k, lw=2)
    ax.scatter([0], [coral[k]], marker='x', color=c, s=50, zorder=3)
ax.scatter([], [], marker='x', color='grey', label='CORAL (0% labels)')
ax.set_xticks(fr, ['0%\n(source only)', '10%', '25%', '50%', '100%\n(target only)'])
ax.set_xlabel('Fraction of labelled autistic training images'); ax.set_ylabel('Autistic macro-F1')
ax.set_ylim(0, .72); ax.legend(frameon=False)
save(f, 'Figure5_3_C2_Label_Efficiency')

# 5.4 end-to-end fine-tuning (MobileNetV3, 3 seeds)
lab = ['Source only\n(0%)', 'DANN\n(0%, unlabelled)', 'FT + target\n(25%)', 'FT + target\n(50%)']
m = [.134, .228, .460, .581]; s = [.028, .016, .036, .002]
am = [.510, .478, .504, .499]; asd = [.007, .006, .013, .013]
f, ax = plt.subplots(figsize=(8, 4.2)); x = np.arange(4); w = .36
b1 = ax.bar(x - w / 2, m, w, yerr=s, capsize=4, color='#b33a3a', label='Autistic macro-F1')
b2 = ax.bar(x + w / 2, am, w, yerr=asd, capsize=4, color='#2b5d8a', label='Adult accuracy (forgetting check)')
ax.bar_label(b1, fmt='%.3f', fontsize=8, padding=3); ax.bar_label(b2, fmt='%.3f', fontsize=8, padding=3)
ax.set_xticks(x, lab); ax.set_ylim(0, .7); ax.set_ylabel('Score (mean ± SD, 3 seeds)'); ax.legend(frameon=False, loc='upper left')
save(f, 'Figure5_4_C2_FineTuning')

# 5.5 C3 AUC per learning state
st = ['Bored\n(54%)', 'Confused\n(33%)', 'Frustrated\n(22%)', 'Low engagement\n(5%)']
emo = [.610, .641, .612, .637]; lo = [.58, .61, .58, .59]; hi = [.64, .67, .64, .69]
emb = [.562, .646, .586, .610]
f, ax = plt.subplots(figsize=(8, 4.2)); x = np.arange(4); w = .36
err = [np.array(emo) - lo, np.array(hi) - emo]
b1 = ax.bar(x - w / 2, emo, w, yerr=err, capsize=4, color='#2e7d3a', label='7-d emotion vector (95% CI)')
b2 = ax.bar(x + w / 2, emb, w, color='#9aa7b4', label='128-d face embedding')
ax.bar_label(b1, fmt='%.3f', fontsize=8, padding=6); ax.bar_label(b2, fmt='%.3f', fontsize=8, padding=2)
ax.axhline(.5, ls='--', c='grey', lw=1); ax.text(-0.45, .505, 'chance (AUC 0.5)', ha='left', fontsize=8, color='grey')
ax.set_xticks(x, st); ax.set_ylim(.45, .75); ax.set_ylabel('Test ROC-AUC'); ax.legend(frameon=False, loc='upper left')
save(f, 'Figure5_5_C3_Learning_States')

# 5.6 C4 prediction vs decision
f, axs = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axs[0]; mods = ['XGBoost', 'DKT (LSTM)']; cog = [.7039, .7189]; fus = [.7052, .7195]; x = np.arange(2); w = .34
b1 = ax.bar(x - w / 2, cog, w, color='#9aa7b4', label='Cognitive only'); b2 = ax.bar(x + w / 2, fus, w, color='#9a7400', label='Cognitive + affect')
ax.bar_label(b1, fmt='%.4f', fontsize=8); ax.bar_label(b2, fmt='%.4f', fontsize=8)
ax.set_xticks(x, mods); ax.set_ylim(.69, .725); ax.set_ylabel('Next-answer AUC')
ax.set_title('(a) Prediction: negligible gain', fontweight='bold'); ax.legend(frameon=False, loc='upper left')
ax = axs[1]; x = np.arange(2)
b1 = ax.bar(x - w / 2, [21.8, 44.7], w, color='#9aa7b4', label='Performance-only rule')
b2 = ax.bar(x + w / 2, [27.0, 43.6], w, color='#6a3d8f', label='Emotion-aware rule')
ax.bar_label(b1, fmt='%.1f%%', fontsize=8); ax.bar_label(b2, fmt='%.1f%%', fontsize=8)
ax.set_xticks(x, ['Recall', 'Precision']); ax.set_ylim(0, 55); ax.set_ylabel('% of upcoming incorrect answers')
ax.set_title('(b) Decision: +5.3 pts recall', fontweight='bold'); ax.legend(frameon=False, loc='upper left')
save(f, 'Figure5_6_C4_Fusion')
print('ok')
