import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


params = {'legend.fontsize': 9,
          'axes.labelsize': 9,
          'axes.titlesize': 9,
          'axes.titleweight': 'bold',
          'xtick.labelsize': 9,
          'ytick.labelsize': 9,
          'legend.title_fontsize': 9}

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42

rcParams.update(params)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7 * .75, 2))

dose = np.linspace(0, 100, 201)
XA, XB = np.meshgrid(dose, dose)


def synergy(x, y, s):
    num = x + y + (s * x * y) / (x + y)
    return num


Z1 = synergy(XA, XB, s=0)
Z2 = synergy(XA, XB, s=1.5)
Z3 = synergy(XA, XB, s=4)

ax1.contour(XA, XB, Z1, colors='black', levels=[25, 50, 75, 100, 125, 150, 175, 200], linewidths=1)
ax2.contour(XA, XB, Z2, colors='black', levels=[25, 50, 75, 100, 125, 150, 175, 200], linewidths=1)
ax3.contour(XA, XB, Z3, colors='black', levels=[25, 50, 75, 100, 125, 150, 175, 200], linewidths=1)

ax1.contourf(XA, XB, Z1, levels=[0, 25, 50, 75, 100, 125, 150, 175, 200], alpha=0.4, cmap='PiYG_r')
ax2.contourf(XA, XB, Z2, levels=[0, 25, 50, 75, 100, 125, 150, 175, 200], alpha=0.4, cmap='PiYG_r')
ax3.contourf(XA, XB, Z3, levels=[0, 25, 50, 75, 100, 125, 150, 175, 200], alpha=0.4, cmap='PiYG_r')

for ax in [ax1, ax2, ax3]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks(np.arange(0, 101, 50))
    ax.arrow(50, 50, 0, -25, width=1.25, color='xkcd:crimson')
    ax.plot(50, 50, 'o', markersize=2, color='xkcd:crimson')

ax2.set_xlabel('Dose of drug A')
ax1.set_ylabel('Dose of drug B')

ax2.yaxis.set_ticklabels([])
ax3.yaxis.set_ticklabels([])

plt.tight_layout()

plt.savefig('isobols.pdf', dpi=600)
