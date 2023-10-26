import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import rcParams


params = {'legend.fontsize': 8,
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'axes.titleweight': 'bold',
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.title_fontsize': 8}

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42

rcParams.update(params)

fig, ax = plt.subplots(1, 1, figsize=(2, 2))

def exp_growth_system(N, t, r1, r2, r3):
    N1, N2, N3 = N
    out1 = r1
    out2 = r2
    out3 = r3
    return [out1, out2, out3]


def bound(value, low=0, high=200):
    return max(low, min(high, value))


n_sims = 1
s = 1
v = [50, 33.33, 16.667]
f1 = np.log10(1/1200)
f2 = np.log10(1/1200)

t = np.linspace(0, 250, 2001)
n = np.log10(2400)
K = 16000
# growth rate without drug
r0 = 1

# Thresholds for surviving cells
low = 125
high = 126

# Reduction in potency in single drug resistant cells
sf1 = 0.5
sf2 = 0.5

n0 = [f1 + n, f2 + n, n - (n + f1) - (n + f2)]
colors = ['black', 'red', 'pink']

benefits = [[], [], []]
for en, s in enumerate([0, 2, 8]):
    blisses = [[] for i in range(n_sims)]
    responses = [[] for i in range(n_sims)]
    responses_capped = [[] for i in range(n_sims)]
    resistances = [[] for i in range(n_sims)]
    deltas = [[] for i in range(n_sims)]
    res1 = [[] for i in range(n_sims)]
    res2 = [[] for i in range(n_sims)]
    sens = [[] for i in range(n_sims)]
    diffs = [[] for i in range(n_sims)]

    for cross in [-1, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, .875, 1]:
        slope_i = [[] for i in range(n_sims)]
        slope_f = [[] for i in range(n_sims)]

        R1i = v[en]
        R2i = v[en]
        R1i_1_res = R1i * sf1
        R2i_1_res = bound(R2i + cross * (R1i - R1i * sf1))
        R2i_2_res = R2i * sf2
        R1i_2_res = R1i
        r1 = r0 - r0 / 100 * (R1i_1_res + R2i_1_res + s * (R1i_1_res * R2i_1_res) / (R1i_1_res + R2i_1_res))
        r2 = r0 - r0 / 100 * (R1i_2_res + R2i_2_res + s * (R1i_2_res * R2i_2_res) / (R1i_2_res + R2i_2_res))
        R3 = R1i + R2i + s * (R1i * R2i) / (R1i + R2i)  # Growth with both
        R3_bliss = R1i + R2i
        r3 = r0 - r0 / 100 * R3
        bliss = R3 - R3_bliss

        sol = odeint(exp_growth_system, n0, t, args=(r1, r2, r3))

        con1 = sol.T[0]
        for i in range(len(t) - len(con1)):
            con1 = np.append(con1, 0)

        con2 = sol.T[1]
        for i in range(len(t) - len(con2)):
            con2 = np.append(con2, 0)

        con3 = sol.T[2]
        for i in range(len(t) - len(con3)):
            con3 = np.append(con3, 0)

        con1[con1 <= -1] = -1
        con2[con2 <= -1] = -1
        con3[con3 <= -1] = -1

        confluence = sum([con1, con2, con3])

        slope_i = confluence[1] - confluence[0]
        slope_f = confluence[2000] - confluence[1999]

        benefit = -(r3 - r1)
        benefits[en].append(benefit)


plt.plot([-1, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, .875, 1],
         benefits[0])
plt.plot([-1, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, .875, 1],
         benefits[1])
plt.plot([-1, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, .875, 1],
         benefits[2])

plt.xlabel('Collateral effect')
plt.ylabel('Fitness benefit')
plt.savefig('resistance_sensitivity.pdf', bbox_inches='tight', dpi=600)
plt.show()
