import numpy as np
from scipy.integrate import odeint
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

fig, ax = plt.subplots(1, 1, figsize=(7 * 0.6, 2))

def exp_growth_system(N, t, r1, r2, r3):
    N1, N2, N3 = N
    out1 = r1
    out2 = r2
    out3 = r3
    return [out1, out2, out3]


n_sims = 25
s = 1
R1 = np.linspace(0.1, 100, n_sims)
R2 = np.linspace(0.1, 100, n_sims)
f1 = np.log10(1/1200)
f2 = np.log10(1/1200)
x1, x2 = np.meshgrid(R1, R2, indexing='ij')

t = np.linspace(0, 250, 2001)
n = np.log10(2400)
K = 16000
# growth rate without drug
r0 = 1
# Reduction in potency in single drug resistant cells
sf1 = 0.5
sf2 = 0.5
# Thresholds for surviving cells
low = 125
high = 126

n0 = [f1 + n, f2 + n, n - (n + f1) - (n + f2)]

for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    blisses = [[] for i in range(n_sims)]
    responses = [[] for i in range(n_sims)]
    responses_capped = [[] for i in range(n_sims)]
    resistances = [[] for i in range(n_sims)]
    deltas = [[] for i in range(n_sims)]
    res1 = [[] for i in range(n_sims)]
    res2 = [[] for i in range(n_sims)]
    sens = [[] for i in range(n_sims)]
    diffs = [[] for i in range(n_sims)]
    slope_i = [[] for i in range(n_sims)]
    slope_f = [[] for i in range(n_sims)]

    for k in range(n_sims):
        for j in range(n_sims):
            R1i = x1[k][j]
            R2i = x2[k][j]
            r1 = r0 - r0 / 100 * (R1i + sf1 * R2i + s * (R1i * sf1 * R2i) / (R1i + sf1 * R2i))
            r2 = r0 - r0 / 100 * (R1i * sf2 + R2i + s * (R1i * sf2 * R2i) / (R1i * sf2 + R2i))
            R3 = R1i + R2i + s * (R1i * R2i) / (R1i + R2i)  # Growth with both
            R3_bliss = R1i + R2i
            r3 = r0 - r0 / 100 * R3
            bliss = R3 - R3_bliss
            diff = R3 - max((R1i + sf1 * R2i + s * (R1i * sf1 * R2i) / (R1i + sf1 * R2i)),
                            (R1i * sf2 + R2i + s * (R1i * sf2 * R2i) / (R1i * sf2 + R2i)))
            diffs[k].append(diff)
            blisses[k].append(bliss)
            responses[k].append(R3_bliss)
            responses_capped[k].append(min(R3, 200))

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

            res1[k].append(con1[200])
            res2[k].append(con2[200])
            sens[k].append(con3[200])

            con1[con1 <= -1] = -1
            con2[con2 <= -1] = -1
            con3[con3 <= -1] = -1

            confluence = sum([con1, con2, con3])

            slope_i[k].append(confluence[1] - confluence[0])
            slope_f[k].append(confluence[2000] - confluence[1999])

    benefit = (np.array(slope_f) - np.array(slope_i)) / 0.125

    plt.plot(s, np.median(benefit), 'ko')
    print(s, np.median(benefit))

plt.xlabel('Synergy parameter (s)')
plt.ylabel('Median fitness benefit')
plt.savefig('increasing synergy.pdf', bbox_inches='tight', dpi=600)
