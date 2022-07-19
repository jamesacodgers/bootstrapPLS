# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..PLS import PLS

d_grid = pd.read_csv("Examples/Data/MA_x.csv", index_col=0)
noiseless_cqa = pd.read_csv("Examples/Data/MA_y.csv", index_col=0)

# %%
pls = PLS(d_grid.to_numpy(), noiseless_cqa.to_numpy(), 2)
pls.gen_boot_params()
y_min = [0.9, -np.inf]
y_max = [np.inf, 0.002]
pls.set_Y_des(y_min, y_max)

resolution = 3

xx,yy = np.meshgrid(np.linspace(10,30, resolution), np.linspace(400,1400, resolution))

probs = np.empty(resolution**2)
probs0 = np.empty(resolution**2)

# %%
E = []
E0 = []
for i in range(resolution):
    for j in range(resolution):
        # print(xx[i,j], yy[i,j])
        print (i,j)
        p,e = pls.boot_MC_prob(np.array([xx[i,j],yy[i,j]]))
        p0,e0 = pls.zeroth_MC_prob(np.array([xx[i,j],yy[i,j]]))
        probs[i*resolution + j] = p
        probs0[i*resolution + j] = p0
        E.append(e)
        E0.append(e0)


prob_square = np.empty((resolution,resolution))
prob_square0 = np.empty((resolution,resolution))
for i in range(resolution):
    for j in range(resolution):
        prob_square[i,j] = probs[i*resolution + j]
        prob_square0[i,j] = probs0[i*resolution + j]



# %%
from scipy.stats import norm

x_flatten = xx.flatten()
y_flatten = yy.flatten()

pos_plot = pd.read_csv("Examples/Data/pos_plot.csv", index_col= 0)

noiseless_cqa_plot = pd.read_csv("Examples/Data/noiseless_cqa.csv", index_col= 0 )

true_prob = np.zeros((resolution,resolution))+1
in_out = np.zeros((resolution,resolution)) + 1 

for i in range(resolution):
    for j in range(resolution):
        l = i*resolution + j
        for k in range(2):
            if y_min[k] != None: 
                true_prob[i,j] = true_prob[i,j]*(1-norm(noiseless_cqa_plot[l,k],error_std[k]).cdf(y_min[k]))
                in_out[i,j] = in_out[i,j]*int(noiseless_cqa_plot[l,k] > y_min[k])
            if y_max[k] != None:
                true_prob[i,j] = true_prob[i,j]*norm(noiseless_cqa_plot[l,k], error_std[k]).cdf(y_max[k])
                in_out[i,j] = in_out[i,j]*int(noiseless_cqa_plot[l,k] < y_max[k])

# %%

fig, ax = plt.subplots(figsize = (8,8))
plt.title("bootstrap method")
cont_pls = ax.contour(xx, yy, prob_square, [0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9])
cont_true = ax.contour(xx, yy, true_prob, [0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9], linestyles = "dashed", color = "black")
ax.clabel(cont_pls, inline = True)
ax.clabel(cont_true, inline = True)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.savefig("MA_DS.pdf", pad_inches = 0, bbox_inches = "tight")
plt.show()

fig, ax = plt.subplots(figsize = (8,8))
plt.title("Zeroth order approx")
cont_pls = ax.contour(xx, yy, prob_square0, [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
cont_true = ax.contour(xx, yy, true_prob, [0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9], linestyles = "dashed", color = "black")
ax.clabel(cont_pls, inline = True)
ax.clabel(cont_true, inline = True)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.savefig("MA_DS_0th.pdf", pad_inches = 0, bbox_inches = "tight")
plt.show()
# %%
