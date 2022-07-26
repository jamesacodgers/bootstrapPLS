#%%

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import numpy.random as rand
from scipy.stats import f
from scipy.optimize import fsolve
from scipy.stats import norm
from scipy.stats import t

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from PLS import PLS

# %%
#define parameters

n_s = 40

mu_1 = 5
mu_2 = 5

s_1 = 5
s_2 = 5

s_e = 0.2 

y_min = 8.0
y_max = 12.0

alpha = 0.90

f = lambda x: norm.cdf(y_max - x, scale = s_e) - norm.cdf(y_min-x, scale =s_e)


#%%
rand.seed(1234)

x1 = rand.normal(mu_1,s_1,n_s)
x2 = rand.normal(mu_2,s_2,n_s)
X = np.vstack((x1,x2)).T
y = np.zeros((n_s,1))
for i in range(n_s):
    y[i,0] = x2[i] + rand.normal(0,s_e)



pls = PLS(X,y,1)
pls.gen_boot_params(n_b = 1000)



# %%

N = 100
plot_range = np.linspace(-10,20,N)
samples = np.empty((N**2,2))

x1x1,x2x2 = np.meshgrid(plot_range,plot_range)

prob_boot = np.empty((N,N))
prob_0th = np.empty((N,N))
true_prob = np.empty((N,N))

pls.set_Y_des([y_min],[y_max])

for i in range(N):
    print(i)
    for j in range(N):
        prob_boot[i,j] = pls.boot_prob([x1x1[i,j],x2x2[i,j]])
        prob_0th[i,j] = pls.zeroth_prob([x1x1[i,j],x2x2[i,j]])
        true_prob[i,j] = f(x2x2[i,j])


# %%



latent_space_negative = pls.X_mean.reshape((2,1)) - pls.W*20
latent_space_postitive = pls.X_mean.reshape((2,1)) + pls.W*20

latent_space_negative_boot = pls.X_mean_boot.reshape((1000,2,1)) - pls.W_boot*20
latent_space_postitive_boot = pls.X_mean_boot.reshape((1000,2,1)) + pls.W_boot*20

point1 = pls.X_mean.reshape((2,1)) - pls.W*5.5
point2 = pls.X_mean.reshape((2,1)) - pls.W*5.5 + 8*np.array([pls.W[1], -pls.W[0]])
point1= point1.reshape(2)
point2 = point2.reshape(2)


# %%
from matplotlib.lines import Line2D
rainbow = plt.get_cmap("brg")

p_list = [0.2,0.9]
fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(X[:,0],X[:,1], marker = "x", c = "black", label = "training data")
for i in range(1000):
    if i == 0:
        ax.plot([latent_space_negative_boot[i][0,0], latent_space_postitive_boot[i][0,0]],
            [latent_space_negative_boot[i][1,0], latent_space_postitive_boot[i][1,0]], 
            linestyle = "-", c = "orange", alpha = 0.02)
    else:
        ax.plot([latent_space_negative_boot[i][0,0], latent_space_postitive_boot[i][0,0]],
            [latent_space_negative_boot[i][1,0], latent_space_postitive_boot[i][1,0]], 
            linestyle = "-", c = "orange", alpha = 0.02)
ax.set_ylim(-10,20)
cont_pls = ax.contour(x1x1, x2x2, prob_boot, p_list)
cont_true = ax.contour(x1x1, x2x2, true_prob, p_list, linestyles = "dashed")
ax.clabel(cont_pls, inline = True )
ax.clabel(cont_true, inline = True)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.scatter(point1[0],point1[1], label = "point 1", color = "red", marker = "x", s = 100 )
ax.scatter(point2[0],point2[1], label = "point 2", color = "purple", marker = "*", s = 100)
handles, labels = plt.gca().get_legend_handles_labels()
labels.insert(0,"Bootstrapped latent space")
handles.insert(0, Line2D([0], [0], label='Bootstrapped Latent Space', color='orange'))
ax.legend(handles = handles, labels = labels, loc = 3)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.savefig("simple_DS.pdf", pad_inches = 0, bbox_inches = "tight")
plt.show()


# %%


fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(X[:,0],X[:,1], marker = "x", c = "black", label = "training data")
ax.plot([latent_space_negative[0,0], latent_space_postitive[0,0]],
        [latent_space_negative[1,0], latent_space_postitive[1,0]], label = "Latent Space", color = "orange")
ax.set_ylim(-10,20)
cont_pls = ax.contour(x1x1, x2x2, prob_0th, p_list)
cont_true = ax.contour(x1x1,x2x2, true_prob, p_list, linestyles = "dashed")
ax.clabel(cont_pls, inline = True)
ax.clabel(cont_true, inline = True)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.scatter(point1[0],point1[1], label = "point 1", color = "red", marker = "x", s = 100 )
ax.scatter(point2[0],point2[1], label = "point 2", color = "purple", marker = "*", s = 100)
ax.legend(loc = 3)
plt.savefig("simple_DS_0th.pdf", pad_inches = 0, bbox_inches = "tight")
plt.show()

# %%
n_p = 200
rv = norm(pls.boot_predict(point1,0), pls.err_cov[0,0])
y_input = np.linspace(pls.zeroth_ppf(point1, 0.0001), pls.zeroth_ppf(point1, 0.9999),n_p).reshape(n_p)




# %%


point1_pdf = np.zeros(n_p)
point2_pdf = np.zeros(n_p)
point1_faber96_pdf = np.zeros(n_p)
point2_faber96_pdf = np.zeros(n_p)

for i in range(n_p):
    point1_faber96_pdf[i] = pls.zeroth_pdf(point1, y_input[i])
    point2_faber96_pdf[i] = pls.zeroth_pdf(point2, y_input[i])
    point1_pdf[i] = pls.boot_pdf(point1, y_input[i])
    point2_pdf[i] = pls.boot_pdf(point2, y_input[i])
    print(i)

# %%


plt.figure(figsize=(10,10))
plt.plot(y_input, point1_pdf, c = "red",label = "point 1  bootstrap")
plt.plot(y_input, point2_pdf, c = "purple", label = "point 2  bootstrap")
plt.plot(y_input, point1_faber96_pdf, c = "red", linestyle = "--", label = "point 1  zeroth order")
plt.plot(y_input, point2_faber96_pdf, c = "purple", linestyle = ":", label = "point 2  zeroth order")
plt.ylim(0,0.75)
plt.fill_betweenx(np.linspace(0,0.75),y_min, y_max, color = "wheat", alpha = 0.5, label = "desired y range")
plt.xlim(y_input[0], y_input[-1])
plt.xlabel("y", fontsize = 15)
plt.ylabel("p(y| x, D)", fontsize = 15)
plt.legend()
plt.savefig("simple_prediction.pdf", pad_inches = 0, bbox_inches = "tight")
plt.show()




