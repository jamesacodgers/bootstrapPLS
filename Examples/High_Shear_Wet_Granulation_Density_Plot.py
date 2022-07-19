#%%
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from  scipy.stats import norm
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
from copy import deepcopy
from scipy.stats import skew

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from PLS import PLS


def find_distance_to_W(x,pls):
    x = x - pls.X_mean
    t_vals = []
    distance_vals = []
    for i in range(pls.n_L):
        t = np.dot(pls.W[:,i], x)
        distance = x - t*pls.W[:,i]
        distance_vals.append(distance)
        t_vals.append(t)
        x = x - t*pls.P[:,i]
    return np.array(t_vals), np.array(distance_vals)

def T_boot(x,pls): 
    T_boot = np.empty((pls.n_b, pls.n_L))
    for i in range(pls.n_b):
        xi = deepcopy(x)
        for j in range(pls.n_L):
            T_boot[i,j] = np.dot(pls.W_boot[i,:,j],xi)
            T_boot[i,j] = T_boot[i,j]
            xi -= T_boot[i,j]*pls.P_boot[i,:,j]
    return T_boot
    


data = pd.read_csv("Vemavarapu_data.csv")

X = np.array(data[['X1','X2','X3','X4','X5','X6']],dtype=float)

y = np.array(data[['Y1']],dtype = float)



#Transform inputs
X[:,0] = np.log(X[:,0])
X[:,2:5] = np.log(X[:,2:5])
X[:,5] = np.cbrt(X[:,5])

y = np.sqrt(y)






n_l = 3
pls = PLS(X,y,n_l)
n_s = pls.n_s
for_plot = np.linspace(min(y),max(y),20)


y_hat = np.zeros(n_s)
y_pos = np.zeros((n_s, 50))
y_pos_zeroth = np.zeros((n_s, 50))
y_dens = np.zeros((n_s,50))
y_dens_zeroth = np.zeros((n_s, 50))


for i in range(pls.n_s):
    print(i)
    Xi = np.vstack((X[:i],X[i+1:]))
    yi = np.vstack((y[:i],y[i+1:]))
    pls = PLS(Xi,yi,2)

    pls.gen_boot_params(n_b = 1000)
    y_hat[i] = pls.predict(X[i])
    f = lambda y: pls.prediction_cdf(X[i], y) - 0.025
    g = lambda y: pls.prediction_cdf(X[i], y) - 0.975
    y_min = fsolve(f, y_hat[i])
    y_max = fsolve(g, y_hat[i])
    y_min_zeroth = pls.zeroth_ppf(X[i], 0.025)
    y_max_zeroth = pls.zeroth_ppf(X[i], 0.975)
    y_pos[i] = np.linspace(y_min, y_max).reshape(50)
    y_pos_zeroth[i] = np.linspace(y_min_zeroth,y_max_zeroth).reshape(50)
    for j in range(50):
        y_dens[i,j] = pls.boot_pdf(X[i],y_pos[i,j])
        y_dens_zeroth[i,j] = pls.zeroth_pdf(X[i], y_pos_zeroth[i,j])


# %%




l = np.argsort(y_hat)
fig,ax1 = plt.subplots(figsize = (14,7))
for i in range(n_s):
    ax1.fill_betweenx(y_pos[l[i]], y_dens[l[i]]*scale + i  + 1 , -y_dens[l[i]]*scale+ i + 1 , color = "blue", alpha = 0.5)
    ax1.fill_betweenx(y_pos_zeroth[l[i]], y_dens_zeroth[l[i]]*scale + i + 1, -y_dens_zeroth[l[i]]*scale + i + 1, color = "orange", alpha = 0.5)
ax1.scatter(np.arange(n_s)+1, y_hat[l], label = "predicted y", color = "grey")
ax1.scatter(np.arange(n_s)+1, y[l], label = "observed output", marker = "x", color = "black")
ax1.set_xlabel("Sample Number")
ax1.set_ylabel("y")
ax1.axhline(np.mean(y), label = "y mean")
ax1.legend(loc = "upper left")
plt.savefig("vemavarapu_prediction.pdf", pad_inches = 0, bbox_inches = "tight")
plt.show()

# %%


