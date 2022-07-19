#%%
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from  scipy.stats import norm
import numpy as np
import numpy.random as rand
from PLS import PLS
from scipy.optimize import fsolve
import pandas as pd



#%%
l = [str(i) for i in range(68)]
Xdf = pd.read_csv("x_snv.csv", names = l  )
ydf = pd.read_csv("ydata.csv", names = ["API frac"])

X = np.array(Xdf)
y = np.array(ydf)

s = 50
N = 1000


X_train = np.array(X[:N])
X_test = np.array(pd.read_csv("x_v_snv.csv"))

y_train = np.array(y[:N])
y_test = np.array(pd.read_csv("y_v.csv"))

n_l = 3

pls = PLS(X_train,y_train,n_l)
pls.gen_boot_params(1000)
# %%
#  Make stuff for plot

N_test = np.shape(X_test)[0]
for_plot = np.linspace(min(y),max(y),20)





# %%
# Make plot
N_plot = 20

y_hat = np.zeros(N_plot)
y_pos = np.zeros((N_plot, 50))
y_pos_faber96 = np.zeros((N_plot, 50))
y_dens = np.zeros((N_plot,50))
y_dens_faber96 = np.zeros((N_plot, 50))

plot_indicies = np.arange(1,N_plot)*(N_test//N_plot)
N_plot = np.shape(plot_indicies)[0]
y_hat_indicies = np.zeros(N_plot)

for i in range(N_plot):
    print(i)
    y_hat_indicies[i] = pls.predict(X_test[plot_indicies[i]])
    f = lambda y: pls.prediction_cdf(X_test[plot_indicies[i]], y) - 0.025
    g = lambda y: pls.prediction_cdf(X_test[plot_indicies[i]], y) - 0.975
    y_min = fsolve(f, y_hat_indicies[i])
    y_max = fsolve(g, y_hat_indicies[i])
    y_min_faber96 = pls.zeroth_ppf(X_test[plot_indicies[i]], 0.025)
    y_max_faber96 = pls.zeroth_ppf(X_test[plot_indicies[i]], 0.975)
    y_pos[i] = np.linspace(y_min, y_max).reshape(50)
    y_pos_faber96[i] = np.linspace(y_min_faber96,y_max_faber96).reshape(50)

    for j in range(50):
        y_dens[i,j] = pls.boot_pdf(X_test[plot_indicies[i]],y_pos[i,j])
        y_dens_faber96[i,j] = pls.zeroth_pdf(X_test[plot_indicies[i]], y_pos_faber96[i,j])

scale = 0.3

l = np.argsort(y_hat_indicies)
# %%
y_test_indicies = y_test[plot_indicies]

plt.figure(figsize = (14,7))
for i in range(N_plot):
    print(i)
    plt.fill_betweenx(y_pos[l[i]], y_dens[l[i]]*scale + i  + 1 , -y_dens[l[i]]*scale+ i + 1 , color = "blue", alpha = 0.5)
    plt.fill_betweenx(y_pos_faber96[l[i]], y_dens_faber96[l[i]]*scale + i + 1, -y_dens_faber96[l[i]]*scale + i + 1, color = "orange", alpha = 0.5)
plt.scatter(np.arange(N_plot)+1, y_hat_indicies[l], label = "predicted y", color = "grey")
plt.scatter(np.arange(N_plot)+1, y_test_indicies[l], label = "observed y", color = "black", marker = "x")
plt.xlabel("Sample Number")
plt.ylabel("y")
plt.legend()
plt.savefig("violin_spectroscopy_{}.pdf".format(N))
plt.show()

# %%
