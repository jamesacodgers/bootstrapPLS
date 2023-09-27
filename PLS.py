from audioop import bias
import numpy as np
import numpy.linalg as lin
from scipy.stats import norm
from numpy.random import randint
from scipy.stats import f
from scipy.stats import chi2
from scipy.stats import t
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_t
import numpy.random as rand

class PLS:
    def __init__(self,X,Y,n_L):
        self.X = X
        self.Y = Y
        self.n_L = n_L
        self.n_s = np.shape(X)[0]
        self.n_x = np.shape(X)[1]
        self.n_y = np.shape(Y)[1]

        self.__boot__ = False


        self.__Y_min__ = None
        self.__Y_max__ = None


        self.W = np.zeros((self.n_x,self.n_L))

        self.T = np.zeros((self.n_s,self.n_L))
        self.P = np.zeros((self.n_x,self.n_L))
        self.Q = np.zeros((self.n_y,self.n_L))

        self.E = np.zeros((self.n_s,self.n_L))
        self.F = np.zeros((self.n_s,self.n_y))

        self.W,self.T,self.P,self.Q,self.X_mean,self.Y_mean= self.__PLS__(X,Y)
        self.find_eigenvalues()


        self.E = self.X - self.T@self.P.T
        self.F = self.Y - self.T@self.Q.T
        self.set_T2_lim()
        self.set_SPE_lim()
        self.find_SE()

    def __PLS__(self, X , Y ):
        W = np.zeros((self.n_x,self.n_L))

        T = np.zeros((self.n_s,self.n_L))
        P = np.zeros((self.n_x,self.n_L))
        Q = np.zeros((self.n_y,self.n_L))


        X_mean = np.mean(X,0)
        Y_mean = np.mean(Y,0)

        X = X - X_mean        #X and Y now used to itterate
        Y = Y - Y_mean

        for j in range(self.n_L):
            eigval, eigvec = lin.eig(X.T@Y@Y.T@X) # Find eigenvector for maximum covariance
            i = np.argmax(eigval)
            w = eigvec[:,i].real

            W[:,j] = w

            T[:,j] = X @ w                                             # find t, p ,q
            P[:,j] = X.T @ T[:,j]/lin.norm(T[:,j])**2
            Q[:,j] = Y.T @ T[:,j]/lin.norm(T[:,j])**2

            X = X - np.outer(P[:,j],T[:,j]).T
            Y = Y - np.outer(Q[:,j],T[:,j]).T
        return W,T,P,Q,X_mean,Y_mean

    def predict(self, x):
        y_new = 0
        x = x - self.X_mean
        for i in range(self.n_L):
          t = x.T@self.W[:,i]
          x = x - t*self.P[:,i]
          y_new += t*self.Q[:,i]
        y_new = y_new + self.Y_mean
        return y_new

    def boot_predict(self,x,j):
        y_new = 0
        x = x - self.X_mean_boot[j]
        for i in range(self.n_L):
          t = x.T@self.W_boot[j][:,i]
          x = x - t*self.P_boot[j][:,i]
          y_new += t*self.Q_boot[j][:,i]
        y_new = y_new+ self.Y_mean_boot[j]
        return y_new                                            
        
    def gen_boot_params(self,n_b = 1000, independent = False):
        self.__boot__ = True
        self.independent = independent
        self.n_b = n_b
        X_boot = np.zeros((self.n_s, self.n_x))
        Y_boot = np.zeros((self.n_s, self.n_y)) # only works for scalar Y


        self.P_boot = np.zeros((n_b,self.n_x,self.n_L))
        self.Q_boot = np.zeros((n_b,self.n_y,self.n_L))
        self.W_boot = np.zeros((n_b,self.n_x,self.n_L))

        self.T_boot = np.zeros((n_b,self.n_s,self.n_L))

        self.Y_mean_boot  = np.zeros((n_b,self.n_y))
        self.X_mean_boot = np.zeros((n_b,self.n_x))

        self.boot_err_cov = np.zeros((n_b, self.n_y, self.n_y) )

        for j in range(n_b):
            #find indicies for bootstrapping
            i_boot = randint(0,self.n_s, self.n_s)
            #Generate X and y data sets

            X_boot = np.zeros((self.n_s,self.n_x))
            Y_boot = np.zeros((self.n_s,self.n_y))

            for i in range(self.n_s):
                X_boot[i] = self.X[i_boot[i]]
                Y_boot[i] = self.Y[i_boot[i]]

            self.W_boot[j],self.T_boot[j],self.P_boot[j],self.Q_boot[j],self.X_mean_boot[j],self.Y_mean_boot[j] = self.__PLS__(X = X_boot, Y = Y_boot)

            prediction_error = np.empty((self.n_s, self.n_y))
            for i in range(self.n_s):
                prediction_error[i] = Y_boot[i] - self.boot_predict(X_boot[i],j)
            if independent == False: 
                self.boot_err_cov[j] = np.cov(prediction_error, bias = True, rowvar = False)
            else: 
                for k in range(self.n_y):
                    self.boot_err_cov[j,k,k] = np.var(prediction_error[:,k])

    def set_Y_des(self,Y_min = None, Y_max = None):
        if Y_min == None:
            self.__Y_min__ = [-np.inf]*self.n_y
        else:
            self.__Y_min__ = np.array(Y_min)
        if Y_max == None: 
            self.__Y_max__ = [np.inf]*self.n_y
        else:
            self.__Y_max__ = np.array(Y_max)
        
    def boot_prob(self,x):
        if self.__boot__ == False:
            raise Exception("bootstrapped paramiters have not yet been generated")
        if self.n_y > 2:
            raise Exception("Method only implimented for n_y <=2 ")
        if self.n_y == 1: 
            P = 1
            if self.__Y_max__ != None:
                P = 0
                for i in range(self.n_b):
                    P += 1/self.n_b*norm.cdf((self.__Y_max__-self.boot_predict(x,i))/np.sqrt(self.boot_err_cov[i]))
            if self.__Y_min__ != None:
                for i in range(self.n_b):
                    P -= 1/self.n_b*norm.cdf((self.__Y_min__-self.boot_predict(x,i))/np.sqrt(self.boot_err_cov[i]))         
        if self.n_y == 2: 
            y_min = [0]*2
            y_max = [0]*2
            for j in range(self.n_y):
                if self.__Y_max__[j] == None:
                    y_max[j] = np.inf
                else: 
                    y_max[j] = self.__Y_max__[j]
                if self.__Y_min__[j] == None:
                    y_min[j] = -np.inf
                else: 
                    y_min[j] = self.__Y_min__[j]
            boot_pdf = lambda y1,y2: self.boot_pdf(x,[y1,y2] )
            P,e = dblquad(boot_pdf, y_min[1], y_max[1], y_min[0], y_max[0],epsrel=1e-2)
        return P

    def find_eigenvalues(self):
        e = np.empty(self.n_L)
        for i in range(self.n_L):
            e[i] = np.var(self.T[:,i])
        self.eigval = e
    
    def set_T2_lim(self, alpha = 0.005):      
        self.T2 = np.zeros(self.n_s)
        for i in range(self.n_s):
            x = self.X[i]-self.X_mean
            t = np.empty(self.n_L)    
            for j in range(self.n_L):
                t[j] = np.dot(x,self.W[:,j])
                x = x - t[j]*self.P[:,j]  
                self.T2[i] += t[j]**2/self.eigval[j]
        T2_lim = self.n_L * (self.n_s**2 -1 )/ (self.n_s*(self.n_s - self.n_L))* f(self.n_L, self.n_s-self.n_L).ppf((1-alpha))
        self.T2_lim = T2_lim

    def set_SPE_lim(self, alpha = 0.005):
        self.spe = np.empty(self.n_s)
        for i in range(self.n_s):
            e = (self.X[i]- self.X_mean) - self.T[i] @ self.P.T
            self.spe[i] = e.T @ e
        mean = np.mean(self.spe)
        var = np.var(self.spe)
        spe_lim = var / (2*mean)*chi2(2*mean**2/var).ppf((1-  alpha))
        self.SPE_lim = spe_lim

    def in_KS(self, x):
        x = x - self.X_mean
        x_orig = x
        t = np.zeros(self.n_L)
        for i in range(self.n_L):
            t[i] = x.T@self.W[:,i]
            x = x - t[i]*self.P[:,i]
        e = x_orig - t@self.P.T
        SPE = np.dot(e,e)
        T2 = 0
        for i in range(self.n_L):
            T2 += t[i]**2/self.eigval[i]
        #print("SPE", self.SPE_lim, SPE, "T2", self.T2_lim, T2)
        if SPE < self.SPE_lim and T2 < self.T2_lim:
            return True
        else:
            return False

    def sampling(self,x):
        p = int(self.in_KS(x))*self.boot_prob(x)
        return(p)

    def boot_pdf(self,x,y):
        p = 0
        for i in range(self.n_b):
            p += 1/self.n_b*multivariate_normal(self.boot_predict(x,i),self.boot_err_cov[i,:,:]).pdf(y)
        return p

    def prediction_cdf(self,x,y):
        p = 0
        for i in range(self.n_b):
            p += 1/self.n_b*norm(self.boot_predict(x,i),np.sqrt(self.boot_err_cov[i,0,0])).cdf(y)
        return p
        
    def find_SE(self, independent = True):
        self.err_cov = np.zeros((self.n_y, self.n_y))
        prediction_error = np.empty((self.n_s, self.n_y))
        for i in range(self.n_s):
            prediction_error[i] = self.Y[i] - self.predict(self.X[i])
        if independent == False: 
            self.err_cov = np.cov(prediction_error, rowvar = False)
        else: 
            for k in range(self.n_y):
                self.err_cov[k,k] = np.var(prediction_error[:,k])
         
    def find_h_obs(self,x):
        t = np.empty(self.n_L)
        x = x - self.X_mean
        for i in range(self.n_L):
            t[i] = x.T@self.W[:,i]
            x = x - t[i]*self.P[:,i]
        h = t.T @ np.diag(1/self.eigval)@t / (self.n_s - 1)
        return h

    def s(self, x): 
        if self.n_y != 1:
            raise Exception("Not implimented for multi dimensional outputs")
        s = np.sqrt(self.err_cov[0,0]) * np.sqrt(1 + 1/self.n_s + self.find_h_obs(x))
        return s
    
    def zeroth_pdf(self, x ,y):
        if self.n_y > 2:
            raise Exception("Not implimented for n_y > 2")
        if self.n_y == 1:
            p = t.pdf((y - self.predict(x))/self.s(x), self.n_s - self.n_L - 1)/self.s(x)
        if self.n_y == 2:
            p = multivariate_t.pdf(y,self.predict(x),self.err_cov)
        return p 

    def zeroth_ppf(self, x , alpha): 
        if self.n_y != 1:
            raise Exception("Not implimented for multi dimensional outputs")
        y_lim = self.predict(x) + self.s(x)*t.ppf(alpha, self.n_s - self.n_L - 1)
        return y_lim
    
    def zeroth_prob(self,x):
        if self.n_y > 2: 
            raise Exception("Not implimented for n_y > 2")
        if self.n_y == 1:
            t_inst= t(self.n_s - self.n_L - 1)
            if self.__Y_min__ == None:
                t_cdf_min = 0 
            else:     
                t_cdf_min = t_inst.cdf((self.__Y_min__ - self.predict(x))/self.s(x))
            if self.__Y_max__ == None:
                t_cdf_max = 1
            else: 
                t_cdf_max = t_inst.cdf((self.__Y_max__ - self.predict(x))/self.s(x))
            prob = t_cdf_max - t_cdf_min
        if self.n_y == 2: 
            for j in range(self.n_y):
                y_min = [0]*2
                y_max = [0]*2
                if self.__Y_max__[j] == None:
                    y_max[j] == np.inf
                else: 
                    y_max[j] == self.__Y_max__[j]
                if self.__Y_min__[j] == None:
                    y_min[j] == -np.inf
                else: 
                    y_min[j] == self.__Y_max__[j]
            zeroth_pdf = lambda y1,y2: self.zeroth_pdf(x,[y1,y2] )
            prob,e = dblquad(zeroth_pdf, y_min[0], y_max[0], y_min[1], y_max[1])

        return prob

    def boot_MC_prob(self, x, n = 100000):
        p = 0
        for i in range(n):
            i = rand.randint(self.n_b)
            y_rand = rand.multivariate_normal(self.boot_predict(x,i), self.boot_err_cov[i])

            if np.all(y_rand<self.__Y_max__) and np.all(y_rand>self.__Y_min__):
                p += 1/n

        e = p*(1-p)/np.sqrt(n)
        return p, e

    def boot_MC_prob_sampling(self, x, n = 100):
        p = 0
        for i in range(n):
            i = rand.randint(self.n_b)
            y_rand = rand.multivariate_normal(self.boot_predict(x,i), self.boot_err_cov[i])

            if np.all(y_rand<self.__Y_max__) and np.all(y_rand>self.__Y_min__):
                p += 1/n
        ks = int(self.in_KS(x))
        return p*ks

    def zeroth_MC_prob(self, x, n = 100000):
        t = multivariate_t(self.predict(x),self.err_cov)
        p = 0
        for i in range(n):
            y_rand = t.rvs()
            if np.all(y_rand<self.__Y_max__) and np.all(y_rand>self.__Y_min__):
                p += 1/n
        e = p*(1-p)/np.sqrt(n)
        return p,e


        
