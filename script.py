pPath = "D:/low_density_classifiers/ldc_python_send"
import os
os.chdir(pPath)
from utils import *
from functs_and_pops import *
from test import *
# from functs_and_pops_work import *
import numpy as np
import time
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
from scipy.integrate import  dblquad, quad
import gzip
import sys
import pickle
from matplotlib import pyplot as plt
f = gzip.open('mnist.pkl.gz', 'rb')
data = pickle.load(f, encoding='bytes')
f.close()
(x_train, y_train), (x_test, y_test) = data

p = 100
n = 1000
factor = 1
p *= factor
n *= factor
x1 = x_train[y_train==4]
x2 = x_train[y_train==9]
A = np.random.randn(p,784)
mu1 = A@sum([x1[i,:,:].reshape(784) for i in range(x1.shape[0])])/x1.shape[0]
mu2 = A@sum([x2[i,:,:].reshape(784) for i in range(x2.shape[0])])/x2.shape[0]

mu1 = mu1.reshape(p,1) / np.linalg.norm(mu1)
mu2 = mu2.reshape(p,1) / np.linalg.norm(mu2)
mu = mu2-mu1
dimf = 2
# ms = [mu1, mu2, mu1, mu2]
ms = [mu, -mu, mu, -mu]
C = 0.5*np.eye(p)
Cs = [C, C, C, C]
ks = [0.02 ,0.02 ,0.48, 0.48]
assert sum(ks) == 1
cl2md = {0 :'su_0', 1 : 'su_1', 2 : 'un_0', 3 : 'un_1'}
pop1 = GaussianPopulation({'p': p, 'n': n, 'Cs' : Cs, 'ms' : ms, 'ks' : ks, 'cl2md' : cl2md})
loss_su = 'softmax'
loss_un = 'sqrt'
gammas_un = np.r_[np.linspace(-2,10,10),np.logspace(1,2,20)]
# gammas_un = np.r_[np.linspace(-2,1,6), np.logspace(0,2,100)]
np_tests = len(gammas_un)
# gammas_un = np.r_[np.logspace(-2,3,np_tests)]
itPs_emp = list(range(0,np_tests,6)) + [np_tests-1]
gammas_vec = [{'su_0' : 1,  'su_1': 1, 'un_0' : g, 'un_1' : g } for g in gammas_un]
funcs1 = Multiloss( {'loss_su' : loss_su, 'loss_un' : loss_un, 'lambdas' : np.array([1,1]), 'gammas' : gammas_vec[0], 'c': 1, 'b' : 1, 'limIt' : 50000, 'dimf': dimf})

testPars = []
for ke, gamma__ in enumerate(gammas_vec):
    # testPars.append({ 'function' : {'loss_su': loss_su, 'loss_un': loss_un, 'lambdas' : np.array([1,2])}, 'population' : {'n' : int(n)}})
    testPars.append({ 'function' : {'loss_su': loss_su, 'loss_un': loss_un, 'gammas' : gammas_vec[ke]}})
test1 = Test(testPars=testPars, T = 500)

start_time = time.time()
test1.draw_predict(pop1, funcs1, theory = True, empiric = False, recalculate= False)
end_time = time.time()
print('time to compute theory : ', end_time - start_time)

start_time = time.time()
test1.draw_predict(pop1, funcs1, theory = False, empiric = True, recalculate= False, itPs = itPs_emp)
end_time = time.time()
print('time to compute empiric results : ', end_time - start_time)





nb_graph = 1
# nb_graph = len(gammas_un)
cl2classif = {0: 0, 1: 1, 2: 0,3: 1}
fig, axs = plt.subplots(ncols=1, nrows=nb_graph, figsize=(10,4 * nb_graph),
                                constrained_layout=True)
ms = [0.1, 0.3, 0.5, 0.7, 0.9]
color_vec = ['r', 'g', 'c', 'k','m', 'b', 'r', 'g', 'k', 'c', 'm', 'b']
clStudy = [2,3]
if nb_graph == 1 : 
    axs = [axs]
    counter = 0


# for counter, loss_un in enumerate(loss_un_vec):
#     axs[counter].set_title(f'loss_un : {loss_un}', fontsize=25)
#     for ils, loss_su in enumerate(loss_su_vec):
pointsx_pr = []
pointsx_th = []
pointsy_pr = []
pointsy_th = []

for itP,tP in enumerate(test1.testPars):
    if itP in itPs_emp:
        pointsx_pr.append( tP['function']['gammas']['un_0'])
        # pointsx_pr.append(np.log(2 + tP['function']['gammas']['un_0']))
        pointsy_pr.append(sum(ks[cl]*funcs1.misclass(cl2classif[cl], test1.drawings[itP][cl]) for cl in clStudy)/ sum([ks[cl] for cl in clStudy]))
    pointsx_th.append(tP['function']['gammas']['un_0'])
    # pointsx_th.append(np.log(2 + tP['function']['gammas']['un_0']))
    # ct = sum(ks[cl]*min(funcs1.misclass_from_stats(cl2classif[cl], test1.mzss[itP][cl],test1.vzss[itP][cl]), funcs1.misclass_from_stats(1-cl2classif[cl], test1.mzss[itP][cl],test1.vzss[itP][cl])) for cl in clStudy)/ sum([ks[cl] for cl in clStudy])
    # pointsy_th.append(ct)
    pointsy_th.append(sum(ks[cl]*funcs1.misclass_from_stats(cl2classif[cl], test1.mzss[itP][cl],test1.vzss[itP][cl]) for cl in clStudy)/ sum([ks[cl] for cl in clStudy]))

axs[counter].plot(pointsx_pr, pointsy_pr, '*', color = color_vec[0], linewidth = 3, label =  f'pr, loss_su : {loss_su}')
axs[counter].plot(pointsx_th, pointsy_th, color = color_vec[0], linestyle = 'dashed', linewidth = 3, label =  f'th, loss_su : {loss_su}')
axs[counter].legend(loc = 'upper right', fontsize = 10)
axs[counter].tick_params(axis='both', which='major', labelsize=10)
axs[counter].grid()
axs[counter].set_ylim(-0.01, 0.6)

fig.savefig(pPath + '/figures/fig.png')


nb_graph = len(itPs_emp)
lbl_dict = {0: 'first entry', 1: 'second entry'}
fig2, ax = plt.subplots(ncols=1, nrows=nb_graph, figsize=(7, 4*nb_graph ), constrained_layout=True)
counter = 0
cl= 2
for itP,tP in enumerate(test1.testPars):
    # print(f'loss_un : {tP["function"]["loss_un"]}')
    # print(f'loss_su : {tP["function"]["loss_su"]}')
    if itP in itPs_emp:
        for kk in range(funcs1.dimf):
            ax[counter].set_title(f'gamma_un = {round(tP["function"]["gammas"]["un_0"], 3)}', fontsize=20)
            # ax[counter].set_title(f'n = {tP["population"]["n"]}, x in class {cl+1}', fontsize=40)
            v_th = float(tP['proj'][kk]@test1.vzss[itP][cl]@tP['proj'][kk].transpose())
            m_th = float(tP['proj'][kk]@test1.mzss[itP][cl])
            u = [m_th + i/30*np.sqrt(v_th) for i in range(-100,100)]
            # print(f'theo : class {cl}, mean: {round(m_th, 5)}, variance: {round(v_th, 5)}')
            gauss_th = [np.exp(-(uu - m_th)**2/2/v_th)/np.sqrt(2*v_th*np.pi) for uu in u]
            ax[counter].plot(u,gauss_th, color = colors[kk], linestyle = '--', label = f'Theoretical dist, {lbl_dict[kk]}')
            # empiric:
            m_pr = (tP['proj'][kk]@test1.drawings[itP][cl]).mean()
            v_pr = (tP['proj'][kk]@test1.drawings[itP][cl]).var()
            u = [m_pr + i/30*np.sqrt(v_pr) for i in range(-100,100)]
            # print(f'prat: class {cl}, mean: {round(m_pr, 5)}, variance: {round(v_pr, 5)}')
            ax[counter].hist((tP['proj'][kk]@test1.drawings[itP][cl]).reshape(-1), density = True, color = colors[kk], rwidth = .7, alpha = 0.5)
            gauss_pr = [np.exp(-(uu - m_pr)**2/2/v_pr)/np.sqrt(2*v_pr*np.pi) for uu in u]
            ax[counter].plot(u,gauss_pr, color = colors[kk], label = f'Empirical dist,  {lbl_dict[kk]}')
            # ax[counter].legend(loc = 'upper right', fontsize = 25)
            ax[counter].tick_params(axis='both', which='major', labelsize=25)
        counter+=1

fig2.savefig(pPath + '/figures/fig_histo.png')
    # plt.show()



# with open("D:\low_density_classifiers\ldc_python_send\script.py", 'r') as file:script_code = file.read()
# exec(script_code)

