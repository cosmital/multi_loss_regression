import numpy as np
from utils import *
colors = {0 : 'tab:blue', 1 : 'tab:red',  2: 'tab:green', 3: 'tab:orange'}

class Test():
    def __init__(self, testPars=[], T = 100):
        self.testPars = []
        self.T = T
        self.ptdict = []
        self.add_tests(testPars)
            # self.ptdict[itP] = {}
                # self.ptdict[itP][classk] = ''.join([ f'{par}: {tP[classk][par]}' for par in tP[classk].keys()] )
    def add_tests(self, testPars):
        l0 = len(self.testPars)
        self.testPars += testPars
        self.tstEnum = enumerate(self.testPars)
        for itP,tP in enumerate(testPars):
            # for classk in tP.keys():
            #     self.ptdict[itP][classk] = ''.join([ f'{par}, ' for par in tP[classk].keys()] ) + tP['absx']
            self.ptdict.append(f'{l0+itP}/{len(self.testPars)}')
            for lbl in ['function', 'population']:
                if lbl not in tP.keys():
                    tP.update({lbl:{}})
                else: self.ptdict[-1] += ''.join([ f'\t[{lbl}] {par}: {tP[lbl][par]}, ' for par in tP[lbl].keys()] )
    def draw_predict(self, pops, funcs, theory = True, empiric = True, recalculate = False, itPs = None):
        if itPs == None: itPs = range(len(self.testPars))
        arg_pt_fx = {}
        for var in ['mzss','vzss','deltass','tQs','mzss_pr','vzss_pr','deltass_pr','tQs_pr', 'drawings']:
            if var not in self.__dict__.keys():
                if var == 'drawings':
                    self.drawings = {itP: {cl : np.zeros((funcs.dimf, self.T)) for cl in pops.rglk} for itP in range(len(self.testPars))}
                else: self.__dict__[var] = {}
        for itP,tP in enumerate(self.testPars):
            if itP in itPs:
                print('')
                print(self.ptdict[itP])
                
                pops.update(tP['population'])
                funcs.update(tP['function'])
                if 'proj' not in tP.keys():
                    tP['proj'] = {cl : np.eye(funcs.dimf)[cl%funcs.dimf].reshape(1,-1) for cl in pops.rglk}
                tP['rglk'] = pops.rglk
                tP['dimf'] = funcs.dimf
                self.theory = theory
                self.empiric = empiric
                if empiric and (recalculate or (itP not in self.mzss_pr.keys())):
                    print('Empirical drawings...')

                    for i in range(self.T):
                        Xs = pops.draw_X()
                        W = funcs.fixed_ptW(Xs, pops.mds, returnQ=False)
                        if i % 50 ==0:
                            print(i, end = ', ')
                        x = {cl : pops.draw_X_cl(cl) for cl in tP['rglk']}
                        for cl in tP['rglk']:
                            self.drawings[itP][cl][:,i] = extend(x[cl],funcs.dimf).T@W
                    # W, Q, _ = funcs.fixed_ptW(Xs, returnQ=True)
                    # self.tQs_pr.append(Q)
                    # self.deltass_pr.append([mat_tr(Q, pops.Sigmas[cl])/pops.n for cl in pops.rglk])
                    self.mzss_pr[itP] = [self.drawings[itP][cl].mean(axis = 1) for cl in tP['rglk']]
                    self.vzss_pr[itP] = [self.drawings[itP][cl]@ self.drawings[itP][cl].T/self.T for cl in tP['rglk']]

                if theory and (recalculate or (itP not in self.mzss.keys())):
                    print('Theoretical prediction...')
                    mzs, vzs, deltas = get_expect_W(pops, funcs, arg = arg_pt_fx, limIt = 30, param_tol = 1)
                    # increase "param_tol" if you want to have a faster but less accurate convergence 
                    self.mzss[itP] = mzs
                    self.vzss[itP] = vzs
                    self.deltass[itP] = deltas
                    arg_pt_fx = {'mzs' : mzs.copy(), 'vzs' : vzs.copy(), 'deltas' : deltas.copy()}
