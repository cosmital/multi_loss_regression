from utils import *
from scipy.special import softmax


class Mapping():
    def check_derivative(self, nbTests = 1, mds = [], hlim = 1, nbIncr2 = 100):
        x0 = np.zeros(self.dimf)
        for md in mds:
            print(f'mode: {md}')
            for _ in range(nbTests):
            # md = np.random.choice(self.mds)
                proj = np.random.rand(self.dimf)
                h = np.random.rand(self.dimf)
                pt = x0 + np.random.randn(self.dimf)
                xs = [pt + (hlim/nbIncr2)*i* h for i in range(-nbIncr2,nbIncr2)]
                prjXs = [proj@x for x in xs]
                ys = [proj@self.f(md, x) for x in xs]
                print(f'eigen values must be negative: {np.linalg.eigvals(self.df(md, pt))}')
                tengent = [proj@(self.f(md, pt) + hsc *self.df(md, pt)@h) for hsc in [-hlim, hlim]]
                # plt.plot(prjXs, ys, legend = f'mode: {md}')
                plt.plot(prjXs, ys, color = 'tab:blue')
                plt.scatter(proj@pt,proj@self.f(md, pt), color = 'tab:red')
                plt.plot([prjXs[0],prjXs[-1]], tengent,  color = 'tab:red', linestyle = 'dashed')
            plt.show()
    def xi(self, md, x, delta):
        T = (1 if len(x.shape)==1 else x.shape[1])
        if ('xi_init' not in self.__dict__.keys()) or (self.xi_init.shape[1] != T):
            self.xi_init = x.copy()
        z = self.__dict__['xi_init'].copy()
        z_aux = z+1
        j = 0
        while (j < self.limIt) & (np.linalg.norm(z - z_aux) > np.sqrt(T)*5e-3/np.sqrt(j+1)):

            z_aux = z
            z = z - (z - self.f_v(md, x + delta @ z))/np.sqrt(j+1)
            j+= 1
        # print('')
        assert j<self.limIt, print(f'in xi, {self.limIt} iterations: no convergence !!')
        return z

    def rdz(self, mz, vz, T, bound = False):
        mzv = mz.reshape(-1,1)
        sqvz = scipy.linalg.sqrtm(vz)
        z = mzv + sqvz @ np.random.randn(self.dimf, int(1.1*T))
        if bound:
            return z[:,np.linalg.norm(z - mzv, axis = 0) < 10* np.linalg.norm(sqvz)][:,:T]
        else: 
            return z[:,:T]
    def z2xi(self, md, z, delta):
        
        
        xi = self.xi(md, z, delta).copy()
        if not np.isnan(xi).any():
            self.xi_init = xi.copy()
        return xi
    
    def Exi(self, xi):
        return xi.mean(axis = 1)
    
    def Exi2(self, xi):
        return xi@xi.T/xi.shape[1]
    
    def dxi(self, md, x, delta):
        T = (1 if len(x.shape)==1 else x.shape[1])
        dfx = self.df(md, x + delta @ self.xi(md, x,delta), compute_mean = False).reshape(self.dimf, T *self.dimf, order = 'F')
        dfxdelta = dfx @ delta
        return sum([np.linalg.inv(np.eye(self.dimf)-dfxdelta[:, (self.dimf* i):(self.dimf*(i+1))]) @ dfx[:, (self.dimf* i):(self.dimf*(i+1))] for i in range(T)])/T
    
    def Edxi(self, md, z, delta, xi):
        dfx = self.df(md, z + delta @ xi, compute_mean = False).reshape(self.dimf, z.shape[1] *self.dimf, order = 'F')
        deltadfx = delta@ dfx
        return sum([np.linalg.inv(np.eye(self.dimf)-deltadfx[:, (self.dimf* i):(self.dimf*(i+1))]) @ dfx[:, (self.dimf* i):(self.dimf*(i+1))] for i in range(z.shape[1])])/z.shape[1]

    def f_v(self, md, Z, axis = 1):
        if axis == 1:
            if Z.shape[0] == self.dimf:
                return self.f(md, Z)
            else: return self.f_v(md, Z.reshape((self.dimf, -1), order = 'F'),  axis = 1).reshape(-1, order = 'F')
        else:
            return self.f_v(md, Z.T,  axis = 1).T
        
    def fixed_ptW(self, Xs, mdsX, returnQ = True):
        p = Xs[mdsX[0]].shape[0]
        # nmds = [X.shape[1] for X in Xs]
        n = sum([Xs[md].shape[1] for md in mdsX])
        X_s = {md: extend(Xs[md],self.dimf) for md in mdsX}
        W = np.ones(p*self.dimf)
        W_aux = np.zeros(p*self.dimf)
        X_Ts = {md: X_s[md].T for md in mdsX}
        j = 0
        while (np.linalg.norm(W-W_aux) > 1e-2 /np.sqrt(j+1)) &( j <self.limIt):
            W_aux = np.copy(W)
            W = W - ( W - sum([X_s[md] @ self.f_v(md, X_Ts[md]@ W, axis = 1)/n for md in mdsX]))/np.sqrt(j+1)
            j+=1
        assert j<self.limIt, print(f'{self.limIt} iterations: no convergence !!')
        if returnQ:
            Wbt = W.reshape(-1,self.dimf, order = 'F').T
            Zs = {md : Wbt@Xs[md] for md in mdsX}
            mdns = {md : Zs[md].shape[1] for md in mdsX}
            Ds = {md:np.concatenate([np.kron(np.array([i==col for i in range(mdns[md])]), self.df(md, Zs[md][:,col])) for col in range(mdns[md])], axis = 0) for md in mdsX}
            return W, np.linalg.inv(np.eye(self.dimf*p) - sum([X_s[md]@Ds[md]@X_Ts[md] for md in mdsX])/n), Ds
        else:
            return W
        
    def misclass_from_stats(self, cl, m, v, T = 10000):
        z = m.reshape(-1,1) + scipy.linalg.sqrtm(v) @ np.random.randn(self.dimf,T)
        return self.misclass(cl, z)
    
def softmaxp(x):
    return softmax(x, axis = 0) - softmax(x, axis = 0)**2
class Multiloss(Mapping):
    def __init__(self, params):
        self.update(params)        

    def update(self, params):
        self.__dict__.update(params)
        self.tol = min(self.lambdas)
        self.J = np.zeros((self.dimf*self.dimf, self.dimf))
        for i in range(self.dimf):
            self.J[i*(self.dimf+1),i] = 1
        print(self.tol)
        self.normdf = max(self.gammas.values()) / min(self.lambdas)
    def f(self, md, x):
        T = (1 if len(x.shape)==1 else x.shape[1])
        x_ = x.reshape(self.dimf, T)
        lambdas_ = self.lambdas.reshape(-1, 1)
        softM = softmax(x, axis = 0).reshape((-1, T))
        if md[:2] == 'su':
            y_s = np.zeros((self.dimf, 1))
            y_s[int(md[3:])] = 1
            if self.loss_su == 'polynoms':
                preFunc =  -2*(np.sum((y_s - softM)*softM, axis=0).reshape((1, -1)).repeat(repeats=self.dimf, axis=0)*softM - softM*(y_s - softM) + self.b*(softM - softM*np.sum(softM, axis=0).reshape((1, -1)).repeat(repeats=self.dimf, axis=0)))
            if self.loss_su == 'softmax':
                preFunc =  -2*(np.sum((y_s - softM)*softM, axis=0).reshape((1, -1)).repeat(repeats=self.dimf, axis=0)*softM - softM*(y_s - softM))
            if self.loss_su == 'cross_entropy':
                preFunc =  - softM +  y_s
            if self.loss_su == 'linear':
                preFunc =  2*(y_s - x_)
        elif md[:2] == 'un':
            if self.loss_un == 'polynoms':
                preFunc = 2*softM**2 - 2*softM*np.sum(softM**2, axis=0).reshape((1, -1)).repeat(repeats=self.dimf, axis=0)+ self.c*(softM - softM*np.sum(softM, axis=0).reshape((1, -1)).repeat(repeats=self.dimf, axis=0))
            if self.loss_un == 'entropy':
                preFunc = softM*(np.log(softM)+1) - softM*np.sum(softM*(np.log(softM) + 1), axis=0).reshape((1, -1)).repeat(repeats=self.dimf, axis=0)
            elif self.loss_un == 'square':
                preFunc = 2*softM**2 - 2*softM*np.sum(softM**2, axis=0).reshape((1, -1)).repeat(repeats=self.dimf, axis=0)
            elif self.loss_un == 'sqrt':
                preFunc = -np.sqrt(softM)/2 + softM*np.sum(np.sqrt(softM)/2, axis=0).reshape((1, -1)).repeat(repeats=self.dimf, axis=0)
            elif self.loss_un == 'quadratic':
                preFunc = 2*x_
        else: assert False
        return (self.gammas[md]*preFunc/lambdas_).reshape(x.shape)
    

    def df(self, md, x, compute_mean = True):
        T = (1 if len(x.shape)==1 else x.shape[1])
        lambdas_ = self.lambdas.reshape(-1, 1)
        lambdas__ = (lambdas_ @ np.ones((1,self.dimf))). reshape(self.dimf**2, 1, order = 'F')
        softM = softmax(x, axis = 0).reshape((-1, T))

        softMp = softmaxp(x).reshape((-1, T))
        sums2 = np.sum(softM**2, axis = 0)
    
        if md[:2] == 'su':
            y_s = np.zeros((self.dimf, 1))
            y_s[int(md[3:])] = 1
            suml2ss = np.sum((y_s - 2*softM)*softM, axis = 0)
            sumlss = np.sum((y_s - softM)*softM, axis = 0)
            if self.loss_su == 'polynoms':
                A = 2*(-softMp*sumlss + softMp*(y_s - 2*softM) -softM*(-softM*suml2ss + softM*(y_s - 2*softM)))
                B_non = 2*(-multcc(self.dimf,(softM*(y_s - 2*softM)),softM) - multcc(self.dimf,softM,softM*(y_s - 2*softM)) + multcc(self.dimf,sumlss*softM, softM) + multcc(self.dimf,suml2ss*softM, softM)) +self.b * (-2*multcc(self.dimf,softM, softM) +multcc(self.dimf,2*softM, softM))

                 
            if self.loss_su == 'softmax':
                A = 2*(-softMp*sumlss + softMp*(y_s - 2*softM) -softM*(-softM*suml2ss + softM*(y_s - 2*softM)))
                B_non = 2*(-multcc(self.dimf,softM*(y_s - 2*softM),softM) - multcc(self.dimf,softM, (softM*(y_s - 2*softM))) + multcc(self.dimf,sumlss*softM,softM) + multcc(self.dimf,suml2ss*softM, softM))
            elif self.loss_su == 'cross_entropy':
                A = -softMp
                B_non = multcc(self.dimf,softM, softM)
            elif self.loss_su == 'linear':
                A = -2*np.ones((self.dimf, ))
                B_non = np.concatenate(self.dimf*[np.zeros((x.shape))], axis = 0)
        else:
            if self.loss_un == 'polynoms': 
                A = 4*softMp*softM - 2*softMp*sums2 - 2*softM*(2*softM**2 - 2*softM*sums2) 
                B_non = -4*multcc(self.dimf,(softM**2),softM) + 2*multcc(self.dimf,sums2*softM,softM) -2*multcc(self.dimf,softM,(2*softM**2)) + 4* multcc(self.dimf,sums2*softM, softM) 
            if self.loss_un == 'entropy':
                sumsl1 = np.sum(softM*(np.log(softM) + 1), axis = 0)
                sumsl2 = np.sum(softM*(np.log(softM)+2), axis = 0)
                A = softMp*(np.log(softM)+2) - softMp*sumsl1 - softM*(softM*(np.log(softM) + 2) - softM*sumsl2)
                B_non = -multcc(self.dimf,(softM*(np.log(softM)+2)), softM) + multcc(self.dimf,(sumsl1)*softM, softM) - multcc(self.dimf,softM, (softM*(np.log(softM)+2))) + multcc(self.dimf,sumsl2*softM, softM)
            elif self.loss_un == 'square':
                A = 4*softMp*softM - 2*softMp*sums2 - 2*softM*(2*softM**2 - 2*softM*sums2)
                B_non = -4*multcc(self.dimf,(softM**2), softM) + 2*multcc(self.dimf,sums2*softM, softM) -2*multcc(self.dimf,softM, (2*softM**2)) + 4* multcc(self.dimf,sums2*softM, softM)
            elif self.loss_un == 'sqrt':
                sumsqs = np.sum(np.sqrt(softM), axis = 0)
                A = -softMp/(4*np.sqrt(softM)) +softMp*sumsqs/2 + softM*(np.sqrt(softM)/4 - (softM/4)*sumsqs)
                B_non = multcc(self.dimf,np.sqrt(softM), softM)/4 - multcc(self.dimf,sumsqs*softM, softM)/2 + multcc(self.dimf,softM, np.sqrt(softM))/4 - multcc(self.dimf,sumsqs*softM, softM)/4
            elif self.loss_un == 'quadratic':
                A = 2*np.ones((self.dimf, ))
                B_non = np.concatenate(self.dimf*[np.zeros((x.shape))], axis = 0)
        B_non = B_non.reshape((self.dimf**2, -1), order = 'F')
        B = B_non - (self.J@ np.ones((self.dimf, 1))) * B_non
        Adg = self.J @ A.reshape((self.dimf, -1), order = 'F')
        if compute_mean:
            return self.gammas[md]*(Adg + B).mean(axis = 1).reshape((self.dimf,self.dimf), order = 'F')/lambdas_
        else:
            return self.gammas[md]*(Adg + B)/lambdas__


    def misclass(self, kk, z):
        return min((np.argmax(z, axis=0) != kk).mean(), 1- (np.argmax(z, axis=0) != kk).mean())



class GaussianPopulation():
    def __init__(self, params):
        self.update(params)
    def update(self, params):
        self.__dict__.update(params)
        self.mds = sorted(set(self.cl2md.values()))
        self.tol = 1/np.sqrt(2*self.p)
        if 'rglk' not in self.__dict__.keys():
            self.rglk = range(len(self.ks))
        self.lk = len(self.rglk)
        assert len(self.ks) == self.lk, print('the number of classes rglk must have the same size as the vector describing the proportion of each class ks')
        self.sqCs = [scipy.linalg.sqrtm(C) for C in self.Cs]
        self.Sigmas = [self.Cs[i] + self.ms[i]@self.ms[i].transpose() for i in self.rglk]
        self.normSigma = sum([self.ks[cl] * np.linalg.norm(Sigma, 2) for cl,Sigma in enumerate(self.Sigmas)])
        self.ns = [int(k * self.n) for k in self.ks]
        self.ns[-1] = self.n - sum(self.ns[:-1])
    def draw_X(self):
        Xs = {md : np.concatenate([self.ms[kk] @ np.ones((1,self.ns[kk])) + self.sqCs[kk] @ np.random.randn(self.p,self.ns[kk]) for kk in self.rglk if self.cl2md[kk]==md], axis = 1) for md in self.mds}
        return Xs
    def draw_X_cl(self, cl, T = 1):
        return self.ms[cl] @ np.ones((1,T)) + self.sqCs[cl] @ np.random.randn(self.p,T)
    

