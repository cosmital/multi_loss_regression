import scipy
import numpy as np
from scipy.integrate import  dblquad
import matplotlib.pyplot as plt

def extend(M,dimf):
    shapeM = M.shape
    if len(shapeM)<2:
        shapeM = [M.shape[0], 1]
    return np.concatenate([np.kron(M.reshape(shapeM), np.array([i==j for i in range(dimf)])) for j in range(dimf)], axis = 0)

def multcc(dimf, A, B, compute_mean = False):
    T = (1 if len(A.shape)==1 else A.shape[1])

    assert dimf == A.shape[0]
    assert A.shape == B.shape,  print('A.shape, B.shape' ,A.shape, B.shape)
    A_aux = A.reshape(dimf, T, order = 'F')
    B_aux = B.reshape(dimf, T, order = 'F')
    AB = np.concatenate([A_aux[:, [i]] @ B_aux[:, [i]].T for i in range(T)], axis = 1).reshape(dimf**2, -1, order = 'F')
    if compute_mean:
        return AB.mean().reshape(dimf,dimf, order = 'F')
    else:
        return AB
    
def mat_tr(Mdimfp, Mp):
    p = Mp.shape[0]
    dimf = Mdimfp.shape[0] //p
    ind_p = [np.arange(p*i, p*(i+1), 1) for i in range(dimf)]
    traceMat = np.zeros((dimf, dimf))
    for a in range(dimf):
        for b in range(dimf):
            traceMat[a, b] = np.trace(Mdimfp[ind_p[a],:][:, ind_p[b]]@Mp)
    return traceMat
def mat_quadr(v1dimfp, Mp):
    return mat_tr(v1dimfp.reshape(-1,1)@ v1dimfp.reshape(1,-1), Mp)


def step_iteration_with_mean(pops, funcs, mzs_aux, vzs_aux, deltas, lp, returnQ = False, tol = 1/100):
    Tmean = int(100/tol)
    mzs_aux = [m.reshape(-1,1) for m in mzs_aux]
    z = {cl : funcs.rdz(mzs_aux[cl], vzs_aux[cl], Tmean) for cl in pops.rglk}
    xi = {cl: funcs.z2xi(pops.cl2md[cl], z[cl], deltas[cl]) for cl in pops.rglk}
    tC = sum([np.kron(pops.ks[cl]*funcs.Edxi(pops.cl2md[cl], z[cl], deltas[cl],xi[cl]), pops.Cs[cl]) for cl in pops.rglk])
    tQ = np.linalg.inv(np.eye(pops.p*funcs.dimf) - tC)
    tmu = sum([np.kron(pops.ks[cl]*funcs.Exi(xi[cl]).reshape(-1,1), pops.ms[cl]) for cl in pops.rglk])  
    mW = tQ@tmu
    mzs = [np.real(extend(pops.ms[cl],funcs.dimf).T @ mW) for cl in pops.rglk]
    mzs = [mzs_aux[cl] - lp*(mzs_aux[cl] - mzs[cl]) for cl in pops.rglk]
    deltas =  [mat_tr(tQ, pops.Sigmas[cl])/pops.n for cl in pops.rglk]
    vzs = [np.zeros((funcs.dimf, funcs.dimf)) for _ in  pops.rglk]
    xis2 = [funcs.Exi2(xi[cl]) for cl in pops.rglk]
    
    tS = sum([ np.kron(pops.ks[cl] * xis2[cl], pops.Sigmas[cl])  for cl in pops.rglk])
    for cl in pops.rglk:
        vzs[cl] = mat_quadr(tQ@tmu, pops.Cs[cl])  + mat_tr(tQ @ tS @ tQ, pops.Sigmas[cl])/pops.n
    if returnQ:
        return mzs, vzs, deltas, tQ
    else:
        return mzs, vzs, deltas

def get_expect_W(pops, funcs, arg = {}, limIt = 30, returnQ = False, param_tol = 0.5):
    # if False:
    if len(arg)>0:
        mzs = arg['mzs'].copy()
        vzs = arg['vzs'].copy()
        deltas = arg['deltas'].copy()
        lr = 0.5
    else:
        mzs = [np.ones((funcs.dimf, 1)) for _ in pops.rglk]
        vzs = [np.ones((funcs.dimf, funcs.dimf)) for _ in pops.rglk]
        deltas = [np.ones((funcs.dimf, funcs.dimf)) for _ in pops.rglk]
        lr = 1

    j = 0
    pow_tol = 0.5
    tol_bse =  param_tol*min(50*min(funcs.lambdas), 1)* pops.tol
    tol = 0.1* tol_bse
    mzs_aux = [m.copy() + 10 *tol * np.ones((funcs.dimf, 1))for m in mzs]
    while (sum([np.linalg.norm(mzs[cl] - mzs_aux[cl]) for cl in pops.rglk])> tol) and (j < limIt):
        # lr = 0.3*(1/(j+1)**pow_tol  + 1/(j+1)**0.5 + 1/(j+1))
        lr = min(lr, 0.5*(1/(j+1)**pow_tol  +  1/(j+1)))
        tol =  tol_bse* lr
        mzs_aux = [m.copy() for m in mzs]
        mzs, vzs, deltas = step_iteration_with_mean(pops, funcs, mzs, vzs, deltas, lr, tol = tol)
        print(f'j={j}/{limIt}:   \t conv: {round(sum([np.linalg.norm(mzs[cl] - mzs_aux[cl])for cl in pops.rglk])/lr, 3)} > {round(tol_bse, 3)} ?    \t lr : {round(lr,4)}')
        j+=1
    if j>=limIt: print(f'in get_expect_W, more than {limIt} iterations: end before convergence !!')
    if returnQ:
        itRes = step_iteration_with_mean(pops, funcs, mzs, vzs, deltas,  lr, tol = tol, returnQ = returnQ)
        return mzs, vzs, deltas, itRes[3]
    else:
        return mzs, vzs, deltas



def objective_lambdas(lambdas, arg = {}):
    print(lambdas, ': ')
    arg['funcs'].update({'lambdas': lambdas})
    try :
        mzs, vzs, _ = get_expect_W(arg['pops'], arg['funcs'])
        erro_class =  sum(arg['pops'].ks[cl]*arg['funcs'].misclass_from_stats(arg['cl2classif'][cl], mzs[cl], vzs[cl]) for cl in arg['clStudy'])/ sum([arg['pops'].ks[cl] for cl in arg['clStudy']])
    except:
        print('error', end = ', ')
        erro_class = 0.5
        pass
    print(f'SCORE : {round(erro_class, 3)}')
    return erro_class