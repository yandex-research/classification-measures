from collections import defaultdict
import warnings

import numpy as np
from sklearn.metrics import confusion_matrix

def count_stats(y_true, y_pred):
    n = len(y_true)
    c2i = list(set(y_true)|set(y_pred))
    m = len(c2i)
    cs = defaultdict(lambda:defaultdict(int))
    for gt,pr in zip(y_true, y_pred):
        cs[gt][pr] += 1
    ts = defaultdict(int)
    ps = defaultdict(int)
    for c1 in range(m):
        for c2 in range(m):
            ts[c1] += cs[c1][c2]
            ps[c2] += cs[c1][c2]
    return n,m,cs,ts,ps

def bin_confmatrix(y_true, y_pred, c):
    tp, fn, fp, tn = 0, 0, 0, 0 
    for y,p in zip(y_true, y_pred):
        if y==p==c:
            tp += 1
        elif y==c:
            fn += 1
        elif p==c:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn

def confent(y_true, y_pred):
    n, m, cs, ts, ps = count_stats(y_true, y_pred)

    pis = defaultdict(lambda:defaultdict(float))
    pjs = defaultdict(lambda:defaultdict(float))
    pijs = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    for c1 in range(m):
        for c2 in range(m):
            if cs[c1][c2]:
                pijs[c1][c1][c2] = cs[c1][c2]/(ts[c1]+ps[c1])
                pijs[c2][c1][c2] = cs[c1][c2]/(ts[c2]+ps[c2])
            else:
                pijs[c1][c1][c2] = 0
                pijs[c2][c1][c2] = 0
    sum1 = 0.
    for c1 in range(m):
        sum2 = 0.
        for c2 in range(m):
            if c1!=c2:
                if pijs[c1][c1][c2]: sum2 += pijs[c1][c1][c2]*np.log(pijs[c1][c1][c2])/np.log(2*m-2)
                if pijs[c1][c2][c1]: sum2 += pijs[c1][c2][c1]*np.log(pijs[c1][c2][c1])/np.log(2*m-2)

        sum1 += sum2*(ts[c1]+ps[c1])/(2.*n)
    return -sum1


def iba_mc(y_true, y_pred):
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    r = []
    for c in list(set(y_true)):
        tp, fn, fp, tn = bin_confmatrix(y_true, y_pred, c)
        t = tp+fn
        p = tp+fp
        if p:
            r.append(tp/p)
        else:
            r.append(0.)
    return np.mean(r)

def sba(y_true, y_pred):
    if tuple(y_true)==tuple(y_pred): return 1
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    min_agr = True
    for i in range(m):
        if cs[i][i]>0:
            min_agr = False
    if min_agr: return 0
    sum1 = 0.
    for i in range(m):
        sum1 += cs[i][i]/ts[i] if ts[i] else ps[i]/n
        sum1 += cs[i][i]/ps[i] if ps[i] else ts[i]/n
    sum1 /= 2*m
    return sum1


def gmr_micro(y_true, y_pred, r=1):
    if len( set(y_true) ) == 1 and len( set(y_pred) ) == 1 and tuple(y_true)==tuple(y_pred): return 1.
    if len( set(y_true) ) == 1 and len( set(y_pred) ) == 1 and set(y_true)!=set(y_pred):
        return -1.

    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    tp, fn, fp, tn = 0, 0, 0, 0 
    for c in list(set(y_true)):
        _tp, _fn, _fp, _tn = bin_confmatrix(y_true, y_pred, c)
        tp += _tp
        fp += _fp
        tn += _tn
        fn += _fn
    t = tp+fn
    p = tp+fp
    t0 = tn+fp
    p0 = tn+fn
    return (n*tp-t*p)/pow( ( pow(t*t0,r)+pow(p*p0,r) )/2., 1./r)

def gmr_weighted(y_true, y_pred, average='macro', r=1):
    if len( set(y_true) ) == 1 and len( set(y_pred) ) == 1 and tuple(y_true)==tuple(y_pred): return 1.
    if len( set(y_true) ) == 1 and len( set(y_pred) ) == 1 and set(y_true)!=set(y_pred): return -1.
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    rr = []
    d = 0.
    for c in list(set(y_true)):
        _tp, _fn, _fp, _tn = bin_confmatrix(y_true, y_pred, c)
        t = _tp+_fn
        p = _tp+_fp
        t0 = _tn+_fp
        p0 = _tn+_fn
        w = (_tp+_fn) if average == 'weighted' else 1
        rr.append( w*(n*_tp-t*p)/pow( ( pow(t*t0,r)+pow(p*p0,r) )/2., 1./r) )
        d += w
    return np.sum(rr)/d


def fowlkes_micro(y_true, y_pred):
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    tp, fn, fp, tn = 0, 0, 0, 0 
    for c in list(set(y_true)):
        _tp, _fn, _fp, _tn = bin_confmatrix(y_true, y_pred, c)
        tp += _tp
        fp += _fp
        tn += _tn
        fn += _fn
    t = tp+fn
    p = tp+fp
    if not t*p: return 0
    return tp/np.sqrt(t*p)

def fowlkes_weighted(y_true, y_pred, average='macro'):
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    c2i = list(set(y_true))
    rr = []
    d = 0.
    for c in list(set(y_true)):
        _tp, _fn, _fp, _tn = bin_confmatrix(y_true, y_pred, c)
        t = _tp+_fn
        p = _tp+_fp
        w = (_tp+_fn) if average == 'weighted' else 1
        if not t*p: 
            rr.append(0)
        else:
            rr.append( w*_tp/np.sqrt(t*p) )
        d += w
    return np.sum(rr)/d

def mcc_default(y_true, y_pred):
    if len( set(y_true) ) == 1 and len( set(y_pred) ) == 1 and tuple(y_true)==tuple(y_pred): return 1.
    if len( set(y_true) ) == 1 and len( set(y_pred) ) == 1 and set(y_true)!=set(y_pred): return -1.
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    sum1 = 0.
    sum2 = 0.
    sum3 = 0.
    for i in range(m):
        sum1 += cs[i][i]*n - ps[i]*ts[i]
        sum2 += ts[i]*ts[i]
        sum3 += ps[i]*ps[i]
    return sum1/np.sqrt((n*n-sum2)*(n*n-sum3))

def mcc_micro(y_true, y_pred):
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    tp, fn, fp, tn = 0, 0, 0, 0 
    for c in list(set(y_true)):
        _tp, _fn, _fp, _tn = bin_confmatrix(y_true, y_pred, c)
        tp += _tp
        fp += _fp
        tn += _tn
        fn += _fn
    t = tp+fn
    p = tp+fp
    t0 = tn+fp
    p0 = tn+fn
    return (tp*tn-fp*fn)/np.sqrt(1.0*p*t*p0*t0)

def mcc_weighted(y_true, y_pred, average='macro'):
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    d = 0.
    rr = []
    for c in list(set(y_true)):
        _tp, _fn, _fp, _tn = bin_confmatrix(y_true, y_pred, c)
        t = _tp+_fn
        p = _tp+_fp
        t0 = _tn+_fp
        p0 = _tn+_fn
        w = (_tp+_fn) if average == 'weighted' else 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rr.append( w*(_tp*_tn-_fp*_fn)/np.sqrt(1.0*p*t*p0*t0) )
        d += w        
    return np.sum(rr)/d

def cd_arccos_weighted(y_true, y_pred, average='macro'):
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    d = 0.
    rr = []
    for c in list(set(y_true)):
        _tp, _fn, _fp, _tn = bin_confmatrix(y_true, y_pred, c)
        t = _tp+_fn
        p = _tp+_fp
        t0 = _tn+_fp
        p0 = _tn+_fn
        w = (_tp+_fn) if average == 'weighted' else 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rr.append(  w*np.arccos( (_tp*_tn-_fp*_fn)/np.sqrt(1.0*p*t*p0*t0) ) )
        d += w        
    return np.sum(rr)/d
