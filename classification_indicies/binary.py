from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

def _wrapper1(y_true, y_pred, fn):
    if len( set(y_true) ) == 1 and set(y_true)==set(y_pred):
        return 1.
    return fn(y_true,y_pred)

def _wrapper2(y_true, y_pred, fn):
    if len( set(y_true) ) == 1 and len( set(y_pred) ) == 1 and set(y_true)!=set(y_pred):
        return -1.
    return _wrapper1(y_true, y_pred, fn)


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

def gmr_bin(y_true, y_pred, r=1):
    if r==0: return matthews_corrcoef(y_true, y_pred)
    if len(set(y_true))==1 or len(set(y_pred))==1: return 0
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=(1,0)).ravel()
    # print('#',y_true, y_pred, tp, fn, fp, tn, '-', ts, ps, '-', ts[1]*ts[0], ps[1]*ps[0], '-', (n*tp-ts[1]*ps[1])/pow( ( pow(ts[1]*ts[0],r)+pow(ps[1]*ps[0],r) )/2., 1./r))
    return (n*tp-ts[1]*ps[1])/pow( ( pow(ts[1]*ts[0],r)+pow(ps[1]*ps[0],r) )/2., 1./r)

def fowlkes_mallows_bin(y_true, y_pred):
    n, m, cs, ts, ps = count_stats(y_true, y_pred)
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=(1,0)).ravel()
    if not ts[1]*ps[1]: return 0
    return tp/np.sqrt(ts[1]*ps[1])

