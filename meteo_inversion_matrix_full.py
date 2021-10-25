from glob import glob
from collections import defaultdict, Counter
import math
import numpy as np
import random

random.seed(42)
EPS  = 1e-5

def alt_mcc_bin(tp, fn, fp, tn):
    n = tp+ fn+ fp+ tn
    c2i = (0,1)
    m = len(c2i)
    cs = defaultdict(lambda:defaultdict(int))
    cs[1][1] = tp
    cs[0][1] = fp
    cs[1][0] = fn
    cs[0][0] = tn

    ts = defaultdict(int)
    ps = defaultdict(int)
    for c1 in range(m):
        for c2 in range(m):
            ts[c1] += cs[c1][c2]
            ps[c2] += cs[c1][c2]

    sum1 = cs[1][1]*cs[0][0]-cs[0][1]*cs[1][0]
    sum2 = ps[1]*ts[1]*ps[0]*ts[0]

    return sum1/np.sqrt(1.*sum2)

def alt_cohen_bin4(tp, fn, fp, tn):
    n = tp+ fn+ fp+ tn
    c2i = (0,1)
    m = len(c2i)
    cs = defaultdict(lambda:defaultdict(int))
    cs[1][1] = tp
    cs[0][1] = fp
    cs[1][0] = fn
    cs[0][0] = tn

    ts = defaultdict(int)
    ps = defaultdict(int)
    for c1 in range(m):
        for c2 in range(m):
            ts[c1] += cs[c1][c2]
            ps[c2] += cs[c1][c2]

    sum1 = 0.
    sum2 = 0.
    for i in range(m):
        sum1 += cs[i][i]
        sum2 += ps[i]*ts[i]

    return (sum1-sum2/n)/(n-sum2/n)

def alt_confent_bin4(tp, fn, fp, tn):
    n = tp+ fn+ fp+ tn
    c2i = (0,1)
    m = len(c2i)
    cs = defaultdict(lambda:defaultdict(int))
    cs[1][1] = tp
    cs[0][1] = fp
    cs[1][0] = fn
    cs[0][0] = tn

    ts = defaultdict(int)
    ps = defaultdict(int)
    for c1 in range(m):
        for c2 in range(m):
            ts[c1] += cs[c1][c2]
            ps[c2] += cs[c1][c2]

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


def alt_sns_bin(tp, fn, fp, tn):
    n = tp+ fn+ fp+ tn
    c2i = (0,1)
    m = len(c2i)
    cs = defaultdict(lambda:defaultdict(int))
    cs[1][1] = tp
    cs[0][1] = fp
    cs[1][0] = fn
    cs[0][0] = tn

    ts = defaultdict(int)
    ps = defaultdict(int)
    for c1 in range(m):
        for c2 in range(m):
            ts[c1] += cs[c1][c2]
            ps[c2] += cs[c1][c2]

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


def alt_gm1_bin(tp, fn, fp, tn):
    r = 1
    n = tp+ fn+ fp+ tn
    c2i = (0,1)
    m = len(c2i)
    cs = defaultdict(lambda:defaultdict(int))
    cs[1][1] = tp
    cs[0][1] = fp
    cs[1][0] = fn
    cs[0][0] = tn

    rr = []
    _tp, _fn, _fp, _tn = tp, fn, fp, tn
    t = _tp+_fn
    p = _tp+_fp
    t0 = _tn+_fp
    p0 = _tn+_fn
    return (n*_tp-t*p)/pow( ( pow(t*t0,r)+pow(p*p0,r) )/2., 1./r)

def alt_CD(tp, fn, fp, tn):
    return -np.arccos(alt_mcc_bin(tp, fn, fp, tn))


metrics_impl = [
('f1', lambda tp, fn, fp, tn: (2*tp)/(2*tp+fp+fn)),
('jaccard', lambda tp, fn, fp, tn: tp/(tp+fp+fn)),
('ba', lambda tp, fn, fp, tn: ((tp)/(tp+fn)+(tn)/(tn+fp))/2.),
('acc', lambda tp, fn, fp, tn: (tp+tn)/(tp+tn+fp+fn)),
('iba', lambda tp, fp, fn, tn: ((tp)/(tp+fn)+(tn)/(tn+fp))/2.),

('gm1', alt_gm1_bin),
('ce', lambda tp, fn, fp, tn:-alt_confent_bin4(tp, fn, fp, tn)),
('sba', alt_sns_bin),

('kappa', alt_cohen_bin4),
('cc', alt_mcc_bin),
('cd',alt_CD),
]

metrics_impl = dict(metrics_impl)

_cache = dict()
def get_bin_indices(tp, fn, fp, tn):
    global _cache
    handle = (tp, fn, fp, tn)
    if handle in _cache:
        return _cache[handle]
    _cache[handle] = dict((m, mfn(tp, fn, fp, tn)) for m, mfn in metrics_impl.items())
    return _cache[handle]

found_examples = defaultdict(list)
discr_examples = defaultdict(list)
sampled_metrics = defaultdict(dict)

seen_dates = set()
seen_fcs = list(range(12))
seen_exps = set()

for fn in glob('data/meteo/prod*.tsv'):
    for idx, line in enumerate(open(fn, encoding='utf-8')):
        if not idx: continue
        exp_group, utc_date, tn, tp, fn, fp = line.strip().split('\t')
        tn = list(map(int,tn.split(',')))
        tp = list(map(int,tp.split(',')))
        fn = list(map(int,fn.split(',')))
        fp = list(map(int,fp.split(',')))

        seen_dates.add( utc_date )
        seen_exps.add( exp_group )

        for fc in range(12): 
            sampled_metrics[(utc_date, fc)][exp_group] = get_bin_indices(tp[fc], fn[fc], fp[fc], tn[fc])


total = set()

for ds in sampled_metrics:
    for i, (a1, m1) in enumerate(sampled_metrics[ds].items()):
        for j, (a2, m2) in enumerate(sampled_metrics[ds].items()):
            markup = ()
            if i<j:
                left_winners = []
                right_winners = []
                draw_cases = []
                for m in metrics_impl:
                    if np.isnan(m1[m]) or np.isnan(m2[m]):
                        continue
                    if m1[m]>m2[m] and abs(m1[m]-m2[m])>EPS:
                        left_winners.append( (m,i,j) )
                    if m1[m]<m2[m] and abs(m1[m]-m2[m])>EPS:
                        right_winners.append( (m,i,j) )
                    if abs(m1[m]-m2[m])<=EPS:
                        draw_cases.append( (m,i,j) )

                handle = frozenset((tuple(markup), (ds,a1), (ds,a2)))
                if left_winners and right_winners:
                    for r1 in left_winners:
                        for r2 in right_winners:
                            found_examples[handle].append( tuple(sorted([r1[0],r2[0]])) )
                            discr_examples[ tuple(sorted([r1[0],r2[0]])) ].append( handle )
                elif left_winners and draw_cases:
                    for r1 in left_winners:
                        for r2 in draw_cases:
                            found_examples[handle].append( tuple(sorted([r1[0],r2[0]])) )
                            discr_examples[ tuple(sorted([r1[0],r2[0]])) ].append( handle )
                elif right_winners and draw_cases:
                    for r1 in right_winners:
                        for r2 in draw_cases:
                            found_examples[handle].append( tuple(sorted([r1[0],r2[0]])) )
                            discr_examples[ tuple(sorted([r1[0],r2[0]])) ].append( handle )
                else:
                    if handle not in found_examples:
                        found_examples[handle] = list()

print('total',len(found_examples))
print('\t'+'\t'.join(sorted(metrics_impl)))
for r1 in sorted(metrics_impl):
    r = [r1]
    for r2 in sorted(metrics_impl):
        n = len(discr_examples[ 
                                tuple(sorted([r1,r2]))
                            ]
                    )
        if n:
            r.append( str( n ) )
        else:
            r.append( '' )
    print('\t'.join(r))


