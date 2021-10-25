from collections import defaultdict, Counter
from glob import glob
import numpy as np
import math
import random

from classification_indicies import NamedMulticlassIndices

random.seed(42)

_cache = dict()
def get_indices(y_true, y_pred):
    global _cache
    handle = (tuple(y_true),tuple(y_pred))
    if handle in _cache:
        return _cache[handle]
    _cache[handle] = dict((m, fn(y_true, y_pred)) for m, fn in NamedMulticlassIndices.items())
    return _cache[handle]

s = 'model'
for m in sorted(NamedMulticlassIndices):
    s += '\t'+m
print(s)

model = None
_gt = 0
for idx, line in enumerate(open('data/sst/real_sent.tsv')):
    if not idx:
        counts = list(map(int,line.strip().split('\t')))
    else:
        if line[0] == '*':
            if model is not None:
                q = get_indices(gt, pr)
                s = f'{model}'
                for m in sorted(NamedMulticlassIndices):
                    s += '\t'+f'{q[m]:.3%}'
                print(s)
                pass
            _, model = line.strip().split(': ')
            gt = []
            pr = []
            _gt = 0
        else:
            freq = list(map(float,line.strip().split('\t')))
            for pred, cnt in enumerate(freq):
                gt.extend( [4-_gt,]*int(np.round(cnt * counts[4-_gt])) )
                pr.extend( [pred,]*int(np.round(cnt * counts[4-_gt])) )
            _gt += 1
q = get_indices(gt, pr)
s = f'{model}'
for m in sorted(NamedMulticlassIndices):
    s += '\t'+f'{q[m]:.3%}'
print(s)
