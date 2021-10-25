from glob import glob
import random
from collections import defaultdict, Counter
from classification_indicies import NamedMulticlassIndices

def get_indices(y_true, y_pred):
    return dict((m, fn(y_true, y_pred)) for m, fn in NamedMulticlassIndices.items())

print('model\t'+'\t'.join(sorted(NamedMulticlassIndices.keys())))
for fn in glob(f'data/imagenet/*.json'):
    fn_short = fn.replace('\\','/').split('/')[-1]

    gt = []
    pr = []
    for line in open(fn):
        g,p,c = map(int,line.strip().split('\t'))
        gt.extend( [g,]*c )
        pr.extend( [p,]*c )

    values = get_indices(gt, pr)
    s = f'{fn_short}'
    for m in sorted(NamedMulticlassIndices):
        s += '\t'+f'{values[m]:.3%}'
    print(s)