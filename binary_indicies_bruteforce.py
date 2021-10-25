import sys
import numpy as np
import random

from collections import defaultdict, Counter

from classification_indicies import NamedBinaryIndices


METRICS = ('f1', 'acc', 'cc', 'ba', 'kappa', 'ce', 'sba', 'gm1')

DONE = defaultdict(set)
DONE2 = defaultdict(set)
SKIPIT = set()

EPS  = 1e-5
SEED = 1
random.seed(SEED)
np.random.seed(SEED)

n = 5
if len(sys.argv)>1: n = int(sys.argv[1])

config = {
    'items' : n,
    'classes' : 2,
    'sizes' : None, # specify classes sizes or set to None for random
}

_cache = dict()
def get_bin_indices(y_true, y_pred):
    global _cache
    handle = (tuple(y_true),tuple(y_pred))
    if handle in _cache:
        return _cache[handle]
    _cache[handle] = dict((m, fn(y_true, y_pred)) for m, fn in NamedBinaryIndices.items() if m in METRICS and m not in SKIPIT)
    return _cache[handle]

def get_random_markup():
    markup = []
    # here we'll repeat until we have all classes presented to avoid trivial markups
    # it's a bit non-deterministic, but it will be okay
    while len(set(markup))<config.get('classes'):
        if not config.get('sizes'):
            markup = random.choices( list(range(config.get('classes'))), k=config.get('items'))
        else:
            markup = []
            for size, class_id in zip( config.get('sizes'), range(config.get('classes')) ):
                markup.extend( [class_id,]*size )
    return markup

found_examples = set()
discr_examples = defaultdict(list)

candidates = [list(map(int,list(bin(n)[2:].zfill(config['items'])))) for n in range(2**config['items'])]
candidates = [c for c in candidates if len(set(c))>1]

print(f'items {config["items"]} classes {config["classes"]}')
print()

sampled_markups = []
total = 0.

found_examples.add(1)
for _i1, markup1 in enumerate(candidates):
    for m in DONE:
        if m in SKIPIT: continue
        if len(DONE[m])==len(METRICS)-1: # and len(DONE2[m])==len(METRICS)-1 :
            print('fully covered',m)
            SKIPIT.add(m)
    for _i2, markup2 in enumerate(candidates):
        for _i3, markup3 in enumerate(candidates):
            if _i2>=_i3: continue
            cmp1 = get_bin_indices( markup1, markup2 )
            cmp2 = get_bin_indices( markup1, markup3 )

            left_winners = []
            right_winners = []
            draw_cases = []
            for m in NamedBinaryIndices:
                if m not in cmp1:
                    continue
                # if m not in METRICS or m in SKIPIT:
                #     continue
                if np.isnan(cmp1[m]) or np.isnan(cmp2[m]):
                    continue
                if cmp1[m]>cmp2[m] and abs(cmp1[m]-cmp2[m])>EPS:
                    left_winners.append( m )
                if cmp1[m]<cmp2[m] and abs(cmp1[m]-cmp2[m])>EPS:
                    right_winners.append( m )
                if abs(cmp1[m]-cmp2[m])<=EPS:
                    draw_cases.append( m )

            handle = ( tuple(markup1), tuple(markup2), tuple(markup3), )

            if left_winners and right_winners:
                for m1 in left_winners:
                    for m2 in right_winners:
                        DONE[m1].add(m2)
                        DONE[m2].add(m1)
                        pair = tuple(sorted((m1,m2)))
                        discr_examples[ pair ].append( handle )
            if left_winners and draw_cases:
                for m1 in left_winners:
                    for m2 in draw_cases:
                        DONE[m1].add(m2)
                        DONE[m2].add(m1)
                        pair = tuple(sorted((m1,m2)))
                        discr_examples[ pair ].append( handle )
            if right_winners and draw_cases:
                for m1 in right_winners:
                    for m2 in draw_cases:
                        DONE[m1].add(m2)
                        DONE[m2].add(m1)
                        pair = tuple(sorted((m1,m2)))
                        discr_examples[ pair ].append( handle )

print()
print(f'metric1\tmetric2\tinvN')
for pair in sorted(discr_examples):
    print(f'{pair[0]}\t{pair[1]}\t{len(discr_examples[pair])}'.replace('.',','))


possible = set()
for _i1, m1 in enumerate(sorted(NamedBinaryIndices)):
    if m1 not in METRICS: # and m not in SKIPIT:
        continue
    for _i2, m2 in enumerate(sorted(NamedBinaryIndices)):
        if m2 not in METRICS: # and m not in SKIPIT:
            continue
        if m1!=m2: # and _i1<_i2:
            possible.add( tuple(sorted([m1,m2])) )
print()
print(f'Finally, disciminated {len(discr_examples)} pairs out of {len(possible)} total.')
print(f'{len(possible-set(discr_examples))} are left:')
print( tuple(sorted(possible-set(discr_examples))) )
print(f'{len(possible-(possible-set(discr_examples)))} are covered:')
print( tuple(sorted(possible-(possible-set(discr_examples)))) )
