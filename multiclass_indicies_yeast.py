import numpy as np
import random
from glob import glob 
import warnings

from collections import defaultdict, Counter

from classification_indicies import NamedMulticlassIndices

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

EPS  = 1e-5
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


cast_fns_base = {
    'float' : lambda x:float(x),
    'int' : lambda x:int(x),
    'float_int' : lambda x:float(x) if '.' in str(x) else float(int(x)),
    'str' : lambda x:str(x),
    'class' : lambda x:str(x)
}


classifiers = (
    ("DecisionTree", DecisionTreeClassifier(max_depth=5)),
    ("ExtraTree", ExtraTreeClassifier(max_depth=5)),
    ("ExtraTreesEnsemble", ExtraTreesClassifier(max_depth=5)),
    ("NearestNeighbors", KNeighborsClassifier(3)),
    ("RadiusNeighbors", RadiusNeighborsClassifier(radius=10.)),
    ("RandomForest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ("BernoulliNB", BernoulliNB()),
    ("GaussianNB", GaussianNB()),
    ("LabelSpreading", LabelSpreading()),
    ("QuadraticDiscriminantAnalysis", QuadraticDiscriminantAnalysis()),
    ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()),
    ("NearestCentroid", NearestCentroid()),
    ("MLPClassifier", MLPClassifier(alpha=1, max_iter=1000)),
    ("LogisticRegression", LogisticRegression(multi_class="multinomial")),
    ("LogisticRegressionCV", LogisticRegressionCV(multi_class="multinomial")),
    ("RidgeClassifier", RidgeClassifier()),
    ("RidgeClassifierCV", RidgeClassifierCV()),
    ("LinearSVC", LinearSVC(multi_class="crammer_singer")),
)

classification_results = dict()
exps = set()

le = LabelEncoder()

for dataset_name in glob(f'data/multiclass/*.tsv'):
    # load and parse dataset
    lines = open(dataset_name, encoding='utf-8').read().split('\n')
    converters = dict()
    fnames = []
    formats = []
    for idx, dtype in enumerate(lines[0].split('\t')):
        converters[idx] = cast_fns_base[dtype]
        fnames.append( f'f{idx:03}' )
        sdtype = dtype.replace('class','str').replace('str','STR')[0]
        if sdtype == 'S': 
            sdtype += '100'
        else:
            sdtype += '8'
        formats.append( sdtype )

    data = np.loadtxt(dataset_name, delimiter='\t', converters=converters, skiprows=1, dtype={'names': fnames, 'formats': formats}, encoding='utf-8')
    recoded_data = []
    for idx, c in enumerate(formats):
        if c[0]!='S': 
            recoded_data.append( list(zip(*data))[idx] )
            continue
        encoded = le.fit_transform(list(zip(*data))[idx])
        recoded_data.append( encoded )

    y = np.array(recoded_data[0])
    X = np.array(list(zip(*recoded_data[1:])))

    # scale and split data
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

    # iterate over classifiers
    for cls_name, clf in classifiers:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # train 
            clf.fit(X_train, y_train)
            # predict
            preds = clf.predict(X_test)
            # measure indicies
            for metric_name, metric_fn in NamedMulticlassIndices.items():
                classification_results[(dataset_name, cls_name, metric_name)] = metric_fn(y_test, preds)
                exps.add( (dataset_name, cls_name) )

metrics = list(sorted(NamedMulticlassIndices.keys()))

found_examples = defaultdict(list)
discr_examples = defaultdict(list)
full_examples = set()

sampled_metrics = defaultdict(dict)
for ds, alg in exps:
    sample = dict()
    for m in metrics:
        sample[m] = classification_results[(ds, alg, m)]
    sampled_metrics[ds][alg] = sample

for ds in sampled_metrics:
    for i, (a1, cmp1) in enumerate(sampled_metrics[ds].items()):
        for j, (a2, cmp2) in enumerate(sampled_metrics[ds].items()):
            if i<j:

                left_winners = []
                right_winners = []
                draw_cases = []

                for m in NamedMulticlassIndices:
                    if np.isnan(cmp1[m]) or np.isnan(cmp2[m]):
                        continue
                    if cmp1[m]>cmp2[m] and abs(cmp1[m]-cmp2[m])>EPS:
                        left_winners.append( (m,i,j) )
                    if cmp1[m]<cmp2[m] and abs(cmp1[m]-cmp2[m])>EPS:
                        right_winners.append( (m,i,j) )
                    if abs(cmp1[m]-cmp2[m])<=EPS:
                        draw_cases.append( (m,i,j) )

                handle = (a1,a2)
                full_examples.add(handle)

                if left_winners and right_winners:
                    for r1 in left_winners:
                        for r2 in right_winners:
                            pair = tuple(sorted((r1[0],r2[0])))
                            found_examples[handle].append( pair )
                            discr_examples[ pair ].append( handle )
                elif left_winners and draw_cases:
                    for r1 in left_winners:
                        for r2 in draw_cases:
                            pair = tuple(sorted((r1,r2)))
                            found_examples[handle].append( pair )
                            discr_examples[ pair ].append( handle )
                elif right_winners and draw_cases:
                    for r1 in right_winners:
                        for r2 in draw_cases:
                            pair = tuple(sorted((r1,r2)))
                            found_examples[handle].append( pair )
                            discr_examples[ pair ].append( handle )
                else:
                    if handle not in found_examples:
                        found_examples[handle] = list()


possible = set()
for m1 in NamedMulticlassIndices:
    for m2 in NamedMulticlassIndices:
        if m1!=m2:
            possible.add( tuple(sorted([m1,m2])) )
print(f'Finally, disciminated {len(discr_examples)} pairs out of {len(possible)} total.')
print(f'{len(possible-set(discr_examples))} are left:', tuple(sorted(possible-set(discr_examples))) )
print()

totals = set()
print('\t'+'\t'.join(metrics))
for r1 in sorted(NamedMulticlassIndices):
    r = [r1]
    for r2 in sorted(NamedMulticlassIndices):
        totals.add(tuple(sorted([r1,r2])))
        n = len(discr_examples[ 
                                tuple(sorted([r1,r2]))
                            ]
                    )
        if n:
            r.append( str( n ) )
        else:
            r.append( '' )
    print('\t'.join(r))

print('total', len(full_examples))