import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import matthews_corrcoef

from .binary import _wrapper1
from .binary import _wrapper2

from .binary import gmr_bin
from .binary import fowlkes_mallows_bin


from .multiclass import confent
from .multiclass import sba
from .multiclass import iba_mc
from .multiclass import gmr_micro
from .multiclass import gmr_weighted
from .multiclass import fowlkes_micro
from .multiclass import fowlkes_weighted
from .multiclass import mcc_default
from .multiclass import mcc_micro
from .multiclass import mcc_weighted
from .multiclass import cd_arccos_weighted

NamedBinaryIndices = {
    'acc': accuracy_score,
    'ba': balanced_accuracy_score,
    'cd': lambda y_true, y_pred: np.arccos(_wrapper2(y_true, y_pred, matthews_corrcoef))/np.pi,
    'kappa': lambda y_true, y_pred: _wrapper1(y_true, y_pred, cohen_kappa_score),
    '-ce': lambda g,p:-confent(g,p),
    'jaccard': lambda y_true, y_pred: _wrapper1(y_true, y_pred, jaccard_score),
    'cc': lambda y_true, y_pred: _wrapper2(y_true, y_pred, matthews_corrcoef),
    'sba': sba,
    'gm1': lambda y_true, y_pred: _wrapper2(y_true, y_pred, gmr_bin),
    'f1': lambda y_true, y_pred: _wrapper1(y_true, y_pred, f1_score),
#    'iba': lambda y_true, y_pred: balanced_accuracy_score(y_pred,y_true),
#    'fm': fowlkes_mallows_bin,
}

NamedMulticlassIndices = {
    'acc': accuracy_score,
    'ba': balanced_accuracy_score,

    'f1_micro': lambda g,p:f1_score(g,p,average='micro'),
    'f1_macro': lambda g,p:f1_score(g,p,average='macro'),
    'f1_weighted': lambda g,p:f1_score(g,p,average='weighted'),

    'jaccard_micro': lambda g,p:jaccard_score(g,p,average='micro'),
    'jaccard_macro': lambda g,p:jaccard_score(g,p,average='macro'),
    'jaccard_weighted': lambda g,p:jaccard_score(g,p,average='weighted'),

    '-ce': lambda g,p:-confent(g,p),
    'kappa': lambda y_true, y_pred: _wrapper1(y_true, y_pred, cohen_kappa_score),

    'sba': sba,

    'gm1_micro':  gmr_micro,
    'gm1_macro':  gmr_weighted,
    'gm1_weighted':    lambda g,p:gmr_weighted(g,p,average='weighted'),

    'cc_default': mcc_default,
    'cc_micro': mcc_micro,
    'cc_macro': mcc_weighted,
    'cc_weighted':   lambda g,p:mcc_weighted(g,p,average='weighted'),

    'cd_default': lambda g,p:np.arccos(mcc_default(g,p))/np.pi,
    'cd_micro': lambda g,p:np.arccos(mcc_micro(g,p))/np.pi,
    'cd_macro':  lambda g,p:cd_arccos_weighted(g,p)/np.pi,
    'cd_weighted':  lambda g,p:cd_arccos_weighted(g,p,average='weighted')/np.pi,
    # 'iba': iba_mc,
    # 'fm_micro':  fowlkes_micro,
    # 'fm_macro':  fowlkes_weighted,
    # 'fm_weighted':    lambda g,p:fowlkes_weighted(g,p,average='weighted'),
}