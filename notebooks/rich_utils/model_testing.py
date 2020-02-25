from typing import Dict, Any, Optional, Callable, List, Tuple, Iterable
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
import xgboost as xgb

from rich_utils.my_roc_auc import my_roc_auc

import logging

logger = logging.getLogger('main.model_testing')



params = dict(
    max_depth=5,
    n_estimators=2,
    learning_rate=0.1,
    min_child_weight=5,
    n_jobs=24
)

logger.info(f"classfier params: {params}")

default_features_mapping = [
    ('Brunel_P'      , 'Brunel_P'           ),
    ('Brunel_ETA'    , 'Brunel_ETA'         ),
    ('nTracks_Brunel', 'nTracks_Brunel'     ),
    ('RichDLLe'      , 'predicted_RichDLLe' ),
    ('RichDLLk'      , 'predicted_RichDLLk' ),
    ('RichDLLmu'     , 'predicted_RichDLLmu'),
    ('RichDLLp'      , 'predicted_RichDLLp' ),
    ('RichDLLbt'     , 'predicted_RichDLLbt'),
]

def get_feature(
        data : np.ndarray,
        feature : str,
        features_mapping : List[Tuple[str, str]] = default_features_mapping
    ) -> np.ndarray:
    """Get single feature column from a dataset composed with a call to merge_dataframes(...)"""

    return data[:,list(zip(*features_mapping))[0].index(feature) + 1]

def _vararg_logical_and(
        *argv : np.ndarray
    ) -> np.ndarray:
    result = argv[0]
    for arr in argv[1:]:
        result = result & arr
    return result

def _is_between(
        arr : np.ndarray,
        left : float,
        right : float
    ) -> np.ndarray:
    return (arr >= left) & (arr < right)

def make_nd_bin_selection(
        *argv : Tuple[str, Iterable[float]],
        features_mapping : List[Tuple[str, str]] = default_features_mapping
    ) -> Tuple[Tuple[Callable[[np.ndarray], np.ndarray]], Tuple[Tuple[float, float]]]:
    """
    Prepare selection functions to be used in single_test(...).


    *argv - tuples of pairs (variable name, list of bin edges)

    features_mapping - list of tuples (real feature, generated feature)


    Returns a tuple of selection functions and a tuple of bin edges for each selection.


    Example usage (make 2x3 bins in P-ETA):
        selection_funcs, bins = \\
                make_nd_bin_selection(('Brunel_P', [10000, 20000, 40000]),
                                      ('nTracks_Brunel', [50, 150, 200, 500]))
    """

    variables, bin_edges = zip(*argv)
    cuts = [zip(single_var_bin_edges[:-1],
                single_var_bin_edges[1: ]) for single_var_bin_edges in bin_edges]

    selection = [
        ((lambda single_cut_capture: (
            lambda array: _vararg_logical_and(*(
                _is_between(get_feature(array, var_name, features_mapping), *single_var_cut)
                for var_name, single_var_cut in zip(variables, single_cut_capture)
            ))
        ))(single_cut), single_cut)
        for single_cut in product(*cuts)
    ]

    selection_funcs, bins = zip(*selection)
    return selection_funcs, bins


def merge_dataframes(
        df_real : pd.DataFrame,
        df_gen : Optional[pd.DataFrame] = None,
        weights_col : str = 'probe_sWeight',
        features_mapping : List[Tuple[str, str]] = default_features_mapping
    ) -> np.ndarray:
    """
    Convert real and generated dataframes to a single numpy array with a
    structure suitable for single_test(...).
    
    
    df_real - pandas dataframe with real data
    
    df_gen - pandas dataframe with generated data. If omitted it's assumed df_real
            has generated features as well.
    
    weights_col - name of the weights column
    
    features_mapping - list of tuples (real feature, generated feature)
    
    
    Returns a numpy array of the following structure:
      weight  feature_1  feature_2 ... feature_n  is_real
      weight  feature_1  feature_2 ... feature_n  is_real
      weight  feature_1  feature_2 ... feature_n  is_real
      ...
    """
    logger.info(f"df_real.shape: {df_real.shape}, df_gen type: {type(df_gen)}")

    feats_real, feats_gen = [
        [weights_col] + list(x)
        for x in zip(*features_mapping)
    ]
        
    data_real = df_real[feats_real].values
    data_gen  = df_real[feats_gen ].values if df_gen is None else \
                df_gen [feats_gen ].values
    
    
    data_real = np.concatenate(
        [data_real, np.ones(shape=(len(data_real), 1), dtype=data_real.dtype)],
        axis=1
    )
    data_gen = np.concatenate(
        [data_gen, np.zeros(shape=(len(data_gen), 1), dtype=data_gen.dtype)],
        axis=1
    )

    if df_gen is None:
        try:
            assert(len(data_gen) == len(data_real))
            ids = np.random.permutation(len(data_real))
            i_half = int(len(data_real) / 2)
            return np.concatenate([data_real[ids[:i_half]],
                                   data_gen [ids[i_half:]]], axis=0)
        except Exception as e:
            logging.exception(e, exc_info=True)


    return np.concatenate([data_real, data_gen], axis=0)


def single_test(
        data_train : np.ndarray,
        data_test : np.ndarray,
        params : Dict[str, Any],
        selection_funcs : Optional[Iterable[Callable[[np.ndarray], np.ndarray]]] = None
    ) -> List[Tuple[float, float]]:
    """
    Run a single XGBoost test. Returns train and test ROC AUC.
    
    
    data_train, data_test - train and test data 2D arrays, respectively
    
    params - parameters dict to be passed to XGBClassifier
    
    selection_funcs - (optional) list of selection functions to be called
            on data that should return a 1D boolean selection mask
    
    
    Returns a list of tuples (train_score, test_score), first one for the global
    data and then one per each of the functions in selection_funcs.
    
    
    Data structure should be: 
      weight  feature_1  feature_2 ... feature_n  class
      weight  feature_1  feature_2 ... feature_n  class
      weight  feature_1  feature_2 ... feature_n  class
      ...
    """
    if selection_funcs is None:
        selection_funcs = []
    
    model = xgb.XGBClassifier(**params)
    model.fit(X=data_train[:,1:-1],
              y=data_train[:,-1],
              sample_weight=data_train[:,0])
    
    preds_train = model.predict_proba(data_train[:,1:-1])[:,1]
    preds_test  = model.predict_proba(data_test [:,1:-1])[:,1]

    _get_score = lambda data, pred, sel=None: (
        my_roc_auc(data[:,-1], pred, data[:,0])
        if sel is None else
        my_roc_auc(data[:,-1][sel], pred[sel], data[:,0][sel])
    )
    scores_global = (_get_score(data_train, preds_train),
                     _get_score(data_test , preds_test ))
    
    logger.info(f"scores global train: {scores_global[0]}, scores global test: {scores_global[1]}")
    
    scores_sel = [
        (_get_score(data_train, preds_train, sfunc(data_train)),
         _get_score(data_test , preds_test , sfunc(data_test )))
        for sfunc in selection_funcs
    ]
    
    return [scores_global] + scores_sel

def kfold_test(
        data : np.ndarray,
        params : Dict[str, Any],
        k : int = 3,
        show_progress : bool = True,
        selection_funcs : Optional[Iterable[Callable[[np.ndarray], np.ndarray]]] = None
    ) -> np.ndarray:
    """
    Run single_test(...) on k folds for cross validation.
    
    
    data - data 2D array
    
    params - parameters dict to be passed to XGBClassifier
    
    k - number of folds
    
    show_progress - if True, call tqdm on the k-fold splits
    
    selection_funcs - (optional) list of selection functions to be called
            on data that should return a 1D boolean selection mask

    
    Returns a numpy array with the resulting scores of the shape:
        (
            k,                         # per each fold
            1 + len(selection_funcs),  # global + each selection from selection_funcs
            2                          # train, test
        )
    """
    
    kf = KFold(n_splits=k, shuffle=True)
    splits = kf.split(data)
    if show_progress:
        splits = tqdm(splits, total=k)

    return np.array([
        single_test(data[i_train], data[i_test], params, selection_funcs)
        for i_train, i_test in splits
    ])


def eval_model(particle, data):
    params = dict(
        max_depth=5,
        n_estimators=200,
        learning_rate=0.1,
        min_child_weight=50,
        n_jobs=24
    )
    selection_funcs, bins = \
        make_nd_bin_selection(('Brunel_P', [0., 10000., 50000., 1000000.]),
                            ('Brunel_ETA', [2., 3., 3.5, 5.]),
                            ('nTracks_Brunel', [0, 150, 250, 1000]))
    if particle == 'electron':
        selection_funcs, bins = \
            make_nd_bin_selection(('Brunel_P', [0., 10000., 50000.]),
                                ('Brunel_ETA', [2., 3., 3.5,]),
                                )
    scores = kfold_test(merge_dataframes(data[particle].test),
                          params, selection_funcs=selection_funcs)
    logger.info(f"mean +- std global train {np.mean(scores[:, 0, 0])} +- {np.std(scores[:, 0, 0])}")
    logger.info(f"mean +- std global test {np.mean(scores[:, 0, 1])} +- {np.std(scores[:, 0, 1])}")
    return scores, bins