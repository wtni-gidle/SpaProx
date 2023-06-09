import pandas as pd
from functools import partial
import numpy as np
from sklearn.utils import check_random_state
from typing import Tuple, List, Callable, Any, Union, Optional
from kneed import KneeLocator
import matplotlib.pyplot as plt
from eli5.sklearn import PermutationImportance
from sklearn.utils import check_random_state
from sklearn.base import clone

from sklearn.model_selection import StratifiedKFold


def iter_shuffled(X, columns_to_shuffle=None, 
                  random_state=None, shuffle_method="mixed"):
    """
    Return an iterator of X matrices which have one or more columns shuffled.
    After each iteration yielded matrix is mutated inplace, so
    if you want to use multiple of them at the same time, make copies.

    ``columns_to_shuffle`` is a sequence of column numbers to shuffle.
    By default, all columns are shuffled once, i.e. columns_to_shuffle
    is ``range(X.shape[1])``.

    If ``pre_shuffle`` is True, a copy of ``X`` is shuffled once, and then
    result takes shuffled columns from this copy. If it is False,
    columns are shuffled on fly. ``pre_shuffle = True`` can be faster
    if there is a lot of columns, or if columns are used multiple times.
    """
    rng = check_random_state(random_state)

    if columns_to_shuffle is None:
        if shuffle_method in ("both", "mixed"):
            columns_to_shuffle = range(int(X.shape[1]/2))
        else:
            columns_to_shuffle = range(X.shape[1])

    X_res = X.copy()
    for columns in columns_to_shuffle:
        # * mixed: 两列混合打乱
        if shuffle_method == "mixed":
            columns = [columns, columns + int(X.shape[1]/2)]
            tmp = X_res[:, columns].reshape(-1)
            rng.shuffle(tmp)
            tmp = tmp.reshape(-1, 2)
            X_res[:, columns] = tmp

        # * both: 两列分别打乱
        # ! 以下做法是两列同时打乱（打乱的顺序一致）
        # * tmp = x[:,[0,2]]
        # * rd.shuffle(tmp)
        # * x[:, [0,2]] = tmp
        if shuffle_method == "both":
            columns = [columns, columns + int(X.shape[1]/2)]
            [rng.shuffle(X_res[:, col]) for col in columns]

        else:
            rng.shuffle(X_res[:, columns])

        yield X_res
        X_res[:, columns] = X[:, columns]


def get_score_importances(
        score_func,  # type: Callable[[Any, Any], float]
        X,
        y,
        n_iter=5,  # type: int
        shuffle_method="mixed", #or mixed
        columns_to_shuffle=None,
        random_state=None
    ):
    # type: (...) -> Tuple[float, List[np.ndarray]]
    """
    Return ``(base_score, score_decreases)`` tuple with the base score and
    score decreases when a feature is not available.

    ``base_score`` is ``score_func(X, y)``; ``score_decreases``
    is a list of length ``n_iter`` with feature importance arrays
    (each array is of shape ``n_features``); feature importances are computed
    as score decrease when a feature is not available.

    ``n_iter`` iterations of the basic algorithm is done, each iteration
    starting from a different random seed.

    If you just want feature importances, you can take a mean of the result::

        import numpy as np
        from eli5.permutation_importance import get_score_importances

        base_score, score_decreases = get_score_importances(score_func, X, y)
        feature_importances = np.mean(score_decreases, axis=0)

    """
    rng = check_random_state(random_state)
    base_score = score_func(X, y)
    scores_decreases = []
    for _ in range(n_iter):
        scores_shuffled = _get_scores_shufled(
            score_func, X, y, columns_to_shuffle=columns_to_shuffle,
            random_state=rng,
            shuffle_method=shuffle_method
        )
        scores_decreases.append(-scores_shuffled + base_score)
    return base_score, scores_decreases


def _get_scores_shufled(score_func, X, y, columns_to_shuffle=None,
                        random_state=None, shuffle_method="mixed"):
    Xs = iter_shuffled(X, columns_to_shuffle, random_state=random_state, shuffle_method=shuffle_method)
    return np.array([score_func(X_shuffled, y) for X_shuffled in Xs])






class RevisedPermutationImportance(PermutationImportance):
    """Meta-estimator which computes ``feature_importances_`` attribute
    based on permutation importance (also known as mean score decrease).

    :class:`~PermutationImportance` instance can be used instead of
    its wrapped estimator, as it exposes all estimator's common methods like
    ``predict``.

    There are 3 main modes of operation:

    1. cv="prefit" (pre-fit estimator is passed). You can call
       PermutationImportance.fit either with training data, or
       with a held-out dataset (in the latter case ``feature_importances_``
       would be importances of features for generalization). After the fitting
       ``feature_importances_`` attribute becomes available, but the estimator
       itself is not fit again. When cv="prefit",
       :meth:`~PermutationImportance.fit` must be called
       directly, and :class:`~PermutationImportance` cannot be used with
       ``cross_val_score``, ``GridSearchCV`` and similar utilities that clone
       the estimator.
    2. cv=None. In this case :meth:`~PermutationImportance.fit` method fits
       the estimator and computes feature importances on the same data, i.e.
       feature importances don't reflect importance of features for
       generalization.
    3. all other ``cv`` values. :meth:`~PermutationImportance.fit` method
       fits the estimator, but instead of computing feature importances for
       the concrete estimator which is fit, importances are computed for
       a sequence of estimators trained and evaluated on train/test splits
       according to ``cv``, and then averaged. This is more resource-intensive
       (estimators are fit multiple times), and importances are not computed
       for the final estimator, but ``feature_importances_`` show importances
       of features for generalization.

    Mode (1) is most useful for inspecting an existing estimator; modes
    (2) and (3) can be also used for feature selection, e.g. together with
    sklearn's SelectFromModel or RFE.

    Currently :class:`~PermutationImportance` works with dense data.

    Parameters
    ----------
    estimator : object
        The base estimator. This can be both a fitted
        (if ``prefit`` is set to True) or a non-fitted estimator.

    scoring : string, callable or None, default=None
        Scoring function to use for computing feature importances.
        A string with scoring name (see scikit-learn `docs`_) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

        .. _docs: https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values

    n_iter : int, default 5
        Number of random shuffle iterations. Decrease to improve speed,
        increase to get more precise estimates.

    random_state : integer or numpy.random.RandomState, optional
        random state

    cv : int, cross-validation generator, iterable or "prefit"
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to disable cross-validation and compute feature importances on the same data as used for training.
            - integer, to specify the number of folds.
            - "prefit" string constant (default).

        If "prefit" is passed, it is assumed that ``estimator`` has been
        fitted already and all data is used for computing feature importances.

    refit : bool
        Whether to fit the estimator on the whole data if cross-validation
        is used (default is True).

    Attributes
    ----------
    feature_importances_ : array
        Feature importances, computed as mean decrease of the score when
        a feature is permuted (i.e. becomes noise).

    feature_importances_std_ : array
        Standard deviations of feature importances.

    results_ : list of arrays
        A list of score decreases for all experiments.

    scores_ : array of float
        A list of base scores for all experiments (with no features permuted).

    estimator_ : an estimator
        The base estimator from which the :class:`~PermutationImportance`
        instance  is built. This is stored only when a non-fitted estimator
        is passed to the :class:`~PermutationImportance`, i.e when ``cv`` is
        not "prefit".

    rng_ : numpy.random.RandomState
        random state
    """

    def __init__(self, estimator, scoring=None, n_iter=5, random_state=None, cv='prefit', refit=True, 
                 shuffle_method="mixed") -> None:
        super().__init__(estimator, scoring, n_iter, random_state, cv, refit)
        self.shuffle_method = shuffle_method

    def _cv_scores_importances(self, X, y, groups=None, **fit_params):
        assert self.cv is not None
        cv = StratifiedKFold(self.cv, shuffle = True, random_state = self.random_state)
        feature_importances = []  # type: List
        base_scores = []  # type: List[float]
        weights = fit_params.pop('sample_weight', None)
        fold_fit_params = fit_params.copy()
        for train, test in cv.split(X, y, groups):
            if weights is not None:
                fold_fit_params['sample_weight'] = weights[train]
            est = clone(self.estimator).fit(X[train], y[train], **fold_fit_params)
            score_func = partial(self.scorer_, est)
            _base_score, _importances = self._get_score_importances(
                score_func, X[test], y[test])
            base_scores.extend([_base_score] * len(_importances))
            feature_importances.extend(_importances)
        return base_scores, feature_importances
    
    def _get_score_importances(self, score_func, X, y):
        return get_score_importances(score_func, X, y, n_iter=self.n_iter, shuffle_method=self.shuffle_method, 
                                     random_state=self.rng_)
    


def show_importance(perm: RevisedPermutationImportance, feature_names):
    data = pd.DataFrame(
        {"Feature": feature_names,
         "Importance": perm.feature_importances_, 
         "std": perm.feature_importances_std_,})
    data.sort_values(by = "Importance", ascending = False, inplace = True)
    return data


def SelectFromModel(feat_imp: pd.DataFrame, threshold = None):
    if threshold == "mean":
        threshold = feat_imp["Importance"].mean()
    
    if threshold == "median":
        threshold = feat_imp["Importance"].median()
    
    if threshold is None:
        threshold = 0

    feat_imp = feat_imp[feat_imp["Importance"] >= threshold].copy()
    feat_imp.sort_values(by = "Importance", ascending = False, inplace = True)
    sel_feat = feat_imp["Feature"].values

    return sel_feat


