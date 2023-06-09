from typing import Optional, Union, Tuple
from collections import Iterable
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.gridspec as gridspec

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import xgboost as xgb

from kneed import KneeLocator


def refit_strategy(cv_results: dict) -> int:
    """
    Define the strategy to select the best estimator.

    The strategy defined here is to filter out all results below a `AUC` threshold
    of `best_auc - 0.005`, select the one with the lowest `loss` from the remaining. 

    Parameters
    ----------
    cv_results : dict
        Cross validation results. A dict with keys as column headers and values as columns, 
        that can be imported into a pandas DataFrame.

    Returns
    -------
    best_index_
        The index of the best estimator.
    """

    cv_results = pd.DataFrame(cv_results)
    cv_results = cv_results[
        [
            "mean_test_loss",
            "mean_test_auc",
            "params",
        ]
    ]
    best_auc = cv_results["mean_test_auc"].max()
    best_auc_threshold = best_auc - 0.005

    high_auc_cv_results = cv_results[
        cv_results["mean_test_auc"] > best_auc_threshold
    ]

    best_index_ = high_auc_cv_results["mean_test_loss"].idxmax()
    print(high_auc_cv_results.loc[best_index_])

    return best_index_



def plotcv(
    cv: GridSearchCV, 
    figname: Optional[str] = None, 
    highlight: bool = True
) -> None:
    """
    Plot the cross validation result using y-axis truncation.
    
    NOTE

    It is difficult to achieve axis truncation.

    Parameters
    ----------
    cv : GridSearchCV

    figname : Optional[str], default=None
        The figure name used to join save path if `figname` is given.

    highlight : bool, default=True
        Whether to highlight the best result.

    Returns
    -------
    `None`.
    """

    best_index = cv.best_index_
    result = cv.cv_results_

    param = list(result["params"][0].keys())[0]
    X_axis = result["param_" + param].data
    x = np.arange(1, 1 + len(X_axis))
    loss = -result["mean_test_loss"]
    auc = result["mean_test_auc"]
    min_auc = round((auc.min() - 0.02)//0.02 * 0.02 + 0.02, 2)
    min_loss = round((loss.min() - 0.05)//0.02 *0.02 + 0.02, 2)
    max_loss = round((loss.max() + 0.04)//0.02 *0.02 - 0.01, 2)
    
    with plt.style.context(['seaborn-bright']):
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize = (5.2, 3.5))
        n = 1.0 - min_auc; m = max_loss - min_loss
        gs = gridspec.GridSpec(2, 1, height_ratios = [n, m], hspace = 0.1)
        ax1 = plt.subplot(gs[0, 0:])
        plt.yticks(fontsize = 12)
        ax1.grid(linestyle = " ")
        ax2 = plt.subplot(gs[1, 0:], sharex = ax1)
        ax2.grid(linestyle = " ")
        ax1.plot(
            x,
            auc,
            color = "blue",
            marker = 'o',
            alpha = 1,
            label = "auc",
        )
        ax2.plot(
            x,
            loss,
            color = "green",
            marker = 'o',
            alpha = 1,
            label = "loss",
        )
        ax1.set_ylim(min_auc, 1.0)
        ax2.set_ylim(min_loss, max_loss)

        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.tick_params(labeltop = False)
        ax2.xaxis.tick_bottom()

        lines = []
        labels = []
        for ax in ax1,ax2:
            axLine, axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)        
        ax1.legend(
            lines, 
            labels, 
            loc = 2, 
            bbox_to_anchor = (1.0, 1.0), 
            handlelength = 2.0
        )
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        ax1.yaxis.set_major_locator(MultipleLocator(0.02))
        ax2.yaxis.set_major_locator(MultipleLocator(0.02))

        # highlight the best result
        if highlight:
            ax2.plot(
                [x[best_index], ] * 2,
                [0, loss[best_index]],
                linestyle = "-.",
                color = "black",
                marker = "x",
                markeredgewidth = 3,
                ms = 8,
            )

        d = 0.01
        kwargs = dict(transform = ax1.transAxes, color = 'k', clip_on = False)
        on = (n + m)/n; om = (n + m)/m
        ax1.plot((-d, d), (-d * on, d * on), **kwargs)        
        ax1.plot((1 - d, 1 + d), (-d * on, d * on), **kwargs)  
        kwargs.update(transform = ax2.transAxes)  
        ax2.plot((-d, d), (1 - d * om, 1 + d * om), **kwargs)  
        ax2.plot((1 - d, 1 + d), (1 - d * om, 1 + d * om), **kwargs)

        plt.xticks(x, X_axis, fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlabel(param, fontsize = 14)
        plt.locator_params("x", nbins = 10)
        if figname:
            plt.savefig(figname + ".svg", bbox_inches = 'tight')
        plt.show()



def plotcv2(
    scores: list, 
    params: Iterable, 
    key: str, 
    figname: Optional[str] = None
) -> None:
    """
    Plot the cross validation result only for one scoring `AUC` curve.

    Parameters
    ----------
    scores : list
        Cross validation `AUC` scores.

    params : Iterable
        An iterable object of parameters.

    key : str
        Parameter name.

    figname : Optional[str], default=None
        The figure name used to join save path if `figname` is given.

    Returns
    -------
    `None`.
    """

    idx = scores.index(max(scores))
    x = np.arange(1, 1 + len(params))
    min_auc = round(min(scores) - 0.05, 2)

    with plt.style.context(['seaborn-bright']):
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize = (5.2, 3.5))
        ax = plt.subplot()
        ax.grid(linestyle = " ")
        ax.plot(
            x,
            scores,
            color = "blue",
            marker = 'o',
            alpha = 1,
            label = "auc"
        )

        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.legend(
            loc = 2, 
            bbox_to_anchor = (1.0, 1.0),
            handlelength = 2.0
        )
        plt.xticks(x, params, fontsize = 12)
        plt.xlabel(key, fontsize = 14)
        plt.ylim(min_auc, 1.0)
        plt.yticks(fontsize = 12)

        ax.plot(
            [x[idx], ] * 2,
            [0, scores[idx]],
            linestyle = "-.",
            color = "black",
            marker = "x",
            markeredgewidth = 3,
            ms = 8,
        )
        if figname:
            plt.savefig(figname + ".svg", bbox_inches = 'tight')
        plt.show()



def random_search(
    model: BaseEstimator, 
    params: dict, 
    train_feature: np.ndarray,
    train_label: np.ndarray,
    n_iter: int = 30, 
    cv: int = 5,
    verbose: int = 0
) -> RandomizedSearchCV:
    """
    Randomized search on hyper parameters.

    Parameters
    ----------
    model : BaseEstimator
        A object of that type is instantiated for each grid point. 
        This is assumed to implement the scikit-learn estimator interface. 
        Either estimator needs to provide a `score` function, or `scoring` must be passed.

    params : dict
        Dictionary with parameters names as keys and lists of parameters to try.

    train_feature : np.ndarray
        2D `np.ndarray` of shape `n_sample` x `n_feature`.

    train_label : np.ndarray
        1D `np.ndarray` of length `n_sample`.

    n_iter : int, default=30
        Number of parameter settings that are sampled.

    cv : int, default=5
        The number of folds in a `StratifiedKFold`.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    Returns
    -------
    random_cv
        Instance of fitted estimator.
    """

    scores = {"auc": "roc_auc", "loss" : "neg_log_loss"}
    # Instantiate StratifiedKFold to ensure the same split.
    skf = StratifiedKFold(n_splits = cv, shuffle = True)
    random_cv = RandomizedSearchCV(
        model, 
        params, 
        n_iter = n_iter, 
        cv = skf, 
        refit = refit_strategy,
        scoring = scores, 
        verbose = verbose, 
        n_jobs = -1
    )
    random_cv.fit(train_feature, train_label)
        
    return random_cv



def grid_search(
    model: BaseEstimator, 
    params: dict, 
    train_feature: np.ndarray,
    train_label: np.ndarray,
    cv: int = 10,
    verbose: int = 0,
    figname: Optional[str] = None,
    highlight: bool = True
)  -> GridSearchCV:
    """
    Exhaustive search over specified parameter values for an estimator.
    Plot the cv result.

    Parameters
    ----------
    model : BaseEstimator
        A object of that type is instantiated for each grid point. 
        This is assumed to implement the scikit-learn estimator interface. 
        Either estimator needs to provide a `score` function, or `scoring` must be passed.

    params : dict
        Dictionary with parameters names as keys and lists of parameters to try.

    train_feature : np.ndarray
        2D `np.ndarray` of shape `n_sample` x `n_feature`.

    train_label : np.ndarray
        1D `np.ndarray` of length `n_sample`.

    cv : int, default=10
        The number of folds in a `StratifiedKFold`.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    figname : Optional[str], default=None
        The figure name used to join save path if `figname` is given.

    highlight : bool, default=True
        Whether to highlight the best result.

    Returns
    -------
    grid_cv
        Instance of fitted estimator.
    """

    scores = {"auc": "roc_auc", "loss" : "neg_log_loss"}
    # Instantiate StratifiedKFold to ensure the same split.
    skf = StratifiedKFold(n_splits = cv, shuffle = True)
    grid_cv = GridSearchCV(
        model, 
        params, 
        cv = skf, 
        refit = refit_strategy,
        scoring = scores, 
        verbose = verbose, 
        n_jobs = -1
    )
    grid_cv.fit(train_feature, train_label)
    if len(params) == 1:
        plotcv(grid_cv, figname = figname, highlight = highlight)

    return grid_cv



def xgbcv(
    init_model: XGBClassifier,
    search_param: dict,
    train_feature: np.ndarray,
    train_label: np.ndarray,
    nfold: int = 10,
    max_round: int = 1000,
    early_stopping: int = 10,
    figname: Optional[str] = None
) -> Tuple[int, int]:
    """
    Cross validate base on `XGBoost`.

    Parameters
    ----------
    init_model : XGBClassifier
        Initial model to cross validate.

    search_param : dict
        Dictionary with parameters names as keys and lists of parameters to try.
        The `key` of `search_param` is either `"num_features"` or `"learning_rate"`.

    train_feature : np.ndarray
        2D `np.ndarray` of shape `n_sample` x `n_feature`.

    train_label : np.ndarray
        1D `np.ndarray` of length `n_sample`.

    nfold : int, default=10
        The number of folds in a `StratifiedKFold`.

    max_round : int, default=1000
        `num_boost_round` in `xgb.cv`.

    early_stopping : int, default=10
        `early_stopping_rounds` in `xgb.cv`.

    figname : Optional[str], default=None
        The figure name used to join save path if `figname` is given.

    Returns
    -------
    Return the following values.

    best_param
        Optimal parameter value.

    best_n_estimators
        Optimal `n_estimators`.
    """

    key = list(search_param.keys())[0]
    n_estimators = []
    scores = []
    params = search_param[key]

    if key == "num_features":
        for each in params:
            skf = StratifiedKFold(n_splits = nfold, shuffle = True)
            start = int(train_feature.shape[1] / 2)
            X = np.concatenate(
                (train_feature[:, :each], train_feature[:, start:(start + each)]),
                1
            )
            dtrain = xgb.DMatrix(X, train_label)
            cvresult = xgb.cv(
                init_model.get_xgb_params(),
                dtrain,
                num_boost_round = max_round, 
                folds = skf,
                metrics = "auc",
                early_stopping_rounds = early_stopping
            )
            n_estimators.append(cvresult.shape[0])
            scores.append(cvresult["test-auc-mean"].iloc[-1])

    if key == "learning_rate":
        init_params = init_model.get_xgb_params()
        for each in params:
            skf = StratifiedKFold(n_splits = nfold, shuffle = True)
            dtrain = xgb.DMatrix(train_feature, train_label)
            init_params["eta"] = each
            cvresult = xgb.cv(
                init_params,
                dtrain,
                num_boost_round = max_round, 
                folds = skf,
                metrics = "auc",
                early_stopping_rounds = early_stopping
            )
            n_estimators.append(cvresult.shape[0])
            scores.append(cvresult["test-auc-mean"].iloc[-1])

    plotcv2(
        scores = scores, 
        params = params, 
        key = key, 
        figname = figname
    )
    idx = scores.index(max(scores))
    best_param = params[idx]
    best_n_estimators = n_estimators[idx]

    return best_param, best_n_estimators



def rfcv(
    init_model, 
    num_features,
    train_feature, 
    train_label, 
    nfold = 10,
    verbose = 0,
    figname = None
) -> int:  
    """
    Cross validate base on `RandomForest`.

    Parameters
    ----------
    init_model : XGBClassifier
        Initial model to cross validate.

    num_features : Iterable
        An iterabel object of `num_features`.

    train_feature : np.ndarray
        2D `np.ndarray` of shape `n_sample` x `n_feature`.

    train_label : np.ndarray
        1D `np.ndarray` of length `n_sample`.

    nfold : int, default=10
        The number of folds in a `StratifiedKFold`.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    figname : Optional[str], default=None
        The figure name used to join save path if `figname` is given.

    Returns
    -------
    best_num_features
        Optimal `num_features`.
    """

    scores = []
    for each in num_features:
        skf = StratifiedKFold(n_splits = nfold, shuffle = True)
        start = int(train_feature.shape[1] / 2)
        X = np.concatenate(
            (train_feature[:, :each], train_feature[:, start:(start + each)]),
            1
        )
        score = cross_val_score(
            init_model, 
            X, 
            train_label, 
            cv = skf, 
            scoring = "roc_auc",
            n_jobs = -1, 
            verbose = verbose
        ).mean()
        scores.append(score)
    
    plotcv2(
        scores = scores,
        params = num_features,
        key = "num_features",
        figname = figname
    )
    idx = scores.index(max(scores))
    best_num_features = num_features[idx]

    return best_num_features



def feature_selection(
    feature_importance: pd.DataFrame, 
    train_feature: np.ndarray, 
    train_label: np.ndarray, 
    nfold: int = 10,
    max_round: int = 1000,
    early_stopping: int = 10,
    threshold: Union[str, int] = "knee", 
    S: float = 1.0,
    figname: Optional[str] = None
) -> int:
    """
    Feature selection on `num` (the number of genes).

    Parameters
    ----------
    feature_importance : pd.DataFrame
        The dataframe that records selected genes and Importance Scores.
        
    train_feature : np.ndarray
        2D `np.ndarray` of shape `n_sample` x `n_feature`.

    train_label : np.ndarray
        1D `np.ndarray` of length `n_sample`.
        
    nfold : int, default=10
        The number of folds in a `StratifiedKFold`.

    max_round : int, default=1000
        `num_boost_round` in `xgb.cv`.

    early_stopping : int, default=10
        `early_stopping_rounds` in `xgb.cv`.
        
    threshold : Union[str, int], default="knee"
        `AUC` threshold to select the optimal number of features.
        Possible inputs for `threshold` are:
        - "knee", to use `KneeLocator` to find `Kneedle`;
        - int, to specify the threshold.

    S : float, default=1.0
        Sensitivity in `KneeLocator`.

    figname : Optional[str], default=None
        The figure name used to join save path if `figname` is given.

    Returns
    -------
    best_num
        Optimal `num`.
    """
    
    scores = []
    for i in range(len(feature_importance)):
        skf = StratifiedKFold(n_splits = nfold, shuffle = True)
        selected = feature_importance.index[:(i+1)]
        start = int(train_feature.shape[1] / 2)
        selected_feature = np.concatenate(
            (train_feature[:, selected], train_feature[:, start + selected]),
            1
        )
        dtrain = xgb.DMatrix(selected_feature, train_label)
        cvresult = xgb.cv(
            {"eta": 0.05},
            dtrain = dtrain,
            num_boost_round = max_round, 
            folds = skf,
            metrics = "auc",
            early_stopping_rounds = early_stopping
        )
        scores.append(cvresult["test-auc-mean"].iloc[-1])

    scores = np.array(scores)
    x = np.arange(len(feature_importance))
    X_axis = x + 1
    min_auc = max(round(min(scores) - 0.25, 2), 0)

    if threshold == "knee":
        idx = KneeLocator(x, scores, S).knee
    else:
        idx = np.where(scores > threshold)[0][0]

    with plt.style.context(['seaborn-bright']):
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize = (5.2, 3.5))
        ax = plt.subplot()
        ax.grid(linestyle = " ")
        ax.plot(
            scores,
            color = "blue",
            label = "auc"
        )
        ax.legend(
            loc = 2, 
            bbox_to_anchor = (1.0, 1.0), 
            handlelength = 1.0
        )
        plt.xticks(x, X_axis, fontsize = 12)
        plt.xlabel("m", fontsize = 14)
        plt.ylim(min_auc, 1.0)
        plt.yticks(fontsize = 12)
        plt.locator_params("x", nbins = 10)
        ticks = plt.xticks()[0].tolist()
        ticks.append(x[idx])
        labels = np.array(ticks) + 1
        plt.xticks(ticks, labels)

        ax.plot(
            [x[idx], ] * 2,
            [0, scores[idx]],
            linestyle = "-.",
            color = "black",
            marker = "x",
            markeredgewidth = 2,
            ms = 8,
        )
        if figname:
            plt.savefig(figname + ".svg", bbox_inches = 'tight')
        plt.show()
    best_num = X_axis[idx]

    return best_num


