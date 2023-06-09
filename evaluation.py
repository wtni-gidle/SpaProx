from typing import Tuple, Optional
from anndata import AnnData

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, 
    matthews_corrcoef,
    roc_curve, 
    auc,
    precision_recall_curve, 
    average_precision_score,
    RocCurveDisplay, 
    PrecisionRecallDisplay
)

import scanpy as sc


def sample_data(
    data: np.ndarray, 
    size: int = 100
) -> Tuple[np.ndarray, int]:
    """
    Sample the data.

    Parameters
    ----------
    data : np.ndarray
        The data to sample.

    size : int, default=100
        The larger the `size`, the more sampled data.

    Returns
    -------
    Return the following variables.

    data_sp
        The sampled data.

    ratio
        The sampling ratio.
    """

    ratio = (size / data.shape[0] * 100) // 5 * 5 / 100
    if ratio >= 0.5:
        data_sp = data
        ratio = None
    else:
        size = int(data.shape[0] * ratio)
        index = np.random.choice(data.shape[0], size, replace = False)
        data_sp = data[index]
    
    return data_sp, ratio



def distance_compare(
    error_pair_distance: np.ndarray, 
    distance: np.ndarray, 
    neighbor_dis: float = 1.0,
    dirname: Optional[str] = None
) -> None:
    """\
    Plot the distance distribution of `False Positive` and `Negative`, and  
    calculate Mann-Whitney U test p-value.

    Parameters
    ----------
    error_pair_distance : np.ndarray
        1D `np.ndarray` of length `n_pair`, record the true distance of misjudged pairs. 
    
    distance : np.ndarray
        1D `np.ndarray` of length :math:`n_obs x (n_obs - 1) / 2`, record the distance.
    
    neighbor_dis : float, default=1.0
        Cell neighborhood radius.

    dirname : Optional[str], defaul=None
        The directory name used to join save path if `dirname` is given.
    
    Returns
    -------
    `None`.
    """

    # Mann-Whitney U test
    data_1 = error_pair_distance[error_pair_distance > neighbor_dis]
    data_2 = distance[distance > neighbor_dis]
    stattest = stats.mannwhitneyu(
        data_1, 
        data_2, 
        alternative = 'less'
    )
    with plt.style.context(['seaborn-bright']):
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        _, ax = plt.subplots(sharex = True, figsize = (5.2, 4.5))
        plt.grid(linestyle = " ")
        sns.kdeplot(
            data_1, 
            color = "#F75050", 
            label = "False Positive", 
            ax = ax
        )
        sns.kdeplot(
            data_2, 
            color = "#81AAFA", 
            label = "Negative", 
            ax = ax
        )
        plt.legend(
            loc = 2, 
            handlelength = 1.5,
            bbox_to_anchor = (1.0, 1.0)
        )
        plt.xlabel("distance")
        plt.ylabel("")

        a, b = "{a:.1e}".format(a = stattest.pvalue).split("e")
        b = str(int(b))
        plt.text(0.61, 0.9, "$p = " + a + r"\times {10}^{" + b + "}$", transform = ax.transAxes)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        if dirname:
            plt.savefig(dirname + "/distance.svg", bbox_inches = 'tight')
        plt.show()
    


def PairPlot(
    adata: AnnData, 
    pixel_pair: np.ndarray, 
    color: str,
    title: str = "",
    figname: Optional[str] = None
) -> None:
    """\
    Plot lines between pairs based on `spatial` plot.

    Parameters
    ----------
    adata : AnnData
        The adata to plot.

    pixel_pair : np.ndarray
        2D `np.ndarray` of shape `n_pair` x 4, record the pixel positions of the pair.

    color : str
        Keys for annotations of observations/cells or variables/genes.

    title : str, default=""
        The `title` of figure.

    figname : Optional[str], default=None
        The figure name used to join save path if `figname` is given.

    Returns
    ------
    `None`.
    """

    # region spatialplot
    _, ax = plt.subplots(constrained_layout = True, figsize = (8, 6))
    pointcolor = cycler(color = ["white"])
    sc.pl.spatial(
        adata, 
        img_key = "hires", 
        size = 1.2,
        show = False, 
        ax = ax, 
        zorder = 1,
        title = title,
        palette = pointcolor,
        color = color,
        groups = color
    )
    del adata.uns[color + "_colors"]
    # endregion

    # region pairlines
    linecolor = "black"
    for lines in pixel_pair:
        lineplot = ax.plot(
            [lines[0], lines[2]], 
            [lines[1], lines[3]],
            alpha = 0.7,
            zorder = 2,
            color = linecolor,
            label = color
        )
    # endregion

    plt.xlabel("")
    plt.ylabel("")
    plt.legend(
        handles = lineplot, 
        bbox_to_anchor = (1.0, 1.0), 
        loc = 2, 
        handlelength = 1.5,
        frameon = False
    )
    if figname:
        plt.savefig(figname + ".svg", bbox_inches = 'tight')
    plt.show()



def PairPlot2(
    adata: AnnData, 
    pixel_pair: np.ndarray,
    color: str, 
    title: str = "",
    ratio: Optional[float] = None,
    figname: Optional[str] = None
) -> None:
    """
    Plot lines between pairs based on `spatial` plot. 

    `Lines`: use different colors to distinguish whether spots belong to different clusters.

    `Spots`: clustering result mapping.

    Parameters
    ----------
    adata : AnnData
        The adata to plot.

    pixel_pair : np.ndarray
        2D `np.ndarray` of shape `n_pair` x 5, record the pixel positions of the pair 
        and whether spots belong to different clusters.

    color : str
        Keys for annotations of observations/cells or variables/genes.

    title : str, default=""
        The `title` of figure.

    ratio : Optional[float], default=None
        Sampling ratio.

    figname : Optional[str], default=None
        The figure name used to join save path if `figname` is given.

    Returns
    -------
    `None`.
    """

    # region spatialplot
    _, ax = plt.subplots(constrained_layout = True, figsize = (8, 6))
    if ratio:
        title = title + "({a}%)".format(a = int(ratio * 100))
    sc.pl.spatial(
        adata, 
        img_key = "hires", 
        size = 1.5,
        alpha = 0.7,
        show = False, 
        ax = ax, 
        zorder = 1,
        title = title,
        color = color
    )
    h, _ = ax.get_legend_handles_labels()
    # endregion

    # region pairlines
    for lines in pixel_pair:
        if lines[-1]:
            b, = ax.plot(
                [lines[0], lines[2]], 
                [lines[1], lines[3]],
                zorder = 2,
                color = "#19198C",
                label = "different clusters"
            ) 
        else:
            c, = ax.plot(
                [lines[0], lines[2]], 
                [lines[1], lines[3]],
                zorder = 2,
                color = "#FFE100",
                label = "same cluster"
            )
    # endregion

    try:        
        h.extend([b, c])
    except:
        try:
            h.extend([b])
        except:
            h.extend([c])
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(
        handles = h,
        bbox_to_anchor = (1.0, 1.0), 
        loc = 2, 
        handlelength = 1.5,
        frameon = False,
        ncol = 2
    )
    if figname:
        plt.savefig(figname + ".svg", bbox_inches = 'tight')
    plt.show()



def PairPlot3(
    adata: AnnData, 
    pixel_pair: np.ndarray, 
    color : str,
    title: str = "",
    ratio: Optional[float] = None,
    figname: Optional[str] = None
) -> None:
    """
    Plot lines between pairs based on `spatial` plot.

    `Spots`: clustering result mapping.

    Parameters
    ----------
    adata : AnnData
        The adata to plot.

    pixel_pair : np.ndarray
        2D `np.ndarray` of shape `n_pair` x 4, record the pixel positions of the pair.

    color : str
        Keys for annotations of observations/cells or variables/genes.

    title : str, default=""
        The `title` of figure.

    ratio : Optional[float], default=None
        Sampling ratio.

    figname : Optional[str], default=None
        The figure name used to join save path if `figname` is given.

    Returns
    -------
    `None`.
    """

    # region spatialplot
    _, ax = plt.subplots(constrained_layout = True, figsize = (8, 6))
    text = title
    if ratio:
        title = title + "({a}%)".format(a = int(ratio * 100))
    sc.pl.spatial(
        adata, 
        img_key = "hires", 
        size = 1.5,
        alpha = 0.7,
        show = False, 
        ax = ax, 
        zorder = 1,
        title = title,
        color = color
    )
    h, _ = ax.get_legend_handles_labels()
    # endregion

    # region pairlines
    for lines in pixel_pair:
        b, = ax.plot(
            [lines[0], lines[2]],
            [lines[1], lines[3]],
            zorder = 2,
            color = "#19198C",
            label = text
        )
    # endregion

    h.extend([b])
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(
        handles = h, 
        bbox_to_anchor = (1.0, 1.0), 
        loc = 2, 
        handlelength = 1.5,
        frameon = False,
        ncol = 2
    )
    if figname:
        plt.savefig(figname + ".svg", bbox_inches = 'tight')
    plt.show()



def evaluate(
    test_label: np.ndarray, 
    predprob: np.ndarray, 
    verbose: bool = True,
    threshold = 0.5
) -> dict:
    """
    Calculate classification metrics for evaluating models.

    Parameters
    ----------
    test_label : np.ndarray
        1D `np.ndarray` of shape `n_samples`, record the test label.

    pred : np.ndarray
        1D `np.ndarray` of shape `n_samples`, record the prediction.

    predprob : np.ndarray
        `np.ndarray` of shape (`n_samples`, ) or (`n_samples`, 2), record the predicted probability.

    verbose : bool
        Whether to plot `ROC` and `PR` curve.
    
    Returns
    -------
    evaluation
        Evaluation result.
    """

    if len(predprob.shape) == 2:
        predprob = predprob[:, 1]
    pred = predprob >= threshold
    confusion = confusion_matrix(test_label, pred)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_Score = 2 * Precision * Recall / (Precision + Recall)
    mcc = matthews_corrcoef(test_label, pred)

    # roc
    fpr, tpr, _ = roc_curve(test_label, predprob)
    roc_auc = auc(fpr, tpr)
    # if verbose:
    #     roc = RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = roc_auc)
    #     roc.plot()
    #     plt.show()

    # pr
    p, r, _ = precision_recall_curve(test_label, predprob)
    AveragePrecision = average_precision_score(test_label, predprob)
    if verbose:
        pr = PrecisionRecallDisplay(precision = p, recall = r, average_precision = AveragePrecision)
        pr.plot()
        plt.show()

    evaluation = {
        "Accuracy": Accuracy,
        "Precision": Precision,
        "Recall": Recall,
        "MCC": mcc,
        "F1_Score": F1_Score,
        "AUC": roc_auc,
        "Average Precision": AveragePrecision,
        "confusion_matrix": confusion
    }

    return evaluation



def advance(
    pred: np.ndarray, 
    test_label: np.ndarray, 
    test_index: np.ndarray, 
    adata: AnnData, 
    scalefactor: float,
    distance: np.ndarray,
    neighbor_dis: float = 1.0,
    map: bool = True,
    sample: bool = True,
    dirname: str = None
) -> None:
    """
    Advanced Evaluation of Model Performance.
    - Plot misjudged samples using spatial plot.
    - Plot the distance distribution of `False Positive` and `Negative`.

    Parameters
    ----------
    pred : np.ndarray
        1D `np.ndarray` of shape `n_samples`, record the prediction.

    test_label : np.ndarray
        1D `np.ndarray` of shape `n_samples`, record the test label.

    test_index : np.ndarray
        2D `np.ndarray` of shape `n_samples` x 2, record the test spot index.

    adata : AnnData
        The adata to plot.

    scalefactor : float
        A scale factor that converts pixel positions in the original full-resolution image 
        to pixel positions in `tissue_hires_image.png`.

    distance : np.ndarray
        1D `np.ndarray` of length :math:`n_obs x (n_obs - 1) / 2`, record the distance.

    neighbor_dis : float, default=1.0
        Cell neighborhood radius.

    map : bool, default=True
        Whether to map clustering results.

    sample : bool, default=True
        Whether to sample.

    dirname : Optional[str], defaul=None
        The directory name used to join save path if `dirname` is given.

    Returns
    -------
    `None`.
    """    

    error_index = np.where(pred != test_label)[0]

    # True Labels of Misjudged Samples
    error_true = test_label[error_index]
    # Pair Index of Misjudged Samples
    error_pair_index = test_index[error_index]

    # region The pixel position of Misjudged Samples
    spa_pixel = adata.obsm["spatial"].copy()
    pixels = np.apply_along_axis(
        lambda x : (spa_pixel[x] * scalefactor).reshape(-1), 
        1, 
        error_pair_index
    )

    pixels_FP = pixels[~error_true]
    pixels_FN = pixels[error_true]

    # Whether spots belong to different clusters
    pair_diff = np.apply_along_axis(
        lambda x : adata.obs["clusters"][x[0]] != adata.obs["clusters"][x[1]],
        1,
        error_pair_index
    )
    pair_diff_FP = pair_diff[~error_true].reshape(-1, 1)
    pair_diff_FN = pair_diff[error_true].reshape(-1, 1)
    # endregion

    # region Add misjudgment type in obs
    spot_FP = list(set(error_pair_index[~error_true].reshape(-1)))
    spot_FN = list(set(error_pair_index[error_true].reshape(-1)))

    adata.obs["False Positive"] = "else"
    adata.obs.loc[adata.obs.index[spot_FP], "False Positive"] = "False Positive"
    adata.obs["False Positive"].astype("category", copy = False)
    

    adata.obs["False Negative"] = "else"
    adata.obs.loc[adata.obs.index[spot_FN], "False Negative"] = "False Negative"
    adata.obs["False Negative"].astype("category", copy = False)
    # endregion

    # region Visualization of Misjudged Samples

    # FP
    PairPlot(adata, pixels_FP, color = "False Positive", title = "False Positive", figname = dirname + "/FP")

    if map:
        pixels_FP = np.concatenate((pixels_FP, pair_diff_FP), axis = 1)
        if sample:
            pixels_FP, ratio = sample_data(pixels_FP)
        else:
            ratio = None
        PairPlot2(adata, pixels_FP, color = "clusters", title = "False Positive", 
                  ratio = ratio, figname = dirname + "/FP_map")
        pt = round((1 - pair_diff_FP.mean()) * 100, 1)
        print("The proportion of spot pairs belonging to the same cluster in the False Positive"
               "is: {x}%".format(x = pt))

    # FN
    PairPlot(adata, pixels_FN, color = "False Negative", title = "False Negative", figname = dirname + "/FN")
    
    if map:
        pixels_FN = np.concatenate((pixels_FN, pair_diff_FN), axis = 1)
        if sample:
            pixels_FN, ratio = sample_data(pixels_FN)
        else:
            ratio = None
        PairPlot2(adata, pixels_FN, color = "clusters", title = "False Negative", 
                  ratio = ratio, figname = dirname + "/FN_map")
        pt = round(pair_diff_FN.mean() * 100, 1)
        print("The proportion of spot pairs belonging to different clusters in the False Negative"
               "is: {x}%".format(x = pt))

    # endregion

    # region Plot the distance distribution

    distance_index = np.apply_along_axis(
        lambda x:(2 * len(adata) - min(x) - 1) * min(x) / 2 + max(x) - min(x) - 1, 
        1, 
        error_pair_index
    )
    distance_index = distance_index.astype("int64")
    # True distance of Misjudged Samples
    error_pair_distance = distance[distance_index]

    distance_compare(error_pair_distance, distance, neighbor_dis, dirname = dirname)
    # endregion
    


def edge_evaluate(
    test_label: np.ndarray, 
    pred: np.ndarray, 
    predprob: np.ndarray, 
    test_index: np.ndarray, 
    adata: AnnData,
    verbose: bool = True
) -> Tuple[dict, np.ndarray]:
    """
    Calculate classification metrics for evaluating models in the different clusters situation.

    Parameters
    ----------
    test_label : np.ndarray
        1D `np.ndarray` of shape `n_samples`, record the test label.

    pred : np.ndarray
        1D `np.ndarray` of shape `n_samples`, record the prediction.

    predprob : np.ndarray
        `np.ndarray` of shape (`n_samples`, ) or (`n_samples`, 2), record the predicted probability.

    test_index : np.ndarray
        2D `np.ndarray` of shape `n_samples` x 2, record the test spot index.

    adata : AnnData
        The adata to evaluate.

    verbose : bool
        Whether to plot `ROC` and `PR` curve.
    
    Returns
    -------
    Return the following variables.

    evaluation
        Evaluation result.
    
    pair_diff
        1D `np.ndarray` of shape `n_samples`, record whether spots belong to different clusters.
    """
    
    if len(predprob.shape) == 2:
        predprob = predprob[:, 1]
    # Whether spots belong to different clusters
    pair_diff = np.apply_along_axis(
        lambda x : adata.obs["clusters"][x[0]] != adata.obs["clusters"][x[1]],
        1,
        test_index
    )
    edge_pred = pred[pair_diff]
    edge_predprob = predprob[pair_diff]
    edge_label = test_label[pair_diff]

    evaluation = evaluate(test_label = edge_label, pred = edge_pred, predprob = edge_predprob, verbose = verbose)

    return evaluation, pair_diff



def edge_advance(
    pair_diff: np.ndarray,
    pred: np.ndarray, 
    test_label: np.ndarray, 
    test_index: np.ndarray, 
    adata: AnnData,
    scalefactor: float,
    sample: bool = True,
    dirname: str = None
) -> None:
    """
    Plot misjudged samples using spatial plot in the different clusters situation.

    Parameters
    ----------
    pair_diff : np.ndarray
        1D `np.ndarray` of shape `n_samples`, record whether spots belong to different clusters.

    pred : np.ndarray
        1D `np.ndarray` of shape `n_samples`, record the prediction.

    test_label : np.ndarray
        1D `np.ndarray` of shape `n_samples`, record the test label.

    test_index : np.ndarray
        2D `np.ndarray` of shape `n_samples` x 2, record the test spot index.

    adata : AnnData
        The adata to evaluate.

    scalefactor : float
        A scale factor that converts pixel positions in the original full-resolution image 
        to pixel positions in `tissue_hires_image.png`.
        
    sample : bool, default=True
        Whether to sample.

    dirname : Optional[str], defaul=None
        The directory name used to join save path if `dirname` is given.

    Returns
    -------
    `None`.
    """

    edge_pred = pred[pair_diff]
    edge_label = test_label[pair_diff]
    edge_index = test_index[pair_diff]

    # region 不同类型样本的连线pixel
    TP_index = np.where(np.logical_and(edge_pred == 1, edge_label == 1))[0]
    FP_index = np.where(np.logical_and(edge_pred == 1, edge_label == 0))[0]
    FN_index = np.where(np.logical_and(edge_pred == 0, edge_label == 1))[0]

    spa_pixel = adata.obsm["spatial"].copy()
    pixels = np.apply_along_axis(
        lambda x : (spa_pixel[x] * scalefactor).reshape(-1), 
        1, 
        edge_index
    )

    pixels_TP = pixels[TP_index]
    pixels_FP = pixels[FP_index]
    pixels_FN = pixels[FN_index]
    # endregion

    # region Visualization of Test Samples
    if sample:
        pixels_TP, ratio = sample_data(pixels_TP, size = 400)
    else:
        ratio = None
    PairPlot3(adata, pixels_TP, color = "clusters", title = "True Positive", ratio = ratio, figname = dirname + "/TP_2")

    if sample:
        pixels_FP, ratio = sample_data(pixels_FP)
    else:
        ratio = None
    PairPlot3(adata, pixels_FP, color = "clusters", title = "False Positive", ratio = ratio, figname = dirname + "/FP_2")

    if sample:
        pixels_FN, ratio = sample_data(pixels_FN)
    else:
        ratio = None
    PairPlot3(adata, pixels_FN, color = "clusters", title = "False Negative", ratio = ratio, figname = dirname + "/FN_2")
    # endregion



def model_compare(**kwargs) -> pd.DataFrame:
    """
    Compare the evaluation results of models.
    
    Parameters
    ----------
    **kwargs
    - keys : Model name.
    - values: A dict of evaluation results.

    Returns
    -------
    df
        The evaluation results of models.
    """

    result = [pd.Series(each) for each in kwargs.values()]
    df = pd.concat(result, axis = 1).T
    df.index = kwargs.keys()
    df.drop("confusion_matrix", axis = 1, inplace = True)
    df = df.astype(float).round(3)

    return df


