from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from colour import Color
import pandas as pd
import numpy as np

from anndata import AnnData

from xgboost import XGBClassifier


def importance(
    xgbmodel: XGBClassifier, 
    spagene: list, 
    threshold: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate Gene Importance Score based on Feature Importance of XGBoost. The steps are as follows.
    - Select genes with both scores greater than or equal to the `threshold`;
    - Take the mean as Importance Score of the gene.


    Parameters
    ----------
    xgbmodel : XGBClassifier
        Trained XGBoost model.

    spagene : list
        The spatially variable genes (also `adata.var_names`).

    threshold : Optional[int], default=None
        The value that determines gene selection. If None, the 80th score is used as the threshold.

    Returns
    -------
    gene_score
        A dataframe that records selected genes and Importance Scores.
    """

    # region gene selection
    feat_imp = pd.Series(xgbmodel.get_booster().get_fscore()).sort_values(ascending = False)
    if threshold is None:
        threshold = feat_imp.iloc[80]
    feat_imp = feat_imp[feat_imp >= threshold]
    feature_string = feat_imp.index.copy()
    feature_index = []
    num_features = xgbmodel.n_features_in_ / 2
    for each in feature_string:
        tmp = int(each.strip("f"))
        if tmp >= num_features:
            tmp = int(tmp - num_features)
        feature_index.append(tmp)
    feat_imp.index = feature_index
    feat_imp = feat_imp[feat_imp.index.value_counts() > 1]
    # endregion

    gene_score = feat_imp.groupby(feat_imp.index).apply(
        lambda x:pd.Series(
            [x.min(), x.max(), x.mean()], 
            ["min", "max", "importance"]
        )
    )
    gene_score = gene_score.unstack(level = -1)
    gene_score.sort_values("importance", ascending = False, inplace = True)
    gene_score["feature"] = np.array(spagene)[gene_score.index]
    gene_score = gene_score[["feature", "min", "max", "importance"]]

    return gene_score



def plot_importance(
    gene_score: pd.DataFrame, 
    num: int = 15, 
    figname: Optional[str] = None
) -> None:
    """
    Plot Top `num` Gene Importance Score.

    Parameters
    ----------
    gene_score : pd.DataFrame
        The dataframe that records selected genes and Importance Scores.

    num : int
        Top `num` genes to plot.

    figname : Optional[str], default=None
        The figure name used to join save path if `figname` is given.

    Returns
    -------
    `None`.
    """

    gene_score = gene_score[:num].copy()
    gene_score.sort_values("importance", ascending = True, inplace = True)

    with plt.style.context(['seaborn-bright']):
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        colors = list(Color("gold").range_to(Color("orangered"), num))
        colors = [each.get_hex() for each in colors]

        _, ax = plt.subplots(figsize = (6, 4.5))
        ax.grid(linestyle = " ")        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.barh(gene_score.feature, gene_score.importance, height = 0.6, color = colors, label = "importance")
        ax.tick_params(left = False)
        
        ax.scatter(gene_score["min"], gene_score["feature"], c = "lightskyblue", label = "min", alpha = 0.7, s = 17)
        ax.scatter(gene_score["max"], gene_score["feature"], c = "dodgerblue", label = "max", alpha = 0.7, s = 17)

        for i in range(len(gene_score)):
            x = gene_score.iloc[i, :][["min", "max"]].values
            y = gene_score.iloc[i, :]["feature"]
            l = mlines.Line2D(x, [y, y], linewidth = 2.2, color = "deepskyblue", alpha = 0.5)
            ax.add_line(l)

        plt.xlabel("Importance Score", fontsize = 14)
        plt.ylabel("Feature", fontsize = 14)
        plt.xlim((gene_score.importance.min() - 30, gene_score.importance.max() + 20))
        plt.yticks(fontsize = 13)
        plt.xticks(fontsize = 12)
        plt.legend()
        if figname:
            plt.savefig(figname + ".svg", bbox_inches = 'tight')
        plt.show()


