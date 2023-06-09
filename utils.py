import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def spot_plot(adata, pair_index, pair_color = "white"):
    spa_pixel = adata.obsm["spatial"].copy()
    scalefactor = adata.uns["spatial"]['V1_Mouse_Brain_Sagittal_Anterior']["scalefactors"]["tissue_hires_scalef"] 
    pixels = np.apply_along_axis(
        lambda x : (spa_pixel[x] * scalefactor).reshape(-1), 
        1, 
        pair_index
    )
    _, ax = plt.subplots(constrained_layout = True, figsize = (8, 6))
    sc.pl.spatial(
        adata, 
        img_key = "hires", 
        size = 1.2,
        show = False, 
        ax = ax, 
        zorder = 1,
        color = "clusters"
    )
    for lines in pixels:
        _ = ax.plot(
            [lines[0], lines[2]], 
            [lines[1], lines[3]],
            alpha = 0.7,
            zorder = 2,
            color = pair_color
        )