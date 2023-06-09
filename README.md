# SpaProx

完成模型训练和预测可以参考[tutorial.ipynb](tutorial.ipynb)

需要在根目录创建logs文件夹，用于记录训练过程。

需要在根目录创建data文件夹存放空间转录组数据，我的data文件夹已上传至[google drive](https://drive.google.com/drive/folders/1Gc8KkX8D7Wdem_suglcaB4vsZ1Udj86r?usp=drive_link)

其中包含的数据集如下所示

1. 公开的 mouse brain sagittal anterior

2. 公开的 human lymph node

3. DLPFC：

[spatial](https://github.com/LieberInstitute/HumanPilot/tree/master/10X)

[h5](https://github.com/LieberInstitute/spatialLIBD)

[ground Truth](https://drive.google.com/drive/folders/10lhz5VY7YfvHrtV40MwaqLmWz56U9eBP?usp=sharing)

4. mouse brain：

[single data](https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_snrna/all_cells_20200625.h5ad)

[annotation](https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_snrna/snRNA_annotation_astro_subtypes_refined59_20200823.csv)

[visium data](https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_visium_wo_cloupe_data.zip)

[region annotation](https://github.com/vitkl/cell2location_paper/blob/master/notebooks/selected_results/mouse_visium_snrna/scvi/v3_lr0002_Adam_signatures_c2l_Cell2location_5experiments_59clusters_14968locations_12809genes_Nonebatch_size_amortisedFalse_20000max_epochs_200alphatest/20200904_regionAnnotation_per_location.csv)

本项目使用 python 和R两种语言，python所需的依赖库见[requirements.txt](requirements.txt)

R所需的包见[spagene.R](spagene.R)
