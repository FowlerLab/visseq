# VIS-seq Analysis
Variant in situ Sequencing (VIS-seq) is a platform for optically profiling thousands of transgenically expressed protein-coding variants simultaneously. VIS-seq comprises a cassette with a promoter expressing a circular RNA containing one or more barcodes that are sequenced in situ to reveal the identity of the variant expressed in each cell and a second promoter expressing the protein variant. We used VIS-seq to create morphological profiles comprising a large set of measurements of the intensity, distribution and shape of different markers for >3,000 variants of lamin A and PTEN from ~11.4 million cell images. Lamin A variants were expressed in U2OS cells and PTEN variants in either iPS cells or derived neurons. Morphological profiles for both _LMNA_ and _PTEN_ variants can be further explored at [visseq.gs.washington.edu](https://visseq.gs.washington.edu).

<figure>
  <img width="1334" src="https://github.com/FowlerLab/visseq/blob/main/FISSEQ_Fig1_website_v2.png">
</figure>
> VIS-seq uses fluorescent in situ sequencing of abundant circular RNA barcodes to genotype cells expressing protein variants. (1) A variant library in the VIS-seq expression cassette is integrated into cells via piggyBac-ase. (2) Cells are fixed; barcodes are reverse transcribed, captured with a padlock probe and amplified; (3) cells are stained and imaged; (4) barcode is sequenced in situ; (5) single cell phenotype-genotype pairs are determined using STARCall; and (6) features for each cell are extracted using CellProfiler. Next, variant-level scores and morphological profiles are computed (this repo). (7) Feature medians and earth-mover distances are computed among all cells expressing each variant; (8) feature selection using pycytominer removes features that were highly correlated, with low variance, or that are biologically irrelevant, and the selected features are z-score normalized to generate profiles. Lastly, (9) variant embeddings are visualized using UMAP following dimensionality reduction with PCA; morphological impact scores for each variant are computed using cosine distance; variant single-cell feature distributions are KS-tested against WT; and AUROC scores for each variant reflect the ability of a model trained to distinguish variant from WT using single cell feature profiles.

This github repository contains the bash+python code (in the folder "analysis_tools") to convert the genotyped Cells x Features matrix output from [STARCall](https://github.com/FowlerLab/starcall-workflow) to:
  1) Variant-level morphological profiles
  2) KS-test p-values for each variant and feature, and
  3) Median and EMD values for each variant and feature,

as well as the Jupyter Notebooks (in folders LMNA and PTEN) that contain the analysis for the paper ["Image-based, pooled phenotyping reveals multidimensional, disease-specific variant effects"](https://www.biorxiv.org/content/10.1101/2025.07.03.663081v1). The code used to generate variant-level AUROC scores is found at [https://github.com/FowlerLab/fisseqtools](https://github.com/FowlerLab/fisseqtools).

First begin by creating a new conda environment (visseq) with Python 3.11 and then using pip -r to install the packages in "requirements.txt". One important dependency is [pycytominer](https://github.com/cytomining/pycytominer) which we use in our generation of variant profiles.

To generate profiles, run generate_profiles.sh with the following inputs:
  1) Experiment name
  2) Cells by features table (.cells_full.parquet file, output of STARCall; each cell should have an associated genotype)
  3) Metadata columns file (optional), specifying which columns of the Cells by Features table are metadata and which are features (default used for PTEN)
  4) Blacklist grep file (optional), specifying which features are blocked (ie do not contain biological information; default used for PTEN)
  5) Thresholds (optional) of number of cells per BC and number of BC per variant (default used for PTEN)
  6) Whether to z-score normalize (optional) to all variants (PTEN) or synonymous variants only (Lamin A)
  7) Name of the column (optional) encoding barcodes (PTEN='virtualBarcode' default; Lamin A='upBarcode')

To directly download the input profiles/p-values/feature summary values/variant curation data needed to run Jupyter Notebooks, use this [zenodo link](https://zenodo.org/records/15787684). Then, run the Jupyter Notebooks to reproduce the paper figures for Lamin A / PTEN portions of the paper (Figs 2-6).
