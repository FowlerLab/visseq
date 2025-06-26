# VIS-seq Analysis
Variant in situ Sequencing (VIS-seq) is a platform for optically profiling thousands of transgenically expressed protein-coding variants simultaneously. VIS-seq comprises a cassette with a promoter expressing a circular RNA containing one or more barcodes that are sequenced in situ to reveal the identity of the variant expressed in each cell and a second promoter expressing the protein variant. We used VIS-seq to create morphological profiles comprising a large set of measurements of the intensity, distribution and shape of different markers for >3,000 variants of lamin A and PTEN from ~11.4 million cell images. Lamin A variants were expressed in U2OS cells and PTEN variants in either iPS cells or derived neurons.

This github repository contains the bash/python code (in the folder "analysis_tools") to convert the genotyped Cells x Features matrix output from STARCall (github.com/FowlerLab/starcall-workflow) to:
  1) Variant-level morphological profiles
  2) KS-test p-values for each variant and feature, and
  3) Median and EMD values for each variant and feature,
as well as the Jupyter Notebooks (in folders LMNA and PTEN) that contain the analysis for the paper "Image-based, pooled phenotyping at nucleotide resolution reveals multidimensional, disease-specific variant effects" (BiorXiv link pending).

Morphological profiles for both _LMNA_ and _PTEN_ variants can be further explored at visseq.gs.washington.edu.

To download the input files needed for the Jupyter Notebooks, use this zenodo link: (pending)
