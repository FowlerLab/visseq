import os
import pandas as pd
import numpy as np
import re
import itertools
from glob import glob
import pickle
from typing import Callable

# ignore mix type warnings from pandas
import warnings
warnings.filterwarnings("ignore")

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import ImageGrid
from adjustText import adjust_text
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib_scalebar.scalebar import ScaleBar

# Read and write images
import skimage.io

# Biopython
from Bio import PDB
from Bio.Seq import Seq

# Data analysis
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.stats import false_discovery_control, zscore, mannwhitneyu, median_abs_deviation, fisher_exact, chi2_contingency, linregress, wasserstein_distance, ks_2samp
import scipy.sparse as sp
import scipy.spatial as spt
import scipy.cluster.hierarchy as hc
from sklearn.impute import KNNImputer
import community as community_louvain
import networkx as nx
from sklearn.neighbors import NearestNeighbors

# define mutation types
mutation_types = ['3nt Deletion', 
                  'Single Missense', 
                  'Synonymous', 
                  'Nonsense', 
                  'Frameshift', 
                  'Other']
variant_type_palette = \
    {'Single Missense':'grey',
     'Synonymous':'darkgreen',
     'Frameshift':'purple', 
     '3nt Deletion':'blue', 
     'Nonsense':'red', 
     'Other':'brown'}

# information about tile sequence
pten_WT_nucseq = \
'ATGACAGCCATCATCAAAGAGATCGTTAGCAGAAACAAAAGGAGATATCAAGAGGATGGATTCGACTTAGACTTGACCTATATTTATCCAAACATTATTGCTATGGGATTTCCTGCAGAAAGACTTGAAGGCGTATACAGGAACAATATTGATGATGTAGTAAGGTTTTTGGATTCAAAGCATAAAAACCATTACAAGATATACAATCTTTGTGCTGAAAGACATTATGACACCGCCAAATTTAATTGCAGAGTTGCACAATATCCTTTTGAAGACCATAACCCACCACAGCTAGAACTTATCAAACCCTTTTGTGAAGATCTTGACCAATGGCTAAGTGAAGATGACAATCATGTTGCAGCAATTCACTGTAAAGCTGGAAAGGGACGAACTGGTGTAATGATATGTGCATATTTATTACATCGGGGCAAATTTTTAAAGGCACAAGAGGCCCTAGATTTCTATGGGGAAGTAAGGACCAGAGACAAAAAGGGAGTAACTATTCCCAGTCAGAGGCGCTATGTGTATTATTATAGCTACCTGTTAAAGAATCATCTGGATTATAGACCAGTGGCACTGTTGTTTCACAAGATGATGTTTGAAACTATTCCAATGTTCAGTGGCGGAACTTGCAATCCTCAGTTTGTGGTCTGCCAGCTAAAGGTGAAGATATATTCCTCCAATTCAGGACCCACACGACGGGAAGACAAGTTCATGTACTTTGAGTTCCCTCAGCCGTTACCTGTGTGTGGTGATATCAAAGTAGAGTTCTTCCACAAACAGAACAAGATGCTAAAAAAGGACAAAATGTTTCACTTTTGGGTAAATACATTCTTCATACCAGGACCAGAGGAAACCTCAGAAAAAGTAGAAAATGGAAGTCTATGTGATCAAGAAATCGATAGCATTTGCAGTATAGAGCGTGCAGATAATGACAAGGAATATCTAGTACTTACTTTAACAAAAAATGATCTTGACAAAGCAAATAAAGACAAAGCCAACCGATACTTTTCTCCAAATTTTAAGGTGAAGCTGTACTTCACAAAAACAGTAGAGGAGCCGTCAAATCCAGAGGCTAGCAGTTCAACTTCTGTAACACCAGATGTTAGTGACAATGAACCTGATCATTATAGATATTCTGACACCACTGACTCTGATCCAGAGAATGAACCTTTTGATGAAGATCAGCATACACAAATTACAAAAGTCTAG'
pten_start_pos=112
pten_end_pos=172
pten_WT_aaseq = \
    str(Seq(pten_WT_nucseq).translate())
pten_WT_aaseq_T3 = \
    pten_WT_aaseq[(pten_start_pos-1):pten_end_pos]
channel_to_dye = {
    'CH0': 'DAPI',
    'CH1': 'PTEN',
    'CH2': 'Phalloidin',
    'CH3': 'pAKT',
    'CH4': 'ConA'
}

# set view in pymol
pten_pymol_view = \
'''set_view (\
    -0.787645280,    0.306422234,   -0.534530580,\
     0.274807274,    0.951202154,    0.140341401,\
     0.551448226,   -0.036352079,   -0.833415985,\
     0.000000000,    0.000000000, -208.299804688,\
    36.423484802,   82.340126038,   31.888553619,\
   164.225158691,  252.374450684,  -20.000000000 )'''

# classify each variant into mutation class
def variant_classification(v):
    if ('fs' in v):
        classification = 'Frameshift'
    elif (v[-1]=='-'):
        vs = v.split('|')
        ncodons_aff = len(vs)
        if ncodons_aff > 2:
            classification = 'Other'
        else:
            if ncodons_aff == 1:
                classification = '3nt Deletion'
            elif ncodons_aff == 2:
                if int(vs[0][1:-1]) == (int(vs[1][1:-1])-1):
                    classification = '3nt Deletion'
                else:
                    classification = 'Other'
            else:
                classification = 'Other'
    elif ('X' in v) | ('*' in v):
        classification = 'Nonsense'
    elif ('WT' in v):
        classification = 'WT'
    else:
        regex_match = re.match(r'([A-Z])(\d+)([A-Z])', v)
        if regex_match is None:
            classification = 'Other'
        elif regex_match.group(1) == regex_match.group(3):
            classification = 'Synonymous'
        else:
            classification = 'Single Missense'
            
    return classification

# Classify each set of Condition(s) on clinvar
def classify_clinvar_entry_pten(entry: str):
    # Split the entry by '|' to handle multiple conditions in one entry
    conditions = entry.split('|')
    
    categories = []
    
    # Check for PHTS (any of the phts keywords)
    # Note: PHTS is defined as including hereditary cancer-predisposing syndrome, Cowden, BRRS, and PTEN hamartoma tumor syndrome.
    
    # Define keyword sets for each category
    phts_keywords = {
        "Hereditary cancer-predisposing syndrome",
        "PTEN hamartoma tumor syndrome",
        "Cowden syndrome 1",
        "Cowden syndrome",
        "PTEN-related disorder",
        "Bannayan-Riley-Ruvalcaba syndrome"  # Add if needed
    }

    somatic_cancer_keywords = {
        "Prostate cancer, hereditary, 1",
        "Neoplasm",
        "Squamous cell carcinoma of the head and neck",
        "Malignant tumor of prostate"
    }

    glioma_keywords = {
        "Glioma susceptibility 2",
        "Glioma"
    }

    macrocephaly_asd_keywords = {
        "Macrocephaly-autism syndrome"
    }
    
    if any(any(k in c for k in phts_keywords) for c in conditions):
        categories.append("PHTS")
    
    # Check for Somatic Cancer
    if any(any(k in c for k in somatic_cancer_keywords) for c in conditions):
        categories.append("Somatic Cancer")
    
    # Check for Glioma susceptibility
        categories.append("Glioma susceptibility")
    
    # Check for Macrocephaly/ASD disorder
    if any(any(k in c for k in macrocephaly_asd_keywords) for c in conditions):
        categories.append("Macrocephaly/ASD")
    
    return categories

# Categorize LMNA diseases for each variant
def LMNA_map_phenotype_to_categories(phenotype: str):
    """
    Returns a dict of booleans for each of the five categories:
    { 'DCM': bool, 'Myopathy': bool, 'Lipodystrophy': bool, 
      'Progeria': bool, 'CMT': bool }
    """
    phenotype_lower = phenotype.lower()
    out = {
        'DCM': False,
        'Myopathy': False,
        'Lipodystrophy': False,
        'Progeria': False,
        'CMT': False
    }
    # DCM category: any mention of DCM, cardiomyopathy, "cardiac disease", etc.
    if any(
        key in phenotype_lower 
        for key in ["dcm", "cardiomyopathy", "cardiac disease", "hcm", "arvc", "cmd1a"]
    ):
        out['DCM'] = True
    # Myopathy category: mentions of myopathy, EDMD, LGMD, muscular dystrophy, etc.
    if any(
        key in phenotype_lower
        for key in ["myopathy", "edmd", "lgmd", "cmd", "muscle", "muscular dystrophy"]
    ):
        out['Myopathy'] = True
    # Lipodystrophy category: mentions of FPLD, lipodystrophy, etc.
    if any(
        key in phenotype_lower
        for key in ["fpld", "fpl", "lipodystrophy", "pld"]
    ):
        out['Lipodystrophy'] = True
    # Progeria category: progeroid, progeria, werner syndrome, HGPS, MADA, etc.
    if any(
        key in phenotype_lower
        for key in ["progeroid", "progeria", "werner", "hgps", "mad"]
    ):
        out['Progeria'] = True
    # CMT category: any mention of CMT
    if "cmt" in phenotype_lower:
        out['CMT'] = True

    return out

def PTEN_combine_phenotype_classes(phenotype_classes):
    """
    Given a list/Series of Phenotype_Class entries in a group,
    decide on the group-level Phenotype_Class.
    
    Rules:
    - 'ASD/DD' if ONLY ASD/DD in entire group
    - 'PHTS'   if ONLY PHTS in entire group
    - 'ASD/DD & PHTS' otherwise
    """
    phenotype_classes_str = [pc for pc in phenotype_classes if isinstance(pc, str)]

    # Does any row mention 'ASD/DD'?
    any_asd = any("ASD/DD" in pc for pc in phenotype_classes_str)
    # Does any row mention 'PHTS'?
    any_phts = any("PHTS" in pc for pc in phenotype_classes_str)
    
    # If we have *both* ASD/DD and PHTS across the group, it's "ASD/DD & PHTS"
    if any_asd and any_phts:
        return "ASD/DD & PHTS"
    elif any_asd:
        return "ASD/DD"
    elif any_phts:
        return "PHTS"
    else:
        # If there's a phenotype that doesn't match either, you might just return
        # something else. But in your example, we only have these classes.
        return "Other"

# Make pie charts for differential features for groups of variants
def plot_grouped_feature_pies(
    pval_df: pd.DataFrame,
    zval_df: pd.DataFrame,
    p_thresh: float,
    z_thresh: float,
    groups: dict[str, list[str]],
    classify_feature: Callable[[list[str]], list[str]],
    palette: dict[str,str],
    title: str = None,
    subtitle: str = None,
    figsize: tuple[int,int] = None,
    autopct_hide_small: Callable[[float], str] = lambda pct: f"{pct:.1f}%" if pct>=3 else ""
):
    """
    For each group of variants, calls (p<p_thresh & |z|>=z_thresh) to
    pick significant features, classifies them via `classify_feature`,
    computes per-variant category proportions, averages them, and plots pies.
    
    Parameters
    ----------
    pval_df : DataFrame
        indexed by Variant, columns are feature-names, values are p-values
    zval_df : DataFrame
        same shape & index as pval_df, values are z-scores
    p_thresh : float
        p-value cutoff (inclusive)
    z_thresh : float
        absolute z-score cutoff (inclusive)
    groups : dict[str, list[str]]
        mapping group-name → list of Variant IDs
    classify_feature : callable
        function(features: list[str]) → list[str] of same length
    palette : dict[str,str]
        mapping each classification → a matplotlib color
    title : str, optional
        suptitle for the figure
    subtitle : str, optional
        subtitle annotation (e.g. f"p<{p_thresh:.1e}, |z|>={z_thresh}")
    figsize : tuple, optional
        overall figure size (width, height)
    autopct : callable, optional
        function mapping percentage→label string in wedges
    
    Returns
    -------
    Matplotlib Figure
    """
    n_groups = len(groups)
    if figsize is None:
        figsize = (4*n_groups, 4)
    fig, axes = plt.subplots(1, n_groups, figsize=figsize)
    
    # iterate each group
    hits_per_group = {}
    for ax, (grp_name, variants) in zip(axes, groups.items()):
        per_variant_props = []
        hit_counts = []
        hits_per_group[grp_name] = {}
        
        for v in variants:
            if v not in pval_df.index or v not in zval_df.index:
                # skip if missing
                continue
            
            p_row = pval_df.loc[v]
            z_row = zval_df.loc[v].abs()
            mask  = (p_row <= p_thresh) & (z_row >= z_thresh)
            hits  = mask.index[mask].tolist()
            
            hits_per_group[grp_name][v] = hits
            hit_counts.append(len(hits))
            if not hits:
                per_variant_props.append(pd.Series(0, index=palette.keys()))
            else:
                cats = classify_feature(hits)
                cnts = pd.Series(cats).value_counts()
                per_variant_props.append(cnts / cnts.sum())
        
        # average proportions
        if per_variant_props:
            avg_prop = pd.DataFrame(per_variant_props).fillna(0).mean(axis=0)
        else:
            avg_prop = pd.Series(0, index=palette.keys())
        avg_prop = avg_prop.reindex(palette.keys(), fill_value=0)
        
        # average hit-count
        avg_hits = float(np.mean(hit_counts)) if hit_counts else 0.0
        
        # draw pie
        wedges, texts, autotexts = ax.pie(
            avg_prop,
            autopct=autopct_hide_small,
            startangle=90,
            colors=[palette[c] for c in avg_prop.index],
            textprops={"color":"white","fontsize":14}
        )
        ax.axis("equal")
        ax.set_title(f"{grp_name}\nn={len(variants)}\navg hits={avg_hits:.1f}", fontsize=12)
    
    # shared legend
    handles = [
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=palette[c], markersize=10)
        for c in palette
    ]
    fig.legend(
        handles, list(palette.keys()),
        title="Category",
        loc="center left",
        bbox_to_anchor=(0.85,0.5),
        title_fontsize=14,
        fontsize=12
    )
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.97)
    if subtitle:
        fig.text(0.5, 0.96, subtitle, ha="center", fontsize=12)
    
    plt.tight_layout(rect=[0,0,0.82,1])
    return (fig, hits_per_group)

# Rescale images so that they are visible
def normalize_image(img, 
                    ignore_zeros=True, 
                    percentile=99.9, 
                    clip_low=None, #manually specify low and high values
                    clip_high=None):
    """
    Revised: explicitly clip to [0,1] before calling img_as_ubyte.
    """
    # Convert to float32
    img = img.astype(np.float32, copy=False)

    if img.ndim == 2:
        # Optionally ignore zeros
        if ignore_zeros:
            valid_pixels = img[img != 0]
        else:
            valid_pixels = img.ravel()

        if valid_pixels.size < 2:
            # No range to scale, return zero 3-channel
            return np.stack([img.astype(np.uint8)]*3, axis=-1)
        if clip_low == None:
            high = np.percentile(valid_pixels, percentile)
            low  = np.percentile(valid_pixels, 100 - percentile)
    
            # Clip to [low, high]
            img_clipped = np.clip(img, low, high)
            # Rescale to [0,1]
            img_rescaled = (img_clipped - low) / (high - low + 1e-8)
            # Explicitly clip to [0,1]
            img_rescaled = np.clip(img_rescaled, 0, 1)
    
            # Convert to 8-bit
            img_8u = skimage.img_as_ubyte(img_rescaled)
            # Make 3-channel
            img_norm = np.stack([img_8u]*3, axis=-1)
        else:
            # Clip to [low, high]
            img_clipped = np.clip(img, clip_low, clip_high)
            # Rescale to [0,1]
            img_rescaled = (img_clipped - clip_low) / (clip_high - clip_low + 1e-8)
            # Explicitly clip to [0,1]
            img_rescaled = np.clip(img_rescaled, 0, 1)
    
            # Convert to 8-bit
            img_8u = skimage.img_as_ubyte(img_rescaled)
            # Make 3-channel
            img_norm = np.stack([img_8u]*3, axis=-1)

    else:
        # NxMxC case
        H, W, C = img.shape
        img_norm = np.zeros((H, W, C), dtype=np.uint8)
        
        for c in range(C):
            channel = img[..., c]
            if ignore_zeros:
                valid_pixels = channel[channel != 0]
            else:
                valid_pixels = channel.ravel()

            if valid_pixels.size < 2:
                # No range to scale
                continue

            if clip_low == None:
                high = np.percentile(valid_pixels, percentile)
                low  = np.percentile(valid_pixels, 100 - percentile)
    
                channel_clipped = np.clip(channel, low, high)
                channel_rescaled = (channel_clipped - low) / (high - low + 1e-8)
                channel_rescaled = np.clip(channel_rescaled, 0, 1)  # <-- crucial line
    
                img_norm[..., c] = skimage.img_as_ubyte(channel_rescaled)
            else:
                channel_clipped = np.clip(channel, clip_low[c], clip_high[c])
                channel_rescaled = (channel_clipped - clip_low[c]) / (clip_high[c] - clip_low[c] + 1e-8)
                channel_rescaled = np.clip(channel_rescaled, 0, 1)  # <-- crucial line
    
                img_norm[..., c] = skimage.img_as_ubyte(channel_rescaled)

    return img_norm

# Function to pull random example cells by variant
def visualize_cells_byvariant(variant_list,
                              genotypes_df, 
                              n_square=5,
                              crop_size=224,
                              plot_middle=100,
                              figure_size=10,
                              output_folder='./crops_T3R2/'):
    
    # mkdir output
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # initialize cellcrops dict
    cell_crops = {}
    for v in variant_list:
        cell_crops[v] = []
        
    # find cell images
    for tile_group in genotypes_df[genotypes_df['aaChanges'].isin(variant_list)]\
        .groupby('aaChanges')\
        .sample(n=n_square**2)\
        .groupby('PhenotypePath'):
        phenotype_image = skimage.io.imread(tile_group[0])
        for k,row_cell in tile_group[1].iterrows():
            centroid = (row_cell['AreaShape_Center_Y'], row_cell['AreaShape_Center_X'])
            variant = row_cell['aaChanges']
            if phenotype_image.shape[0] == 4:
                cell_crops[variant].append(
                    np.moveaxis(
                        phenotype_image[:,(round(centroid[0])-crop_size//2):(round(centroid[0])+crop_size//2),
                            (round(centroid[1])-crop_size//2):(round(centroid[1])+crop_size//2)].copy(), 
                                0, -1)
                                            ) # need copy here to copy this data out
            else:
                cell_crops[variant].append(
                    phenotype_image[(round(centroid[0])-crop_size//2):(round(centroid[0])+crop_size//2),
                        (round(centroid[1])-crop_size//2):(round(centroid[1])+crop_size//2),:].copy()
                                            ) # need copy here to copy this data out
        
    # plot
    bounds=((crop_size-plot_middle)//2, crop_size-(crop_size-plot_middle)//2)
    for v in variant_list:
        fig = plt.figure(figsize=(figure_size, figure_size))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(n_square,n_square),  # creates grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cell_crops[v]):
            # Iterating over the grid returns the Axes.
            ax.imshow(normalize_image(im[bounds[0]:bounds[1],bounds[0]:bounds[1],0:3]))
            ax.grid(False)
        fig.suptitle(v, fontsize=20)
        plt.savefig(os.path.join(output_folder,v+'.png'))
        plt.show()
    
    # return cell crops dict
    return cell_crops

def visualize_cells_byvariant2(variant_list,
                               genotypes_df, 
                               n_square=5,
                               crop_size=224,
                               plot_middle=100,
                               figure_size=10,
                               cellcenterx='AreaShape_Center_X',
                               cellcentery='AreaShape_Center_Y',
                               output_folder='./crops/'):
    
    # mkdir output
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # initialize cellcrops dict
    cell_crops = {}
    for v in variant_list:
        cell_crops[v] = []
        
    # find cell images
    for tile_group in genotypes_df[genotypes_df['aaChanges'].isin(variant_list)]\
        .groupby('aaChanges')\
        .sample(n=n_square**2)\
        .groupby('PhenotypePath'):
        phenotype_image = skimage.io.imread(tile_group[0], mode='r')
        for k,row_cell in tile_group[1].iterrows():
            centroid = (row_cell[cellcentery], row_cell[cellcenterx])
            variant = row_cell['aaChanges']
            cell_crops[variant].append(
                np.moveaxis(
                    phenotype_image[(round(centroid[0])-crop_size//2):(round(centroid[0])+crop_size//2),
                        (round(centroid[1])-crop_size//2):(round(centroid[1])+crop_size//2)].copy(), 
                            0, -1)
                                      )
        
    # plot
    bounds=((crop_size-plot_middle)//2, crop_size-(crop_size-plot_middle)//2)
    for v in variant_list:
        fig = plt.figure(figsize=(figure_size, figure_size))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(n_square,n_square),  # creates grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cell_crops[v]):
            # Iterating over the grid returns the Axes.
            ax.imshow(normalize_image(im[bounds[0]:bounds[1],bounds[0]:bounds[1]]))
            ax.grid(False)
        fig.suptitle(v, fontsize=20)
        plt.savefig(os.path.join(output_folder,v+'.png'))
        plt.show()
    
    # return cell crops dict
    return cell_crops

# Visualize by variant, with DAPI, cellmask, scalebar
def visualize_cells_byvariant3(
        variant_list,
        genotypes_df, 
        n_square=5,
        crop_size=224,
        plot_middle=100,
        figure_size=10,
        pixel_size=0.108,  # microns per pixel (example)
        cellcenterx='AreaShape_Center_X',
        cellcentery='AreaShape_Center_Y',
        output_folder='./crops/',
        show_images=False,
        clip_phenotype=None,
        show_DAPI=True,
        clip_DAPI=None,
        crop_cell=True
    ):
    """
    For each variant in `variant_list`, randomly sample n_square^2 cells, 
    load Phenotype + DAPI + CellMask images, crop & mask each cell, 
    normalize, then display with scale bars in a 5x5 grid.
    
    Args:
        variant_list (list): List of variants to visualize.
        genotypes_df (pd.DataFrame): Must have columns 
            ['aaChanges', 'PhenotypePath', 'DAPIPath', 'CellMaskPath', cellcenterx, cellcentery].
        pixel_size (float): Size of each pixel in microns, for scale bar.
    """
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # Initialize dict to store crops for each variant
    cell_crops = {v: [] for v in variant_list}
    
    # We will sample n_square^2 rows per variant, then group by PhenotypePath
    # so we can load each tile only once. 
    # However, the simplest approach is as the original code: 
    #   - group by 'aaChanges'
    #   - sample rows
    #   - group by 'PhenotypePath'
    # If you want to avoid multiple loads of the same path for different variants,
    # you'd restructure the code. But we'll keep the same logic for now.
    
    sub_df = genotypes_df[genotypes_df['aaChanges'].isin(variant_list)]
    
    # For each variant, sample n_square^2, then group by PhenotypePath
    # If some variant has fewer than 25 rows total, sample with replace=True or handle it otherwise
    for variant, group_var in sub_df.groupby('aaChanges'):
        if len(group_var) < n_square**2:
            sampled_rows = group_var.sample(n=n_square**2, replace=True, random_state=0)
        else:
            sampled_rows = group_var.sample(n=n_square**2, replace=False, random_state=0)
        
        # Now group this subset by PhenotypePath
        for phen_path, tile_group in sampled_rows.groupby('PhenotypePath'):
            
            # The tile_group has the same DAPIPath & CellMaskPath
            if show_DAPI:
                dapi_path = tile_group['DAPIPath'].iloc[0]
            mask_path = tile_group['CellMaskPath'].iloc[0]

            # Load images
            phenotype_img = skimage.io.imread(phen_path)  # e.g., 16-bit
            if show_DAPI:
                dapi_img      = skimage.io.imread(dapi_path)  # e.g., 16-bit
            cellmask_img  = skimage.io.imread(mask_path)  # label image

            # For each row in this tile
            for k, row_cell in tile_group.iterrows():
                centroid_y = int(round(row_cell[cellcentery]))
                centroid_x = int(round(row_cell[cellcenterx]))
                
                # Crop bounds
                ymin = centroid_y - crop_size//2
                ymax = centroid_y + crop_size//2
                xmin = centroid_x - crop_size//2
                xmax = centroid_x + crop_size//2

                # If out of bounds, skip
                if (ymin < 0 or xmin < 0 or
                    ymax > phenotype_img.shape[0] or
                    xmax > phenotype_img.shape[1]):
                    continue

                # Crop 
                crop_phen  = phenotype_img[ymin:ymax, xmin:xmax].copy()
                if show_DAPI:
                    crop_dapi  = dapi_img[ymin:ymax, xmin:xmax].copy()
                crop_mask  = cellmask_img[ymin:ymax, xmin:xmax].copy()

                # Normalize before cropping
                if clip_phenotype == None:
                    phen_norm = normalize_image(crop_phen)  # shape (H, W)
                else:
                    phen_norm = normalize_image(crop_phen, 
                                                clip_low=clip_phenotype[0],
                                                clip_high=clip_phenotype[1])  # shape (H, W)
                if show_DAPI and clip_DAPI == None:
                    dapi_norm = normalize_image(crop_dapi)  # shape (H, W)
                elif show_DAPI:
                    dapi_norm = normalize_image(crop_dapi,
                                                clip_low=clip_DAPI[0],
                                                clip_high=clip_DAPI[1])  # shape (H, W)

                # Identify label in cell mask 
                # local centroid in the cropped region
                local_y = centroid_y - ymin
                local_x = centroid_x - xmin
                label_id = crop_mask[local_y, local_x]
                
                if label_id == 0:
                    # Could skip if centroid is background
                    continue

                # Create boolean mask for that label
                bool_mask = (crop_mask == label_id)
                
                # (Assuming phen_norm, dapi_norm are grayscale)
                phen_3ch = phen_norm
                if show_DAPI:
                    dapi_3ch = dapi_norm

                # Apply the mask
                if crop_cell:
                    phen_masked = phen_3ch * bool_mask[..., None]
                    if show_DAPI:
                        dapi_masked = dapi_3ch * bool_mask[..., None]
                else:
                    phen_masked = phen_3ch
                    if show_DAPI:
                        dapi_masked = dapi_3ch

                # Composite: R=0, G=phen, B=dapi
                H, W, _ = phen_masked.shape
                rgb_crop = np.zeros((H, W, 3), dtype=np.uint8)
                rgb_crop[..., 1] = phen_masked[..., 0]  # green channel
                if show_DAPI:
                    rgb_crop[..., 2] = dapi_masked[..., 0]  # blue channel
                
                # Store in dict
                cell_crops[variant].append(rgb_crop)
    
    # Now plot each variant's crops in a 5x5 grid
    bounds = ((crop_size - plot_middle)//2, crop_size - (crop_size - plot_middle)//2)
    for v in variant_list:
        fig = plt.figure(figsize=(figure_size, figure_size))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(n_square, n_square),
                         axes_pad=0.2)

        # plot each of the up to 25 crops
        for ax, im_rgb in zip(grid, cell_crops[v]):
            # Possibly show only the center region
            im_center = im_rgb[bounds[0]:bounds[1], bounds[0]:bounds[1], :]

            ax.imshow(im_center)
            ax.set_xticks([]); ax.set_yticks([])

            # Create and add a scale bar (1 pixel = pixel_size micrometers)
            scalebar = ScaleBar(pixel_size,  # pixel size
                                "um",        # units
                                location="lower right",
                                box_alpha=0,
                                label=None,
                                color='white',
                                scale_loc="bottom",
                                label_loc=None,
                                pad=0.01
                               )
            scalebar.label_formatter = lambda value, unit: "" # make sure the text is removed
            ax.add_artist(scalebar)
        
        # If we have fewer than 25 images, the rest of the subplots will be blank
        fig.suptitle(v, fontsize=20)
        plt.savefig(os.path.join(output_folder, v + '.png'))
        if show_images:
            plt.show()
        else:
            plt.close()
    
    return cell_crops

# Visualize by variant, with CH0-3 with mask and scalebar
def visualize_cells_byvariant4(
        variant_list,
        genotypes_df, 
        n_square=5,
        crop_size=224,
        plot_middle=100,
        figure_size=10,
        pixel_size=0.108,  # microns per pixel (example)
        cellcenterx='AreaShape_Center_X',
        cellcentery='AreaShape_Center_Y',
        output_folder='./crops/',
        show_images=False,
        clip_values=None,
        crop_cell=True
    ):
    """
    For each variant in `variant_list`, randomly sample n_square^2 cells, 
    load Phenotype + DAPI + CellMask images, crop & mask each cell, 
    normalize, then display with scale bars in a 5x5 grid.
    
    Args:
        variant_list (list): List of variants to visualize.
        genotypes_df (pd.DataFrame): Must have columns 
            ['aaChanges', 'PhenotypePath', 'DAPIPath', 'CellMaskPath', cellcenterx, cellcentery].
        pixel_size (float): Size of each pixel in microns, for scale bar.
    """
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # Initialize dict to store crops for each variant
    cell_crops = {v: [] for v in variant_list}
    
    # We will sample n_square^2 rows per variant, then group by PhenotypePath
    # so we can load each tile only once. 
    # However, the simplest approach is as the original code: 
    #   - group by 'aaChanges'
    #   - sample rows
    #   - group by 'PhenotypePath'
    # If you want to avoid multiple loads of the same path for different variants,
    # you'd restructure the code. But we'll keep the same logic for now.
    
    sub_df = genotypes_df[genotypes_df['aaChanges'].isin(variant_list)]
    
    # For each variant, sample n_square^2, then group by PhenotypePath
    # If some variant has fewer than 25 rows total, sample with replace=True or handle it otherwise
    for variant, group_var in sub_df.groupby('aaChanges'):
        if len(group_var) < n_square**2:
            sampled_rows = group_var.sample(n=n_square**2, replace=True, random_state=0)
        else:
            sampled_rows = group_var.sample(n=n_square**2, replace=False, random_state=0)
        
        # Now group this subset by PhenotypePath
        for phen_path, tile_group in sampled_rows.groupby('PhenotypePath'):
            
            # The tile_group has the same DAPIPath & CellMaskPath
            dapi_path  = tile_group['DAPIPath'].iloc[0]
            akt_path   = tile_group['pAKTPath'].iloc[0]
            stain_path = tile_group['CH2Path'].iloc[0]
            mask_path  = tile_group['CellMaskPath'].iloc[0]

            # Load images
            phenotype_img = skimage.io.imread(phen_path)  # e.g., 16-bit
            dapi_img      = skimage.io.imread(dapi_path)  # e.g., 16-bit
            pakt_img      = skimage.io.imread(akt_path)   # e.g., 16-bit
            stain_img     = skimage.io.imread(stain_path) # e.g., 16-bit
            cellmask_img  = skimage.io.imread(mask_path)  # label image

            # For each row in this tile
            for k, row_cell in tile_group.iterrows():
                centroid_y = int(round(row_cell[cellcentery]))
                centroid_x = int(round(row_cell[cellcenterx]))
                
                # Crop bounds
                ymin = centroid_y - crop_size//2
                ymax = centroid_y + crop_size//2
                xmin = centroid_x - crop_size//2
                xmax = centroid_x + crop_size//2

                # If out of bounds, skip
                if (ymin < 0 or xmin < 0 or
                    ymax > phenotype_img.shape[0] or
                    xmax > phenotype_img.shape[1]):
                    continue

                # Crop 
                crop_akt   = pakt_img[ymin:ymax, xmin:xmax].copy()
                crop_stain = stain_img[ymin:ymax, xmin:xmax].copy()
                crop_phen  = phenotype_img[ymin:ymax, xmin:xmax].copy()
                crop_dapi  = dapi_img[ymin:ymax, xmin:xmax].copy()
                crop_mask  = cellmask_img[ymin:ymax, xmin:xmax].copy()

                # Normalize before cropping
                if clip_values == None:
                    akt_norm   = normalize_image(crop_akt)   # shape (H, W)
                    stain_norm = normalize_image(crop_stain) # shape (H, W)
                    phen_norm  = normalize_image(crop_phen)  # shape (H, W)
                    dapi_norm  = normalize_image(crop_dapi)  # shape (H, W)
                else:
                    akt_norm   = normalize_image(crop_akt, 
                                                 clip_low=clip_values[3][0],
                                                 clip_high=clip_values[3][1])  # shape (H, W)
                    stain_norm = normalize_image(crop_stain,
                                                 clip_low=clip_values[2][0],
                                                 clip_high=clip_values[2][1])  # shape (H, W)
                    phen_norm  = normalize_image(crop_phen, 
                                                 clip_low=clip_values[1][0],
                                                 clip_high=clip_values[1][1])  # shape (H, W)
                    dapi_norm  = normalize_image(crop_dapi,
                                                 clip_low=clip_values[0][0],
                                                 clip_high=clip_values[0][1])  # shape (H, W)

                # Identify label in cell mask 
                # local centroid in the cropped region
                local_y = centroid_y - ymin
                local_x = centroid_x - xmin
                label_id = crop_mask[local_y, local_x]
                
                if label_id == 0:
                    # Could skip if centroid is background
                    continue

                # Create boolean mask for that label
                bool_mask = (crop_mask == label_id)
                
                # Apply the mask
                if crop_cell:
                    akt_masked   = akt_norm   * bool_mask[..., None]
                    stain_masked = stain_norm * bool_mask[..., None]
                    phen_masked  = phen_norm  * bool_mask[..., None]
                    dapi_masked  = dapi_norm  * bool_mask[..., None]
                else:
                    akt_masked   = akt_norm
                    stain_masked = stain_norm
                    phen_masked  = phen_norm
                    dapi_masked  = dapi_norm

                # Composite: R=akt, Y=phalloidin, G=protein, B=DAPI
                H, W, _ = phen_masked.shape
                rgb_crop = np.zeros((H, W, 3), dtype=np.uint8)
                rgb_crop[..., 0] = (akt_masked[..., 0]*0.69+stain_masked[..., 0]*0.29).astype('uint8')  # red channel
                rgb_crop[..., 1] = (phen_masked[..., 0]*0.69+stain_masked[..., 0]*0.29).astype('uint8')  # green channel
                rgb_crop[..., 2] = dapi_masked[..., 0]  # blue channel
                # --- Add the fourth image as a yellow overlay ---
                # Adding arr4 to R and G channels will produce a yellow tint
                # Use np.clip to avoid overflow above 255
                #rgb_crop[..., 0] = np.clip(rgb_crop[..., 0] + stain_masked[..., 0], 0, 255)
                #rgb_crop[..., 1] = np.clip(rgb_crop[..., 1] + stain_masked[..., 0], 0, 255)
                
                # Store in dict
                cell_crops[variant].append(rgb_crop)
    
    # Now plot each variant's crops in a 5x5 grid
    bounds = ((crop_size - plot_middle)//2, crop_size - (crop_size - plot_middle)//2)
    for v in variant_list:
        fig = plt.figure(figsize=(figure_size, figure_size))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(n_square, n_square),
                         axes_pad=0.2)

        # plot each of the up to 25 crops
        for ax, im_rgb in zip(grid, cell_crops[v]):
            # Possibly show only the center region
            im_center = im_rgb[bounds[0]:bounds[1], bounds[0]:bounds[1], :]

            ax.imshow(im_center)
            ax.set_xticks([]); ax.set_yticks([])

            # Create and add a scale bar (1 pixel = pixel_size micrometers)
            scalebar = ScaleBar(pixel_size,  # pixel size
                                "um",        # units
                                location="lower right",
                                box_alpha=0,
                                label=None,
                                color='white',
                                scale_loc="bottom",
                                label_loc=None,
                                pad=0.01
                               )
            scalebar.label_formatter = lambda value, unit: "" # make sure the text is removed
            ax.add_artist(scalebar)
        
        # If we have fewer than 25 images, the rest of the subplots will be blank
        fig.suptitle(v, fontsize=20)
        plt.savefig(os.path.join(output_folder, v + '.png'))
        if show_images:
            plt.show()
        else:
            plt.close()
    
    return cell_crops

# Function to pull random example cells by barcode
def visualize_cells_bybarcode(barcode_list,
                              genotypes_df, 
                              n_square=3,
                              crop_size=224,
                              plot_middle=100,
                              figure_size=10,
                              output_folder='./crops_T3R2_barcode/'):
    
    # mkdir output
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # initialize cellcrops dict
    cell_crops = {}
    for v in barcode_list:
        cell_crops[v] = []
        
    # find cell images
    for tile_group in genotypes_df[genotypes_df['virtualBarcode'].isin(barcode_list)]\
        .groupby('virtualBarcode')\
        .sample(n=n_square**2)\
        .groupby('PhenotypePath'):
        phenotype_image = skimage.io.imread(tile_group[0])
        for k,row_cell in tile_group[1].iterrows():
            centroid = (row_cell['AreaShape_Center_Y'], row_cell['AreaShape_Center_X'])
            bc = row_cell['virtualBarcode']
            if phenotype_image.shape[0] == 4:
                cell_crops[bc].append(
                    np.moveaxis(
                        phenotype_image[:,(round(centroid[0])-crop_size//2):(round(centroid[0])+crop_size//2),
                            (round(centroid[1])-crop_size//2):(round(centroid[1])+crop_size//2)].copy(), 
                                0, -1)
                                            ) # need copy here to copy this data out
            else:
                cell_crops[bc].append(
                    phenotype_image[(round(centroid[0])-crop_size//2):(round(centroid[0])+crop_size//2),
                        (round(centroid[1])-crop_size//2):(round(centroid[1])+crop_size//2),:].copy()
                                            ) # need copy here to copy this data out
        
    # plot
    bounds=((crop_size-plot_middle)//2, crop_size-(crop_size-plot_middle)//2)
    for v in barcode_list:
        fig = plt.figure(figsize=(figure_size, figure_size))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(n_square,n_square),  # creates grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cell_crops[v]):
            # Iterating over the grid returns the Axes.
            ax.imshow(normalize_image(im[bounds[0]:bounds[1],bounds[0]:bounds[1],0:3]))
            ax.grid(False)
        fig.suptitle(v, fontsize=20)
        plt.savefig(os.path.join(output_folder,v+'.png'))
        plt.close()
    
    # return cell crops dict
    return cell_crops

# Visualize by feature, with DAPI images, cell masks, and scale bar
def visualize_cells_byfeature(feature_list,
                              genotypes_df,
                              n_quantiles=10,
                              n_square=5,
                              crop_size=224,
                              plot_middle=100,
                              figure_size=10,
                              axes_pad=0.2,
                              pixel_size=0.108,     # micrometers per pixel, example
                              cellcenterx='AreaShape_Center_X',
                              cellcentery='AreaShape_Center_Y',
                              output_folder='./feature_crops/',
                              show_images=False,
                              phenotype_col="PhenotypePath",
                              DAPI_col="DAPIPath",
                              mask_col="CellMaskPath",
                              phenotype_ch=1, #green
                              clip_phenotype=None,
                              clip_DAPI=None,
                              crop_cell=True
                             ):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    needed_indices = {}  # { feature: {decile_label: set_of_indices} }
    cell_crops = {}      # { feature: {decile_label: [(crop_rgb, variant), ...]} }

    # 1) Create decile bins & sample
    for f in feature_list:
        decile_col = f"_decile_{f}"
        genotypes_df[decile_col] = pd.qcut(
            genotypes_df[f],
            q=n_quantiles,
            duplicates='drop'
        )
        needed_indices[f] = {}
        cell_crops[f] = {}

        for decile_label, grp_df in genotypes_df.groupby(decile_col):
            if pd.isna(decile_label):
                continue
            target_count = n_square**2
            if len(grp_df) < target_count:
                samp = grp_df.sample(n=target_count, replace=True, random_state=0)
            else:
                samp = grp_df.sample(n=target_count, replace=False, random_state=0)
            needed_indices[f][decile_label] = set(samp.index)
            cell_crops[f][decile_label] = []

    # 2) Load images once per PhenotypePath
    for phenotype_path, tile_group in genotypes_df.groupby(phenotype_col):
        # Check if there's anything needed in this tile
        relevant_rows = tile_group.index
        found_needed = False
        for f in feature_list:
            for decile_label in needed_indices[f]:
                if len(needed_indices[f][decile_label].intersection(relevant_rows)) > 0:
                    found_needed = True
                    break
            if found_needed:
                break
        if not found_needed:
            continue

        # Shared paths
        cellmask_path = tile_group[mask_col].iloc[0]
        dapi_path     = tile_group[DAPI_col].iloc[0]

        # Load images
        phenotype_img = skimage.io.imread(phenotype_path)  # 16-bit
        cellmask_img  = skimage.io.imread(cellmask_path)   # label
        dapi_img      = skimage.io.imread(dapi_path)       # 16-bit

        for row_idx, row_cell in tile_group.iterrows():
            y_cent = int(round(row_cell[cellcentery]))
            x_cent = int(round(row_cell[cellcenterx]))

            for f in feature_list:
                decile_col = f"_decile_{f}"
                decile_label = row_cell[decile_col]
                if pd.isna(decile_label):
                    continue
                if row_idx not in needed_indices[f][decile_label]:
                    continue

                ymin = y_cent - crop_size//2
                ymax = y_cent + crop_size//2
                xmin = x_cent - crop_size//2
                xmax = x_cent + crop_size//2

                # Edge checks
                if (ymin < 0 or xmin < 0 or
                    ymax > phenotype_img.shape[0] or
                    xmax > phenotype_img.shape[1]):
                    needed_indices[f][decile_label].remove(row_idx)
                    continue

                # Crop
                crop_phen16 = phenotype_img[ymin:ymax, xmin:xmax].copy()
                crop_mask   = cellmask_img[ymin:ymax, xmin:xmax].copy()
                crop_dapi16 = dapi_img[ymin:ymax, xmin:xmax].copy()

                # (A) Normalize AFTER cropping
                #     Use ignore_zeros=True so background won't shift the percentile to 0
                # Normalize AFTER cropping
                if clip_phenotype == None:
                    phen_norm_3ch = normalize_image(crop_phen16)  # shape (H, W)
                else:
                    phen_norm_3ch = normalize_image(crop_phen16, 
                                                    clip_low=clip_phenotype[0],
                                                    clip_high=clip_phenotype[1])  # shape (H, W)
                if clip_DAPI == None:
                    dapi_norm_3ch = normalize_image(crop_dapi16)  # shape (H, W)
                else:
                    dapi_norm_3ch = normalize_image(crop_dapi16,
                                                    clip_low=clip_DAPI[0],
                                                    clip_high=clip_DAPI[1])  # shape (H, W)
                #phen_norm_3ch = normalize_image(crop_phen16, ignore_zeros=True, percentile=99.9)
                #dapi_norm_3ch = normalize_image(crop_dapi16, ignore_zeros=True, percentile=99.9)

                # (B) Identify label
                crop_y = y_cent - ymin
                crop_x = x_cent - xmin
                label_id = crop_mask[crop_y, crop_x]
                if label_id == 0:
                    needed_indices[f][decile_label].remove(row_idx)
                    continue

                bool_mask = (crop_mask == label_id)

                # (C) Apply mask to normalized images
                # phen_norm_3ch is NxMx3, so multiply channel-wise
                if crop_cell:
                    phen_masked = phen_norm_3ch * bool_mask[..., None]
                    dapi_masked = dapi_norm_3ch * bool_mask[..., None]
                else:
                    phen_masked = phen_norm_3ch
                    dapi_masked = dapi_norm_3ch

                # (D) Build final R/G = phen_masked, B=dapi_masked
                h, w, _ = phen_masked.shape
                color_img = np.zeros((h, w, 3), dtype=phen_masked.dtype)
                color_img[..., phenotype_ch] = phen_masked[..., 0]  # green default
                color_img[..., 2] = dapi_masked[..., 0]  # blue

                variant = row_cell['aaChanges']
                cell_crops[f][decile_label].append((color_img, variant))
                needed_indices[f][decile_label].remove(row_idx)

                if len(cell_crops[f][decile_label]) >= (n_square**2):
                    break

    # 3) PLOTTING
    half_display = (crop_size - plot_middle) // 2
    bounds = (half_display, crop_size - half_display)

    for f in feature_list:
        for decile_label, crop_list in cell_crops[f].items():
            if len(crop_list) < (n_square**2):
                print(f"[Warning] Decile '{decile_label}' for feature '{f}'"
                      f" has only {len(crop_list)} cells.")

            fig = plt.figure(figsize=(figure_size, figure_size))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(n_square, n_square),
                             axes_pad=axes_pad)

            for ax, (crop_rgb, variant) in zip(grid, crop_list[:n_square**2]):
                # Optionally show only the center region
                crop_center = crop_rgb[bounds[0]:bounds[1], bounds[0]:bounds[1], :]
                
                # Display the image
                ax.imshow(crop_center)
                ax.set_title(str(variant), fontsize=7, pad=5)
                ax.axis('off')

                # Create and add a scale bar (1 pixel = pixel_size micrometers)
                scalebar = ScaleBar(pixel_size,  # pixel size
                                    "um",        # units
                                    location="lower right",
                                    box_alpha=0,
                                    label=None,
                                    color='white',
                                    scale_loc="bottom",
                                    label_loc=None,
                                    pad=0.01
                                   )
                scalebar.label_formatter = lambda value, unit: "" # make sure the text is removed
                ax.add_artist(scalebar)

            fig.suptitle(f"{f} - {decile_label}", fontsize=14)
            outname = f"{f}_{str(decile_label).replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_folder, outname),
                        dpi=150, bbox_inches='tight')
            if show_images:
                plt.show()
            else:
                plt.close()

    return cell_crops

# Visualize by feature, with more channels, for PTEN
def visualize_cells_byfeature2(feature_list,
                               genotypes_df,
                               n_quantiles=10,
                               n_square=5,
                               crop_size=224,
                               plot_middle=100,
                               figure_size=10,
                               axes_pad=0.2,
                               pixel_size=0.108,     # micrometers per pixel, example
                               cellcenterx='AreaShape_Center_X',
                               cellcentery='AreaShape_Center_Y',
                               output_folder='./feature_crops/',
                               show_images=False,
                               clip_values=None,
                               crop_cell=True
                              ):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    needed_indices = {}  # { feature: {decile_label: set_of_indices} }
    cell_crops = {}      # { feature: {decile_label: [(crop_rgb, variant), ...]} }

    # 1) Create decile bins & sample
    for f in feature_list:
        decile_col = f"_decile_{f}"
        genotypes_df[decile_col] = pd.qcut(
            genotypes_df[f],
            q=n_quantiles,
            duplicates='drop'
        )
        needed_indices[f] = {}
        cell_crops[f] = {}

        for decile_label, grp_df in genotypes_df.groupby(decile_col):
            if pd.isna(decile_label):
                continue
            target_count = n_square**2
            if len(grp_df) < target_count:
                samp = grp_df.sample(n=target_count, replace=True, random_state=0)
            else:
                samp = grp_df.sample(n=target_count, replace=False, random_state=0)
            needed_indices[f][decile_label] = set(samp.index)
            cell_crops[f][decile_label] = []

    # 2) Load images once per PhenotypePath
    for phenotype_path, tile_group in genotypes_df.groupby('PhenotypePath'):
        # Check if there's anything needed in this tile
        relevant_rows = tile_group.index
        found_needed = False
        for f in feature_list:
            for decile_label in needed_indices[f]:
                if len(needed_indices[f][decile_label].intersection(relevant_rows)) > 0:
                    found_needed = True
                    break
            if found_needed:
                break
        if not found_needed:
            continue

        # Shared paths
        cellmask_path = tile_group["CellMaskPath"].iloc[0]
        dapi_path     = tile_group["DAPIPath"].iloc[0]
        akt_path      = tile_group['pAKTPath'].iloc[0]
        stain_path    = tile_group['CH2Path'].iloc[0]

        # Load images
        phenotype_img = skimage.io.imread(phenotype_path)  # 16-bit
        cellmask_img  = skimage.io.imread(cellmask_path)   # label
        dapi_img      = skimage.io.imread(dapi_path)       # 16-bit
        pakt_img      = skimage.io.imread(akt_path)   # e.g., 16-bit
        stain_img     = skimage.io.imread(stain_path) # e.g., 16-bit

        for row_idx, row_cell in tile_group.iterrows():
            y_cent = int(round(row_cell[cellcentery]))
            x_cent = int(round(row_cell[cellcenterx]))

            for f in feature_list:
                decile_col = f"_decile_{f}"
                decile_label = row_cell[decile_col]
                if pd.isna(decile_label):
                    continue
                if row_idx not in needed_indices[f][decile_label]:
                    continue

                ymin = y_cent - crop_size//2
                ymax = y_cent + crop_size//2
                xmin = x_cent - crop_size//2
                xmax = x_cent + crop_size//2

                # Edge checks
                if (ymin < 0 or xmin < 0 or
                    ymax > phenotype_img.shape[0] or
                    xmax > phenotype_img.shape[1]):
                    needed_indices[f][decile_label].remove(row_idx)
                    continue

                # Crop 
                crop_akt   = pakt_img[ymin:ymax, xmin:xmax].copy()
                crop_stain = stain_img[ymin:ymax, xmin:xmax].copy()
                crop_phen  = phenotype_img[ymin:ymax, xmin:xmax].copy()
                crop_dapi  = dapi_img[ymin:ymax, xmin:xmax].copy()
                crop_mask  = cellmask_img[ymin:ymax, xmin:xmax].copy()

                # (A) Normalize AFTER cropping
                #     Use ignore_zeros=True so background won't shift the percentile to 0
                # Normalize AFTER cropping
                # Normalize before cropping
                if clip_values == None:
                    akt_norm   = normalize_image(crop_akt)   # shape (H, W)
                    stain_norm = normalize_image(crop_stain) # shape (H, W)
                    phen_norm  = normalize_image(crop_phen)  # shape (H, W)
                    dapi_norm  = normalize_image(crop_dapi)  # shape (H, W)
                else:
                    akt_norm   = normalize_image(crop_akt, 
                                                 clip_low=clip_values[3][0],
                                                 clip_high=clip_values[3][1])  # shape (H, W)
                    stain_norm = normalize_image(crop_stain,
                                                 clip_low=clip_values[2][0],
                                                 clip_high=clip_values[2][1])  # shape (H, W)
                    phen_norm  = normalize_image(crop_phen, 
                                                 clip_low=clip_values[1][0],
                                                 clip_high=clip_values[1][1])  # shape (H, W)
                    dapi_norm  = normalize_image(crop_dapi,
                                                 clip_low=clip_values[0][0],
                                                 clip_high=clip_values[0][1])  # shape (H, W)

                # (B) Identify label
                crop_y = y_cent - ymin
                crop_x = x_cent - xmin
                label_id = crop_mask[crop_y, crop_x]
                if label_id == 0:
                    needed_indices[f][decile_label].remove(row_idx)
                    continue

                bool_mask = (crop_mask == label_id)

                # (C) Apply mask to normalized images
                # phen_norm_3ch is NxMx3, so multiply channel-wise
                if crop_cell:
                    phen_masked  = phen_norm * bool_mask[..., None]
                    dapi_masked  = dapi_norm * bool_mask[..., None]
                    akt_masked   = akt_norm * bool_mask[..., None]
                    stain_masked = stain_norm * bool_mask[..., None]
                else:
                    phen_masked  = phen_norm
                    dapi_masked  = dapi_norm
                    stain_masked = stain_norm
                    akt_masked   = akt_norm

                # (D) Build final R=0, G=phen_masked, B=dapi_masked
                h, w, _ = phen_masked.shape
                color_img = np.zeros((h, w, 3), dtype=phen_masked.dtype)
                color_img[..., 0] = (akt_masked[..., 0]*0.69+stain_masked[..., 0]*0.29).astype('uint8')  # red channel
                color_img[..., 1] = (phen_masked[..., 0]*0.69+stain_masked[..., 0]*0.29).astype('uint8')  # green channel
                color_img[..., 2] = dapi_masked[..., 0]  # blue channel

                variant = row_cell['aaChanges']
                cell_crops[f][decile_label].append((color_img, variant))
                needed_indices[f][decile_label].remove(row_idx)

                if len(cell_crops[f][decile_label]) >= (n_square**2):
                    break

    # 3) PLOTTING
    half_display = (crop_size - plot_middle) // 2
    bounds = (half_display, crop_size - half_display)

    for f in feature_list:
        for decile_label, crop_list in cell_crops[f].items():
            if len(crop_list) < (n_square**2):
                print(f"[Warning] Decile '{decile_label}' for feature '{f}'"
                      f" has only {len(crop_list)} cells.")

            fig = plt.figure(figsize=(figure_size, figure_size))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(n_square, n_square),
                             axes_pad=axes_pad)

            for ax, (crop_rgb, variant) in zip(grid, crop_list[:n_square**2]):
                # Optionally show only the center region
                crop_center = crop_rgb[bounds[0]:bounds[1], bounds[0]:bounds[1], :]
                
                # Display the image
                ax.imshow(crop_center)
                ax.set_title(str(variant), fontsize=7, pad=5)
                ax.axis('off')

                # Create and add a scale bar (1 pixel = pixel_size micrometers)
                scalebar = ScaleBar(pixel_size,  # pixel size
                                    "um",        # units
                                    location="lower right",
                                    box_alpha=0,
                                    label=None,
                                    color='white',
                                    scale_loc="bottom",
                                    label_loc=None,
                                    pad=0.01
                                   )
                scalebar.label_formatter = lambda value, unit: "" # make sure the text is removed
                ax.add_artist(scalebar)

            fig.suptitle(f"{f} - {decile_label}", fontsize=14)
            outname = f"{f}_{str(decile_label).replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_folder, outname),
                        dpi=150, bbox_inches='tight')
            if show_images:
                plt.show()
            else:
                plt.close()

    return cell_crops

# Plot regression line between two columns
def scatter_with_regression(
        ax,
        data,
        x,
        y,
        hue,
        palette,
        x_label,
        y_label=None,
        legend=False,
        line_color='black',
        r_xloc=0.05,
        r_yloc=0.9,
        show_line=True
    ):
    """
    Plots a scatterplot and a regression line on the given Axes object.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object on which to plot.
    data : DataFrame
        The dataframe containing the data to be plotted.
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.
    hue : str
        Column name for grouping data (color).
    palette : dict or list
        Color palette or mapping.
    x_label : str
        Label for the x-axis.
    y_label : str, optional
        Label for the y-axis. Only used if provided (default: None).
    legend : bool, optional
        Whether to draw legend on this axis.
    line_color : str, optional
        Color of the regression line (default: 'black').
    """

    # Scatter
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        ax=ax,
        legend=legend
    )

    # Regression line (no scatter points from regplot)
    if show_line:
        sns.regplot(
            data=data,
            x=x,
            y=y,
            scatter=False,
            ci=None,
            color=line_color,
            ax=ax
        )

    # Compute R^2
    # Drop NaNs in the relevant columns
    df_no_na = data[[x, y]].dropna()
    if not df_no_na.empty:
        slope, intercept, r_value, p_value, std_err = linregress(
            df_no_na[x],
            df_no_na[y]
        )
        r_squared = r_value**2
    else:
        r_squared = float('nan')

    # Add R^2 text in the top-left corner
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Display R^2 on the plot
    ax.text(
        r_xloc, r_yloc, 
        f'$R = {r_value:.2f}$', 
        transform=ax.transAxes, 
        fontsize=20, 
        bbox=dict(facecolor='white', alpha=0.5)
    )

    # Labels
    ax.set_xlabel(x_label, fontsize=14)
    if y_label:
        ax.set_ylabel(y_label, fontsize=14)

    # Font size
    ax.tick_params(axis='both',labelsize=14)

# Plot feature scatterplots colored by other features
def plot_two_features_coloredbyactivityabundance(df, 
                                                 xcol, 
                                                 ycol,
                                                 color1='Intensity_MeanIntensity_CH1_iPSC',
                                                 color2='Intensity_MeanIntensity_CH3_Corrected_iPSC',
                                                 label1=None,
                                                 label2=None,
                                                 variants_to_annotate=[],
                                                 variantpalette=variant_type_palette,
                                                 cbar_label='Abundance/pAKT Scores iPSC',
                                                 save_name=False,
                                                 vmin=-10,
                                                 vmax=10,
                                                 xlim=(-8,3),
                                                 ylim=(-8,3),
                                                 legend_display=False,
                                                 annopalette={},
                                                 annolabel_withtext=False):
    fig,axs=plt.subplots(figsize=(18,5), ncols=3, sharex=True, sharey=True)
    # ------------------------------------
    # Create colorbar for the numeric data
    # ------------------------------------
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="bwr")
    sm.set_array([])  # needed in some older MPL versions
    
    if legend_display:
        sns.scatterplot(
            data=df,
            x=xcol,
            y=ycol,
            hue='Variant_Class',
            palette=variantpalette,
            ax=axs[0],
            legend=False
        )
        # axs[0].legend(loc=legend_display,
        #               title='Variant Type',
        #               markerscale=2,
        #               fontsize=12,
        #               title_fontsize=14)
    else:
        sns.scatterplot(
            data=df,
            x=xcol,
            y=ycol,
            hue='Variant_Class',
            palette=variantpalette,
            ax=axs[0],
            legend=False
        )
    # Add dashed lines at x=0 and y=0
    axs[0].axhline(y=0, color='grey', linestyle='--', linewidth=1)
    axs[0].axvline(x=0, color='grey', linestyle='--', linewidth=1)
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[0].tick_params(axis='both',labelsize=12)
    if label1 != None:
        axs[0].set_xlabel(label1, fontsize=18)
    if label2 != None:
        axs[0].set_ylabel(label2, fontsize=18)

    sns.scatterplot(
        data=df,
        x=xcol,
        y=ycol,
        hue=color1,
        palette='bwr',
        hue_norm=norm,
        ax=axs[1],
        legend=False
    )
    # Add dashed lines at x=0 and y=0
    axs[1].axhline(y=0, color='grey', linestyle='--', linewidth=1)
    axs[1].axvline(x=0, color='grey', linestyle='--', linewidth=1)
    axs[1].tick_params(axis='both',labelsize=12)
    if label1 != None:
        axs[1].set_xlabel(label1, fontsize=18)
    if label2 != None:
        axs[1].set_ylabel(label2, fontsize=18)
    
    sns.scatterplot(
        data=df,
        x=xcol,
        y=ycol,
        hue=color2,
        palette='bwr',
        hue_norm=norm,
        ax=axs[2],
        legend=False
    )
    # Add dashed lines at x=0 and y=0
    axs[2].axhline(y=0, color='grey', linestyle='--', linewidth=1)
    axs[2].axvline(x=0, color='grey', linestyle='--', linewidth=1)
    axs[2].tick_params(axis='both',labelsize=12)
    if label1 != None:
        axs[2].set_xlabel(label1, fontsize=18)
    if label2 != None:
        axs[2].set_ylabel(label2, fontsize=18)

    # Annotate variants
    if len(variants_to_annotate) > 0:
        variants_anno = df.query('Variant in @variants_to_annotate')
        for ax in axs:
            sns.scatterplot(
                data=variants_anno,
                x=xcol,
                y=ycol,
                hue='Label',
                palette=annopalette,
                ax=ax,
                s=300,
                marker='^',
                legend=False
            )
            if annolabel_withtext:
                text_annos = []
                for index,row in variants_anno.iterrows():
                    text_annos.append(
                        ax.text(row[xcol], row[ycol], row['Variant'], fontsize=16)
                    )
                adjust_text(text_annos,
                    expand_points=(2,2),
                    arrowprops=dict(
                        arrowstyle="-", 
                        color='black', 
                        lw=1
                        ),
                    ax=ax)
                
    cbar = plt.colorbar(sm, 
                        ax=axs[-1], 
                        fraction=0.04, 
                        pad=0.04, 
                        orientation='vertical'
                       )
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=18)
    plt.show()

    return fig
    
# Function to plot heatmap of scores
def plot_heatmap(df_heatmap,
                 wt_aa_seq=pten_WT_aaseq,
                 start_pos=pten_start_pos,
                 end_pos=pten_end_pos,
                 figure_size=(30,15),
                 tick_spacing=5,
                 plot_title="PTEN Tile 3 Rep 2 Morphological Scores after FDR Corr",
                 output_file_name="PTENT3R2.morphologicalscoreFDR.heatmap.pdf",
                 vmin=-2,
                 vmax=10,
                 vcenter=0,
                 stops=True,
                 threentdels=False,
                 save_plots=True,
                 show_plots=True,
                 domain_colors={},
                 other_anno_cols=[],
                 other_anno_colors=[{}],
                 plot_position=True,
                 domain_height=0.03,
                 anno_height=0.015
                ):
    
    #reset defaults
    sns.reset_defaults()
    plt.rcdefaults()
                  
    #amino acid ordering
    aliphatic = ["A", "V", "I", "L", "G"]
    aromatic = ["F", "Y", "W"]
    nonpolar = ["C", "M", "P"]
    polar = ["S", "T", "N", "Q"]
    acidic = ["D", "E"]
    basic = ["H", "K", "R"]
    aa_type = aliphatic+aromatic+nonpolar+polar+acidic+basic
    if stops:
        aa_type.append("*")
    if threentdels:
        # Add aatypes to possible mutations
        aa_type=aa_type+['del+0','del+1','del+2']
    
    # set amino acid order and position order
    df_heatmap['Mut'] = pd.Categorical(df_heatmap['Mut'], 
                                       categories=aa_type, 
                                       ordered=True)
    df_heatmap['position'] = pd.Categorical(df_heatmap['position'], 
                                            categories=list(range(start_pos,end_pos+1)), 
                                            ordered=True)
    
    # use wtseq, subsetting for T3
    wt_seq = wt_aa_seq[(start_pos-1):end_pos]

    # compute averaged scores per position
    position_averaged_scores = \
        df_heatmap\
            .query('Mut not in ["del+1", "del+2"] & WT != Mut')\
            [['position','score']]\
            .groupby('position')\
            .agg(np.mean)\
            .reset_index()
    position_averaged_scores['position'] = \
        position_averaged_scores['position'].map(int)
    position_averaged_scores = \
        position_averaged_scores.dropna(subset=['score'])

    # If domain info is present in df_heatmap, map it onto the averaged scores.
    if "domain" in df_heatmap.columns:
        # Use the unique mapping from position to domain.
        pos2dom = dict(df_heatmap[['position','domain']].drop_duplicates().values)
        position_averaged_scores['domain'] = position_averaged_scores['position'].map(pos2dom)
        position_averaged_scores['color'] = \
            position_averaged_scores['domain'].map(domain_colors).map(mcolors.to_hex)
    else:
        position_averaged_scores['color'] = 'black'

    #prepare the pivot table
    pivot_table = df_heatmap.pivot_table(
        index=['Mut'],
        columns='position',
        values='score',
        aggfunc='mean',
        dropna=False,
        observed=False,
        sort=True
    )

    #create the heatmap
    plt.figure(figsize=figure_size)
    #create a colormap that includes grey for NaN values
    cmap = sns.color_palette("bwr", as_cmap=True)

    #plot the heatmap and explicitly get the axis object (ax)
    hm = sns.heatmap(pivot_table, 
                     cmap=cmap, 
                     annot=False, 
                     fmt=".2f", 
                     linewidths=.5, 
                     vmin=vmin, vmax=vmax, center=vcenter, 
                     mask=pivot_table.isna(),  # Mask NaN values
                     cbar_kws={'extend':'both', 'shrink': 0.5, 'pad': 0.001},  # Shrink color bar and reduce padding
                     square=True, 
                     zorder=1)  # Ensure heatmap has a lower zorder

    #set the facecolor of the plot to grey where NaNs are present
    plt.gca().patch.set_facecolor('grey')

    #customize colorbar tick label size
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)  # Set colorbar tick labels' font size
    
    for i in range(len(wt_seq)):
        wt = wt_seq[i]
        pos = start_pos + i

        #get the x and y positions for the heatmap
        if wt in pivot_table.index and pos in pivot_table.columns:
            y = pivot_table.index.get_loc(wt)
            x = pivot_table.columns.get_loc(pos)
            #add black dot with higher zorder
            hm.scatter(x + 0.5, y + 0.5, s=30, color='black', marker='o', zorder=2)

    #get the current tick positions and labels
    current_ticks = plt.xticks()[0]

    #replace the labels with integers starting from your intended value (no start codon in the library)
    start_tick = (start_pos//tick_spacing + 1)*tick_spacing
    new_labels = [start_pos + int(tick) for tick in range(len(current_ticks))]
    every_spaced_tick = current_ticks[(start_tick-start_pos)::tick_spacing]  #select every tick
    every_spaced_label = new_labels[(start_tick-start_pos)::tick_spacing]    #select corresponding labels for those ticks

    if plot_position:
        hm.set_xticks([])
        hm.set_xticklabels([])
        plt.yticks(fontsize=20)
        hm.tick_params(axis='y', labelrotation=0)
        
    else:
        #set the new x-axis tick labels and font size as well as y font size
        plt.xticks(ticks=every_spaced_tick, labels=every_spaced_label, fontsize=25)
        hm.set_xticklabels(every_spaced_label, fontsize=25, rotation=0)
        plt.yticks(fontsize=20) 
        hm.tick_params(axis='y', labelrotation=0)
    
    #plt.title(plot_title, fontsize=50)
    plt.xlabel("")
    plt.ylabel("")
    #plt.ylabel('Amino Acid Mutations', fontsize=40, labelpad=0)

    ## DOMAIN ANNOTATION START ##
    if 'domain' in df_heatmap.columns:
        # Group by domain to find domain start and end (min, max positions)
        domain_info = \
            df_heatmap[['domain', 'position']]\
                .drop_duplicates()\
                .dropna(subset=['domain'])\
                .groupby('domain')['position']\
                .agg(['min','max'])\
                .reset_index()
        
        # Create a small axis on top (no sharex=ax here)
        hm_pos = hm.get_position()
        domain_ax = hm.figure.add_axes([
            hm_pos.x0,
            hm_pos.y0 + hm_pos.height + 0.005,
            hm_pos.width,
            domain_height
        ])
        domain_ax.set_xlim(hm.get_xlim())
        
        # We don't need any ticks or labels on domain_ax
        domain_ax.set_ylim([0, 1])
        domain_ax.set_yticks([])
        domain_ax.set_yticklabels([])
        domain_ax.set_xticks([])
        domain_ax.set_xticklabels([])

        # Hide spines on the top axis
        domain_ax.spines['top'].set_visible(False)
        domain_ax.spines['right'].set_visible(False)
        domain_ax.spines['left'].set_visible(False)
        domain_ax.spines['bottom'].set_visible(False)

        # For each domain, draw a rectangle from domain_start to domain_end
        for _, row in domain_info.iterrows():
            dname  = row['domain']
            dstart = row['min']
            dend   = row['max']
            # Convert domain start/end to pivot_table column indices
            if dstart in pivot_table.columns and dend in pivot_table.columns:
                xstart = pivot_table.columns.get_loc(dstart)
                xend   = pivot_table.columns.get_loc(dend)
                
                # Draw a rectangle spanning these positions
                width = xend - xstart + 1  # +1 to include the end
                domain_ax.add_patch(Rectangle(
                    (xstart, 0),     # (x, y) of the lower-left corner
                    width,           # width of the rectangle
                    1,               # height (we fill from y=0 to y=1)
                    color=domain_colors.get(dname, "grey"),
                    alpha=1,
                    zorder=10
                ))
                
                # Add a text label in the middle of the domain
                domain_ax.text(
                    xstart + width/2, 
                    0.5, 
                    dname, 
                    ha='center', 
                    va='center', 
                    fontsize=30, 
                    zorder=11
                )
    ## DOMAIN ANNOTATION END ##

    ## OTHER ANNOTATION START ##
    # If other annotation info is present, add colored patches without text.
    for i,anno_col_name in enumerate(other_anno_cols):
        if anno_col_name in df_heatmap.columns:
            hm_pos = hm.get_position()
            # If domain exists, then shift
            if 'domain' in df_heatmap.columns:
                y_offset = hm_pos.y0 + hm_pos.height + 0.005 + 0.03 + (anno_height+0.005)*i + 0.005  # domain axis + gap
            else:
                y_offset = hm_pos.y0 + hm_pos.height + anno_height*i + 0.005
            # Create a new axis above the domain annotation axis
            other_ax = hm.figure.add_axes([
                hm_pos.x0,
                y_offset,
                hm_pos.width,
                anno_height
            ])
            other_ax.set_xlim(hm.get_xlim())
            other_ax.set_ylim([0, 1])
            other_ax.set_xticks([])
            other_ax.set_yticks([])
            # Hide spines for a clean look
            for spine in other_ax.spines.values():
                spine.set_visible(False)
            
            # Loop through each unique other annotation
            for ann in df_heatmap[anno_col_name].dropna().unique():
                # Get all positions for which this annotation applies
                pos_list = sorted(df_heatmap.loc[df_heatmap[anno_col_name] == ann, 'position'].unique())
                # Group positions into contiguous segments
                groups = []
                current_group = []
                for pos in pos_list:
                    if not current_group:
                        current_group = [pos]
                    elif pos == current_group[-1] + 1:
                        current_group.append(pos)
                    else:
                        groups.append(current_group)
                        current_group = [pos]
                if current_group:
                    groups.append(current_group)
                
                # Draw a rectangle for each contiguous group
                for group in groups:
                    group_start = group[0]
                    group_end = group[-1]
                    if group_start in pivot_table.columns and group_end in pivot_table.columns:
                        xstart = pivot_table.columns.get_loc(group_start)
                        xend = pivot_table.columns.get_loc(group_end)
                        width = xend - xstart + 1
                        other_ax.add_patch(Rectangle(
                            (xstart, 0),
                            width,
                            1,
                            color=other_anno_colors[i].get(ann, "grey"),
                            alpha=1,
                            zorder=10
                        ))
    ## OTHER ANNOTATION END ##

    # Add position averaged plot
    ## POSITION ANNOTATION START ##
    if plot_position:
        # Create a small axis on bottom (no sharex=ax here)
        hm_pos = hm.get_position()
        pos_ax = hm.figure.add_axes([
            hm_pos.x0,
            hm_pos.y0 - 0.05,
            hm_pos.width,
            0.045
        ])
        pos_ax.set_xlim([start_pos-0.5,end_pos+0.5])
        ylim_posavg_scores = \
            [int(min(position_averaged_scores["score"]))-2, 
             int(max(position_averaged_scores["score"]))+2]
        pos_ax.set_ylim()

        # Plot your scores as a black line
        pos_ax.plot(
            position_averaged_scores["position"], 
            position_averaged_scores["score"], 
            color="black", 
            lw=2,
            zorder=1)

        # Then scatter on top to see the points
        pos_ax.scatter(position_averaged_scores["position"], 
                       position_averaged_scores["score"], 
                       c=position_averaged_scores["color"], 
                       s=50,
                       zorder=2)

        # Set axhline
        pos_ax.axhline(y=0,xmin=0,xmax=1,c='red',alpha=0.5,ls='--')

        # Set the x-axis tick labels below the position plot
        plt.xticks(ticks=every_spaced_label, labels=every_spaced_label, fontsize=25)

        # Set yticks for position plot
        pos_ax.set_yticks(ticks=[ylim_posavg_scores[0],0,ylim_posavg_scores[1]],
                          labels=[ylim_posavg_scores[0],0,ylim_posavg_scores[1]],
                          fontsize=18)
        
    ## POSITION ANNOTATION END ##
    
    if save_plots:
        plt.savefig(output_file_name, dpi=600)
    if show_plots:
        plt.show()
    else:
        plt.close()

# Function to visualize scores on structure
def paint_structure(avg_scores,
                    output_suffix='PTENNeuronT3R2_morphologyscoresFDR',
                    output_dir="./R2_plots/pdb/",
                    input_pdb='/net/fowler/vol1/home/pendyala/FISSEQ/PTEN_iPSC_T3_pycytominer_120524/1d5r.pdb',
                    pymol_view=pten_pymol_view,
                    start_pos=pten_start_pos,
                    end_pos=pten_end_pos,
                    pymol_exec="$HOME/pymol/pymol -c",
                    extra_pymol="show surface, sele; set transparency=0.5; color green, resi 1352\n",
                    image_size=(1800,3000),
                    dpi=1000,
                    vmin=0,
                    vmax=10,
                    coloring="blue_white_red",
                    discrete_coloring=False,
                    discrete_colors = ["#CFCFCF","#FF0000"] # hex colors
                   ):

    # Read in the average scores into a dictionary for fast lookup
    score_dict = dict(zip(avg_scores["position"], avg_scores["score"]))

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("PTEN", input_pdb)

    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()[1]  # residue position number
                if res_id in score_dict:
                    value = score_dict[res_id]
                    for atom in residue:
                        atom.set_bfactor(value)
                else:
                    # If no score available, set to 0 or some default
                    for atom in residue:
                        atom.set_bfactor(0.0)

    # Save structure
    io = PDB.PDBIO()
    io.set_structure(structure)
    filename_out = os.path.join(output_dir, os.path.basename(input_pdb).split('.')[0] + '_' + output_suffix)
    io.save(filename_out + ".pdb")

    # if B-factor discrete coloring
    if discrete_coloring:
        # Save script
        with open(filename_out + ".pml", "w") as f:
            f.write(pymol_view + "\n")
            f.write("select resi " + str(start_pos) + "-" + str(end_pos) + "\n")
            f.write("remove solvent\n")
            f.write("color grey, not sele\n")
            
            f.write("spectrum b, " + " ".join(["0x"+s[1:] for s in discrete_colors]) + ", sele, minimum=" + str(0) + ", maximum=" + str(len(discrete_colors)-1) + "\n")
            f.write(extra_pymol)
            #f.write("ray " + str(image_size[0]) + "," + str(image_size[1]) + "\n")
            f.write("ray " + "\n")
            f.write("png " + filename_out + ".png, " + str(image_size[0]) + ", " + str(image_size[1]) + ", dpi="+str(dpi))
        f.close()
    else:
        # Save script
        with open(filename_out + ".pml", "w") as f:
            f.write(pymol_view + "\n")
            f.write("select resi " + str(start_pos) + "-" + str(end_pos) + "\n")
            f.write("remove solvent\n")
            f.write("color grey, not sele\n")
            f.write("spectrum b, " + coloring + ", sele, minimum=" + str(vmin) + ", maximum=" + str(vmax) + "\n")
            f.write(extra_pymol)
            #f.write("ray " + str(image_size[0]) + "," + str(image_size[1]) + "\n")
            f.write("ray " + "\n")
            f.write("png " + filename_out + ".png, " + str(image_size[0]) + ", " + str(image_size[1]) + ", dpi="+str(dpi))
        f.close()
    
    pymol_command = pymol_exec + " " + filename_out + ".pdb " + filename_out + ".pml"
    return pymol_command

# Function to plot features as histogram, umap, heatmap, and pdb
def plot_feature_dist_umap(variant_scores,
                           umap_coords,
                           feature_to_heatmap,
                           plot_title,
                           plot_name,
                           dis_path='./R2_plots/distributions/',
                           umap_path='./R2_plots/heatmap/',
                           highlight_variants=False,
                           vmin=-10,
                           vmax=10,
                           vcenter=0,
                           save_plots=True,
                           show_plots=True,
                           dist_plot=True,
                           umap_plot=True,
                           score_col="ZScore",
                           remove_nonclassified_variants=True,
                           umap1_col='UMAP1',
                           umap2_col='UMAP2'):
    
    # merge scores and umap locations
    merged_data = variant_scores.merge(umap_coords, on='Variant')
    
    # remove 'Other' variants
    if remove_nonclassified_variants:
        merged_data = \
            merged_data[merged_data['Variant_Class']!='Other']
        merged_data.Variant_Class = \
            merged_data.Variant_Class.cat.remove_categories('Other')

    # Highlight variants on the density plot
    if highlight_variants:
        
        # Set class to default if class is not present
        if 'HighlightClass' not in merged_data.columns:
            merged_data['HighlightClass'] = np.nan
            merged_data[merged_data['Highlight']]['HighlightClass'] = 'Default'
        variants_to_highlight = merged_data[merged_data['Highlight']]['Variant'].values
        
        # Create a color mapping for variant classes
        highlight_classes = merged_data[merged_data['Highlight']]["HighlightClass"].unique()
        variant_classes = variant_scores['Variant_Class'].cat.categories
        palette = sns.color_palette("tab10", n_colors=len(variant_classes)+len(highlight_classes))
        palette = palette[len(variant_classes):]
        class_color_map = dict(zip(highlight_classes, palette))
    
    if dist_plot:
        # plot histograms over different types of variants
        feature_displot = \
            sns.displot(data=merged_data, 
                        x=feature_to_heatmap, 
                        hue="Variant_Class", 
                        kind='kde',
                        common_norm=False)

        ax_dis = feature_displot.ax
        
        # Determine vertical limits and reserve space at the top for labels
        ylim = ax_dis.get_ylim()
        xlim = ax_dis.get_xlim()
        ax_dis.set_ylim(ylim[0], ylim[1] * 1.4)  # Expand y-limit for text space
        ylim = ax_dis.get_ylim()
        line_top = ylim[1] * 0.85
        label_line_y = ylim[1] * 0.925
        #base_text_y = ylim[1] * 0.9

        # Highlight variants on the density plot
        if highlight_variants:

            for var in variants_to_highlight:
                # Sort highlighted variants by feature value to handle them in order
                highlighted_data = merged_data[merged_data['Variant'].isin(variants_to_highlight)]
                highlighted_data = highlighted_data.assign(xval=highlighted_data[feature_to_heatmap])
                highlighted_data = highlighted_data.sort_values('xval')

                # Keep track of placed texts to avoid overlap
                label_min_spacing = (xlim[1] - xlim[0]) * 0.02
                last_label_x_end = -np.inf
                label_positions = []
                #placed_text_positions = []  # Will store (x, y) of placed texts
                #horizontal_threshold = (ax_dis.get_xlim()[1] - ax_dis.get_xlim()[0]) * 0.03  # Adjust as needed
                #vertical_step = (ax_dis.get_ylim()[1] - ax_dis.get_ylim()[0]) * 0.06

                for _, row in highlighted_data.iterrows():
                    val = row['xval']
                    var = row['Variant']
                    var_class = row['HighlightClass']
                    var_color = class_color_map[var_class]

                    # Draw a limited vertical line
                    ax_dis.plot([val, val], [ylim[0], line_top], color=var_color, linestyle='--', linewidth=2)

                    # Determine a suitable text position
                    label_x = val
                    #text_x = val
                    #text_y = base_text_y
                    label_str = f"{var}"
                    
                    # Check if placing at label_x would overlap the previous label
                    # Approximate text width by string length times a factor
                    # (Assuming ~0.02 * text length in data units as a rough guess; adjust as needed)
                    text_width = len(label_str) * (xlim[1] - xlim[0]) * 0.003
                    # Ensure label_x is placed so that (label_x - text_width/2) > last_label_x_end + label_min_spacing
                    while (label_x - text_width/2) < (last_label_x_end + label_min_spacing):
                        label_x += label_min_spacing  # shift to the right until no overlap
                        
                    # Now place the text
                    text = ax_dis.text(label_x, 
                                       label_line_y, 
                                       label_str,
                                       color=var_color,
                                       horizontalalignment='center',
                                       verticalalignment='bottom',
                                       fontsize=8,
                                       rotation=90)
                    
                    # Update last_label_x_end to the end of this label
                    last_label_x_end = label_x + text_width/2

                    # Store label position and variant position for arrow
                    label_positions.append((val, line_top, label_x, label_line_y, var_color))

                    # Update last_label_x_end to the end of this label
                    last_label_x_end = label_x + text_width/2

                    # Store label position and variant position for arrow
                    label_positions.append((val, line_top, label_x, label_line_y, var_color))
            
                # Draw arrows connecting labels to variant lines
                # We'll use a simple arrow drawn as a line with arrowstyle
                for (val, top, lx, ly, c) in label_positions:
                    # Draw arrow from label (lx, ly) to slightly above line_top to avoid touching line
                    ax_dis.annotate("",
                                    xy=(val, top),
                                    xytext=(lx, ly - 0.005*(ylim[1]-ylim[0])),  # slightly above label line_y
                                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5),
                                    ha='center', va='bottom')
                    
                # Add title and save distribution plot
                ax_dis.set_title(plot_title, fontsize=15)
                feature_displot.savefig(os.path.join(dis_path, plot_name + '.pdf'))
                    
        # Use scientific notation if necessary
        xlim = ax_dis.get_xlim()
        if abs(xlim[0]) < 0.01:
            ax_dis.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

        plt.title(plot_title, fontsize=15)
        if save_plots:
            feature_displot.savefig(os.path.join(dis_path, plot_name+'.pdf'))
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # plot umap
    if umap_plot:
        if (type(umap1_col) == str) or (len(umap1_col) == 1):
            if type(umap1_col) == list:
                umap1_col = umap1_col[0]
                umap2_col = umap2_col[0]
            fig, ax = plt.subplots(figsize=(5, 4))
            norm = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
            sns.scatterplot(data=merged_data, 
                            x=umap1_col, 
                            y=umap2_col, 
                            hue=score_col, 
                            palette="bwr",
                            s=50,
                            legend=False,
                            hue_norm=norm,
                            ax=ax)
    
            # Highlight variants on the UMAP plot (if requested)
            if highlight_variants:
                highlighted_data = merged_data[merged_data['Variant'].isin(merged_data[merged_data['Highlight']]['Variant'].values)]
                for class_ in highlighted_data['HighlightClass'].unique():
                    class_subset = highlighted_data[highlighted_data['HighlightClass'] == class_]
                    var_color = class_subset['HighlightClass'].iloc[0]  # adjust if you have a mapping
                    sns.scatterplot(
                        data=class_subset,
                        x=umap1_col,
                        y=umap2_col,
                        color=var_color,
                        s=150,
                        marker='*',
                        edgecolor='black',
                        linewidth=1.5,
                        ax=ax,
                        legend=False
                    )
                    for _, row in class_subset.iterrows():
                        ax.text(row[umap1_col], 
                                row[umap2_col] + 0.05, 
                                f"{row['Variant']}", 
                                color='black', 
                                ha='center', 
                                va='bottom', 
                                fontsize=12)
    
            # Create a ScalarMappable for the colorbar
            sm = plt.cm.ScalarMappable(cmap="bwr", norm=norm)
            sm.set_array([])
            # Instead of attaching the colorbar to the existing axis, create a new one.
            # Adjust the coordinates [left, bottom, width, height] as needed.
            cax = fig.add_axes([1, 0.15, 0.03, 0.7])
            cbar = plt.colorbar(sm, cax=cax, extend="both", spacing="proportional", boundaries=np.linspace(vmin, vmax))
            cbar.ax.set_yscale('linear')
    
            # Format the UMAP plot
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('UMAP 1', fontsize=16)
            ax.set_ylabel('UMAP 2', fontsize=16)
            plt.tight_layout()
    
            if save_plots:
                fig.savefig(os.path.join(umap_path, plot_name + '.pdf'), dpi=600)
            if show_plots:
                plt.show()
            else:
                plt.close()
                x
        else:
            fig, axs = plt.subplots(figsize=(4 * len(umap1_col), 4), ncols=len(umap1_col))
            norm = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
            for i in range(len(umap1_col)):
                sns.scatterplot(data=merged_data, 
                                x=umap1_col[i], 
                                y=umap2_col[i], 
                                hue=score_col, 
                                palette="bwr",
                                s=50,
                                legend=False,
                                hue_norm=norm,
                                ax=axs[i])
        
                if highlight_variants:
                    highlighted_data = merged_data[merged_data['Variant'].isin(merged_data[merged_data['Highlight']]['Variant'].values)]
                    for class_ in highlighted_data['HighlightClass'].unique():
                        class_subset = highlighted_data[highlighted_data['HighlightClass'] == class_]
                        var_color = class_subset['HighlightClass'].iloc[0]  # adjust as needed
                        sns.scatterplot(
                            data=class_subset,
                            x=umap1_col[i],
                            y=umap2_col[i],
                            color=var_color,
                            s=150,
                            marker='*',
                            edgecolor='black',
                            linewidth=1.5,
                            ax=axs[i],
                            legend=False
                        )
                        for _, row in class_subset.iterrows():
                            axs[i].text(
                                row[umap1_col[i]], 
                                row[umap2_col[i]] + 0.05, 
                                f"{row['Variant']}", 
                                color='black', 
                                ha='center', 
                                va='bottom', 
                                fontsize=12
                            )
    
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].set_xlabel(umap1_col[i], fontsize=16)
                axs[i].set_ylabel(umap2_col[i], fontsize=16)
    
            # Create a ScalarMappable for the colorbar
            sm = plt.cm.ScalarMappable(cmap="bwr", norm=norm)
            sm.set_array([])
            # For multiple subplots, create a dedicated axis on the right side of the figure.
            cax = fig.add_axes([1, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(sm, cax=cax, extend="both", spacing="proportional", boundaries=np.linspace(vmin, vmax))
            cbar.ax.set_yscale('linear')
            
            plt.tight_layout()
    
            if save_plots:
                fig.savefig(os.path.join(umap_path, plot_name + '.pdf'), dpi=600)
            if show_plots:
                plt.show()
            else:
                plt.close()

def plot_feature(variant_scores,
                 umap_coords,
                 feature_to_heatmap,
                 plot_title,
                 plot_name,
                 dist_plot=True,
                 umap_plot=True,
                 heatmap_plot=True,
                 plot_position_heatmap=True,
                 pdb_plot=True,
                 highlight_variants=False,
                 save_plots=True,
                 show_plots=True,
                 stops=True,
                 threentdels=False,
                 syn_zscore=True,
                 domain_map={},
                 domain_colors={},
                 other_anno_map=[],
                 other_anno_colors=[],
                 start_pos=pten_start_pos,
                 end_pos=pten_end_pos,
                 wt_nuc_seq=pten_WT_nucseq,
                 wt_aa_seq=pten_WT_aaseq,
                 remove_nonclassified_variants=True,
                 umap1_col='UMAP1',
                 umap2_col='UMAP2',
                 dis_path='./R2_plots/distributions/',
                 heatmap_path='./R2_plots/heatmap/',
                 umap_path='./R2_plots/umap/',
                 pdb_path='./R2_plots/pdb/',
                 input_pdb='/net/fowler/vol1/home/pendyala/FISSEQ/PTEN_iPSC_T3_pycytominer_120524/1d5r.pdb',
                 pymol_view=pten_pymol_view,   
                 pymol_exec="$HOME/pymol/pymol -c",
                 extra_pymol="show surface, sele; set transparency=0.5; color green, resi 1352\n",
                 image_size=(1800,3000),
                 dpi=1000,
                 domain_height=0.03,
                 anno_height=0.015,
                 heatmap_figure_size=(30,15),
                 vmin=-10,
                 vmax=10,
                 vcenter=0):
    
    # Average feature over aggregated_df
    variant_scores = variant_scores.copy()
    if syn_zscore:
        mean_syn = variant_scores.query('Variant_Class=="Synonymous"')[feature_to_heatmap].mean()
        std_syn = variant_scores.query('Variant_Class=="Synonymous"')[feature_to_heatmap].std()
        score_col = 'ZScore'
        variant_scores[score_col] = (variant_scores[feature_to_heatmap]-mean_syn)/std_syn
    else:
        score_col = feature_to_heatmap

    # Filter for variants we want to plot
    variant_classes_to_heatmap = ["Single Missense","Synonymous"]
    if stops:
        variant_classes_to_heatmap.append("Nonsense")
    piperegex=r"\|"
    variant_scores_filtered = variant_scores\
        .query('Variant_Class in @variant_classes_to_heatmap')\
        .query('~Variant.str.contains(@piperegex)')

    # Transform into the format for the heatmap
    variant_scores_transformed = \
        pd.DataFrame({
            "WT": variant_scores_filtered["Variant"].str[0],  # Extract the first letter
            "position": variant_scores_filtered["Variant"].str[1:-1].astype(int),  # Extract the numeric part and convert to int
            "Mut": variant_scores_filtered["Variant"].str[-1],  # Extract the last letter
            "score": variant_scores_filtered[score_col]  # Retain the score
    })
    variant_scores_transformed_filt = \
        variant_scores_transformed\
            .query('position >= @start_pos & position <= @end_pos')
    
    # If include 3nt deletions
    if threentdels:
        nuc_seq_tile = wt_nuc_seq[3*(start_pos-1):3*(end_pos+1)]
        aa_seq_tile = wt_aa_seq[(start_pos-1):(end_pos+1)]

        # Map all possible 3nt deletions and classify them
        threent_deletions_classification = []
        for start_i in range(len(nuc_seq_tile) - 4):
            mutated_nucseq = nuc_seq_tile[:start_i] + nuc_seq_tile[start_i+3:]
            mutated_aa = str(Seq(mutated_nucseq).translate(to_stop=False))

            pos_first = start_i//3+start_pos
            pos_second = pos_first+1
            aa_first = mutated_aa[start_i//3]
            if start_i%3 == 0:
                del_name = aa_seq_tile[start_i//3]+str(pos_first)+'-'
            else:
                del_name = aa_seq_tile[start_i//3]+str(pos_first)+aa_first+\
                               '|'+aa_seq_tile[start_i//3+1]+str(pos_second)+'-'
            threent_deletions_classification.append([del_name, aa_seq_tile[start_i//3],pos_first,'del+'+str(start_i%3)])
        threent_deletions_classification_df = \
            pd.DataFrame(threent_deletions_classification, columns=["Variant", "WT", "position", "Mut"])
        variant_scores_transformed_3ntdel = \
            variant_scores\
                .query('Variant_Class == "3nt Deletion"')\
                .merge(threent_deletions_classification_df, on="Variant", how="inner")\
                .rename(columns={score_col:'score'})\
                [['WT',"position","Mut","score"]]

        # Combine with morphology scores for other variants
        variant_scores_transformed_filt = \
            pd.concat([variant_scores_transformed_filt,variant_scores_transformed_3ntdel])
    
    # plot distribution over variant classes and feature on umap
    if (dist_plot|umap_plot):
        plot_feature_dist_umap(variant_scores=variant_scores,
                               umap_coords=umap_coords,
                               feature_to_heatmap=feature_to_heatmap,
                               plot_title=plot_title,
                               plot_name=plot_name,
                               dis_path=dis_path,
                               umap_path=umap_path,
                               highlight_variants=highlight_variants,
                               vmin=vmin,
                               vmax=vmax,
                               vcenter=vcenter,
                               save_plots=save_plots,
                               show_plots=show_plots,
                               dist_plot=dist_plot,
                               umap_plot=umap_plot,
                               score_col=score_col,
                               remove_nonclassified_variants=remove_nonclassified_variants,
                               umap1_col=umap1_col,
                               umap2_col=umap2_col
                              )
    
    # plot heatmap
    # add domain information if it is provided
    variant_scores_transformed_filt_heatmap = variant_scores_transformed_filt.copy()
    if len(domain_map.keys()) > 0:
        # check whether every position has a domain
        pos_check=0
        for pos in range(start_pos,end_pos+1):
            if pos not in domain_map.keys():
                pos_check=1
                print('Domain dictionary does not have every position!')
        if not pos_check:
            variant_scores_transformed_filt_heatmap['domain'] = \
                variant_scores_transformed_filt_heatmap['position'].map(domain_map)
    if len(other_anno_map) > 0:
        for i,anno_dict in enumerate(other_anno_map):
            # check whether every position has a domain
            pos_check=0
            for pos in range(start_pos,end_pos+1):
                if pos not in anno_dict.keys():
                    pos_check=1
                    print('Annotation dictionary does not have every position!')
            if not pos_check:
                variant_scores_transformed_filt_heatmap['anno'+str(i+1)] = \
                    variant_scores_transformed_filt_heatmap['position'].map(anno_dict)
    if heatmap_plot:
        plot_heatmap(variant_scores_transformed_filt_heatmap,
                     plot_title=plot_title,
                     output_file_name=os.path.join(heatmap_path, plot_name+'.pdf'),
                     vmin=vmin,
                     vmax=vmax,
                     vcenter=vcenter,
                     save_plots=save_plots,
                     show_plots=show_plots,
                     stops=stops,
                     threentdels=threentdels,
                     start_pos=start_pos,
                     end_pos=end_pos,
                     wt_aa_seq=wt_aa_seq,
                     domain_colors=domain_colors,
                     other_anno_cols=['anno'+str(i+1) for i in range(len(other_anno_map))],
                     other_anno_colors=other_anno_colors,
                     plot_position=plot_position_heatmap,
                     domain_height=domain_height,
                     anno_height=anno_height,
                     figure_size=heatmap_figure_size)
    
    # plot pdb
    if pdb_plot:
        avg_scores_bypos = variant_scores_transformed.groupby("position")["score"].mean().reset_index()
        pymol_command = paint_structure(avg_scores_bypos,
                                        output_dir=pdb_path,
                                        output_suffix=plot_name,
                                        input_pdb=input_pdb,
                                        pymol_view=pymol_view,   
                                        start_pos=start_pos,
                                        end_pos=end_pos,
                                        pymol_exec=pymol_exec,
                                        extra_pymol=extra_pymol,
                                        image_size=image_size,
                                        dpi=dpi,
                                        vmin=vmin,
                                        vmax=vmax)
    else:
        pymol_command=""
    
    return variant_scores_transformed_filt, pymol_command

# Convert p. Arg216His format into R216H format
def convert_variant(hgvs_variant):
    # Mapping from three-letter amino acid codes to single-letter codes.
    # Includes 'Ter' mapping to '*'.
    aa_map = {
        "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
        "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
        "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
        "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
        "Ter": "*"
    }
    
    # Ensure the variant starts with 'p.' and then strip it
    if not hgvs_variant.startswith("p."):
        raise ValueError("Variant must start with 'p.'")
    variant_str = hgvs_variant[2:]
    
    # The variant is typically in the form of {ThreeLetterCode}{Position}{ThreeLetterCode}
    # e.g. "Arg335Ter" or "Trp274Leu"
    # We can use a regex or a simple approach since we know three-letter codes are fixed length.
    pattern = r"^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$"
    match = re.match(pattern, variant_str)
    if not match:
        raise ValueError("Variant format does not match the expected pattern (e.g. Arg335Ter).")
    
    ref_aa, pos, alt_aa = match.groups()

    # Convert three-letter codes to single-letter codes
    if ref_aa not in aa_map or alt_aa not in aa_map:
        raise ValueError("Unknown amino acid code encountered.")
    
    ref_single = aa_map[ref_aa]
    alt_single = aa_map[alt_aa]
    
    # Build the final short form variant
    return f"{ref_single}{pos}{alt_single}"

# Function to classify features into channels
def classify_features_with_dye(features,channel_to_dye=channel_to_dye):
    classifications = []
    for feature in features:
        # Find all channels in the feature name
        channels = re.findall(r'CH\d', feature)
        channels = sorted(set(channels))  # Get unique channels, sorted
        if len(channels) == 1:
            dye = channel_to_dye.get(channels[0], 'Unknown')
            classification = {
                'feature': feature,
                'type': 'Single Channel',
                'channels': channels,
                'associated_dye': dye
            }
        else:
            dyes = [channel_to_dye.get(ch, 'Unknown') for ch in channels]
            classification = {
                'feature': feature,
                'type': 'Multiple Channels',
                'channels': channels,
                'associated_dye': ', '.join(dyes)
            }
        classifications.append(classification)
    return classifications

# Function to classify features into compartments
def classify_compartment(features):
    classifications = []
    for feature_name in features:
        if "Nucl" in feature_name:
            classifications.append("Nuclei")
        elif "Cytoplasm" in feature_name:
            classifications.append("Cytoplasm")
        else:
            classifications.append("Whole Cell")
    return classifications

# Function to make correlation plots over positions
def plot_correlation_positions(
        pos_embeddings,
        metric='cosine',
        annotation_cols=[],
        annotation_palettes=[],
        label_col='position',
        perform_knn_imputation=True,
        fillnaval=0,
        n_clust_pos=7,
        annotate_clusters=True,
        dendrogram_method='average',
        cluster_palette=sns.color_palette('tab10'),
        cmap_cluster="magma",
        cluster_spacing=0.1,
        cluster_names=None,  #if provided, should be a dict mapping cluster id to name
        cluster_text_color=None, #if not provided, all white
        cluster_text_fontsize=12,
        cbar_pos=(0.05, 0.15, 0.03, 0.6),
        cbar_kws={'extend':'both', 'shrink': 0.5, 'pad': 0.001},
        plot_filename="./consensus_plots/LMNAT3.profilecosine.averagedprofiles.aapos.pdf",
        y_pos_fontsize=6,
        ax_row_dendro=False,
        ax_col_dendro=False,
        figsize=(30,30),
        vmin=-1,
        vmax=1
    ):
    
    # Get embedding dimensions.
    embedding_dims = [x for x in pos_embeddings.columns if x not in annotation_cols and x != label_col]

    # Perform KNN imputation if True.
    if perform_knn_imputation:
        imputer = KNNImputer(n_neighbors=3, weights='uniform')
        pos_embeddings_filled = imputer.fit_transform(pos_embeddings[embedding_dims])
    else:
        pos_embeddings_filled = pos_embeddings[embedding_dims].fillna(fillnaval).values

    # Compute similarity matrix and dendrogram.
    dist_matrix = pairwise_distances(pos_embeddings_filled, metric=metric)
    similarity_matrix = 1 - dist_matrix
    for i in range(len(dist_matrix)):
        dist_matrix[i, i] = 0
    linkage = hc.linkage(spt.distance.squareform(dist_matrix), method=dendrogram_method)
    similarity_df = pd.DataFrame(data=similarity_matrix,
                                 columns=pos_embeddings[label_col],
                                 index=pos_embeddings[label_col])
    
    # Perform clustering.
    pos_cluster = hc.cut_tree(linkage, n_clusters=n_clust_pos)
    pos_embeddings['Cluster'] = pos_cluster[:, 0]

    # Create clustermap without built-in row_colors.
    aapos_cluster_plot = sns.clustermap(
        similarity_df, 
        figsize=figsize,
        row_linkage=linkage,
        col_linkage=linkage,
        cmap=cmap_cluster,
        xticklabels=False,
        yticklabels=True,
        cbar_pos=cbar_pos,
        cbar_kws=cbar_kws,
        vmin=vmin,
        vmax=vmax,
        row_colors=None
    )
    
    # Remove the row and col (optional) dendrogram.
    aapos_cluster_plot.ax_row_dendrogram.set_visible(ax_row_dendro)
    aapos_cluster_plot.ax_col_dendrogram.set_visible(ax_col_dendro)
    
    # Adjust the ytick labels’ fontsize.
    for label in aapos_cluster_plot.ax_heatmap.get_yticklabels():
        label.set_fontsize(y_pos_fontsize)
    
    # Get the heatmap's bounding box and the figure handle.
    heatmap_pos = aapos_cluster_plot.ax_heatmap.get_position()
    fig = aapos_cluster_plot.fig

    # Define widths (in figure coordinate units) and spacing.
    spacing = 0.005       # space between annotation axes.
    cluster_width = 0.03  # width for the cluster annotation on the right.

    # The ordering of rows as drawn in the heatmap.
    ordered_ind = aapos_cluster_plot.dendrogram_row.reordered_ind
    
    # ---------------------------
    # Add extra annotations on the TOP
    # ---------------------------
    # We use the column ordering from the dendrogram.
    ordered_cols = aapos_cluster_plot.dendrogram_col.reordered_ind
    # Define height (in figure coordinates) for each top annotation and spacing.
    anno_height = 0.02   # height for each extra annotation
    top_spacing = 0.005   # space between annotation axes
    # Start at the top edge of the heatmap.
    current_top = heatmap_pos.y1
    for anno, pal in zip(annotation_cols, annotation_palettes):
        current_top = current_top + top_spacing
        # Get annotation values for each column.
        # First, set the index so that we can align by label.
        anno_series = pos_embeddings.set_index(label_col)[anno]
        # Ensure the order matches the columns in similarity_df.
        anno_color_list = [pal[val] for val in anno_series.loc[similarity_df.columns]]
        # Reorder using the dendrogram order.
        anno_color_ordered = np.array(anno_color_list)[ordered_cols]
        # Convert colors to RGB; shape becomes (1, n_cols, 3)
        anno_rgb = np.array([mcolors.to_rgb(c) for c in anno_color_ordered]).reshape(1, -1, 3)
        # Create a new axis spanning the top of the heatmap.
        ax_top = fig.add_axes([heatmap_pos.x0, current_top, heatmap_pos.width, anno_height])
        ax_top.imshow(anno_rgb, aspect='auto')
        ax_top.axis('off')
        current_top += anno_height

    # ---- Add extra annotations on the LEFT ----
    # Start positioning from the left edge of the heatmap.
    # base_left = heatmap_pos.x0
    # current_left = base_left

    # for anno, pal in zip(annotation_cols, annotation_palettes):
    #     current_left = current_left - (anno_width + spacing)
    #     anno_colors = pos_embeddings[anno].map(lambda x: pal[x]).values
    #     # Reorder colors to match the heatmap.
    #     anno_colors_ordered = anno_colors[ordered_ind]
    #     # Convert colors to RGB.
    #     anno_rgb = np.array([mcolors.to_rgb(c) for c in anno_colors_ordered]).reshape(-1, 1, 3)
    #     ax_anno = fig.add_axes([current_left, heatmap_pos.y0, anno_width, heatmap_pos.height])
    #     ax_anno.imshow(anno_rgb, aspect='auto')
    #     ax_anno.axis('off')
    
    # ---- Add cluster annotation on the RIGHT ----
    if annotate_clusters:
        # Compute the far right edge of all current axes in the figure.
        max_x = max(ax.get_position().x1 for ax in fig.axes)
        # Place cluster annotation immediately to the right of this edge.
        cluster_left = max_x + spacing + cluster_spacing
        ordered_rows = aapos_cluster_plot.dendrogram_row.reordered_ind
        cluster_colors = pos_embeddings['Cluster'].map(lambda x: cluster_palette[x]).values
        cluster_colors_ordered = np.array(cluster_colors)[ordered_rows]
        # Reorder colors to match the heatmap.
        n_rows = len(ordered_rows)
        cluster_colors_ordered = cluster_colors[ordered_ind]
        cluster_rgb = np.array([mcolors.to_rgb(c) for c in cluster_colors_ordered]).reshape(-1, 1, 3)
        ax_cluster = fig.add_axes([cluster_left, heatmap_pos.y0, cluster_width, heatmap_pos.height])
        ax_cluster.imshow(cluster_rgb, aspect='auto')
        ax_cluster.axis('off')

        # If cluster_names is provided, annotate each contiguous cluster group.
        if cluster_names is not None:
            # Get the cluster assignments in the order of the heatmap rows.
            ordered_clusters = np.array(pos_embeddings['Cluster'])[ordered_rows]
            unique_clusters = np.unique(ordered_clusters)
            # For each cluster, find the indices in the ordered list and compute a center.
            for clust in unique_clusters:
                indices = np.where(ordered_clusters == clust)[0]
                if len(indices) > 0:
                    # Compute the center of the block.
                    y_center = (indices[0] + indices[-1] + 1) / 2.0
                    # Get the name from the provided dict, or use the cluster id as fallback.
                    name = cluster_names.get(clust, str(clust)) if isinstance(cluster_names, dict) else str(clust)
                    if cluster_text_color is not None:
                        ax_cluster.text(0, y_center, name, ha='center', va='center',
                                        color=cluster_text_color[clust], fontsize=cluster_text_fontsize, weight='bold', rotation=90)
                    else:
                        ax_cluster.text(0, y_center, name, ha='center', va='center',
                                        color='white', fontsize=cluster_text_fontsize, weight='bold', rotation=90)
    
    
    # Save and close the figure.
    aapos_cluster_plot.savefig(plot_filename, dpi=600)
    plt.close()
    
    return pos_embeddings, similarity_df, linkage
    
# set file paths
def construct_phenotype_path(replicate_phenotyping_path, 
                             well, 
                             tile_x, 
                             tile_y, 
                             wellprefix='well',
                             wellsuffix='_seqgrid5_phenogrid20',
                             tileprefix='tile',
                             filename='channel1.tif'):
    output_path = \
        os.path.join(replicate_phenotyping_path, 
                     wellprefix+str(well)+wellsuffix, 
                     tileprefix+str(tile_x).zfill(2)+'x'+str(tile_y).zfill(2)+'y',
                     'cellprofiler',
                     filename)
    return output_path

# average together z-scores from two replicates between features
def prepare_consensus_variant_scores(dfs_list, 
                                     variant_col='aaChanges'):

    # get features and variants to average
    total_features = [c for c in dfs_list[0].columns if c not in [variant_col,'Metadata_Object_Count']]
    var_common = [list(df[variant_col].values) for df in dfs_list]
    var_common = list(set.intersection(*[set(list) for list in var_common]))
    zscore_array_list = []

    # perform zscoring over syn variants for each feature
    for df in dfs_list:
        if variant_col != "Variant":
            df = df.rename(columns={variant_col:'Variant'})
        df.index = df['Variant']
        df['Variant_Class'] = \
            pd.Categorical(
                df['Variant'].astype(str).apply(variant_classification),
                    categories=mutation_types, 
                    ordered=True)
        df.drop(columns='Variant',inplace=True)
        zscores = []
        syn_df = df.query('Variant_Class == "Synonymous"')
        for f in total_features:
            mean_syn = syn_df[f].mean()
            std_syn = syn_df[f].std()
            zscores.append((df.loc[var_common,f].values - mean_syn)/std_syn)
        zscore_array_list.append(np.vstack(zscores).T)

    variant_scores = pd.DataFrame(np.mean(zscore_array_list, axis=0),
                                  index=var_common,
                                  columns=total_features)\
                        .reset_index()\
                        .rename(columns={'index':'Variant'})
    return variant_scores

# make boxplots for different classes of variants
def boxplot_with_significance(
    df,
    x_col="Variant Designation",
    y_col="Morphological Impact Score",
    hue_col=None,
    order=None,
    palette=None,
    alpha=0.001,
    pairs=None,            # list of (group1, group2) pairs to compare
    shared_label="***",    # text to show if any comparison is p < alpha
    offset=0.02,
    anno_sample_sizes=True,
    xlabels=None,
    xlabel_fontsize=11,
    ax=None
    ):
    """
    Create a Seaborn boxplot, draw significance brackets that start
    just above the highest non-outlier (upper whisker) point + offset,
    place a shared label if any p < alpha, and append (n=...) under each x-label.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    # Determine order if not provided
    if order is None:
        order = sorted(df[x_col].dropna().unique())

    # 1) Draw boxplot
    sns.boxplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        order=order,
        ax=ax
    )
    # remove dup legend
    if (hue_col is not None) and (hue_col == x_col) and ax.legend_:
        ax.legend_.remove()

    # 2) Annotate sample sizes
    if anno_sample_sizes:
        counts = df[x_col].value_counts()
        if xlabels is None:
            new_labels = [f"{cat}\n(n={int(counts.get(cat,0))})" for cat in order]
            ax.set_xticklabels(new_labels, fontsize=xlabel_fontsize)
        else:
            new_labels = [f"{xlabels[j]}\n(n={int(counts.get(cat,0))})" for j,cat in enumerate(order)]
            ax.set_xticklabels(new_labels, fontsize=xlabel_fontsize)
    else:
        ax.tick_params(axis='x',labelsize=xlabel_fontsize)

    if not pairs:
        return ax

    data_max = df[y_col].max()
    data_min = df[y_col].min()
    height_so_far    = -10 # initialize
    vertical_offset  = offset * (data_max - data_min)
    bracket_thickness = 0.01 * (data_max - data_min)
    any_significant = False

    for (g1, g2) in pairs:
        data1 = df.loc[df[x_col] == g1, y_col].dropna()
        data2 = df.loc[df[x_col] == g2, y_col].dropna()
        if data1.empty or data2.empty:
            continue

        # test
        _, pval = mannwhitneyu(data1, data2, alternative='two-sided')
        if pval >= alpha:
            continue

        any_significant = True

        # --- new: compute upper whiskers for each group ---
        def upper_whisker(series):
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            cap = q3 + 1.5 * iqr
            return series[series <= cap].max()

        w1 = upper_whisker(data1)
        w2 = upper_whisker(data2)
        whisker_max = max(w1, w2)

        # start bracket just above that whisker
        bracket_y = max(height_so_far + vertical_offset,
                        whisker_max + vertical_offset)
        height_so_far = bracket_y

        # x‐coords
        x1, x2 = order.index(g1), order.index(g2)
        if x1 > x2:
            x1, x2 = x2, x1

        ax.plot(
            [x1, x1, x2, x2],
            [bracket_y,
             bracket_y + bracket_thickness,
             bracket_y + bracket_thickness,
             bracket_y],
            color='black', lw=1.5
        )

    # expand top limit
    ax.set_ylim(top=height_so_far + vertical_offset*(len(pairs)+3))

    # shared label
    if any_significant and shared_label:
        ax.text(
            0.02, 0.98, shared_label,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=14, fontweight='bold'
        )

    return ax

# make stacked bar charts with significance
def stacked_bar_with_proportion_significance(counts, 
                                             pairs, 
                                             alpha=0.05,
                                             shared_label="***",
                                             offset=0.02,
                                             ax=None,
                                             colorpalette=None,
                                             title="Proline Clusters by Domain",
                                             ylabel="Num Positions",
                                             xlabel="Domain",
                                             xtick_rotation=0,
                                             figsize=(3,3),
                                             pseudocount=1):
    """
    Create a stacked bar plot from a DataFrame 'counts' (with index as the x-axis categories
    and columns as groups to stack, e.g. clusters). For each pair of categories specified in 'pairs',
    perform a chi-square test on the full set of counts across clusters (after adding a pseudocount of 1)
    to test if the two domains have the same proportions. If the p-value is below alpha, draw a significance
    bracket between the bars. If any pair is significant, add a shared significance label on the left side.

    Parameters
    ----------
    counts : pd.DataFrame
        DataFrame with index as categories (e.g. Domain) and columns as groups (e.g. Clusters).
    pairs : list of tuples
        List of pairs of categories to compare, e.g. [('Coil 1B','Linker 12'), ('Coil 1B','Coil 2A')].
    alpha : float, optional
        Significance threshold (default is 0.05).
    shared_label : str, optional
        Label to add on the left if any pair is significant.
    offset : float, optional
        Fraction of the data range used as vertical offset for brackets.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    colorpalette : dict, optional
        Mapping from column labels to colors.
    title : str, optional
        Title for the plot.
    ylabel : str, optional
        Y-axis label.
    xlabel : str, optional
        X-axis label.
    xtick_rotation : int, optional
        Rotation angle for x-tick labels.
    figsize : tuple, optional
        Figure size if ax is None.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes with the plot.
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Preserve the order of categories (domains)
    categories = counts.index.tolist()

    # Plot the stacked bars
    bottom = pd.Series(0, index=categories)
    if colorpalette is None:
        # If no palette is provided, use default colors
        colorpalette = {col: None for col in counts.columns}

    for col in sorted(counts.columns):
        ax.bar(categories, counts[col], bottom=bottom,
               color=colorpalette.get(col, None),
               label=f"Cluster {col}")
        bottom += counts[col]

    # Basic formatting
    ax.set_title(title, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel)
    plt.xticks(rotation=xtick_rotation, fontsize=12)
    plt.tight_layout()

    # Get the total counts per category (used for bracket placement)
    totals = counts.sum(axis=1)
    data_max = totals.max()
    data_min = 0  # bars start at 0
    offset_pixels = offset * (data_max - data_min)
    bracket_thickness = 0.01 * (data_max - data_min)
    height_so_far = data_max  # start above the highest bar
    any_significant = False

    # Loop through each specified pair and perform a chi-square test with pseudocount adjustment
    for (cat1, cat2) in pairs:
        if cat1 not in categories or cat2 not in categories:
            continue

        # Construct the contingency table for the two domains and add a pseudocount if necessary
        table_df = counts.loc[[cat1, cat2]] + pseudocount
        table = table_df.values
        
        try:
            chi2, pval, dof, expected = chi2_contingency(table)
        except Exception as e:
            print(f"Error running chi-square test for {cat1} vs {cat2}: {e}")
            continue

        if pval < alpha:
            any_significant = True
            # Determine the vertical position: above both bars
            local_max = max(totals.loc[cat1], totals.loc[cat2])
            bracket_y = max(height_so_far + offset_pixels, local_max + offset_pixels)
            height_so_far = bracket_y  # update for subsequent brackets

            # Determine x-axis positions based on category order
            x1 = categories.index(cat1)
            x2 = categories.index(cat2)
            if x1 > x2:
                x1, x2 = x2, x1

            # Draw the bracket lines
            ax.plot(
                [x1, x1, x2, x2],
                [bracket_y, bracket_y + bracket_thickness,
                 bracket_y + bracket_thickness, bracket_y],
                color='black',
                lw=1.5
            )

    # Adjust y-limit to ensure all brackets are visible
    ax.set_ylim(top=height_so_far + offset_pixels)

    # Add the shared significance label if any comparison was significant
    if any_significant and shared_label:
        ax.text(
            0.02, 0.98, shared_label,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=14, color='black', fontweight='bold'
        )

    return ax

# Make heatmaps while still retaining cluster identity
def cluster_rows_within_groups(
    df, 
    group_labels, 
    method='average', 
    metric='euclidean'
):
    """
    For each group in group_labels, compute a hierarchical clustering
    of that subset of rows. Then stack these subsets back together in
    the order of group_labels' unique values.

    Parameters
    ----------
    df : pd.DataFrame
        Rows = samples, Columns = features.
    group_labels : pd.Series or array-like
        Group membership for each row (same index/length as df).
    method : str
        Linkage method, e.g. 'average', 'complete', 'ward', etc.
    metric : str
        Distance metric, e.g. 'euclidean', 'correlation', etc.

    Returns
    -------
    df_reordered : pd.DataFrame
        The DataFrame rows re-ordered so that rows are grouped by group,
        and within each group, sorted by hierarchical clustering.
    group_labels_ordered : pd.Series
        The group labels in the new row order (same index as df_reordered).
    """
    # Make sure df and group_labels share the same index
    df = df.copy()
    if not isinstance(group_labels, pd.Series):
        group_labels = pd.Series(group_labels, index=df.index, name="group")

    # Keep track of final row order
    final_index = []

    # We’ll just use the order groups appear in group_labels.unique()
    # If you want a custom group order, specify it explicitly.
    unique_groups = group_labels.unique()

    for g in unique_groups:
        # Subset rows for group g
        subset_idx = group_labels[group_labels == g].index
        subset_df = df.loc[subset_idx]

        # If only one row in this group, no actual clustering needed
        if len(subset_df) == 1:
            final_index.extend(subset_idx)
            continue

        # Compute linkage for rows within this group
        Z = hc.linkage(subset_df, method=method, metric=metric)
        # Get the leaf order from the linkage
        leaves = hc.leaves_list(Z)

        # Reindex this subset by the leaf order
        subset_ordered = subset_idx[leaves]
        final_index.extend(subset_ordered)

    # Reorder the entire df
    df_reordered = df.loc[final_index]
    group_labels_ordered = group_labels.loc[final_index]

    return df_reordered, group_labels_ordered

# Make heatmaps while still retaining channel identity
def cluster_columns_within_groups(df, col_groups, method='average', metric='correlation'):
    """
    df:          DataFrame of shape (n_samples, n_features)
                 rows = observations (cells, wells, etc.)
                 columns = features
    col_groups:  pd.Series of length == number of columns of df
                 e.g. col_groups[col_name] = 'CH0', 'CH1', 'Multi', or 'None'
    method:      linkage method, e.g. 'average', 'ward', ...
    metric:      distance metric, e.g. 'euclidean', 'correlation', ...
    """
    unique_groups = col_groups.cat.categories
    unique_groups = [g for g in unique_groups if g in col_groups.values]
    
    final_col_order = []
    
    for grp in unique_groups:
        # Subset columns for this group:
        these_cols = col_groups.index[col_groups == grp]
        df_sub = df[these_cols]  # shape: (n_samples, subset_of_columns)
        
        # We cluster columns, so we do linkage on df_sub.T
        # (df_sub.T => shape: (subset_of_columns, n_samples))
        linkage_result = hc.linkage(df_sub.T, method=method, metric=metric)
        col_leaves = hc.leaves_list(linkage_result)
        
        # Reorder columns according to the hierarchical clustering leaves
        these_cols_ordered = df_sub.columns[col_leaves]
        
        # Append to final order
        final_col_order.extend(these_cols_ordered)
    
    # Return the reordered DataFrame and the corresponding group labels
    df_reordered = df[final_col_order]
    col_groups_reordered = col_groups[final_col_order]
    
    return df_reordered, col_groups_reordered

# Add gaps to split heatmaps
def add_gaps_between_groups(df, group_series):
    """
    Insert a column of all NaN between each distinct group to create a blank visual spacer.
    """
    final_cols_with_spacers = []
    unique_groups = group_series.unique()
    
    for i, grp in enumerate(unique_groups):
        these_cols = group_series.index[group_series == grp]
        final_cols_with_spacers.extend(these_cols)
        # Add a spacer column except after the last group
        if i < len(unique_groups) - 1:
            spacer_col = f"SPACER_{grp}"
            df[spacer_col] = np.nan
            final_cols_with_spacers.append(spacer_col)
    
    return df[final_cols_with_spacers]

# perform louvain clustering
def cluster_louvain(df_pca, 
                    k=50, 
                    resolution=1.0,
                    metric='euclidean', 
                    weighted=True):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric=metric).fit(df_pca)
    # This returns the (k) nearest neighbors and distances for each sample
    distances, indices = nbrs.kneighbors(df_pca)
    
    # Create a sparse weight (adjacency) matrix in CSR format
    # We'll store 1 / distance as a weight if you want a similarity,
    # or simply store 1 to indicate an edge (unweighted).
    # We'll also do a small trick to avoid division by zero (neighbors=0).
    n_samples = df_pca.shape[0]
    row_list = []
    col_list = []
    weight_list = []
    
    for i in range(n_samples):
        for j_idx, dist in zip(indices[i, :], distances[i, :]):
            row_list.append(i)
            col_list.append(j_idx)
            # Example: define weights as 1 / distance (or just 1 if unweighted)
            if dist == 0:
                weight_list.append(1.0)
            else:
                if weighted:
                    weight_list.append(1.0 / dist)
                else:
                    weight_list.append(1.0)
    
    # Build a symmetric adjacency matrix
    W = sp.coo_matrix((weight_list, (row_list, col_list)), 
                      shape=(n_samples, n_samples))
    
    # Make it symmetric by taking the max or average of W and W.T
    # (Or just do an OR if unweighted)
    W = 0.5 * (W + W.T)
    
    # Now W is your adjacency (sparse) matrix in COO format, 
    # you could convert to CSR with W.tocsr().
    W = W.tocsr()
    
    # Convert the sparse adjacency matrix into a NetworkX graph
    G = nx.from_scipy_sparse_array(W)
    
    # Now run Louvain
    # You can pass a 'resolution' argument here to control granularity
    # (default is 1.0; >1 leads to more clusters, <1 leads to fewer).
    partition = community_louvain.best_partition(G, resolution=resolution)
    
    # This returns a dictionary of node -> cluster_id
    labels = np.array([partition[i] for i in range(n_samples)])
    nclust = np.sort(np.unique(labels))[-1] + 1
    return labels,nclust

# perform differential feature testing by cluster
def differential_testing_bycluster(df_profiles,
                                   features,
                                   cluster_col="Louvain Cluster"
                                  ):

    # Perform Mann-Whitney U-testing on features per cluster
    nclust = np.sort(np.unique(df_profiles[cluster_col]))[-1]+1
    rank_sum_bycluster = []
    for i in range(nclust):
        feature_test = {}
        for f in features:
            clust_feat_values = df_profiles[df_profiles[cluster_col]==i][f].values
            clust_feat_values = clust_feat_values[~np.isnan(clust_feat_values)]
            nonclust_feat_values = df_profiles[df_profiles[cluster_col]!=i][f].values
            nonclust_feat_values = nonclust_feat_values[~np.isnan(nonclust_feat_values)]
            u,p = mannwhitneyu(clust_feat_values,
                                  nonclust_feat_values)
            if p<0:
                p=0
            elif p>1:
                p=1
            median_clust = np.median(clust_feat_values)
            mad_clust =  median_abs_deviation(clust_feat_values)
            median_nonclust = np.median(nonclust_feat_values)
            mad_nonclust = median_abs_deviation(nonclust_feat_values)
            median_tot = np.median(np.concatenate((clust_feat_values, nonclust_feat_values)))
            mad_tot = median_abs_deviation(np.concatenate((clust_feat_values, nonclust_feat_values)))
            median_syn = np.median(df_profiles\
                                      .query('Variant_Class == "Synonymous"')[f].values)
            mad_syn = median_abs_deviation(df_profiles\
                                                  .query('Variant_Class == "Synonymous"')[f].values)
            feature_test[f] = (u,p,
                               median_clust,mad_clust,
                               median_nonclust,mad_nonclust,
                               median_tot,mad_tot,
                               median_syn,mad_syn)
        rank_sum_bycluster.append(feature_test)

    # Make df
    rank_sum_bycluster_df = pd.concat(
        [
            pd.DataFrame.from_dict(rank_sum_bycluster[i], 
                                   orient='index', 
                                   columns=['Mann_Whitney_U','Mann_Whitney_p',
                                           'Median_Clust','MAD_Clust',
                                           'Median_NonClust','MAD_NonClust',
                                           'Median_Tot', 'MAD_Tot',
                                           'Median_Syn', 'MAD_Syn']
                                  ).assign(Cluster=i)
                for i in range(nclust)
        ], axis=0)
    rank_sum_bycluster_df = \
        rank_sum_bycluster_df\
            .reset_index()\
            .rename(columns={'index': 'Feature'})
    rank_sum_bycluster_df['Robust_Z'] = \
        (rank_sum_bycluster_df['Median_Clust'] - rank_sum_bycluster_df['Median_Tot'])\
            / rank_sum_bycluster_df['MAD_Tot']
    rank_sum_bycluster_df['Robust_Z_Syn'] = \
        (rank_sum_bycluster_df['Median_Clust'] - rank_sum_bycluster_df['Median_Syn'])\
            / rank_sum_bycluster_df['MAD_Syn']

    return rank_sum_bycluster_df

# get distances between chain C and other chains in PDB objects
def get_nearest_distances(pdb_model, ref_chain='C'):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_model)
    
    # Identify ref chain and all other chains.
    chain_c = None
    other_chains = []
    for chain in structure.get_chains():
        if chain.id == ref_chain:
            chain_c = chain
        else:
            other_chains.append(chain)
    if chain_c is None:
        raise ValueError("Chain " + ref_chain + " not found in structure")
    
    # Helper function: return the beta-carbon atom (CB), or CA for glycine.
    def get_beta_atom(residue):
        if "CB" in residue:
            return residue["CB"]
        elif residue.get_resname() == "GLY" and "CA" in residue:
            return residue["CA"]
        else:
            return None

    # Get list of (residue, beta-atom) for chain C.
    chain_c_atoms = []
    for residue in chain_c.get_residues():
        beta_atom = get_beta_atom(residue)
        if beta_atom is not None:
            chain_c_atoms.append((residue, beta_atom))
    
    # Build a result list where each entry corresponds to one residue in chain C.
    # For each Chain C residue, we store its nearest neighbor in each other chain.
    results = []
    for res_c, atom_c in chain_c_atoms:
        # This dict will map each other chain's id to a dict with the neighbor info.
        neighbors = {}
        for other_chain in other_chains:
            # Build a list of (residue, beta-atom) for the other chain.
            other_atoms = []
            for residue in other_chain.get_residues():
                beta_atom = get_beta_atom(residue)
                if beta_atom is not None:
                    other_atoms.append((residue, beta_atom))
            
            # Find the nearest beta-carbon in this other chain.
            min_distance = float("inf")
            nearest_res = None
            nearest_atom = None
            for res_other, atom_other in other_atoms:
                distance = atom_c - atom_other
                if distance < min_distance:
                    min_distance = distance
                    nearest_res = res_other
                    nearest_atom = atom_other
            
            if nearest_res is not None:
                neighbors[other_chain.id] = {
                    "residue": nearest_res,
                    "atom": nearest_atom,
                    "distance": min_distance
                }
        results.append({
            "chain_ref_residue": res_c,
            "chain_ref_atom": atom_c,
            "neighbors": neighbors
        })
    
    return results

# convert to df
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}
def get_nearest_distance_dataframe(pdb_model, ref_chain='C'):
    """
    Runs get_distances() on the given PDB model file and converts the result into
    a pandas DataFrame with columns:
    
    - 'Ref_residue'
    - For each other chain X:
      - 'Nearest_residue_chainX'
      - 'Dist_residue_chainX'
    """
    distances = get_nearest_distances(pdb_model, ref_chain=ref_chain)
    
    # Determine the set of other chain IDs from the neighbors dictionary.
    chain_ids = set()
    for entry in distances:
        chain_ids.update(entry["neighbors"].keys())
    chain_ids = sorted(chain_ids)  # Sort for consistent column ordering.
    
    rows = []
    for entry in distances:
        row = {}
        # Format the chain C (reference) residue as "RESNAME RESSEQ"
        res_c = entry["chain_ref_residue"]
        one_letter = THREE_TO_ONE.get(res_c.resname, "X")
        row["Ref_residue"] = f"{one_letter}{res_c.id[1]}"
        
        # Add nearest neighbor info for each other chain.
        for chain_id in chain_ids:
            neighbor = entry["neighbors"].get(chain_id)
            if neighbor is not None:
                neighbor_res = neighbor["residue"]
                one_letter_neighbor = THREE_TO_ONE.get(neighbor_res.resname, "X")
                row[f"Nearest_residue_chain{chain_id}"] = f"{one_letter_neighbor}{neighbor_res.id[1]}"
                row[f"Dist_residue_chain{chain_id}"] = neighbor["distance"]
            else:
                row[f"Nearest_residue_chain{chain_id}"] = None
                row[f"Dist_residue_chain{chain_id}"] = None
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

# Plot ROC AUC vs number of training examples
def graph_auc_examples(
    score_df,
    img_save_path   = None,
    experiment_name = None,
    xlim            = None,
    ylim            = None,
    graph_column    = "AUC ROC",
    count_column    = "Example Count",
    x_label         = "Number of Training Cells",
    y_label         = "AUROC",
    colorby_column  = "Variant_Class",
    legend_title    = "Variant Type",
    colorby_palette = variant_type_palette,
    show_legend     = False,
    log_axis        = False,
    rx              = 0.6,
    ry              = 0.9,
    figsize         = (5,4)
):
    data_df = score_df.copy()
    data_df = data_df.dropna()
    title = "Num Training Examples vs. ROC AUC"
    if experiment_name is not None:
        title = f"{title}: {experiment_name}"

    title += f" (n = {len(data_df)})"
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=data_df,
        x=count_column,
        y=graph_column,
        hue=colorby_column,
        palette=colorby_palette,
        ax=ax
    )
    if log_axis:
        ax.set_xscale('log')
        # Compute linear regression in log space
        x = np.log10(data_df[count_column])
        y = data_df[graph_column]
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        # Prepare line values for the best-fit line
        line_x_log = np.linspace(x.min(), x.max(), 100)
        line_x = np.power(10, line_x_log)
        line_y = slope * line_x_log + intercept
    else:
        # Compute linear regression
        x = data_df[count_column]
        y = data_df[graph_column]
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        # Prepare line values for the best-fit line
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = slope * line_x + intercept
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.plot(line_x, line_y, color='red')
    ax.set_xlabel(x_label, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylabel(y_label, fontsize=14)
    if show_legend:
        ax.legend(loc='upper left', 
                  markerscale=2, 
                  fontsize=12, 
                  title_fontsize=14, 
                  title=legend_title,
                  bbox_to_anchor=(1,1))
    else:
        ax.legend().set_visible(False)
    
    # Display R^2 on the plot
    ax.text(
        rx, ry, 
        f'$R = {r_value:.2f}$', 
        transform=ax.transAxes, 
        fontsize=20, 
        bbox=dict(facecolor='white', alpha=0.5)
    )
    plt.tight_layout()
    if img_save_path is not None:
        fig.savefig(img_save_path)

# Plot spheres on structure counting variants
def plot_structure_spheres(counts_df,
                           output_suffix='PTEN_nuclearMisloc_counts',
                           output_dir="./consensus_plots/pdb/",
                           input_pdb='/net/fowler/vol1/home/pendyala/FISSEQ/PTEN_iPSC_T3_pycytominer_120524/1d5r.pdb',
                           pymol_view=pten_pymol_view,
                           start_pos=None,
                           end_pos=None,
                           pymol_exec="$HOME/pymol/pymol -c",
                           extra_pymol="color green, resi 1352\n",
                           image_size=(1800,3000),
                           dpi=1000,
                           base_radius=0.3,
                           scale_factor=0.1):
    """
    counts_df : pandas.DataFrame with columns ['position','count']
    for each residue position, how many variants you saw.
    
    This writes:
      * a .pml script that loads `input_pdb` as object "pten"
      * for each pos: pseudoatom sphere_{pos} with vdw=base_radius + scale_factor*count
      * labels each sphere with its count
      * shows the cartoon, hides solvent, colors pten grey, spheres red
      * does a ray trace and writes a PNG
    Returns the shell command to run PyMOL.
    """
    # 1) build a dict of counts
    count_dict = dict(zip(counts_df['position'], counts_df['count']))

    # 2) parse the PDB and grab CA coords for any pos in count_dict
    parser = PDB.PDBParser(QUIET=True)
    struct = parser.get_structure("PTEN", input_pdb)
    coords = {}
    for model in struct:
        for chain in model:
            for res in chain:
                pos = res.get_id()[1]
                if pos in count_dict and 'CA' in res:
                    coords[pos] = res['CA'].get_coord()

    # 3) prepare output paths
    base = os.path.splitext(os.path.basename(input_pdb))[0]
    prefix = os.path.join(output_dir, f"{base}_{output_suffix}")
    pml_path = prefix + ".pml"
    print(base)

    # 4) write the PML
    with open(pml_path, "w") as f:
        f.write(f"load {input_pdb}, {base}\n")
        f.write(pymol_view + "\n")
        #f.write("hide everything, all\n")
        f.write(f"show cartoon, {base}\n")
        f.write(f"color grey80, {base}\n")
        if start_pos and end_pos:
            f.write(f"select region, resi {start_pos}-{end_pos}\n")
            f.write("remove solvent\n")
        # sphere creation
        for pos, cnt in count_dict.items():
            if pos not in coords:
                continue
            x, y, z = coords[pos]
            radius = base_radius + scale_factor * cnt
            name = f"sphere_{pos}"
            # create the pseudoatom
            f.write(
                f"pseudoatom {name}, pos=[{x:.3f},{y:.3f},{z:.3f}], "
                f"vdw={radius:.3f}\n"
            )
            # color & show
            f.write(f"color red, {name}\n")
            f.write(f"show spheres, {name}\n")
            f.write(f"set sphere_transparency, 0.3, {name}\n")
        # any extra PyMOL commands
        if extra_pymol:
            f.write(extra_pymol)
        # render
        f.write("ray\n")
        f.write(f"png {prefix}.png, {image_size[0]}, {image_size[1]}, dpi={dpi}\n")

    # 5) return the shell command
    return f"{pymol_exec} {input_pdb} {pml_path}"

        