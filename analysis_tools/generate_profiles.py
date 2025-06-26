#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from pathlib import Path
from pycytominer import normalize, feature_select
import fastparquet
from utils import *

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment',          required=True)
    p.add_argument('--aggregated-median',   required=True)
    p.add_argument('--ks-dir',              required=True)
    p.add_argument('--metrics-dir',         required=True)
    p.add_argument('--blocklist-file',      required=True)
    p.add_argument('--nonblocked-file',     required=True)
    p.add_argument('--output-dir',          required=True)
    p.add_argument('--normalize-to', choices=['all', 'synonymous'], default='all')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    exp = args.experiment
    ks_dir = Path(args.ks_dir)
    metrics = Path(args.metrics_dir)
    os.makedirs(metrics, exist_ok=True)

    # 1) load median aggregated
    df_med = pd.read_parquet(args.aggregated_median)
    if 'aaChanges' in df_med.columns:
        df_med.rename(columns={'aaChanges':'Variant'}, inplace=True)

    # 2) merge piecewise outputs and load EMDs
    pattern_types = ['KS', 'p', 'loc', 'EMD']
    for t in pattern_types:
        files = sorted(glob(f"{args.ks_dir}/results/{exp}.{t}.*.csv"))
        dfs = [pd.read_csv(f, index_col=0) for f in files]
        df_all = pd.concat(dfs, axis=1)
        out_file = os.path.join(metrics, f"{exp}.{t}_all.csv")
        df_all.to_csv(out_file)
    emd_csv = metrics / f"{exp}.EMD_all.csv"
    df_emd = pd.read_csv(emd_csv, index_col=None).rename(columns={"Unnamed: 0":"Variant"})
    df_emd.index = df_emd['Variant']
    df_emd = df_emd.drop('Variant', axis=1)

    # 3) remove non-reproducible EMD
    # Perform WT bootstrapping and remove features with poor reproducibility
    df_WT_list = sorted(glob(f"{args.ks_dir}/{exp}.WT.*.csv"))
    dfs_WT = [pd.read_csv(f, index_col=0) for f in df_WT_list]
    df_WT = pd.concat(dfs_WT, axis=1)
    
    # compute WT median and MAD for features
    non_blacklisted_features = []
    with open(args.nonblocked_file, 'r') as f:
        for line in f:
            non_blacklisted_features.append(line.strip()) 
    feat_wt_median = df_WT[non_blacklisted_features].median()
    feat_wt_MAD = pd.Series(median_abs_deviation(df_WT[non_blacklisted_features],axis=0), index=non_blacklisted_features)

    # Remove infinite features
    df_emd_rescaled = \
        df_emd / feat_wt_MAD
    inf_feat_emd = df_emd_rescaled.columns\
                        [df_emd_rescaled.isin([np.inf, -np.inf, np.nan]).any()]
    df_emd_rescaled_filt = \
        df_emd_rescaled.drop(columns=inf_feat_emd)
    removed_inf_emd_feat = df_emd_rescaled_filt.columns

    # Run bootstraps
    n_bootstraps = 25
    df_emd_rep_list = []
    for rep in range(n_bootstraps):
        # Shuffle the DataFrame
        df_WT_shuffled = df_WT.loc[:,removed_inf_emd_feat].sample(frac=1)
        df_WT_shuffled_list = np.array_split(df_WT_shuffled, 2)
        list_e = []
        for feat in removed_inf_emd_feat:
            list_e.append(wasserstein_distance(df_WT_shuffled_list[0][feat], df_WT_shuffled_list[1][feat]))
        df_emd_rep_list.append(pd.Series(list_e, index=removed_inf_emd_feat))

    # Concat and save
    df_emd_rep = pd.concat(df_emd_rep_list, axis=1)
    df_emd_rep = \
        df_emd_rep\
            .rename(columns={i:('Replicate'+str(i+1)) for i in range(n_bootstraps)})\
            .T
    df_emd_rep = df_emd_rep / feat_wt_MAD[removed_inf_emd_feat]
    wt_rep_file = ks_dir/"results"/f"{exp}_WTreplicates.csv"
    df_emd_rep.to_csv(wt_rep_file)

    # Set threshold
    df_emd_rep_avg = df_emd_rep.mean(axis=0)
    df_emd_rep_avg_sorted = df_emd_rep_avg.sort_values(ascending=True)
    emd_score_WT_Q3 = np.percentile(df_emd_rep_avg_sorted,75)
    emd_score_WT_Q1 = np.percentile(df_emd_rep_avg_sorted,25)
    emd_score_WT_IQR = emd_score_WT_Q3 - emd_score_WT_Q1
    emd_score_WT_thresh = 1.5*emd_score_WT_IQR + emd_score_WT_Q1 # 1.5 x IQR plus Q1
    features_emd = list(df_emd_rep_avg.index[df_emd_rep_avg <= emd_score_WT_thresh])
    reproducible_EMD_path = os.path.join(args.output_dir, 'reproducible_EMD.txt')
    with open(reproducible_EMD_path, 'w') as f:
        for feat in features_emd:
            f.write(feat + '\n')
    print(f"Reproducible features by EMD: {reproducible_EMD_path}")

    # 4) merge median + EMD
    profiles = (
        df_med
        .merge(df_emd[features_emd].rename(columns={c: f"{c}_EMD" for c in features_emd}).reset_index(), # add EMD suffix
               on='Variant', how='inner')
    )
    profiles['Variant_Class'] = \
        pd.Categorical(
            profiles['Variant'].astype(str).apply(variant_classification),
            categories=mutation_types,
            ordered=True
        )

    # 5) normalize
    meta_cols = ['Variant','Variant_Class','Metadata_Object_Count']
    feat_med = [c for c in df_med.columns if c not in meta_cols]
    feat_emd = [c for c in profiles.columns if c.endswith('_EMD')]
    all_feats = feat_med + feat_emd
    
    if args.normalize_to == 'synonymous':
        sample_filter = "Variant_Class == 'Synonymous'"
    else:
        sample_filter = 'all'

    norm_file = Path(args.output_dir) / f"{exp}_normalized.parquet"
    normalize(
        profiles=profiles,
        features=all_feats,
        image_features=False,
        meta_features=meta_cols,
        method="standardize",
        samples=sample_filter,
        output_type='parquet',
        output_file=norm_file
    )
    print(f"Normalized features written to {norm_file}")

    # 6) feature selection
    sel_file = Path(args.output_dir) / f"{exp}_selected.parquet"
    feature_select(
        profiles=norm_file,
        features=all_feats,
        image_features=False,
        samples="all",
        operation=['variance_threshold','blocklist','correlation_threshold'],
        blocklist_file=args.blocklist_file,
        output_type='parquet',
        output_file=sel_file
    )
    print(f"Selected profiles written to {sel_file}")

if __name__ == '__main__':
    main()
