#!/usr/bin/env python3
import argparse, os
import pandas as pd
import numpy as np
from pathlib import Path
from pycytominer import aggregate
import pickle
import fastparquet

def load_list_file(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith('#')]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment',            required=True            )
    p.add_argument('--cell-features',         required=True            )
    p.add_argument('--output-dir',            required=True            )
    p.add_argument('--metadata-columns-file'                           )
    p.add_argument('--blacklist-grep-file'                             )
    p.add_argument('--bc-threshold',    type=int, default=10           )
    p.add_argument('--variant-bc-threshold', type=int, default=4       )
    p.add_argument('--barcode-name', type=str, default='virtualBarcode')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load cells
    if Path(args.cell_features).suffix == '.parquet':
        full_df = pd.read_parquet(args.cell_features)
    else:
        full_df = pd.read_csv(args.cell_features)

    # metadata columns
    if args.metadata_columns_file:
        metadata_columns = load_list_file(args.metadata_columns_file)
    else:
        metadata_columns = [
            'Unnamed: 0','Unnamed: 0.1','file_index','file_path','tile_index',
            'tile_x','tile_y','xpos','ypos','bbox_x1','bbox_y1','bbox_x2','bbox_y2',
            'count_0','barcode_0','quality_0','count_1','barcode_1','quality_1',
            'count_2','barcode_2','quality_2','count_3','barcode_3','quality_3',
            'count_4','barcode_4','quality_4','count_5','barcode_5','quality_5',
            'count_6','barcode_6','quality_6','count_7','barcode_7','quality_7',
            'count_8','barcode_8','quality_8','count_9','barcode_9','quality_9',
            'count_10','barcode_10','quality_10','count_11','barcode_11','quality_11',
            'count_12','barcode_12','quality_12','count_13','barcode_13','quality_13',
            'count_14','barcode_14','quality_14','count_15','barcode_15','quality_15',
            'count','virtualBarcode','upBarcode','size','collision','upTagCollision',
            'geno','hgvsc','hgvsp','reads','codonChanges','codonHGVS','aaChanges',
            'aaChangeHGVS','offTarget','variantType','editDistance','well'
        ]

    feature_columns = [c for c in full_df.columns if c not in metadata_columns]

    # blacklist patterns
    if args.blacklist_grep_file:
        patterns = load_list_file(args.blacklist_grep_file)
    else:
        patterns = [
            'ImageNumber','ObjectNumber','BoundingBox','Center',
            'AngularSecondMoment','_X','_Y','_Z','Orientation','.1'
        ]
    feature_blacklist = [c for c in feature_columns if any(p in c for p in patterns)]
    non_blacklisted_features = [f for f in feature_columns if f not in feature_blacklist]

    # write blocklist
    blocklist_path = os.path.join(args.output_dir, 'feature_blocklist.txt')
    with open(blocklist_path, 'w') as f:
        f.write('blocklist\n')
        for feat in feature_blacklist:
            f.write(feat + '\n')
    print(f"Blocklist: {blocklist_path}")

    # write nonblocked features
    nonblocked_path = os.path.join(args.output_dir, 'nonblocked_features.txt')
    with open(nonblocked_path, 'w') as f:
        for feat in non_blacklisted_features:
            f.write(feat + '\n')
    print(f"Non-blocked features: {nonblocked_path}")

    # count & filter
    full_df['FullData_ObjectNumber'] = full_df.index
    by_bc = full_df.groupby(args.barcode_name).agg(
        count=('FullData_ObjectNumber','count'),
        aaChanges=('aaChanges','first'),
        variantType=('variantType','first')
    )
    bc_ok = by_bc.query('count >= @args.bc_threshold').index
    bpv = by_bc.loc[bc_ok].reset_index().groupby('aaChanges').agg(
        barcode_num=(args.barcode_name,'count'),
        total_cells=('count','sum'),
        variantType=('variantType','first')
    )
    variants_ok = bpv.query('barcode_num >= @args.variant_bc_threshold').index

    df_called = full_df.query('editDistance in [0,1]')
    df_filt   = df_called[df_called[args.barcode_name].isin(bc_ok) & df_called['aaChanges'].isin(variants_ok)]

    # save population for metrics
    pop_file = os.path.join(args.output_dir, f"{args.experiment}_population.parquet")
    df_filt.to_parquet(pop_file)
    print(f"Saved population: {pop_file}")

    # aggregate median + counts
    out_parquet = os.path.join(args.output_dir, f"{args.experiment}_aggregated.parquet")
    aggregate(
        population_df=df_filt,
        strata='aaChanges',
        features=feature_columns,
        operation='median',
        output_file=out_parquet,
        output_type='parquet',
        compute_object_count=True,
        object_feature='FullData_ObjectNumber'
    )
    print(f"Wrote: {out_parquet}")

    # split features into 100 pieces for KS testing
    non_blocked = [f for f in feature_columns if f not in feature_blacklist]
    features_to_test = [f for f in non_blocked]
    nparts = int(np.ceil(len(features_to_test) / 100))
    ks_dir = os.path.join(args.output_dir, "KS_EMD")
    if not os.path.exists(ks_dir):
        os.mkdir(ks_dir)
    for i in range(nparts):
        piece = features_to_test[i*100:(i+1)*100]
        df_piece = df_filt[['aaChanges'] + piece]
        grouped = df_piece.groupby('aaChanges')
        pkl_path = os.path.join(ks_dir, f"{args.experiment}.groupedvariants.piece{i}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(grouped, f)
        wt_csv = os.path.join(ks_dir, f"{args.experiment}.WT.piece{i}.csv")
        df_piece.query('aaChanges == "WT"').to_csv(wt_csv, index=False)
        print(f"Wrote KS piece {i}: {pkl_path}, {wt_csv}")
 
if __name__ == '__main__':
    main()
