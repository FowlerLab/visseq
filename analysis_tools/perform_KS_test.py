import os
import sys
import pandas as pd
import numpy as np
import pickle
import scipy.stats as ss

def calculate_stats(feature_list, df_variants_grouped, df_WT, p_value):
    variant_list = [v for v,unused_df in df_variants_grouped]
    df_p_values_feature = pd.DataFrame(index=variant_list,columns=feature_list)
    df_KS_values_feature = pd.DataFrame(index=variant_list,columns=feature_list)
    df_EMD_values_feature = pd.DataFrame(index=variant_list,columns=feature_list)
    df_loc_values_feature = pd.DataFrame(index=variant_list,columns=feature_list)
    
    variant_counter = 0
    
    for feat in feature_list:
        print('Processing: ' + feat)
        list_p = []
        list_u = []
        list_l = []
        list_e = []
        wt_feat_finite = df_WT[feat]
        wt_feat_finite = wt_feat_finite[np.isfinite(wt_feat_finite)].values
        wt_feat_finite = np.sort(wt_feat_finite)
        for variant,df_group in df_variants_grouped: 
            variant_counter += 1
            grp_feat_finite = df_group[feat]
            grp_feat_finite = grp_feat_finite[np.isfinite(grp_feat_finite)]
            if ((len(wt_feat_finite) == 0) | (len(grp_feat_finite) == 0)):
                list_p.append(1)
                list_u.append(0)
                list_l.append(0)
                list_e.append(0)
            else:
                res = ss.kstest(rvs=grp_feat_finite,
                                cdf=wt_feat_finite)
                #print(res)
                emd = ss.wasserstein_distance(grp_feat_finite,
                                              wt_feat_finite)
                #print(emd)
                list_p.append(res.pvalue)
                list_u.append(res.statistic*res.statistic_sign)
                list_l.append(res.statistic_location)
                list_e.append(emd)
        
        df_p_values_feature[feat] = list_p
        df_KS_values_feature[feat] = list_u
        df_loc_values_feature[feat] = list_l
        df_EMD_values_feature[feat] = list_e
    
    df_KS_values_feature = df_KS_values_feature.apply(pd.to_numeric)
    df_p_values_feature = df_p_values_feature.apply(pd.to_numeric)
    df_EMD_values_feature = df_EMD_values_feature.apply(pd.to_numeric)
    df_loc_values_feature = df_loc_values_feature.apply(pd.to_numeric)
    
    df_p_values_feature.loc['sig_gene_count'] = 0
    
    for i in range(len(df_p_values_feature.columns)):
        count = 0
        for j in range(len(df_p_values_feature.index)-1):
            if df_p_values_feature.iloc[j,i] <= p_value:
                count +=1
        df_p_values_feature.iloc[len(df_p_values_feature.index)-1,i] = count

    return df_KS_values_feature, df_p_values_feature, df_loc_values_feature, df_EMD_values_feature


# Get files from command line
variant_pklfile = sys.argv[1]
WT_csvfile = sys.argv[2]
filenameprefix = (os.path.basename(variant_pklfile)).split('.')[0]
filenamesuffix = os.path.basename(WT_csvfile).split('.')[-2:]
filedir = os.path.dirname(variant_pklfile)

p_value_set = 0.001

# import data
WT_df = pd.read_csv(WT_csvfile)
with open(variant_pklfile, 'rb') as f:
    grouped_variants = pickle.load(f)
f.close()

# set feature columns
feature_list_totest = \
    [c for c in WT_df.columns if c not in ['Unnamed: 0','aaChanges']]

# perform KS-test
df_KS, df_p, df_loc, df_EMD = calculate_stats(feature_list_totest, grouped_variants, WT_df, p_value_set)

# save test output 
filename_out_KS = '.'.join(([filenameprefix,'KS'] + filenamesuffix))
filename_out_loc = '.'.join(([filenameprefix,'loc'] + filenamesuffix))
filename_out_EMD = '.'.join(([filenameprefix,'EMD'] + filenamesuffix))
filename_out_p = '.'.join(([filenameprefix,'p'] + filenamesuffix))
df_KS.to_csv(os.path.join(filedir,'results',filename_out_KS))
df_loc.to_csv(os.path.join(filedir,'results',filename_out_loc))
df_EMD.to_csv(os.path.join(filedir,'results',filename_out_EMD))
df_p.to_csv(os.path.join(filedir,'results',filename_out_p))
