import os
import sys
import pickle

sys.path.append(os.path.abspath("../analysis_tools/"))
from utils import * 

# Get files from command line
cluster_featurefile = sys.argv[1]
filenameprefix = (os.path.basename(cluster_featurefile)).split('.')[0]
filenamesuffix = os.path.basename(cluster_featurefile).split('.')[-2:]
filedir = os.path.dirname(cluster_featurefile)

# import merged_profiles
df_profiles = \
    pd.read_csv(cluster_featurefile)
df_profiles.drop(columns='Unnamed: 0',inplace=True)

# Set features
features_fromdf = \
    [f for f in df_profiles.columns \
         if f not in ['Variant', 'Louvain Cluster','Variant_Class']]

# Get differential features
differential_features_bycluster_df = \
    differential_testing_bycluster(df_profiles,
                                   features_fromdf,
                                   cluster_col="Louvain Cluster")

# Save
filename_out = '.'.join(([filenameprefix,'MW'] + filenamesuffix))
differential_features_bycluster_df.to_csv(os.path.join(filedir,'results',filename_out))