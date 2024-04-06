import datetime
from numpy import int64
import py_entitymatching as em
import pandas as pd
import os, sys

#READ LABELED DATA
A = em.read_csv_metadata('../newdata/tableA_downsampled.csv',key='id')
B = em.read_csv_metadata('../newdata/tableB_downsampled.csv',key='id')
C = em.read_csv_metadata('../sample/candidate.csv',key='_id')
G = em.read_csv_metadata('../sample/labeled.csv',
                         key='_id',
                         ltable=A,
                         rtable=B,
                         fk_ltable='ltable_id',
                         fk_rtable='rtable_id') #DO THIS MANUALLY
# print(G.head(len(G)))

# GENERATE FEATURES
feature_table = em.get_features_for_matching(A,B,validate_inferred_attr_types=False)
# print(matching_features.head(10))


# CONVERT THE LABELED DATA TO FEATURE VECTORS USING THE FEATURE TABLE
attrs_from_table = ['ltable_Song_Name', 'ltable_Artist_Name', 'ltable_Released',
                    'rtable_Song_Name', 'rtable_Artist_Name', 'rtable_Released']


# Convert the I into a set of feature vectors using F
H = em.extract_feature_vecs(G,
                            feature_table=feature_table,
                            attrs_before = attrs_from_table,
                            attrs_after='label',
                            show_progress=True)
# print(H)


rf = em.RFMatcher()

# Get the attributes to be projected while training
attrs_to_be_excluded = []
attrs_to_be_excluded.extend(['_id', 'ltable_id', 'rtable_id', 'label'])
attrs_to_be_excluded.extend(attrs_from_table)

# Train using feature vectors from the labeled data.
rf.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='label')

# Convert the cancidate set to feature vectors using the feature table
L = em.extract_feature_vecs(C, feature_table=feature_table,
                             attrs_before= attrs_from_table,
                             show_progress=True, n_jobs=-1)

# # Predict the matches
# predictions = rf.predict(table=L, exclude_attrs=attrs_to_be_excluded,                          
#               append=True, target_attr='predicted', inplace=False)

# print(predictions.head())
