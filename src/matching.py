import datetime
from numpy import int64
import py_entitymatching as em
import pandas as pd
import os, sys

#READ LABELED DATA
A = em.read_csv_metadata('../newdata/tableA_downsampled.csv',key='id')
B = em.read_csv_metadata('../newdata/tableB_downsampled.csv',key='id')
G = em.read_csv_metadata('../sample/labeled.csv',
                         key='_id',
                         ltable=A,
                         rtable=B,
                         fk_ltable='ltable_id',
                         fk_rtable='rtable_id') #DO THIS MANUALLY
# print(G.head(len(G)))

# GENERATE FEATURES
blocking_features = em.get_features_for_blocking(A,B,True)
matching_features = em.get_features_for_matching(A,B,True)
# print(matching_features.head(10))


# CONVERT THE LABELED DATA TO FEATURE VECTORS USING THE FEATURE TABLE
attrs_from_table = ['ltable_Song_Name', 'ltable_Artist_Name', 'ltable_Released',
                    'rtable_Song_Name', 'rtable_Artist_Name', 'rtable_Released']

H = em.extract_feature_vecs(G,
                            feature_table=matching_features,
                            attrs_before = attrs_from_table,
                            attrs_after='label',
                            show_progress=True)

print(H)