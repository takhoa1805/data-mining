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
matching_features = em.get_features_for_matching(A,B,validate_inferred_attr_types=False)
# print(matching_features.head(10))


# CONVERT THE LABELED DATA TO FEATURE VECTORS USING THE FEATURE TABLE
attrs_from_table = ['ltable_Song_Name', 'ltable_Artist_Name', 'ltable_Released',
                    'rtable_Song_Name', 'rtable_Artist_Name', 'rtable_Released']

# Split S into development set (I) and evaluation set (J)
IJ = em.split_train_test(G, train_proportion=0.5, random_state=0)
I = IJ['train']
J = IJ['test']

# Convert the I into a set of feature vectors using F
H = em.extract_feature_vecs(I,
                            feature_table=matching_features,
                            attrs_before = attrs_from_table,
                            attrs_after='label',
                            show_progress=True)
# print(H)

# Create a set of ML-matchers
dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')


# # Select the best ML matcher using CV
# result = em.select_matcher([dt, rf, svm, ln, lg], table=H, 
#         exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
#         k=5,
#         target_attr='label', metric_to_select_matcher='precision', random_state=0)
# result['cv_stats']

# # print(result)


# Get the attributes to be projected while training
attrs_to_be_excluded = []
attrs_to_be_excluded.extend(['_id', 'ltable_id', 'rtable_id', 'label'])
attrs_to_be_excluded.extend(attrs_from_table)

# Train using feature vectors from the labeled data.
rf.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='label')



