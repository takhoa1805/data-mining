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


# CONVERT THE LABELED DATA TO FEATURE VECTORS USING THE FEATURE TABLE
attrs_from_table = ['ltable_Song_Name', 'ltable_Artist_Name', 'ltable_Released',
                    'rtable_Song_Name', 'rtable_Artist_Name', 'rtable_Released']

# Split S into I an J
IJ = em.split_train_test(G, train_proportion=0.7, random_state=0)
I = IJ['train']
J = IJ['test']


# Create a set of ML-matchers
dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')


feature_table = em.get_features_for_matching(A,B,False)

H = em.extract_feature_vecs(I,feature_table=feature_table,attrs_after='label',show_progress=True)

# Select the best ML matcher using CV
result = em.select_matcher([dt, rf, svm, ln, lg], table=H, 
        exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
        k=4,
        target_attr='label', metric_to_select_matcher='f1', random_state=0)

print(result['cv_stats'])


# Convert J into a set of feature vectors using feature table
L = em.extract_feature_vecs(J, feature_table=feature_table,
                            attrs_after='label', show_progress=False)


# Train using feature vectors from I 
lg.fit(table=H, 
       exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'], 
       target_attr='label')

# Predict on L 
predictions = lg.predict(table=L, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'], 
              append=True, target_attr='predicted', inplace=False)


predictions.to_csv('../sample/predict.csv')

# Evaluate the predictions
eval_result = em.eval_matches(predictions, 'label', 'predicted')
em.print_eval_summary(eval_result)




################################# PRODUCTION #################################

def integrate():
    # # MERGE 2 TABLES
    # table = tableA.merge(tableB, left_on='ltable_id', right_on='rtable_id')
    # table.columns = ['ltable_id','rtable_id','ltable_Song_Name','ltable_Artist_Name','ltable_Released','rtable_Song_Name','rtable_Artist_Name','rtable_Released']

    tableA = em.read_csv_metadata('../data/tableA.csv',key='id')
    tableB = em.read_csv_metadata('../data/tableB.csv',key='id')

    ## Create new data contains some attributes
    tableA = tableA[['id', 'Song_Name', 'Artist_Name','Released']]
    tableB = tableB[['id', 'track_name', 'artist_name','release_date']]
    
    # ## Rename columns 
    tableA.columns = ['id', 'Song_Name', 'Artist_Name', 'Released']
    tableB.columns = ['id', 'Song_Name', 'Artist_Name', 'Released']

    ## Drop NA values
    tableA = tableA.dropna()
    tableB = tableB.dropna()

    # CONVERT A TABLE COLUMNS TO LOWERCASE IF DATA IS STRNG TYPE
    tableA['Song_Name'] = tableA['Song_Name'].str.lower()
    tableB['Artist_Name'] = tableB['Artist_Name'].str.lower()

    # Set key attribute for new data
    em.set_key(tableA, 'id')
    em.set_key(tableB, 'id')

    ## sort the data by id
    tableA = tableA.sort_values(by=['id'])
    tableB = tableB.sort_values(by=['id'])


    ## Convert A1 released time to year
    tableA['Released'] = pd.to_datetime(tableA['Released']).dt.year.astype(int)

    

    ab = em.AttrEquivalenceBlocker()
    C = ab.block_tables(tableA, tableB, 'Song_Name', 'Song_Name', l_output_attrs=['Song_Name','Artist_Name','Released'], r_output_attrs=['Song_Name','Artist_Name','Released'])
    # print(C.head())
    

integrate()