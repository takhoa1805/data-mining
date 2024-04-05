import datetime
from numpy import int64
import py_entitymatching as em
import pandas as pd
import os, sys

# READ FILES
A = em.read_csv_metadata('../newdata/tableA_downsampled.csv',key='id')
B = em.read_csv_metadata('../newdata/tableB_downsampled.csv',key='id')


# OVERLAP BLOCKER IS CHOSEN ==> INCOMPATIBILITY IN GENRE ATTRIBUTE BETWEEN 2 FILES
# CREATE OVERLAP BLOCK OBJECT
ob = em.OverlapBlocker()
ab = em.AttrEquivalenceBlocker()


# CANDIDATE SETS => PICK ONE => ATTRIBUTE EQUIVALENCE
# Cob = ob.block_tables(A, B, 'Song_Name', 'Song_Name', overlap_size=2, l_output_attrs=['Song_Name','Artist_Name','Genre','Released'], r_output_attrs=['Song_Name','Artist_Name','Genre','Released'] )
Cab = ab.block_tables(A, B, 'Song_Name', 'Song_Name', l_output_attrs=['Song_Name','Artist_Name','Genre','Released'], r_output_attrs=['Song_Name','Artist_Name','Genre','Released'])


# CHOOSE SUITABLE BLOCKER
# print(Cob.head(10)) ===> THIS ONE IS NOT SUITABLE
print(Cab.head(10)) 

# LABEL
S = em.sample_table(Cab,len(Cab))
# SAVE LABEL FILE
S.to_csv('../sampled/labeled.csv')
# G = em.label_table(S,'is_match')