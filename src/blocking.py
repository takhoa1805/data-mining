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
Cab = ab.block_tables(A, B, 'Song_Name', 'Song_Name', l_output_attrs=['Song_Name','Artist_Name','Released'], r_output_attrs=['Song_Name','Artist_Name','Released'])
Cob = ob.block_tables(A, B, 'Artist_Name', 'Artist_Name', overlap_size=3, l_output_attrs=['Song_Name','Artist_Name','Released'], r_output_attrs=['Song_Name','Artist_Name','Released'] )


# CHOOSE SUITABLE BLOCKER
# print(Cob.head(10)) ===> THIS ONE IS NOT SUITABLE
# print(Cab.head(10)) 

# COMBINE 2 BLOCKERS
# C = em.combine_blocker_outputs_via_union([Cab,Cob])

# LABEL
S = em.sample_table(Cab,len(Cab))
# S = em.sample_table(Cob,len(Cob))
# S = em.sample_table(C,len(C))

# SAVE LABEL FILE
S.to_csv('../sample/candidate.csv')

# # FIXING MANUALLY AND RE-LOAD FILE 
# FINISHING LABELING
G = em.read_csv_metadata('../sample/labeled.csv',key='_id') #DO THIS MANUALLY
print(G.head(10))
