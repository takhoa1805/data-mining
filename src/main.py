import py_entitymatching as em
import pandas as pd
import os, sys
import profiler

# READ INPUT FILES
A = em.read_csv_metadata('../data/tableA.csv',key='id')
B = em.read_csv_metadata('../data/tableB.csv',key='id')

## List all attributes in the input tables
print("Attributes in A: ", A.columns)
print("Attributes in B: ", B.columns)


## Create new data contains some attributes
newA = A[['id', 'Song_Name', 'Artist_Name', 'Genre', 'Released']]
newB = B[['id', 'track_name', 'artist_name', 'genre', 'release_date']]

## Rename columns
newA.columns = ['id', 'Song_Name', 'Artist_Name', 'Genre', 'Released']
newB.columns = ['id', 'Song_Name', 'Artist_Name', 'Genre', 'Released']


# EVALUATE FILE SIZE
print('Length tableA: ', len(newA))
print('Length tableB: ', len(newB))
print('Length matches: ', len(newA) * len(newB))

# INVESTIGATE DATA
print(newA.head(5))
print(newB.head(5))

# Set key attribute for new data
em.set_key(newA, 'id')
em.set_key(newB, 'id')

# DOWNSAMPLE FILES
A1, B1 = em.down_sample(newA,newB,500,1,show_progress=True)
print(A1.head(5))
print(B1.head(5))
print("File length after down sampling: ")
print(len(A1))
print(len(B1))

em.get_key(A1)
em.get_key(B1)


## export the downsampled file
# os.makedirs('../newdata', exist_ok=True)
# A1.to_csv('../newdata/tableA_downsampled.csv',index=False)
# B1.to_csv('../newdata/tableB_downsampled.csv',index=False)


# PROFILE DATA
