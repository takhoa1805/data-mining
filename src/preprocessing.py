import datetime
from numpy import int64
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

## Drop NA values
newA = newA.dropna()
newB = newB.dropna()

# CONVERT A TABLE COLUMNS TO LOWERCASE IF DATA IS STRNG TYPE
newA['Song_Name'] = newA['Song_Name'].str.lower()
newA['Artist_Name'] = newA['Artist_Name'].str.lower()
newA['Genre'] = newA['Genre'].str.lower()



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
A1, B1 = em.down_sample(newA,newB,1000,1,show_progress=True)

# NO NEED DOWN SAMPLING
# A1,B1=newA,newB

print("File length after down sampling: ")
print(len(A1))
print(len(B1))

em.get_key(A1)
em.get_key(B1)

## sort the data by id
A1 = A1.sort_values(by=['id'])
B1 = B1.sort_values(by=['id'])


## Convert A1 released time to year
A1['Released'] = pd.to_datetime(A1['Released']).dt.year.astype(int64)



print(A1.head(5))
print(B1.head(5))




## export the downsampled file
os.makedirs('../newdata', exist_ok=True)
A1.to_csv('../newdata/tableA_downsampled.csv',index=False)
B1.to_csv('../newdata/tableB_downsampled.csv',index=False)


# PROFILE DATA
