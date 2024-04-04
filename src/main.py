import py_entitymatching as em
import pandas as pd
import os, sys
import profiler

# READ INPUT FILES
A = em.read_csv_metadata('../data/tableA.csv',key='id')
B = em.read_csv_metadata('../data/tableB.csv',key='id')

# EVALUATE FILE SIZE
print(len(A))
print(len(B))
print(len(A) * len(B))

# INVESTIGATE
print(A.head(2))
print(B.head(2))

# DOWNSAMPLE FILES
A1, B1 = em.down_sample(A,B,500,1,show_progress=True)
print(A1.head(10))
print(B1.head(10))
print("File length after down sampling: ")
print(len(A1))
print(len(B1))
