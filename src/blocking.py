import datetime
from numpy import int64
import py_entitymatching as em
import pandas as pd
import os, sys

A = em.read_csv_metadata('../newdata/tableA_downsampled.csv',key='id')
B = em.read_csv_metadata('../newdata/tableB_downsampled.csv',key='id')

print(A.head(10))
print(B.head(10))