import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading the data for ex1
f = open('ex1data1.txt', mode='r')
data = f.read()
data = data.split('\n')
data = data[:len(data) - 1]
rows = []
for i in data:
    temp = i.split(',')
    temp[0] = float(temp[0])
    temp[1] = float(temp[1])
    rows.append(temp)
