#!/usr/bin/env python
import math
import numpy as np
import csv
from sklearn.decomposition import PCA
import sklearn

seasonDate = []
def PCAFunction(pcaforseasondata):
  print('Running PCA for data : ')
  pca = PCA(n_components=2)
  global floatdataofseasonArray
  floatdataofseasonArray = sklearn.preprocessing.normalize(floatdataofseasonArray, axis=0, copy = True, norm='l2')
  pcaforseasondata = pca.fit_transform(floatdataofseasonArray)
  return pcaforseasondata

def readdatafrom1314():
  print('reading data from NBA season 13-14 :')

  with open('/Users/xiaoy/Desktop/FNYP/data.csv', newline='') as r:
    readfile = csv.reader(r)
    for row in readfile:
      if seasonDate == []:
        seasonDate.append(row)
      else:
        seasonDate.append(row)

  seasonArray = np.array(seasonDate)
  return seasonArray


seasonArray = readdatafrom1314()
print(seasonArray)
seasonArray_num = np.c_[seasonArray]
floatdataofseasonArray = seasonArray_num.astype(float)
pcaforseasondata = PCAFunction(floatdataofseasonArray)
print(pcaforseasondata)





print('step 1:')
print('\n')
name = ("Lebron James", "Kyrie Irving", "Kevin Love")

print('The player name :', name)
print('\n')
Dataforplayer = [
  [ [1954,1816,1142],[1920,1041,1234],[1743,1628,1228]],   # score
  [ [646,418,116],[514,249,186],[511,389,168]],   # assist
  [ [92,83,53],[104,56,58],[109,114,51] ],   # steal
  [ [44,24,21],[49,18,41],[49,20,49]]    # block
]

print(Dataforplayer)
print('\n')
Weights1 = [
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0]
]

Weights2 = [
  [0.67615722,0.27570596,0.46761184],
  [0.4313165,0.05153551,0.40296709],
  [0.31824524,0.13857329,0.32407946],
  [0.31031147,0.14832782,0.3125419]
]

Normalization = [
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0]
]

SumWorstandBest  = [
  [0.0, 0.0],
  [0.0, 0.0],
  [0.0, 0.0]
]

WorstandBest = [
  [0.0, 0.0],
  [0.0, 0.0],
  [0.0, 0.0]
]
time = [1.0]
diw = [0.0, 0.0, 0.0]
dib = [0.0, 0.0, 0.0]
sumdiwdib = [0.0, 0.0 ,0.0]
Similarity = [0.0, 0.0, 0.0]
print('step 2:(normalised to form the matrix)')
for i in range(len(Dataforplayer)):
  for j in range(len(Dataforplayer[i])):
    Weights1[i][j] = float(sum(Dataforplayer[i][j])/math.sqrt(math.pow(len(Dataforplayer[i]), 2)))
print(Weights1)
print('\n')

print('step 3:(Data Normalization)')
for i in range(len(Weights2)):
  for j in range(len(Weights2[i])):
    Normalization[i][j] = Weights2[i][j]/sum(Weights2[j])

print(Weights2)
print('\n')
print(Normalization)
print('\n')

print('step 4:(Determine the worst alternative and the best alternative)')
for i in Normalization:
  for j in range(len(i)):
    SumWorstandBest[j][0] += math.pow(i[j]-min(i), 2)
    SumWorstandBest[j][1] += math.pow(i[j]-max(i), 2)
print(SumWorstandBest)
print('\n')

print('step 5:(Calculate the L-2 distance beteween the target  alternative i and the worst condition)')
for i in range(len(SumWorstandBest)):
  for j in range(len(SumWorstandBest[i])):
    WorstandBest[i][j] = math.sqrt(SumWorstandBest[i][j])
print(WorstandBest)
print('\n')

print('step 6:(Calculate the similarity to the worst condition:)')
for i in range(len(WorstandBest)):
  dib[i] = WorstandBest[i][0]/sum(WorstandBest[i])
print('dib =', dib)
print('\n')

for i in range(len(time)):
  for j in range(len(SumWorstandBest[i])):
    diw[j] = WorstandBest[0][j]/sum(WorstandBest[j])
print('diw =', diw)
print('\n')

for i in range(len(WorstandBest)):
  sumdiwdib[i] = dib[i] + diw[i]
print('sumdiwdib =', sumdiwdib)
print('\n')

for i in range(len(WorstandBest)):
  if sumdiwdib[i] == 0:
    i += 1
    continue
  else:
    Similarity[i] = dib[i]/sumdiwdib[i]

print(Similarity)
print('\n')

print('Ideal Alternative')
print (name[Similarity.index(max(Similarity))])


