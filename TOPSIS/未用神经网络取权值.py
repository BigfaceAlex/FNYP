#!/usr/bin/env python
import math

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
Weights = [
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0]
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
    Weights[i][j] = float(sum(Dataforplayer[i][j])/math.sqrt(math.pow(len(Dataforplayer[i]), 2)))
print(Weights)
print('\n')

print('step 3:(Data Normalization)')
for i in range(len(Weights)):
  for j in range(len(Weights[i])):
    Normalization[i][j] = Weights[i][j]/sum(Weights[j])

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


