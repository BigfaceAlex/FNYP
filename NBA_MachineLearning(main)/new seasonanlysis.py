# -*- coding: utf-8 -*-
import numpy as np
import sklearn
from sklearn.decomposition import PCA
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import random
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import cross_val_score

global iteration
global teamstat
global WL
global TNlist
global TPlist
global FNlist
global FPlist
global CV
global teamwins1314
seasonDate = []
seasonDate2 = []


def PCAFunction(seasonArray_PCA):
	print('Running PCA for data : ')
	pca = PCA(n_components=2)
	global seasonArray_float
	seasonArray_float = sklearn.preprocessing.normalize(seasonArray_float, axis=0, copy = True, norm='l2')
	seasonArray_PCA = pca.fit_transform(seasonArray_float)
	return seasonArray_PCA

def readdatafrom1314():
	print('reading data from NBA season 13-14 :')

	with open('/Users/xiaoy/Downloads/NBA_MachineLearning(main)/2013_2014_cumulative.csv', newline='') as r:
		readfile = csv.reader(r)
		for row in readfile:
			if seasonDate == []:
				seasonDate.append(row)
			elif int(row[4]) > 20 and row[0] != seasonDate[-1][0] and row[7]>'0':
				seasonDate.append(row)
			else:
				print('######')

	seasonArray = np.array(seasonDate)
	return seasonArray


def readatafrom1415():
	print('reading data from NBA season 14-15 :')
	with open('/Users/xiaoy/Downloads/NBA_MachineLearning(main)/2014_2015_cumulative.csv', newline='') as r:
		read_file = csv.reader(r)
		for row in read_file:
			if seasonDate2 == []:
				seasonDate2.append([row[0], row[3]])
			elif row[4] >= '20' and row[0] != seasonDate2[-1][0]:
				seasonDate2.append([row[0], row[3]])
			else:
				print('######')
	seasonArray2 = np.array(seasonDate2)
	print(np.shape(seasonArray2))
	return seasonArray2

def organizeteam(seasonArray2):
	teamstat = []
	for Teamname in seasonArray2[:, 1]:
		if Teamname in teamstat:
			pass
		else:
			teamstat.append(Teamname)
			print(Teamname)
	teamstat = {el:[0,0] for el in teamstat}
	pcaTeam = list(zip(seasonArray2[:, 1], seasonArray_PCA[:, 0], seasonArray_PCA[:, 1]))
	playerpaca = dict(zip(seasonArray[:, 0], pcaTeam))
	for t in playerpaca:
		teamstring = playerpaca[t][0]
		teamstat[teamstring][0] += playerpaca[t][1]
		teamstat[teamstring][1] += playerpaca[t][2]
	return teamstat


def organizing1415team(teamstat):
	WL = []
	with open('/Users/xiaoy/Downloads/NBA_MachineLearning(main)/2014_2015.csv', newline='') as r:
		readfiles = csv.reader(r)
		for row in readfiles:
			WL.append(row)
	WL = [[teamstat.get(item, item) for item in data] for data in WL]
	print (WL)
	for memory, row in enumerate(WL):
		if row[1] > row[3]:
			row.append([1])
		else:
			row.append([0])
		del row[1]
		del row[2]
		WL[memory] = sum(row,[])
	return WL

def trainmodelSVC():
	CV = []
	TNlist = []
	FPlist = []
	TNlist = []
	FNlist = []
	svmModel = svm.SVC(kernel = 'rbf', C = grid.best_params_[0], gamma  = grid.best_params_[1])
	svmModel.fit(X_test, Y_test)
	for item in X_cv:
		CV.append(svmModel.predict([item]))

	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for estimatedata, action in zip(CV, Y_cv):
		if np.round(estimatedata) == 1:
			TP += 1*(np.round(estimatedata) == action)
			FP += 1*(np.round(estimatedata) != action)
		elif np.round(estimatedata) == 0:
			TN += 1*(np.round(estimatedata) == action)
			FN += 1*(np.round(estimatedata) != action)
		else:
			print('######')

	TNlist.append(TN)
	TPlist.append(TP)
	FNlist.append(FN)
	FPlist.append(FP)
	FPlist = [x.tolist() for x in FPlist if type(x) == np.ndarray]
	TPlist = [x.tolist() for x in TPlist if type(x) == np.ndarray]
	FNlist = [x.tolist() for x in FNlist if type(x) == np.ndarray]
	TNlist = [x.tolist() for x in TNlist if type(x) == np.ndarray]

	confusionmat = np.array([[np.mean(TPlist), np.mean(FPlist)], [np.mean(FNlist), np.mean(TNlist)]])
	print('ploting the result of SVM: ')
	xx, yy = np.meshgrid(np.linspace(min(X_test[:,0]), max(X_test[:,0]), 500), np.linspace(min(X_test[:,2]), max(linspace(X_test[:,2]))))
	plt.figure()
	z = svmModel.decision_function(np.c_[xx.ravel(), xx.ravel(), yy.ravel(), yy.ravel()])
	z = z.shape(xx.shape)
	plt.imshow(Z, interpolation='nearest',
			extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
			origin='lower', cmap=plt.cm.coolwarm)
	contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
						linetypes='--')
	plt.scatter(X_test[:, 0], X_test[:, 2], s=30, c=Y_test, cmap=plt.cm.coolwarm)
	plt.scatter(X_cv[:, 0], X_cv[:, 2], s=30, c=Y_cv, cmap=plt.cm.coolwarm, edgecolors='w')
	plt.title('Cross Section of 4-D Decision Boundary for Game Prediction SVM', fontsize = 18)
	plt.xlabel('Team 1 PCA Dimension 1')
	plt.ylabel('Team 2 PCA Dimension 1')
	plt.show()
	return confusionmat

def kmeanofwin(seasonArray, seasonArray_PCA):
	teamwins1314 = []
	with open('/Users/xiaoy/Downloads/NBA_MachineLearning(main)/Records2013-2014.csv', newline = '') as r:
		read_files = csv.reader(r)
		for row in read_files:
			teamwins1314.update({row[0]:row[1]})

	winsmins = np.array([int(teamWin2013[team])*int(mins) for mins,team in zip(seasonArray[:,6], seasonArray[:,3])])
	seasonArraywins = np.c_[seasonArray_PCA, winsmins]
	st = KMeans(n_clusters = 5, n_init = 20)
	st.fit(seasonArraywins)
	label = st.labels_
	fig2 = plt.figure(2)
	labelfloat = label.astype(float)
	ax = Axes3D(fig2, rect = [0, 0, 0.95, 1], elev = 48, azim = 30)
	X = seasonArraywins[:, 0]
	Y = seasonArraywins[:, 1]
	Z = seasonArraywins[:, 2]
	name = seasonArray[:, 0]
	ax.scatter(X, Y, Z, c = labelfloat, cmap = colormap, edgecolor = 'k')
	ax.set_title('The result of the win', fontsize = 15)
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_zlabel('win time!')
	fig7 = plt.figure(7)
	ax = Axes3D(fig7, rect=[0, 0, .95, 1], elev=48, azim=30)
	ax.scatter(X, Y, Z, c=labels2.astype(np.float), cmap = colormap, edgecolor='k')
	for i in range(len(X)):
		ax.text(X[i], Y[i], Z[i],  names[i], zorder=1)
	ax.set_title('KMeans with Wins', fontsize = 18)
	ax.set_xlabel('PCA Dimension 1')
	ax.set_ylabel('PCA Dimension 2')
	ax.set_zlabel('Win-Minutes')
	plt.show()

seasonArray = readdatafrom1314()
seasonArray_num = np.c_[seasonArray[:,4], seasonArray[:,6], seasonArray[0:, 27:]]
seasonArray_float = seasonArray_num.astype(float)
seasonArray_PCA = PCAFunction(seasonArray_float)

print('For plotting result of PCAFunction : ')
fig, ax = plt.subplots(1, 1, figsize = (8,8))

#set the position of x & y
X = seasonArray_PCA[:,0]
Y = seasonArray_PCA[:,1]
PER = seasonArray[:, 7].astype(float)
PER = 100*PER/np.max(PER)
label = seasonArray[:, 0]

for x,y,lab in zip(X, Y, label):
	ax.scatter(x,y,label=lab)

plt.title('The result that it reduced dimensionality data o13-14 season & there are player names on label: ', fontsize=18)
plt.xlabel('Offensive')
plt.ylabel('Defensive')

colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
colors = [colormap(i) for i in np.linspace(0, 1,len(ax.collections))]
for i,j in enumerate(ax.lines):
    j.set_color(colors[i])
for i,txt in enumerate(label):
	ax.annotate(txt, (X[i], Y[i]), xytext=(X[i]+.005,Y[i]+.0010))


def PCAFunction(seasonArray_PCA):
	print('Running PCA for data : ')
	pca = PCA(n_components=2)
	seasonArray_float = sklearn.preprocessing.normalize(seasonArray_float, axis=0, copy = True, norm='l2')
	seasonArray_PCA = pca.fit_transform(seasonArray_float)
	return seasonArray_PCA

#plt.show()
#end1
iteration = 500
est = KMeans(n_clusters = 8, n_init=50, n_jobs = 4, max_iter = iteration)
est.fit(seasonArray_PCA)
#estimate the number of classify
labels = est.labels_
r1 = pd.Series(est.labels_).value_counts()
r2 = pd.DataFrame(est.cluster_centers_)
r = pd.concat([r2, r1], axis =1)
print(r)


#Multi-graph layering
fig1, ax1 = plt.subplots(1,1)
ax1.scatter(X, Y, c=labels.astype(np.float), cmap = colormap, edgecolor = 'K')
plt.title('The Kmeans of 8 clusters ', fontsize=18)
plt.xlabel('Offensive')
plt.ylabel('Defensive')
#plt.show()
#end2


print('For determine position: ')
x_min, x_max = seasonArray_PCA[:, 0].min(), seasonArray_PCA[:, 0].max()
y_min, y_max = seasonArray_PCA[:, 1].min(), seasonArray_PCA[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1600), np.linspace(y_min, y_max, 1600))

# Obtain labels for each point in mesh. Use last trained model.
Z = est.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
# plt.figure(3)
fig3, ax3 = plt.subplots(1,1, figsize=(6,6))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=colormap,
           aspect='auto', origin='lower')
X = seasonArray_PCA[:, 0]
Y = seasonArray_PCA[:, 1]
plt.plot(X, Y, 'k.', markersize=2)
# Plot the centroids as a white X
centroids = est.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)

plt.title('K-means clustering on the PCA-reduced Statistics from the 2013-2014 season \n'
          'Centroids are marked with white cross', fontsize = 18)
plt.xlabel('Offensive')
plt.ylabel('Defensive')
plt.xlim(1.15*x_min, 1.15*x_max)
plt.ylim(1.15*y_min, 1.15*y_max)
print(centroids[:, 0], centroids[:, 1])
#plt.show()
#end3


seasonArray2 = readatafrom1415()
teamstat = organizeteam(seasonArray2)
WL = organizing1415team(teamstat)
#end4--next srart svm


L = len(WL)
testl = int(.9*L)
teamDataWL = [row[:-1] for row in WL]
homeDataWL = [row[-1] for row in WL]
randNums = random.sample(range(L), testl)
teamDataWL_test = [teamDataWL[x] for x in randNums]
homeTeamW_test = [homeDataWL[x] for x in randNums]
teamDataWL_csv = [teamDataWL[int(idofdata)] for idofdata, val in enumerate(teamDataWL) if idofdata not in randNums]
homeTeamW_csv = [homeDataWL[int(idofdata)] for idofdata, val in enumerate(teamDataWL) if idofdata not in randNums]

X_test = np.array(teamDataWL_test)
Y_test = np.array(homeTeamW_test)
X_cv = np.array(teamDataWL_csv)
Y_cv = np.array(homeTeamW_csv)


C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_test, Y_test)
"""
clf.fit(X_test, Y_test)
print(clf.score(X_test, Y_test))
y_hat = clf.predict(X_test)
show_accuracy(y_hat, Y_test, 'train data accuracy')
print(clf.score(X_cv, Y_cv))
y_hat = clf.predict(X_cv)
show_accuracy(y_hat, Y_cv, 'test data accuracy')
print(grid.best_params_[1])
"""
print('Start for confusionmat: ')
confusionmat = trainmodelSVC(X_test, Y_test, X_cv, Y_cv)
print(confusionmat)

teamwin1314 = kmeanofwin(seasonArray, seasonArray_PCA)

print('The result of code')
seasonArray_num = np.c_(seasonArray[:,6], seasonArray_num)
floatsea = seasonArray_num.astype(float)
floatsea = sklearn.preprocessing.normalize(floatsea, axis = 0)
pca = PCA(n_components = 1)
PCA2 = pca.fit_reansform(floatsea[:, :])
floatper = seasonArray[:, 7].astype(float)
plt.figure(6)
plt.clf()
z = np.polyfit(floatper, PCA2, 1)
plt.plot(floatper, PCA2, 'bo')
plt.title('The result of every players for PER vs PCA ', fontsize = 18)
plt.xlabel("PER")
plt.ylabel('PCA')



p = floatper.tolist()
s = [x for y in PCA2.tolist() for x in y]
r = np.corrcoef(p, s)[0,1]
print(r)
plt.show()