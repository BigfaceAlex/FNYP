# -*- coding: utf-8 -*-
"""
Founded in February 2018 on Sunday

@author: xiaoY
"""

import math
import pandas as pd
import numpy as np
import csv
from sklearn import linear_model
import random
from sklearn.model_selection import cross_val_score

fund_elo = 1600
team_elo = {}

def InitData(Mpos, Opos, Tpos):
	realM = Mpos.drop(['Rk', 'Age', 'Arena'], axis=1)
	realO = Opos.drop(['Rk', 'G', 'MP'], axis=1)
	realT = Tpos.drop(['Rk', 'G', 'MP'], axis=1)

	meargeMO = pd.merge(realM, realO, how = 'left', on = 'Team')
	finalexcl = pd.merge(meargeMO, realT, how = 'left', on = 'Team')
	return finalexcl.set_index('Team', drop = True, append = False)

def getelovalue(team):
	try:
		return team_elo[team]
	except:
		team_elo[team] = fund_elo
		return team_elo[team]
	finally:
		return 0

def calelo(winteam, loseteam):
	RA = getelovalue(winteam)
	RB = getelovalue(loseteam)

	EA = 1/(1 + math.pow(10, (RB - RA)/400))
	EB = 1/(1 + math.pow(10, (RA - RB)/400))

	if RA >= 2400:
		K = 16
	elif RA <= 2400:
		K = 32
	else:
		K = 24

	RAnewelo = round(RA + K*(1 - EA))
	RBnewelo = round(RB + K*(1 - EB))

	return RAnewelo, RBnewelo

def trainning(data, result):
	X = []
	y = []

	for index, rows in result.iterrows():
		winteam = rows['WTeam']
		loseteam = rows['LTeam']
		winelo = getelovalue(winteam)
		loseelo = getelovalue(loseteam)

		if rows['WLoc'] == 'H':
			winelo += 120
		else:
			loseelo += 120

		win_fea = [winelo]
		lose_fea = [loseelo]

		for key, value in data.loc[winteam].iteritems():
			win_fea.append(value)
		for key, value in data.loc[loseteam].iteritems():
			lose_fea.append(value)

		if np.random.random() > 0.5:
			X.append(win_fea + lose_fea)
			y.append(0)
		else:
			X.append(lose_fea + win_fea)
			y.append(1)


		win_new_score, lose_new_score = calelo(winteam, loseteam)
		team_elo[winteam] = win_new_score
		team_elo[loseteam] = lose_new_score
	return np.nan_to_num(X), y


def predict(schedule, info):
	X = []
	for index, rows in schedule.iterrows():
		team1 = rows['Vteam']
		team2 = rows['Hteam']
		team1elo = getelovalue(team1)
		team2elo = getelovalue(team2)

		X1 = [team1elo]
		X2 = [team2elo+120]

		for key, value in info.loc[team1].iteritems():
			X1.append(value)
		for key, value in info.loc[team2].iteritems():
			X2.append(value)
		X.append(X1 + X2)

	return np.nan_to_num(X)


if __name__ == '__main__':
	Mpos = pd.read_csv('/Users/xiaoy/Downloads/FNYP/pythonNBA/15-16Miscellaneous_Stat.csv')
	Opos = pd.read_csv('/Users/xiaoy/Downloads/FNYP/pythonNBA/15-16Opponent_Per_Game_Stat.csv')
	Tpos = pd.read_csv('/Users/xiaoy/Downloads/FNYP/pythonNBA/15-16Team_Per_Game_Stat.csv')
	team_result = pd.read_csv('/Users/xiaoy/Downloads/FNYP/pythonNBA/2015-2016_result.csv')

	value = InitData(Mpos, Opos, Tpos)
	value.to_csv('/Users/xiaoy/Downloads/FNYP/pythonNBA/trainning_data.csv')
	X,y = trainning(value, team_result)

	trainmodel = linear_model.LogisticRegression()
	trainmodel.fit(X,y)
	pre_schedule = pd.read_csv('/Users/xiaoy/Downloads/FNYP/pythonNBA/16-17Schedule.csv')

	predictX = predict(pre_schedule, value)
	predictresult = []

	pre_data = pd.read_csv('/Users/xiaoy/Downloads/FNYP/pythonNBA/16-17Schedule.csv')
	pre_X = predict(pre_data, value)
	pre_y = trainmodel.predict_proba(pre_X)

	for index, rows in pre_data.iterrows():
		contianer = [rows['Vteam'], pre_y[index][0], rows['Hteam'], pre_y[index][1]]
		predictresult.append(contianer)
	print(predictresult)

	with open('/Users/xiaoy/Downloads/FNYP/pythonNBA/predictresult of 16-17.csv', 'w') as f:
		writers = csv.writer(f)
		writers.writerow(['V_Team', 'The winning probability', 'H_Team', 'The winning probability'])
		writers.writerows(predictresult)

	print("Doing cross-validation..")
	print("The accuracy of prediction: ")
	print(cross_val_score(trainmodel, X, y, cv = 10, scoring='accuracy', n_jobs=-1).mean())










