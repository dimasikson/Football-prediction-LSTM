
import numpy as np
import pandas as pd
import sys
import wget
import os
import math
from hyperparams import Hyperparams as hp

firstSeason = int(sys.argv[1])
lastSeason = int(sys.argv[2])

leagues = {
    'E0': 5
    # 'D1': 6,
    # 'I1': 5,
    # 'SP1': 5,
    # 'F1': 5,
    # 'N1': 17,
    # 'P1': 17,
    # 'T1': 17
    }

def downloadFiles(firstSeason, lastSeason, leagues):

    for league in leagues:

        firstSeasonLeague = max(firstSeason, leagues[league])

        for i in range(firstSeasonLeague, lastSeason+1):

            yearStart = '0' + str(i)
            yearEnd = '0' + str(i+1)

            season = yearStart[-2:] + yearEnd[-2:]

            url = r'https://www.football-data.co.uk/mmz4281/' + season + '/' + league + '.csv'

            filename = league + '-' + season
            print(filename)

            wget.download(url, hp.rawDataFpath + filename + '.csv')

            df = pd.read_csv(hp.rawDataFpath + filename + '.csv', engine='python')
            df = df.loc[df['FTHG'] == df['FTHG']]
            df.to_csv(hp.rawDataFpath + filename + '.csv')


# downloadFiles(firstSeason, lastSeason, leagues)

def preProcess(firstSeason, lastSeason, leagues):

    for league in leagues:

        firstSeasonLeague = max(firstSeason, leagues[league])

        leagueDf = None

        for i in range(firstSeasonLeague, lastSeason+1):

            yearStart = '0' + str(i)
            yearEnd = '0' + str(i+1)
            season = yearStart[-2:] + yearEnd[-2:]
            filename = league + '-' + season
            print(filename)

            df = pd.read_csv(hp.rawDataFpath + filename + '.csv', engine='python')

            if i >= 19:
                df = df.rename(columns={'AvgH': 'BbAvH', 'AvgD': 'BbAvD', 'AvgA': 'BbAvA'})

            df = df[[
                'Div',
                'Date',
                'HomeTeam', 'AwayTeam',
                'FTHG', 'FTAG',
                'HTHG', 'HTAG',
                'FTR', 'HTR',
                'HS', 'AS',
                'HST', 'AST',
                'BbAvH', 'BbAvD', 'BbAvA'
                ]]

            df.columns = [
                    'Div',
                    'Date',
                    'HomeTeam', 'AwayTeam',
                    'GoalsFor_H', 'GoalsFor_A',
                    'GoalsForHT_H', 'GoalsForHT_A',
                    'FTR', 'HTR',
                    'ShotsTaken_H', 'ShotsTaken_A',
                    'TShotsTaken_H', 'TShotsTaken_A',
                    'HomeOdds', 'DrawOdds', 'AwayOdds'
                ]

            for col in ['HomeOdds', 'DrawOdds', 'AwayOdds', 'ShotsTaken_H', 'ShotsTaken_A','TShotsTaken_H', 'TShotsTaken_A']:
                df[col] = df[col].apply(lambda x: 1. if x > 20 else x/20)

            for col in ['GoalsFor_H', 'GoalsFor_A','GoalsForHT_H', 'GoalsForHT_A']:
                df[col] = df[col].apply(lambda x: 1. if x > 10 else x/10)

            df['Points_H'] = df['FTR'].apply(lambda x: 1. if x == 'H' else (1/3 if x == 'D' else 0))
            df['Points_A'] = df['FTR'].apply(lambda x: 1. if x == 'A' else (1/3 if x == 'D' else 0))

            table = pd.DataFrame(df.HomeTeam.sort_values().unique(), columns=['Team'])

            table['T_TablePosition'] = np.arange(len(table.Team))+1

            # COLUMN NAMING SYSTEM: T/H/A + statName + H/A
            # example: total goals for the season by the home team -> T_GoalsFor_H

            tableColumns = [
                # main table
                'T_GoalsFor', 'T_GoalsAg','T_GoalsForHT', 'T_GoalsAgHT','T_ShotsTaken', 'T_ShotsFaced','T_TShotsTaken', 'T_TShotsFaced',
                'T_Points','T_GamesPlayed','T_GoalDif','T_VAR',
                'H_GoalsFor', 'H_GoalsAg','H_ShotsTaken', 'H_ShotsFaced','H_TShotsTaken', 'H_TShotsFaced','H_Points','H_GamesPlayed', 'H_VAR',
                'A_GoalsFor', 'A_GoalsAg','A_ShotsTaken', 'A_ShotsFaced','A_TShotsTaken', 'A_TShotsFaced','A_Points','A_GamesPlayed', 'A_VAR'
            ]

            for col in tableColumns:
                table[col] = 0

            dfColumns = [
                #goals, all season
                'T_GamesPlayed_H','T_GamesPlayed_A',
                'T_GoalsFor_H','T_GoalsAg_H','T_GoalsFor_A','T_GoalsAg_A',
                'T_ShotsTaken_H','T_ShotsFaced_H','T_ShotsTaken_A','T_ShotsFaced_A',
                'T_TShotsTaken_H','T_TShotsFaced_H','T_TShotsTaken_A','T_TShotsFaced_A',
                'T_Points_H','T_Points_A',
                'T_TablePosition_H','T_TablePosition_A',
                'T_GoalsForHT_H','T_GoalsAgHT_H','T_GoalsForHT_A','T_GoalsAgHT_A',
                'T_VAR_H','T_VAR_A',
                'H_GoalsFor_H','H_GoalsAg_H','H_ShotsTaken_H','H_ShotsFaced_H','H_TShotsTaken_H','H_TShotsFaced_H','H_Points_H','H_VAR_H',
                'A_GoalsFor_A','A_GoalsAg_A','A_ShotsTaken_A','A_ShotsFaced_A','A_TShotsTaken_A','A_TShotsFaced_A','A_Points_A','A_VAR_A',
                'L6H_Goals_HA'
            ]

            for col in dfColumns:
                df[col] = 0

            df['L6H_Goals_HA'] = df['L6H_Goals_HA'].apply(lambda x: [[] for i in range(6)])

            qGoalsFor, qGoalsAg = {i: [None]*6 for i in df.HomeTeam.unique().tolist()}, {i: [None]*6 for i in df.HomeTeam.unique().tolist()}

            for row in range(len(df)):

                if row % 10 == 0:
                    print(row)

                HTeam = df['HomeTeam'].iloc[row]
                ATeam = df['AwayTeam'].iloc[row]

                tempDfRow = df.iloc[row]
                tempTableHome = table[table['Team']==HTeam]
                tempTableAway = table[table['Team']==ATeam]

                # update df for home team
                if tempTableHome['T_GamesPlayed'].values[0] > 0:

                    cols = ['GoalsFor', 'GoalsAg', 'GoalsForHT', 'GoalsAgHT', 'ShotsTaken', 'TShotsTaken', 'ShotsFaced', 'TShotsFaced', 'TablePosition', 'Points', 'VAR']
                    for col in cols:
                        tempDfRow['T_' + col + '_H'] = tempTableHome['T_' + col].values[0] / tempTableHome['T_GamesPlayed'].values[0]

                    if table[table['Team']==HTeam]['H_GamesPlayed'].values[0] > 0:
                        cols = ['GoalsFor', 'GoalsAg', 'ShotsTaken', 'TShotsTaken', 'ShotsFaced', 'TShotsFaced', 'Points', 'VAR']
                        for col in cols:
                            tempDfRow['H_' + col + '_H'] = tempTableHome['H_' + col].values[0] / tempTableHome['H_GamesPlayed'].values[0]

                tempDfRow['T_GamesPlayed_H'] = tempTableHome['T_GamesPlayed'].values[0]

                # update df for away team
                if tempTableAway['T_GamesPlayed'].values[0] > 0:

                    cols = ['GoalsFor', 'GoalsAg', 'GoalsForHT', 'GoalsAgHT', 'ShotsTaken', 'TShotsTaken', 'ShotsFaced', 'TShotsFaced', 'TablePosition', 'Points', 'VAR']
                    for col in cols:
                        tempDfRow['T_' + col + '_A'] = tempTableAway['T_' + col].values[0] / tempTableAway['T_GamesPlayed'].values[0]

                    if tempTableAway['A_GamesPlayed'].values[0] > 0:
                        cols = ['GoalsFor', 'GoalsAg', 'ShotsTaken', 'TShotsTaken', 'ShotsFaced', 'TShotsFaced', 'Points', 'VAR']
                        for col in cols:
                            tempDfRow['A_' + col + '_A'] = tempTableAway['A_' + col].values[0] / tempTableAway['A_GamesPlayed'].values[0]

                tempDfRow['T_GamesPlayed_A'] = tempTableAway['T_GamesPlayed'].values[0]

                # unpack L6H_Goals
                if tempTableHome['T_GamesPlayed'].values[0] >= 3 and tempTableAway['T_GamesPlayed'].values[0] >= 3:
                    for i in range(6):
                        tempDfRow['L6H_Goals_HA'][i].append(qGoalsFor[HTeam][i])
                        tempDfRow['L6H_Goals_HA'][i].append(qGoalsAg[HTeam][i])
                        tempDfRow['L6H_Goals_HA'][i].append(qGoalsFor[ATeam][i])
                        tempDfRow['L6H_Goals_HA'][i].append(qGoalsAg[ATeam][i])

                # update table for home team
                tempTableHome['T_GamesPlayed'] += 1
                tempTableHome['H_GamesPlayed'] += 1

                cols = ['GoalsFor', 'ShotsTaken', 'TShotsTaken', 'Points']
                for col in cols:
                    tempTableHome['T_' + col] += tempDfRow[col + '_H']
                    tempTableHome['H_' + col] += tempDfRow[col + '_H']

                tempTableHome['T_GoalsAg'] += tempDfRow['GoalsFor_A']
                tempTableHome['T_ShotsFaced'] += tempDfRow['ShotsTaken_A']
                tempTableHome['T_TShotsFaced'] += tempDfRow['TShotsTaken_A']

                tempTableHome['T_GoalsForHT'] += tempDfRow['GoalsForHT_H']
                tempTableHome['T_GoalsAgHT'] += tempDfRow['GoalsForHT_A']

                tempTableHome['H_GoalsAg'] += tempDfRow['GoalsFor_A']
                tempTableHome['H_ShotsFaced'] += tempDfRow['ShotsTaken_A']
                tempTableHome['H_TShotsFaced'] += tempDfRow['TShotsTaken_A']

                # update table for away team
                tempTableAway['T_GamesPlayed'] += 1
                tempTableAway['A_GamesPlayed'] += 1

                cols = ['GoalsFor', 'ShotsTaken', 'TShotsTaken', 'Points']
                for col in cols:
                    tempTableAway['T_' + col] += tempDfRow[col + '_A']
                    tempTableAway['A_' + col] += tempDfRow[col + '_A']

                tempTableAway['T_GoalsAg'] += tempDfRow['GoalsFor_H']
                tempTableAway['T_ShotsFaced'] += tempDfRow['ShotsTaken_H']
                tempTableAway['T_TShotsFaced'] += tempDfRow['TShotsTaken_H']

                tempTableAway['T_GoalsForHT'] += tempDfRow['GoalsForHT_A']
                tempTableAway['T_GoalsAgHT'] += tempDfRow['GoalsForHT_H']

                tempTableAway['A_GoalsAg'] += tempDfRow['GoalsFor_H']
                tempTableAway['A_ShotsFaced'] += tempDfRow['ShotsTaken_H']
                tempTableAway['A_TShotsFaced'] += tempDfRow['TShotsTaken_H']

                # update qGoalsFor, qGoalsAg
                qGoalsFor[HTeam] = qGoalsFor[HTeam][1:] + [tempDfRow['GoalsForHT_H']]
                qGoalsFor[HTeam] = qGoalsFor[HTeam][1:] + [tempDfRow['GoalsFor_H'] - tempDfRow['GoalsForHT_H']]
                qGoalsFor[ATeam] = qGoalsFor[ATeam][1:] + [tempDfRow['GoalsForHT_A']]
                qGoalsFor[ATeam] = qGoalsFor[ATeam][1:] + [tempDfRow['GoalsFor_A'] - tempDfRow['GoalsForHT_A']]

                qGoalsAg[HTeam] = qGoalsAg[HTeam][1:] + [tempDfRow['GoalsForHT_A']]
                qGoalsAg[HTeam] = qGoalsAg[HTeam][1:] + [tempDfRow['GoalsFor_A'] - tempDfRow['GoalsForHT_A']]
                qGoalsAg[ATeam] = qGoalsAg[ATeam][1:] + [tempDfRow['GoalsForHT_H']]
                qGoalsAg[ATeam] = qGoalsAg[ATeam][1:] + [tempDfRow['GoalsFor_H'] - tempDfRow['GoalsForHT_H']]

                # sort table + update T_TablePosition
                df.iloc[row] = tempDfRow
                table[table['Team']==HTeam] = tempTableHome
                table[table['Team']==ATeam] = tempTableAway

                table['T_GoalDif'] = table['T_GoalsFor'] - table['T_GoalsAg']

                table = table.sort_values(['T_Points', 'T_GoalDif', 'T_GoalsFor', 'Team'], ascending=[False,False,False,True])
                table['T_TablePosition'] = np.arange(len(table.Team))+1

            if leagueDf is None:
                leagueDf = df
            else:
                leagueDf = leagueDf.append(df)

        leagueDf.to_csv('processedData/' + league + '.csv')


preProcess(firstSeason, lastSeason, leagues)


