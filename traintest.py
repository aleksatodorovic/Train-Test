"""
Alex Todorovic CMPS-3240
Using pandas for data processing, LinearSVC from sklearn.svm for training and testing, metrics from sklearn for evaluating accuracy

Note:
Although I planned on using several seasons of games for training I've only included one for training and one for testing
in this program. This was mainly for simplicity and ease of use as several .csv files include NaN values and values that
are not processable that cause errors. In the future I plan on adding multiple seasons of games for training and testing,
but I must first find and clean the .csv files causing problems.

This model still yields a surprisingly high accuracy score with just 380 games to train on.
I've gotten as low as 0.45 and as high as 0.63.
"""
import pandas as pd
from sklearn import metrics
from sklearn.svm import LinearSVC


'''Training data processing'''
league = pd.read_csv('E0-3.csv', header=None,
                                                #these are the names of the columns
                         names=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
                                'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                                'bet1', 'bet2', 'bet3', 'bet4', 'bet5', 'bet6', 'bet7', 'bet8', 'bet9', 'bet10', 'bet11',
                                'bet12', 'bet13', 'bet14', 'bet15', 'bet16', 'bet17', 'bet18', 'bet19', 'bet20', 'bet21',
                                'bet22', 'bet23','bet24', 'bet25', 'bet26', 'bet27', 'bet28', 'bet29', 'bet30', 'bet31',
                                'bet32','bet33', 'bet34', 'bet35', 'bet36', 'bet37', 'bet38', 'bet39', 'bet40', 'bet41',
                                'bet42'])


'''Features list: home shots, away shots, home shots on target, away shots on target,
home fouls, away fouls, home corners, away corners, home yellow cards, away yellow
cards, home red cards, away red cards'''
features = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',]
x_train = league[features]
x_train = x_train[1:] #removing labels
y_train = league[['FTR']]
y_train = y_train[1:] #removing labels


'''     Testing data processing     '''
league = pd.read_csv('E0-2.csv', header=None,
                         names=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
                                'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                                'bet1', 'bet2', 'bet3', 'bet4', 'bet5', 'bet6', 'bet7', 'bet8', 'bet9', 'bet10', 'bet11',
                                'bet12', 'bet13', 'bet14', 'bet15', 'bet16', 'bet17', 'bet18', 'bet19', 'bet20', 'bet21',
                                'bet22', 'bet23','bet24', 'bet25', 'bet26', 'bet27', 'bet28', 'bet29', 'bet30', 'bet31',
                                'bet32','bet33', 'bet34', 'bet35', 'bet36', 'bet37', 'bet38', 'bet39', 'bet40', 'bet41',
                                'bet42'])
'''Doing the same thing in training data processing, except I'm using a different .csv file'''
x_test = league[features]
x_test = x_test[1:]
y_test = league[['FTR']]
y_test = y_test[1:]


svm = LinearSVC() #initiating

svm.fit(x_train, y_train) #training

p = svm.predict(x_test) #predcting

accuracy = metrics.accuracy_score(p, y_test) #measuring accuracy

print(accuracy)





