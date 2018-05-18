
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import matplotlib.cm as cm
import re
sns.set_style("darkgrid")
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
dataset = pd.read_csv("C:/Users/prana/Desktop/652/LAB 8/CompleteDataset.csv", low_memory=False)
dataset.columns
dataset.head()

def conversion(money_str):
    notes = ''
    # Find the numbers and append
    for letter in money_str:
        if letter in '1234567890.':
            notes = notes + letter
        else:
            pass
    # Divide by 1000 to convert K to M for value
    if 'K' in money_str:
        return (float(notes)/1000)    
    else:
        return float(notes)

def wage_conversion(money_str):
    notes = ''
    # Find the numbers and append
    for letter in money_str:
        if letter in '1234567890.':
            notes = notes + letter
        else:
            pass
    
    return float(notes)

def convert_attributes(number_str):
    if type(number_str) == str:
        if '+' in number_str:
            return float(number_str.split('+')[0])
        elif '-' in number_str:
            return float(number_str.split('-')[0])
        else:
            return float(number_str)


#Data Conversion
dataset['Wage'] = dataset['Wage'].apply(wage_conversion) # Units = K
print(dataset['Wage'][-10:].dtype)
dataset['Value'] = dataset['Value'].apply(conversion) # Units = M
print(dataset['Value'][-10:].dtype)

dataset['Remaining Potential'] = dataset['Potential'] - dataset['Overall']

dataset['Preferred Position'] = dataset['Preferred Positions'].str.split().str[0]


###Best 11 based on overall rating in fifa data set
def formation_best_squad(position):
    dataset_copy = dataset.copy()
    store = []
    for i in position:
        store.append([i,dataset_copy.loc[[dataset_copy[dataset_copy["Preferred Position"] == i]["Overall"].idxmax()]]['Name'].to_string(index = False), dataset_copy[dataset_copy['Preferred Position'] == i]['Overall'].max()])
        dataset_copy.drop(dataset_copy[dataset_copy['Preferred Position'] == i]['Overall'].idxmax(), inplace = True)
    #return store
    return pd.DataFrame(np.array(store).reshape(11,3), columns = ['Position', 'Player', 'Overall']).to_string(index = False)

# 4-3-3
formation433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']
print ('4-3-3')
print (formation_best_squad(formation433))

#3-5-2
formation352 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']
print ('3-5-2')
print (formation_best_squad(formation352))

##4-2-3-1
formation4231=['GK','LB','CB','CB','RB','CDM','CDM','LM','CAM','RM','ST']
print('4-2-3-1')
print(formation_best_squad(formation4231))
##potential against overall rating based on age parameter
##basic visualization


dataset_potential = dataset.groupby(['Age'])['Potential'].mean()
dataset_overall = dataset.groupby(['Age'])['Overall'].mean()

dataset_summary = pd.concat([dataset_potential, dataset_overall], axis=1)

ax = dataset_summary.plot(color='C0,C1')
ax.set_ylabel('Player_Rating',color='r')
ax.set_title('Rating by Age',color='b')
plt.show()


###Top potential low rated players 
dataframe=dataset
dataframe['growth']=dataframe['Potential']-dataframe['Overall']
high_potential=dataframe[['Name','Overall','growth','Club','Preferred Positions']]
Top_Growths=high_potential.sort_values(by=['growth','Overall'],ascending=False)
print(Top_Growths[:10])

####Top 20 players
Top=dataset[['Name','Age','Preferred Positions','Overall']]
Top_20=Top.sort_values(by=['Overall'],ascending=False)
print(Top_20[:20])

#######Machine Learning

mldf=dataset[['Name','Value','Overall','Age','Finishing']]


##To remove non-numeric values in Finishing column

def numeric_values(s):
    try:
        n = int(s)
        return (1 <= n and n <= 99)
    except ValueError:
        return False
 
#remove not valid entries for Finishing
mldf = mldf.loc[mldf['Finishing'].apply(lambda x: numeric_values(x))]
 
#now we can define Finishing as integers
mldf['Finishing'] = mldf['Finishing'].astype('int')

####Distribution of players 
plt.hist(dataset.Overall, bins=16, alpha=0.6, color='r')
plt.title("#Distribution of Players based on Overall rating")
plt.xlabel("Overall")
plt.ylabel("Count")
 
plt.show()

##Machine Learning(Linear regression)
##Data Slicing
##Dividing data using model selection 
from sklearn.model_selection import train_test_split
 
train, test = train_test_split(mldf, test_size=0.20, random_state=99)
 
xtrain = train[['Value']]
ytrain = train[['Overall']]
 
xtest = test[['Value']]
ytest = test[['Overall']]

regression = linear_model.LinearRegression()
regression.fit(xtrain, ytrain)

#Predicting y using test data set
y_predictions = regression.predict(xtest)

print("Mean squared error using linear regression: %.2f" % mean_squared_error(ytest, y_predictions))
plt.plot(y_predictions,ytest)
plt.show()

####Using another model support vector regression (radial basis function for non linear problems) 
SVR_dataset = SVR(kernel='rbf', gamma=1e-3, C=100, epsilon=0.1)
SVR_dataset.fit(xtrain, ytrain.values.ravel())

radial_function = SVR_dataset.predict(xtest)
plt.plot(radial_function,ytest)
plt.show()

print("Mean squared error using svr: %.2f" % mean_squared_error(ytest, radial_function))

