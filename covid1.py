import pandas as pd
import sklearn
import pickle

data = pd.read_csv('data(1).csv')

x = data.iloc[:, :6]
y = data.iloc[:, -1]

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
