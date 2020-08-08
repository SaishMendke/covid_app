import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LogisticRegression

app = Flask(__name__) #Initialize the flask App
#model = pickle.load(open('model.pkl', 'rb'))

data = pd.read_csv('data(1).csv')
#data = np.loadtxt('data(1).csv')

x = data.iloc[:, :6]
y = data.iloc[:, -1]

regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(x, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = regressor.predict(final_features)
        #prob = model.predict_proba(final_features)

        output1 = round(prediction[0], 2)
        
    except:
        return render_template('apology.html')
    
    return render_template('index.html', prediction_text='The prediction of COVID-19 is {}'.format(output1))

if __name__ == "__main__":
    app.run(debug=True)
