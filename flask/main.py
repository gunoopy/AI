from flask import Flask, render_template, request
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models



#
# Load Files
#

# model
file_dir = 'files/'
MODEL = models.load_model(file_dir + 'iris_model.h5')

# feature name information
with open(file_dir + 'label_info.pickle', 'rb') as f :
    LABELS = pickle.load(f)

# standard scaler
with open(file_dir + 'standard_scaler.pickle', 'rb') as f :
    SCALER = pickle.load(f)

FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


#
# Functions
#

def processing(new_data) :
    '''
    - new_data : 1D-array
        ex) [6.1, 2.8, 4.7, 1.2]
    - return : predict result(label, string) & plot result(figure)
    '''

    # scaling & predict
    new_data = SCALER.transform(new_data.reshape(1, -1))
    pred_prob = MODEL.predict(new_data)[0]
    pred_class = LABELS[pred_prob.argmax()]

    # result dataframe
    pred_prob = pd.DataFrame(pred_prob, columns=['prob']).rename(index=LABELS)
    pred_prob = pred_prob.reset_index().sort_values('prob', ascending=True)

    # plot
    fig = px.bar(pred_prob, x='prob', y='index', labels={'index': '', 'prob': 'Probability'})

    return pred_class, fig


#
# Flask
#

app = Flask(__name__)

@app.route('/')
def home() :
    return render_template('index.html', feature_names = FEATURE_NAMES)


@app.route('/result', methods = ['POST', 'GET'])
def result() :
    if request.method == 'POST' :
        # label 순서에 맞게 1D array로 변환
        result = np.array([request.form[feature_name] for feature_name in FEATURE_NAMES])
        # result = np.array([6.1, 2.8, 4.7, 1.2]) # sample

        prediction, fig = processing(result)

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('result.html', graphJSON = graphJSON, prediction = prediction)
    else :
        return render_template('index.html')



if __name__ == '__main__' :
    app.run(debug=True)