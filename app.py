from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
filename = 'children anemia.pkl'
model = pickle.load(open(filename, 'rb'))

prediction_group = {0: 'Mild', 1:'Moderate', 2:'Not anemic', 3:'Severe'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Hemoglobin_level = request.form['Hemoglobin level']
    Living_place = request.form['Type of place of residence']
    Wealth_level = request.form['Wealth index combined']
    Age_1st_birth = request.form['Age of respondent at 1st birth']
    pred = model.predict(np.array([[Hemoglobin_level, Living_place, Wealth_level, Age_1st_birth ]]))

    predicted_category = prediction_group.get(pred[0], 'Unknown Category')

    return render_template('index.html', predict=str(pred), predicted_category=predicted_category)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
