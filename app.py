from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
filename = 'children anemia.pkl'
model = pickle.load(open(filename, 'rb'))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Hemoglobin_level = request.form['Hemoglobin level']
    Living_place = request.form['Type of place of residence']
    Wealth_level = request.form['Wealth index combined']
    Age_1st_birth = request.form['Age of respondent at 1st birth']
    pred = model.predict(np.array([[Hemoglobin, Living_place, Wealth_level, Age_1st_birth ]]))
    #print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(debug=True)
