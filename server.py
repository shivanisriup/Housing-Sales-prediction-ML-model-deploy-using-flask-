import numpy as np
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    ini_feature=[int(x) for x in request.form.values()]
    final_feature=np.array([ini_feature])
    prediction=model.predict(final_feature)
    return render_template('index.html',prediction_text='Housing sales price should be $ {}'.format(prediction))


app.run(debug=True)