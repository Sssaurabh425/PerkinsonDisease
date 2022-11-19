import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
##load the model
model = pickle.load(open('classification4.pkl','rb'))
scalar = pickle.load(open('scaler.pkl','rb'))
pca = pickle.load(open('pca.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    new_data= pca.transform(new_data)
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    new_data= pca.transform(final_input)
    output = model.predict(new_data)[0]
    if(output == 0):
        a = " Not "
    else:
        a = " "

    return render_template('home.html',prediction_text="The Person is{}affected with Parkinson Disease".format(a))

if __name__ =="__main__":
    app.run(debug=True)

