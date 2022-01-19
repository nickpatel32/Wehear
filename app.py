import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import jsonify
from joblib import load
import json

# Initialize the flask App
app = Flask(__name__)

# Mapping Dictionary

tab_map = {
     "combiflame": 0,
     "dolo_650": 1,
     "neurobin_vit_b_pink_red": 2,
     "ecosprin": 3,
     "crocin": 4,
     "paracetemol 500mg": 5,
     "danp": 6,
     "lth_unison": 7,
     "lemolate_gold": 8,
     "paracetamol_650mg": 9,
     "Vita C":10,
     "Diclowin T":11,
     "Ratntac":12,
     "Vomistop":13,
     "Lopamid":14,
     "Metacin":15,
     "Brompeph":16,
     "CTZ":17,
     "Zgfoly":18,
     "Asirst":19,
     "Spasmani":20
}

milk_map = {
    
     "Gold_50%":0 ,
     "Gold_60%":1,
     "Gold_70%":2,
     "Gold_80%":3,
     "Gold_90%":4,
     "Gold_100%":5,
     "Taaza_50%":6,
     "Taaza_60%":7,
     "Taaza_70%":8,
     "Taaza_80%":9,
     "Taaza_90%":10,
     "Taaza_100%":11,
     "Shakti_50%":12,
     "Shakti_60%":13,
     "Shakti_70%":14,
     "Shakti_80%":15,
     "Shakti_90%":16,
     "Shakti_100%":17
}

tab_op_map = {value: key for key, value in tab_map.items()}
milk_op_map = {value: key for key, value in milk_map.items()}
##
tab_model = load('tab_model.pkl')
milk_model = load('milk_model.pkl')


# default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


# milk

@app.route('/predict-milk', methods=['GET', 'POST'])
def prediction_milk_api():
    if request.method == 'GET':
        arr = json.loads(request.args.get('arr', None))
    else:
        arr = request.get_json()
    arr = np.array(arr['arr'])
    preds = milk_model.predict_proba(arr.reshape(1, -1))
    acc = np.amax(preds)
    output = milk_op_map.get(preds.argmax())
    return jsonify(milk_name=output, acc=str(acc * 100))


# tab

@app.route('/predict-tab', methods=['GET', 'POST'])
def prediction_tab_api():
    if request.method == 'GET':
        arr = json.loads(request.args.get('arr', None))
    else:
        arr = request.get_json()
    arr = np.array(arr['arr'])
    preds = tab_model.predict_proba(arr.reshape(1, -1))
    acc = np.amax(preds)
    output = tab_op_map.get(preds.argmax())
    return jsonify(med_name=output, acc=str(acc * 100))


if __name__ == "__main__":
     app.run(debug=True)
