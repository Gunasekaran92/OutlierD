import flask
from flask_cors import CORS, cross_origin
from flask import request, jsonify
import pandas as pd
from sklearn.externals import joblib

from datetime import timedelta  
from flask import Flask, make_response, request, current_app  
from functools import update_wrapper
# initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/foo": {"origins": "http://127.0.0.1:5000"}})


@app.route('/api/model', methods=['GET'])
@cross_origin(origin='localhost',headers=['Content-type','application/json'])
def api_filter():
    query_parameters = request.args

    TransactionAmount = query_parameters.get('TransactionAmount')
    NonFuelTaxAmount = query_parameters.get('NonFuelTaxAmount')
    ModeofPayment = query_parameters.get('ModeofPayment')
    
    if (ModeofPayment == 'Cash'):
        CreditCard_Trans = 0
        Cash_Trans = 1
    else:
        CreditCard_Trans = 1
        Cash_Trans = 0
        
    
    #Dummy Data
    dict1 = {"Transaction Amount":TransactionAmount,"NonFuel Tax Amount":NonFuelTaxAmount,"CreditCard_Trans":CreditCard_Trans,"Cash_Trans":Cash_Trans}
    new_Df=pd.DataFrame.from_dict(dict1,orient='index')
    #new_Df.columns = [''] * len(new_Df.columns)
    
    new_Df = new_Df.T
    
    loaded_model = joblib.load('finalized_model.sav')
   
    result = loaded_model.predict_proba(new_Df)
    
    y_pred_prob = pd.DataFrame(data = result)
    print(y_pred_prob)
    output = round((y_pred_prob[1]*100),2).to_string(index=False) + ' %'
    
    print(output)
    
    #L = result[:,-1:]*100
    #makeitastring = ''.join(map(str, L))
    #print(makeitastring)
    #print(makeitastring[-11:-7] + '%')

    #return jsonify(makeitastring[-11:-7] + '%')
    return jsonify(output)

app.run()