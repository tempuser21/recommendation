
from recommendation import *


# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import sys

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    try:
        
        reques_json = request.get_json(force=True)#request.json
        reco_list = get_final_recommendation(reques_json[0])

        print(reco_list)

        jsonfiles = reco_list.to_json(orient='records')

        return jsonfiles

    except:

        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
        
    app.run(port=port, debug=True, use_reloader=False)