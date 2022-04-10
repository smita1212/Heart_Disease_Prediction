#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
# from keras.models import model_from_json
import pickle
from sklearn.preprocessing import StandardScaler


# In[2]:


app = Flask(__name__, template_folder='templates', static_folder='static')

#loading all models

#Random Forest Classifier
RFC_model = pickle.load(open('RFC_model.pkl', 'rb'))

#Support Vector Classifier
SVC_model = pickle.load(open('SVC_model.pkl', 'rb'))

#Decision Tree 
DT_model = pickle.load(open('DT_model.pkl', 'rb'))

#Neural Network
NN_model = pickle.load(open('NN_model.pkl', 'rb'))

scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    print(features)
    final_features = [np.array(features)]
    final_features = scaler.fit_transform(final_features)    
    
    #Random Forest Model Prediction
    rfc_predicted_class = RFC_model.predict(final_features)
    rfc_predicted_prob = RFC_model.predict_proba(final_features)
    print("Predicted Class :", rfc_predicted_class[0])
    rfc_confidence = rfc_predicted_prob[0][1] if rfc_predicted_class[0] == 1 else rfc_predicted_prob[0][0] 
    print("Confidence :", rfc_confidence)
    
    #Support Vector Model Prediction
    svc_predicted_class = SVC_model.predict(final_features)
    svc_predicted_prob = SVC_model.predict_proba(final_features)
    print("Predicted Class :", svc_predicted_class[0])
    svc_confidence = svc_predicted_prob[0][1] if svc_predicted_class[0] == 1 else svc_predicted_prob[0][0] 
    print("Confidence :", svc_confidence)
    
    #Decision Tree Model Prediction
    dtc_predicted_class = DT_model.predict(final_features)
    dtc_predicted_prob = DT_model.predict_proba(final_features)
    print("Predicted Class :", dtc_predicted_class[0])
    dtc_confidence = dtc_predicted_prob[0][1] if dtc_predicted_class[0] == 1 else dtc_predicted_prob[0][0] 
    print("Confidence :", dtc_confidence)
    
    #Neural Network Model Prediction
    nn_predicted_class = NN_model.predict(final_features)
    nn_predicted_prob = NN_model.predict_proba(final_features)
    print("Predicted Class :", nn_predicted_class[0][0])
    nn_confidence = nn_predicted_prob[0][1] if nn_predicted_class == 1 else nn_predicted_prob[0][0] 
    print("Confidence :", nn_confidence)
    
    return render_template('index.html', rfc_prediction_class=rfc_predicted_class[0], rfc_prediction_prob =rfc_confidence,
                                         svc_prediction_class=svc_predicted_class[0], svc_prediction_prob=svc_confidence,
                                         dtc_prediction_class=dtc_predicted_class[0], dtc_prediction_prob=dtc_confidence,
                                         nn_prediction_class=nn_predicted_class,      nn_prediction_prob=nn_confidence)
        
@app.route('/predict_api',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


# In[ ]:


if __name__ == '__main__':
    app.run()


# In[ ]:




