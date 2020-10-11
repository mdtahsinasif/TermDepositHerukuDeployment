# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 02:11:53 2020

@author: TahsinAsif
"""

from flask import Flask, jsonify, request, render_template
import joblib
#from sklearn.externals import joblib
import pandas as pd
from pandas.tests.groupby.test_value_counts import df
from sklearn import tree, model_selection, ensemble
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree, model_selection, preprocessing, ensemble, feature_selection, neighbors, naive_bayes
from sklearn.svm import SVC # "Support Vector Classifier"
import pytest
import os
import json


app = Flask(__name__,template_folder='template')
#MODEL_FILE = 'C:/Users/TahsinAsif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/Heroku_Yogen_API/grid_classifier_modelVersion1.pkl'
MODEL_FILE = 'C:/Users/TahsinAsif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/Heroku_Yogen_API/knn_estimator_model.pkl'
log_estimator = joblib.load(MODEL_FILE)

@app.route('/')
def home():
    global log_estimator
    return render_template('index.html')


@app.route('/termDepositPrediction', methods=['POST','GET'])
def termDepositPrediction():
    output = ''
    json_= request.json
    url_test = pd.DataFrame([json_])
    default_json = request.args.get('default') 
    default_json = 0
    print('default_json:------------>', default_json)    
    url_test['default'] = pd.DataFrame([default_json])
    
    housing_json = request.args.get('housing') 
    housing_json = 0
    url_test['housing'] = pd.DataFrame([housing_json])
    print('housing_json:------------>', housing_json)
    
    loan_json = request.args.get('loan') 
    loan_json = 0
    url_test['loan'] = pd.DataFrame([loan_json])
    print('loan_json:--------------->', loan_json)
    
    age_json = request.args.get('age') 
    age_json = 34
    print('age_json:---------------->', age_json)
    url_test['age'] = pd.DataFrame([age_json])
    
    job_json = request.args.get('job') 
    job_json = 'job_admin'
    print('job_json:---------------->', job_json)
    job_admin_value = 0
    job_blue_collar_value = 0
    job_entrepreneur_value = 0
    job_housemaid_value = 0
    job_management_value = 0
    job_retired_value = 0
    job_self_employed_value = 0
    job_services_value = 0
    job_student_value = 0
    job_technician_value = 0
    job_unemployed_value = 0
    job_unknown_value = 0
      
    if(job_json == "job_admin"):
        job_admin_value = 1   
        print("job_admin_value----------------->",job_admin_value)
    elif(job_json == "job_blue_collar"):
        job_blue_collar_value = 1
        print("job_blue_collar_value----------------->",job_blue_collar_value)
    elif(job_json == "job_entrepreneur"):
        job_entrepreneur_value = 1
        print("job_entrepreneur_value----------------->",job_entrepreneur_value)
    elif(job_json == "job_housemaid"):
        job_housemaid_value = 1
        print("job_housemaid_value----------------->",job_housemaid_value)
    elif(job_json == "job_management"):
        job_management_value = 1
        print("job_management_value----------------->",job_management_value)
    elif(job_json == "job_retired"):
        job_retired_value = 1
        print("job_retired_value----------------->",job_retired_value)
    elif(job_json == "job_self_employed"):
        job_self_employed_value = 1
        print("job_self_employed_value----------------->",job_self_employed_value)
    elif(job_json == "job_services"):
        job_services_value = 1
        print("job_services_value----------------->",job_services_value)
    elif(job_json == "job_student"):
        job_student_value = 1
        print("job_student_value----------------->",job_student_value)
    elif(job_json == "job_technician"):
        job_technician_value = 1
        print("job_technician_value----------------->",job_technician_value)
    elif(job_json == "job_unemployed"):
        job_unemployed_value = 1
        print("job_unemployed_value----------------->",job_unemployed_value)
    elif(job_json == "job_unknown"):
        job_unknown_value = 1 
        print("job_unknown_value----------------->",job_unknown_value)
            
         
    url_test['job_admin'] = pd.DataFrame([job_admin_value])
    url_test['job_blue_collar'] = pd.DataFrame([job_blue_collar_value])
    url_test['job_entrepreneur'] = pd.DataFrame([job_entrepreneur_value])
    url_test['job_housemaid'] = pd.DataFrame([job_housemaid_value])
    url_test['job_management'] = pd.DataFrame([job_management_value])
    url_test['job_retired'] = pd.DataFrame([job_retired_value])
    url_test['job_self_employed'] = pd.DataFrame([job_self_employed_value])
    url_test['job_services'] = pd.DataFrame([job_services_value])
    url_test['job_student'] = pd.DataFrame([job_student_value])
    url_test['job_technician'] = pd.DataFrame([job_technician_value])
    url_test['job_unemployed'] = pd.DataFrame([job_unemployed_value])
    url_test['job_unknown'] = pd.DataFrame([job_unknown_value])
    
    marital_divorced_Value = 0
    marital_married_Value = 0
    marital_single_Value = 0
    
    marital_json = request.args.get('marital') 
    marital_json = 'marital_married'
    print('marital_json:', marital_json)
    if(marital_json == "marital_divorced"):
        marital_divorced_Value = 1 
        print('marital_divorced_Value:', marital_divorced_Value)
    elif(marital_json == "marital_married"):
        marital_married_Value = 1 
        print('marital_married_Value:', marital_married_Value)
    elif(marital_json == "marital_single"):
        marital_single_Value = 1
        print('marital_single_Value:', marital_single_Value)
               
    url_test['marital_divorced'] = pd.DataFrame([marital_divorced_Value])
    url_test['marital_married'] = pd.DataFrame([marital_married_Value])
    url_test['marital_single'] = pd.DataFrame([marital_single_Value])
    
    education_primary_Value = 0
    education_secondary_Value = 0
    education_tertiary_Value = 0
    education_unknown_Value = 0
    
    education_json = request.args.get('education') 
    education_json = 'education_secondary'
    print('education_json:', education_json)
    if(education_json == "education_primary"):
        education_primary_Value = 1 
        print('education_primary_Value:', education_primary_Value)
    elif(education_json == "education_secondary"):
        education_secondary_Value = 1 
        print('education_secondary_Value:', education_secondary_Value)
    elif(education_json == "education_tertiary"):
        education_tertiary_Value = 1 
        print('education_tertiary_Value:', education_tertiary_Value)
    elif(education_json == "education_unknown"):
        education_unknown_Value = 1 
        print('education_unknown_Value:', education_unknown_Value)
    
    url_test['education_primary'] = pd.DataFrame([education_primary_Value])
    url_test['education_secondary'] = pd.DataFrame([education_secondary_Value])
    url_test['education_tertiary'] = pd.DataFrame([education_tertiary_Value])
    url_test['education_unknown'] = pd.DataFrame([education_unknown_Value])
    contact_cellular_Value = 0
    contact_telephone_Value = 0
    contact_unknown_Value = 0
    
    contact_json = request.args.get('contact')
    contact_json = 'contact_unknown'
    print('contact_json:', contact_json)
    if(contact_json == "contact_cellular"):
        contact_cellular_Value = 1 
        print('contact_cellular_Value:', contact_cellular_Value)
    elif(contact_json == "contact_telephone"):
        contact_telephone_Value = 1 
        print('contact_telephone_Value:', contact_telephone_Value)
    elif(contact_json == "contact_unknown"):
        contact_unknown_Value = 1 
        print('contact_unknown_Value:', contact_unknown_Value)
        
    url_test['contact_cellular'] = pd.DataFrame([contact_cellular_Value])
    url_test['contact_telephone'] = pd.DataFrame([contact_telephone_Value])
    url_test['contact_unknown'] = pd.DataFrame([contact_unknown_Value])   
    
    balance_json = request.args.get('balance') 
    balance_json = 869
    print('balance_json:---------------->', balance_json)
    url_test['balance'] = pd.DataFrame([balance_json])
    
    campaign_json = request.args.get('campaign') 
    campaign_json = 1
    print('campaign_json:---------------->', campaign_json)
    url_test['campaign'] = pd.DataFrame([campaign_json])
    
    duration_json = request.args.get('duration') 
    duration_json = 1677
    print('duration_json:---------------->', duration_json)
    url_test['duration'] = pd.DataFrame([duration_json])
    
    day_json = request.args.get('day')
    day_json = 6
    print('day_json:---------------->', day_json)
    url_test['day'] = pd.DataFrame([day_json])
    
    pdays_json = request.args.get('pdays') 
    pdays_json = -1
    print('pdays_json:---------------->', pdays_json)
    url_test['pdays'] = pd.DataFrame([pdays_json])
    
    previous_json = request.args.get('previous') 
    previous_json = 0
    print('previous_json:---------------->', previous_json)
    url_test['previous'] = pd.DataFrame([previous_json])
    
    poutcome_failure_Value = 0
    poutcome_other_Value = 0
    poutcome_success_Value = 0
    poutcome_unknown_Value = 0
    
    poutcome_json = request.args.get('poutcome') 
    poutcome_json = 'poutcome_unknown'
    print('poutcome_json:', poutcome_json)
    if(poutcome_json == "poutcome_failure"):
        poutcome_failure_Value = 1 
        print('poutcome_failure_Value:', poutcome_failure_Value)
    elif(poutcome_json == "poutcome_other"):
        poutcome_other_Value = 1 
        print('poutcome_other_Value:', poutcome_other_Value)
    elif(poutcome_json == "poutcome_success"):
        poutcome_success_Value = 1 
        print('poutcome_success_Value:', poutcome_success_Value)
    elif(poutcome_json == "poutcome_unknown"):
        poutcome_unknown_Value = 1 
        print('poutcome_unknown_Value:', poutcome_unknown_Value)
        
    url_test['poutcome_failure'] = pd.DataFrame([poutcome_failure_Value])
    url_test['poutcome_other'] = pd.DataFrame([poutcome_other_Value])
    url_test['poutcome_success'] = pd.DataFrame([poutcome_success_Value])   
    url_test['poutcome_unknown'] = pd.DataFrame([poutcome_unknown_Value]) 
        
    #Predictor Variables  
    print("LIST VALUE ----------->",list(url_test.columns.values))
    x = url_test[['job_admin','job_blue_collar','job_entrepreneur','job_housemaid',
                'job_management','job_retired','job_self_employed','job_services',
                'job_student','job_technician','job_unemployed','job_unknown',
                'marital_divorced','marital_married','marital_single',
                'education_primary','education_secondary','education_tertiary',
                'education_unknown','contact_cellular','contact_telephone',
                'contact_unknown','age','loan','housing','default',
                'balance','campaign','duration','day','pdays','previous',
                'poutcome_failure','poutcome_other','poutcome_success',
                'poutcome_unknown']] 
    print('query_df_array::----->',url_test)
    prediction = log_estimator.predict(x)
    print('Predicted Value;--->',prediction)
    if(prediction == 1):
        output = "Yes"
    else:
        output = "No"
   # return jsonify (pd.Series(prediction).to_json(orient='values'))
    return render_template('index.html', prediction_text='Predicted Term  Probablity is :---> {}'.format(output))
    
if __name__ == '__main__':
    app.run(debug=True)

