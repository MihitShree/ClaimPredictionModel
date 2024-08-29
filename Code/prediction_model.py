from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd
import numpy as np

#IMPORTIN DATA
train_beneficiary_data = pd.read_csv('/Users/prabhavagrawal/Desktop/Prabhav/Lumiq/Data Science/Health Claims/Train_Beneficiarydata-1542865627584.csv')
test_beneficiary_data = pd.read_csv('/Users/prabhavagrawal/Desktop/Prabhav/Lumiq/Data Science/Health Claims/Test_Beneficiarydata-1542969243754.csv')

train_inpatient_data = pd.read_csv('/Users/prabhavagrawal/Desktop/Prabhav/Lumiq/Data Science/Health Claims/Train_Inpatientdata-1542865627584.csv')
test_inpatient_data = pd.read_csv('/Users/prabhavagrawal/Desktop/Prabhav/Lumiq/Data Science/Health Claims/Test_Inpatientdata-1542969243754.csv')

"""
train_outpatient_data = pd.read_csv('/Users/prabhavagrawal/Desktop/Prabhav/Lumiq/Data Science/Health Claims/Train_Outpatientdata-1542865627584.csv')
test_outpatient_data = pd.read_csv('/Users/prabhavagrawal/Desktop/Prabhav/Lumiq/Data Science/Health Claims/Test_Outpatientdata-1542865627584.csv')

train_beneficiary_data = train_beneficiary_data.merge(train_inpatient_data[['BeneID', 'Provider','AdmissionDt' , 'DischargeDt' , 'DiagnosisGroupCode',
'ClmDiagnosisCode_1', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10','ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4'
,'ClmProcedureCode_5','ClmProcedureCode_6']], on='BeneID', how='left')

test_beneficiary_data = test_beneficiary_data.merge(test_inpatient_data[['BeneID', 'Provider','AdmissionDt' , 'DischargeDt' , 'DiagnosisGroupCode',
'ClmDiagnosisCode_1', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10','ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4'
,'ClmProcedureCode_5','ClmProcedureCode_6']], on='BeneID', how='left')

"""
"""
train_beneficiary_data = train_beneficiary_data.merge(train_inpatient_data[['BeneID','AdmissionDt','DischargeDt']], on = 'BeneID', how = 'left' )
test_beneficiary_data = test_beneficiary_data.merge(test_inpatient_data[['BeneID','AdmissionDt','DischargeDt']], on = 'BeneID', how = 'left' )
train_beneficiary_data['AdmissionDt'] = pd.to_datetime(train_beneficiary_data['AdmissionDt'])
train_beneficiary_data['DischargeDt'] = pd.to_datetime(train_beneficiary_data['DischargeDt'])
test_beneficiary_data['AdmissionDt'] = pd.to_datetime(test_beneficiary_data['AdmissionDt'])
test_beneficiary_data['DischargeDt'] = pd.to_datetime(test_beneficiary_data['DischargeDt'])
train_beneficiary_data['DaysIP'] = (train_beneficiary_data['DischargeDt'] - train_beneficiary_data['AdmissionDt']).dt.days
test_beneficiary_data['DaysIP'] = (test_beneficiary_data['DischargeDt'] - test_beneficiary_data['AdmissionDt']).dt.days

train_beneficiary_data['DaysIP'].fillna(0, inplace=True)
test_beneficiary_data['DaysIP'].fillna(0, inplace=True)
"""
train_beneficiary_data['RenalDiseaseIndicator'] = train_beneficiary_data['RenalDiseaseIndicator'].replace({'Y': 1, 0: 0})
test_beneficiary_data['RenalDiseaseIndicator'] = test_beneficiary_data['RenalDiseaseIndicator'].replace({'Y': 1, 0: 0})

current_year = pd.Timestamp.now().year

train_beneficiary_data['DOB'] = pd.to_datetime(train_beneficiary_data['DOB'])
test_beneficiary_data['DOB'] = pd.to_datetime(test_beneficiary_data['DOB'])
train_beneficiary_data['Age'] = current_year - train_beneficiary_data['DOB'].dt.year
test_beneficiary_data['Age'] = current_year - test_beneficiary_data['DOB'].dt.year

"""


Label_Encoder = LabelEncoder()
for col in ('DiagnosisGroupCode',
'ClmDiagnosisCode_1', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10','ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4'
,'ClmProcedureCode_5','ClmProcedureCode_6'):
    train_beneficiary_data[col] = train_beneficiary_data[col].astype('category')
    test_beneficiary_data[col] = test_beneficiary_data[col].astype('category')
    
    train_beneficiary_data[col] = Label_Encoder.fit_transform(train_beneficiary_data[col])
    test_beneficiary_data[col] = Label_Encoder.fit_transform(test_beneficiary_data[col])
  
#Tried adding more parameters so as to accuracy by picking them from the IP data but the values in Diagnosis code and Provider and Procedure codes are different so even if i try to encode them and make categories it will not work 

x_train = train_beneficiary_data[['DaysIP', 'DiagnosisGroupCode',
'ClmDiagnosisCode_1', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10','ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4'
,'ClmProcedureCode_5','ClmProcedureCode_6','County','Gender','Race','RenalDiseaseIndicator','Age', 'ChronicCond_Alzheimer','ChronicCond_Heartfailure','ChronicCond_KidneyDisease','ChronicCond_Cancer','ChronicCond_ObstrPulmonary','ChronicCond_Depression','ChronicCond_Diabetes','ChronicCond_IschemicHeart','ChronicCond_Osteoporasis','ChronicCond_rheumatoidarthritis','ChronicCond_stroke']]

x_test = test_beneficiary_data[['DaysIP' , 'DiagnosisGroupCode',
'ClmDiagnosisCode_1', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10','ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4'
,'ClmProcedureCode_5','ClmProcedureCode_6','County','Gender','Race','RenalDiseaseIndicator','Age','ChronicCond_Alzheimer','ChronicCond_Heartfailure','ChronicCond_KidneyDisease','ChronicCond_Cancer','ChronicCond_ObstrPulmonary','ChronicCond_Depression','ChronicCond_Diabetes','ChronicCond_IschemicHeart','ChronicCond_Osteoporasis','ChronicCond_rheumatoidarthritis','ChronicCond_stroke']]
"""

x_train = train_beneficiary_data[['County','Gender','Race','RenalDiseaseIndicator','Age','ChronicCond_Alzheimer','ChronicCond_Heartfailure','ChronicCond_KidneyDisease','ChronicCond_Cancer','ChronicCond_ObstrPulmonary','ChronicCond_Depression','ChronicCond_Diabetes','ChronicCond_IschemicHeart','ChronicCond_Osteoporasis','ChronicCond_rheumatoidarthritis','ChronicCond_stroke']]
x_test = test_beneficiary_data[['County','Gender','Race','RenalDiseaseIndicator','Age','ChronicCond_Alzheimer','ChronicCond_Heartfailure','ChronicCond_KidneyDisease','ChronicCond_Cancer','ChronicCond_ObstrPulmonary','ChronicCond_Depression','ChronicCond_Diabetes','ChronicCond_IschemicHeart','ChronicCond_Osteoporasis','ChronicCond_rheumatoidarthritis','ChronicCond_stroke']]

y_train_ip =  train_beneficiary_data[['IPAnnualReimbursementAmt']]
y_train_op =  train_beneficiary_data[['OPAnnualReimbursementAmt']]
y_train_ip = y_train_ip.values.ravel()
y_train_op = y_train_op.values.ravel()

y_test_ip =  test_beneficiary_data[['IPAnnualReimbursementAmt']]
y_test_op =  test_beneficiary_data[['OPAnnualReimbursementAmt']]
y_test_ip = y_test_ip.values.ravel()
y_test_op = y_test_op.values.ravel()

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(x_train, y_train_ip)
y_pred_ip = rf_model.predict(x_test)
rf_model.fit(x_train, y_train_op)
y_pred_op = rf_model.predict(x_test)


r2sip = r2_score(y_test_ip,y_pred_ip)
r2sop = r2_score(y_test_op,y_pred_op)

print("R2 Score for IP", r2sip)
print("R2 Score for OP", r2sop)

# Create a DataFrame with test y values and predicted y values
resultsip = pd.DataFrame({'Test Y': y_test_ip, 'Predicted Y': y_pred_ip})
resultsop = pd.DataFrame({'Test Y': y_test_op, 'Predicted Y': y_pred_op})



# Print the DataFrame
print("Inpatient \n",  resultsip)
print("Outpatient \n", resultsop)

import joblib

# Train the model
rf_model_ip = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model_ip.fit(x_train, y_train_ip)
joblib.dump(rf_model_ip, 'rf_model_ip.pkl')

rf_model_op = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model_op.fit(x_train, y_train_op)
joblib.dump(rf_model_op, 'rf_model_op.pkl')

print("Models have been serialized and saved.")



