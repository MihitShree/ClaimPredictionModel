#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load datasets
beneficiary_data = pd.read_csv('/Users/prabhavagrawal/Desktop/Prabhav/Lumiq/Data Science/Health Claims/Train_Beneficiarydata-1542865627584.csv')
inpatient_data = pd.read_csv('/Users/prabhavagrawal/Desktop/Prabhav/Lumiq/Data Science/Health Claims/Train_Inpatientdata-1542865627584.csv')
outpatient_data = pd.read_csv('/Users/prabhavagrawal/Desktop/Prabhav/Lumiq/Data Science/Health Claims/Train_Outpatientdata-1542865627584.csv')
claims_data = pd.read_csv('/Users/prabhavagrawal/Desktop/Prabhav/Lumiq/Data Science/Health Claims/Train-1542865627584.csv')

# Convert date columns to datetime
beneficiary_data['DOB'] = pd.to_datetime(beneficiary_data['DOB'])
beneficiary_data['DOD'] = pd.to_datetime(beneficiary_data['DOD'])
inpatient_data['ClaimStartDt'] = pd.to_datetime(inpatient_data['ClaimStartDt'])
inpatient_data['ClaimEndDt'] = pd.to_datetime(inpatient_data['ClaimEndDt'])
inpatient_data['AdmissionDt'] = pd.to_datetime(inpatient_data['AdmissionDt'])
inpatient_data['DischargeDt'] = pd.to_datetime(inpatient_data['DischargeDt'])
outpatient_data['ClaimStartDt'] = pd.to_datetime(outpatient_data['ClaimStartDt'])
outpatient_data['ClaimEndDt'] = pd.to_datetime(outpatient_data['ClaimEndDt'])
# Convert categorical columns to category dtype
beneficiary_data['Gender'] = beneficiary_data['Gender'].astype('category')
beneficiary_data['Race'] = beneficiary_data['Race'].astype('category')
beneficiary_data['State'] = beneficiary_data['State'].astype('category')
beneficiary_data['County'] = beneficiary_data['County'].astype('category')
inpatient_data['Provider'] = inpatient_data['Provider'].astype('category')
outpatient_data['Provider'] = outpatient_data['Provider'].astype('category')
claims_data['Provider'] = claims_data['Provider'].astype('category')
claims_data['PotentialFraud'] = claims_data['PotentialFraud'].astype('category')
# Handle missing values
beneficiary_data['DOD'].fillna(pd.NaT, inplace=True)
inpatient_data.fillna({'AttendingPhysician': 'Unknown', 'OperatingPhysician': 'Unknown', 'OtherPhysician': 'Unknown'}, inplace=True)
for col in inpatient_data.columns:
    if 'ClmDiagnosisCode' in col or 'ClmProcedureCode' in col:
        inpatient_data[col].fillna('Unknown', inplace=True)
outpatient_data.fillna({'AttendingPhysician': 'Unknown', 'OperatingPhysician': 'Unknown', 'OtherPhysician': 'Unknown'}, inplace=True)
for col in outpatient_data.columns:
    if 'ClmDiagnosisCode' in col or 'ClmProcedureCode' in col:
        outpatient_data[col].fillna('Unknown', inplace=True)
# Remove duplicates
beneficiary_data.drop_duplicates(inplace=True)
inpatient_data.drop_duplicates(inplace=True)
outpatient_data.drop_duplicates(inplace=True)
claims_data.drop_duplicates(inplace=True)



inpatient_data['Month-Year'] = inpatient_data['ClaimStartDt'].dt.to_period('M')
month_counts = inpatient_data['Month-Year'].value_counts().sort_index()
month_counts.plot(kind='bar', edgecolor='black')
plt.xlabel('Month-Year')
plt.ylabel('Number of Claims')
plt.grid(True)






# Show the plot
plt.show()
#Find Time
current_year = pd.Timestamp.now().year
beneficiary_data['Age'] = current_year - beneficiary_data['DOB'].dt.year
#Merging data
inpatient_data = inpatient_data.merge(beneficiary_data[['BeneID', 'Age', 'Gender', 'Race']], on='BeneID', how='left')
outpatient_data = outpatient_data.merge(beneficiary_data[['BeneID', 'Age', 'Gender', 'Race']], on='BeneID', how='left')
#Age Range Graph
bins = [0, 20, 40, 60, 80, 100, 120]
labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-120']
beneficiary_data['AgeRange'] = pd.cut(beneficiary_data['Age'], bins=bins, labels=labels, right=False)
inpatient_data['AgeRange'] = pd.cut(inpatient_data['Age'], bins=bins, labels=labels, right=False)
outpatient_data['AgeRange'] = pd.cut(outpatient_data['Age'], bins=bins, labels=labels, right=False)
age_reimbursement_inpatient = inpatient_data.groupby('AgeRange')['InscClaimAmtReimbursed'].sum()
age_reimbursement_outpatient = outpatient_data.groupby('AgeRange')['InscClaimAmtReimbursed'].sum()
plt.figure(figsize=(10, 5))
plt.bar(age_reimbursement_inpatient.index.astype(str), age_reimbursement_inpatient, alpha=0.6, label='Inpatient')
plt.bar(age_reimbursement_outpatient.index.astype(str), age_reimbursement_outpatient, alpha=0.6, label='Outpatient', bottom=age_reimbursement_inpatient)
plt.xlabel('Age Range')
plt.ylabel('Total Reimbursement Amount')
plt.title('Total Reimbursement Amount by Age Range')
plt.legend()
plt.show()
#Gender Wise Reimbursement Amount 
gender_reimbursement_inpatient = inpatient_data.groupby('Gender')['InscClaimAmtReimbursed'].sum()
gender_reimbursement_outpatient = outpatient_data.groupby('Gender')['InscClaimAmtReimbursed'].sum()
plt.figure(figsize=(10, 5))
plt.bar(gender_reimbursement_inpatient.index.astype(str), gender_reimbursement_inpatient, alpha=0.6, label='Inpatient')
plt.bar(gender_reimbursement_outpatient.index.astype(str), gender_reimbursement_outpatient, alpha=0.6, label='Outpatient', bottom=gender_reimbursement_inpatient)
plt.xlabel('Gender')
plt.ylabel('Total Reimbursement Amount')
plt.title('Total Reimbursement Amount by Gender')
plt.legend()
plt.show()
#Distribution of Beneficiary age 
plt.figure(figsize=(10, 5))
sns.histplot(beneficiary_data['Age'], bins=30, kde=True)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Beneficiary Ages')
plt.show()
# Top Diagonosis code
top_diagnosis_codes_inpatient = inpatient_data['ClmDiagnosisCode_1'].value_counts().head(10)
top_procedure_codes_inpatient = inpatient_data['ClmProcedureCode_1'].value_counts().head(10)
top_diagnosis_codes_outpatient = outpatient_data['ClmDiagnosisCode_1'].value_counts().head(10)
top_procedure_codes_outpatient = outpatient_data['ClmProcedureCode_1'].value_counts().head(10)
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# Inpatient Diagnosis Codes
sns.barplot(x=top_diagnosis_codes_inpatient.index, y=top_diagnosis_codes_inpatient.values, ax=axs[0])
axs[0].set_title('Top 10 Inpatient Diagnosis Codes')
axs[0].set_ylabel('Frequency')
axs[0].set_xlabel('Diagnosis Code')
# Outpatient Diagnosis Codes
sns.barplot(x=top_diagnosis_codes_outpatient.index, y=top_diagnosis_codes_outpatient.values, ax=axs[1])
axs[1].set_title('Top 10 Outpatient Diagnosis Codes')
axs[1].set_ylabel('Frequency')
axs[1].set_xlabel('Diagnosis Code')
plt.tight_layout()
plt.show()
inpatient_data['Month-Year'] = inpatient_data['ClaimStartDt'].dt.to_period('M')
outpatient_data['Month-Year'] = outpatient_data['ClaimStartDt'].dt.to_period('M')
inpatient_claims_over_time = inpatient_data['Month-Year'].value_counts().sort_index()
outpatient_claims_over_time = outpatient_data['Month-Year'].value_counts().sort_index()
plt.figure(figsize=(15, 5))
plt.plot(inpatient_claims_over_time.index.astype(str), inpatient_claims_over_time, label='Inpatient', color='red')
plt.plot(outpatient_claims_over_time.index.astype(str), outpatient_claims_over_time, label='Outpatient', color='blue')
plt.xlabel('Month-Year')
plt.ylabel('Number of Claims')
plt.title('Number of Claims Over Time')
plt.legend()
plt.show()






