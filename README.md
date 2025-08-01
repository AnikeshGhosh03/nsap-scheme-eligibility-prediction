📌 Project Overview: 
The National Social Assistance Programme (NSAP) is a flagship social welfare initiative by the Government of India that provides financial support to the elderly, widows, and persons with disabilities belonging to Below Poverty Line (BPL) households.
This project leverages machine learning to predict the most appropriate NSAP scheme for an applicant using their demographic and socio-economic data, thereby automating the eligibility classification process and reducing manual errors.
Key Highlights:

Multi-class classification of NSAP schemes using AI_KOSH dataset
Achieved 97% accuracy with a Decision Tree Classifier
Model evaluated using confusion matrix and classification report
Prepared for deployment via IBM Cloud / Streamlit

⚙️ Tech Stack: 
Python 3.8+
Jupyter Notebook / IBM Cloud
Libraries:
  pandas, numpy
  scikit-learn
  matplotlib, seaborn
  joblib

📊 Model Workflow: 
Data Collection
AI_KOSH NSAP dataset with state, district, caste, Aadhaar, and beneficiary data
Data Preprocessing
Handle missing values and encode categorical variables
Feature engineering for caste distribution and digital identity
Model Training
Decision Tree Classifier for multi-class prediction
Tuned using GridSearchCV for optimal accuracy
Evaluation
Achieved ~97% accuracy
Confusion matrix shows reliable multi-class performance
Deployment on IBM Cloud Lite

🚀 Future Scope: 
Integrate real-time applicant-level data for finer eligibility predictions
Deploy model on IBM Cloud with live dashboard for government agencies
Expand to other welfare schemes and add explainable AI (SHAP)
streamlit (optional for UI)

📚 References: 
AI-KOSH NSAP Dataset
National Social Assistance Programme
Scikit-learn Documentation
IBM Cloud

