#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='VBuqWf_MMPsvG0FHmqy9FyTa3tIkPOfv86_3kuGss5tD',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/identity/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.direct.us-south.cloud-object-storage.appdomain.cloud')

bucket = 'nsapmlpredictiveassistanceforsoci-donotdelete-pr-o384ffqh8r9jgv'
object_key = 'nsapallschemes.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_1 = pd.read_csv(body)
df_1.head(10)


# In[7]:


df_1.info()
df_1.describe(include='all')


# In[8]:


print("Missing values:\n", df_1.isnull().sum())
print("\nDuplicate rows:", df_1.duplicated().sum())


# In[9]:


df_1 = df_1.drop_duplicates()


# In[10]:


target = 'schemecode'  # change this if needed

numeric_cols = df_1.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df_1.select_dtypes(exclude=["number"]).drop(columns=[target], errors="ignore").columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib


# In[12]:


X = df_1.drop(columns=[target])
y = df_1[target].astype(str)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# In[13]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_cols = [
    'lgdstatecode', 'lgddistrictcode', 'totalbeneficiaries', 'totalmale', 'totalfemale', 'totaltransgender', 'totalsc', 'totalst', 'totalgen', 'totalobc', 'totalaadhaar', 'totalmobilenumber'
]

categorical_cols = ['finyear', 'statename', 'districtname']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Fit and transform again
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)


# In[14]:


import os, json
import numpy as np
import joblib

os.makedirs("artifacts", exist_ok=True)

np.save("artifacts/X_train.npy", X_train_prep)
np.save("artifacts/X_test.npy", X_test_prep)
np.save("artifacts/y_train.npy", y_train.to_numpy())
np.save("artifacts/y_test.npy", y_test.to_numpy())

joblib.dump(preprocessor, "artifacts/preprocessor.pkl")

ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
ohe_features = ohe.get_feature_names_out(categorical_cols).tolist()
all_features = numeric_cols + ohe_features

with open("artifacts/feature_names_after_ohe.json", "w") as f:
    json.dump(all_features, f, indent=2)

class_names = sorted(y.unique())
with open("artifacts/class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

print("Rebuilt and saved preprocessing artifacts!")


# In[15]:


print(os.listdir("artifacts"))


# In[16]:


import numpy as np
import joblib
import json

X_train = np.load("artifacts/X_train.npy")
X_test = np.load("artifacts/X_test.npy")
y_train = np.load("artifacts/y_train.npy", allow_pickle=True)
y_test = np.load("artifacts/y_test.npy", allow_pickle=True)

with open("artifacts/class_names.json") as f:
    class_names = json.load(f)


# In[17]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, "artifacts/decision_tree_model.pkl")
print(" Model trained and saved as decision_tree_model.pkl")


# In[18]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=class_names)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[19]:


import matplotlib.pyplot as plt
import numpy as np
import json

# Load features
with open("artifacts/feature_names_after_ohe.json") as f:
    feature_names = json.load(f)

# Get importance scores
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Top 15 features
plt.figure(figsize=(10, 6))
plt.title("Top Feature Importances (Decision Tree)")
plt.bar(range(15), importances[indices[:15]], align="center")
plt.xticks(range(15), [feature_names[i] for i in indices[:15]], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[20]:


import pandas as pd
import joblib

# Load model + preprocessor
model = joblib.load("artifacts/decision_tree_model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

# Must include all original columns used in training
sample_applicant = {
    "finyear": "2022-23",
    "lgdstatecode": 10,             # dummy value, required for shape
    "statename": "Odisha",
    "lgddistrictcode": 302,         # dummy value, required for shape
    "districtname": "Khurda",
    "schemecode": "IGNOAPS",        # will be ignored — this is target
    "totalbeneficiaries": 3,
    "totalmale": 1,
    "totalfemale": 2,
    "totaltransgender": 0,
    "totalsc": 0,
    "totalst": 1,
    "totalgen": 2,
    "totalobc": 0,
    "totalaadhaar": 3,
    "totalmobilenumber": 2
}

# Create DataFrame and drop schemecode (target) column
input_df = pd.DataFrame([sample_applicant]).drop(columns=["schemecode"])

# Transform and predict
X_input = preprocessor.transform(input_df)
prediction = model.predict(X_input)[0]

print("✅ Predicted NSAP Scheme Code:", prediction)


# In[21]:



# In[ ]:




