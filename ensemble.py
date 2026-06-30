import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop("customerID",axis=1,inplace=True)
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"]=df["Churn"].map({"No":0,"Yes":1})
df=pd.get_dummies(df,drop_first=True)

X=df.drop("Churn",axis=1)
y=df["Churn"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

models={
"Decision Tree":DecisionTreeClassifier(random_state=42),
"Random Forest":RandomForestClassifier(random_state=42),
"Extra Trees":ExtraTreesClassifier(random_state=42),
"AdaBoost":AdaBoostClassifier(random_state=42),
"Gradient Boosting":GradientBoostingClassifier(random_state=42),
"XGBoost":XGBClassifier(random_state=42,eval_metric="logloss"),
"LightGBM":LGBMClassifier(random_state=42,verbose=-1)
}

results=[]

for name,model in models.items():
    print("="*60)
    print(name)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    acc=accuracy_score(y_test,pred)
    pre=precision_score(y_test,pred)
    rec=recall_score(y_test,pred)
    f1=f1_score(y_test,pred)
    results.append([name,acc,pre,rec,f1])
    print("Accuracy:",acc)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))

result_df=pd.DataFrame(results,columns=["Model","Accuracy","Precision","Recall","F1"])
print(result_df.sort_values("Accuracy",ascending=False))

plt.figure(figsize=(10,5))
plt.bar(result_df["Model"],result_df["Accuracy"])
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()
