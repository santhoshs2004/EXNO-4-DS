# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT :
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/b78ffaf3-f836-4579-a9df-e12d32ba0c9f)
```
df.head()
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/d580afdc-c7f9-4314-8a34-a721edcd0c8a)
```
df.dropna()
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/7297741e-b89a-4bf9-8b4b-d08fbc49ad07)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/f7e2222c-3c57-4961-a1ff-e8af2a29ec14)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/06792929-7c03-46a0-a3a3-79a0ebe41e0f)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/d03d1951-c2fd-4720-9641-68fcdd4fcdea)
```
from sklearn.preprocessing import Normalizer
scale=Normalizer()
df[['Height','Weight']]=scale.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/572cd77f-a9ef-4ef8-b6d5-6fe8ab50ad6a)
```
from sklearn.preprocessing import MaxAbsScaler
scalen=MaxAbsScaler()
df[['Height','Weight']]=scalen.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/1fc8f3c6-9ef2-4e5a-bede-db07045e9769)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/40bbdf66-e0d2-4daa-941d-893e5818b550)
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/c52801f5-4321-4ead-9f2f-b32d1c775a1b)
```
data.isnull().sum()
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/79a2ebcd-13c4-4e4e-b5e5-68a974467e75)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/012a48b7-fbed-4ebf-a853-01b0b0f81119)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/009afd1c-ebd3-459f-b701-ca0a84a583fa)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/13b85a6f-83d9-449b-a7b0-179575d1b3f6)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/c93ee8b1-2929-433d-b5ab-402e9ac4b644)
```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/91f1e536-1bb4-46f9-9e3e-237617972aa5)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/94198e3a-1c93-4869-b630-8f64a61df10d)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/78ee9aae-c749-43c2-95f7-6e3efda4dad3)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/526ab729-df5d-43cd-a7de-43a67e2f6682)
```
x=new_data[features].values
x
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/d10d5efc-5e1f-48fe-ba98-e4b890424a6c)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/2e7c4571-880d-485c-9a79-7e68e8655d51)
```
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/a982d92a-0e4d-41c9-b3e6-e8d31241d000)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/0bfde99d-e27c-4396-9e53-5d2945c68621)
```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/99f9e88c-185b-46b1-84d8-8535cb72529d)
```
data.shape
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/1f15c920-b9d9-4334-b765-ee2f7856b194)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/8da0072b-14c7-4267-a8d4-33fd1936f467)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/a08f594f-c667-4a2e-8bb1-5cd464ba8ac7)
```
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/d271ed52-cc58-45b5-8323-442fe0fcbe70)
```
chi2,p, _, _ =chi2_contingency(contigency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/23005529/EXNO-4-DS/assets/139842207/dc43f2e5-3115-4ea7-8ccb-15d0d6f3e6a1)

# RESULT:

Thus perform Feature Scaling and Feature Selection process and save the data to a file successfully.
