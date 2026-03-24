# EXNO:2
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
<img width="930" height="327" alt="311642720-1d9eff83-45dd-45f6-8e05-a60443a28c93" src="https://github.com/user-attachments/assets/7f6ae866-7ab3-4135-ac11-5f1e613a1fb3" />


```
dt.info()
```
<img width="432" height="420" alt="311643075-9d3a9f34-afc7-4575-9700-4a3a9d06bcf2" src="https://github.com/user-attachments/assets/e06e2606-843b-4483-84ec-4ef4a2d5fc24" />


```
dt.shape
```
<img width="172" height="57" alt="311643348-2e552cda-c340-4c14-81bb-ce0a638212c6" src="https://github.com/user-attachments/assets/ceaac3ed-3565-490e-b485-d4ef38a01899" />


```
dt.set_index("PassengerId",inplace=True)
dt.describe()
```

<img width="771" height="370" alt="311643632-2e3db065-da65-4184-81e9-09b195e0b91d" src="https://github.com/user-attachments/assets/fb877fdf-bbec-4468-a803-3960dd0b03e9" />

```
dt.nunique()
```
<img width="248" height="277" alt="311644018-97151d1c-ef1a-44df-885a-15f77cade99a" src="https://github.com/user-attachments/assets/3a2777e8-92e6-4a2a-81c6-f05bddfe3562" />


```
dt["Survived"].value_counts()
```
<img width="442" height="117" alt="311644271-a547fdac-8e3f-43f2-9932-fa2c2a265134" src="https://github.com/user-attachments/assets/f0652ab8-30ff-4fb6-9e7c-a4be9e6ea0ff" />

```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```

<img width="427" height="92" alt="311644581-8eced09f-2a18-4e3f-bd13-243701048fc5" src="https://github.com/user-attachments/assets/28b6809c-e98a-458a-8b7f-72ef81bf77e5" />


```
sns.countplot(data=dt,x="Survived")
```
<img width="931" height="580" alt="311644875-34adfa76-c868-440d-b0c6-d97ebe93e351" src="https://github.com/user-attachments/assets/8f22fd44-a535-4512-b172-69f99c59dd7e" />


```
dt
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/ff6a85fa-f2ce-4769-93c8-b596fcaaad17)
```
dt.Pclass.unique()
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/58fef3e0-6e22-489a-a2b9-85fc559daffa)
```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/8bb0fc68-5f7a-4005-80c6-b4706fd4cd3b)
```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/9af5b685-0281-43c4-bcda-dff99e437f3c)
```
sns.catplot(x='Survived',hue="Gender",data=dt,kind='count')
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/8555cc95-07c9-4436-a2a7-f2eb918d171d)
```
dt.boxplot(column="Age",by="Survived")
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/472e3846-e3dc-4391-813e-cd7d2eaccaf9)
```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/2d4244b8-65d1-46f3-a848-c4e5c57928f5)
```
sns.jointplot(x="Age",y="Fare",data=dt)
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/25fae9cf-16d8-46b0-b427-b9bd7b47f027)
```
fig,ax1=plt.subplots(figsize=(8,5))
sns.boxplot(ax=ax1,x="Pclass",y="Age",hue="Gender",data=dt)
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/839a046e-ac0d-483c-b09f-544f1608dde0)
```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/61b5359f-119c-41a3-bbc9-a301c5cc5196)
```
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/1a358d7b-b6ed-45b6-8266-18c249421323)
```
sns.pairplot(dt)
```
![image](https://github.com/Kalpanareshma/EXNO2DS/assets/122040453/a32237e1-689e-43d6-8765-d5deed872071)





















# RESULT
   Thus, the Exploratory Data Analysis on the given data set was performed successfully. 
