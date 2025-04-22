## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

![Screenshot 2025-04-22 105027](https://github.com/user-attachments/assets/e36179c0-7c7d-470d-8a03-4d9292eb33e9)


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![Screenshot 2025-04-22 105226](https://github.com/user-attachments/assets/c258c7fa-af08-43e4-bd6a-8a6b38035851)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![Screenshot 2025-04-22 105428](https://github.com/user-attachments/assets/31f9c8b4-49b0-47f7-abad-5e56b5485d53)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![Screenshot 2025-04-22 105548](https://github.com/user-attachments/assets/9fa30969-7bfc-4199-b3de-c0c956b808eb)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```

![Screenshot 2025-04-22 105655](https://github.com/user-attachments/assets/b4f6cb47-d640-480a-b997-b1519e6705e0)

```
pd.get_dummies(df2,columns=["nom_0"])
```

![Screenshot 2025-04-22 105825](https://github.com/user-attachments/assets/0933715a-08a9-419d-a947-3913c46b7a4a)

```
pip install --upgrade category_encoders
```

![Screenshot 2025-04-22 105956](https://github.com/user-attachments/assets/bdc7c7fa-86ea-4ef1-aef1-5177f29bf042)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

![Screenshot 2025-04-22 110227](https://github.com/user-attachments/assets/8757b218-e688-4312-a863-c8be93d725bc)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

![Screenshot 2025-04-22 110306](https://github.com/user-attachments/assets/1c79ef8b-e254-4f52-88b0-656db4921659)

```
dfb=pd.concat([df,nd],axis=1)
dfb
```

![Screenshot 2025-04-22 110429](https://github.com/user-attachments/assets/156ea7aa-551f-4362-962d-c471f7252ed6)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![Screenshot 2025-04-22 110533](https://github.com/user-attachments/assets/d816f9f0-fc08-4ac1-8e06-7f48960c087c)


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

![Screenshot 2025-04-22 110621](https://github.com/user-attachments/assets/9d8b5153-6b8c-4344-a981-66bef4e3021a)


```
df.skew()
```

![Screenshot 2025-04-22 110713](https://github.com/user-attachments/assets/eefc73bb-50d6-4292-9cad-4802a2878341)

```
np.log(df["Highly Positive Skew"])
```

![Screenshot 2025-04-22 110803](https://github.com/user-attachments/assets/840a8403-aefd-4db4-9a70-9920fa01609f)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![Screenshot 2025-04-22 110859](https://github.com/user-attachments/assets/a307363c-f9fa-482d-87ec-5328de7fafa4)

```
np.sqrt(df["Highly Positive Skew"])
```

![Screenshot 2025-04-22 110938](https://github.com/user-attachments/assets/d9ff6929-47f9-401a-ac86-1b6cc2e55922)

```
np.square(df["Highly Positive Skew"])
```

![Screenshot 2025-04-22 111123](https://github.com/user-attachments/assets/327b5eda-c650-406d-82f6-e4b8ca65cc81)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![Screenshot 2025-04-22 111223](https://github.com/user-attachments/assets/df4df308-0cd5-48d0-a184-a1e22dfba679)

```
df.skew()
```

![Screenshot 2025-04-22 111301](https://github.com/user-attachments/assets/c4f9ba0f-d908-4d36-9778-dcb36d50a081)

```
df["Highly Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![Screenshot 2025-04-22 111342](https://github.com/user-attachments/assets/b0941032-2e70-43ef-b1e2-839d57fe34c5)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![Screenshot 2025-04-22 111501](https://github.com/user-attachments/assets/b3122aee-d433-4233-87ce-606813692c1e)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![Screenshot 2025-04-22 111615](https://github.com/user-attachments/assets/81bd15f4-7b2a-4790-8f22-05612fcee1f4)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![Screenshot 2025-04-22 111716](https://github.com/user-attachments/assets/33552eb6-ee49-4d0c-8d47-62deca2632bd)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```

```
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
```

```
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![Screenshot 2025-04-22 111908](https://github.com/user-attachments/assets/6e6ff578-e1f0-4175-8eb5-28da1a0ab4cc)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![Screenshot 2025-04-22 112002](https://github.com/user-attachments/assets/8367a26e-f76d-4430-aaea-7606014c35d7)

```
dt=pd.read_csv("titanic_dataset.csv")
dt
```

![Screenshot 2025-04-22 112105](https://github.com/user-attachments/assets/0b176bcc-8519-40cf-b6f1-9494af709f4e)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![Screenshot 2025-04-22 112208](https://github.com/user-attachments/assets/067f66eb-aec0-40f8-aa4e-060cec7d5d2b)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![Screenshot 2025-04-22 112259](https://github.com/user-attachments/assets/9e0b7fbb-64d2-48cd-8a6e-2987c213a0b7)


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.

       
