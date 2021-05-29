import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

health= pd.read_csv("E:\Citihealth\data.csv")
health.head()

health.info()

health.isnull().sum() #checking for misisng values
health.Value.value_counts() #checking the values present in each columns

health.corr()

health.Indicator.value_counts()

health.Year.value_counts()

health["Race/ Ethnicity"].value_counts()

health["Indicator Category"].value_counts()
health.Place.value_counts()

health["Value"]= health["Value"].fillna(health["Value"].mean())
for column in ["Source","BCHC Requested Methodology"]:
    health[column].fillna(health[column].mode()[0], inplace= True)

health= health.drop(columns=["Methods","Notes"])
health.head()

health.isnull().sum()

# Data Visualization
sns.countplot(y= health["Indicator Category"])
plt.show()

sns.countplot(y= health["Year"])
plt.show()


sns.countplot(y= health["Race/ Ethnicity"])
#plt.figure(figsize=(100,100))
plt.show()


groupvalues=health.groupby('Indicator Category').sum().reset_index()
groupvalues

plt.figure(figsize=(15,10))
sns.set(style="darkgrid")
g = sns.barplot(groupvalues["Indicator Category"],groupvalues['Value'])
for index, row in groupvalues.iterrows():
    g.text(row.name,row.Value, round(row.Value,2), color='black', ha="center")
    g.set_xticklabels(g.get_xticklabels(),rotation= 90, fontsize= 18)
    g.set_xlabel("Indicator Category", fontsize=18)
plt.show()

sns.countplot(health["Gender"])
plt.show()

plt.figure(figsize = (10,5))
labels = 'Male', 'Female', 'Both'
sizes = np.array([12.4, 17.9, 69.6])
colors = ['yellowgreen', 'violet', 'yellow']

p, tx, autotexts = plt.pie(sizes, labels=labels, colors=colors,
        autopct="", shadow=True)

for i, a in enumerate(autotexts):
    a.set_text("{}".format(sizes[i]))

plt.axis('equal')
plt.show()


plt.figure(figsize=(40,10))
sns.set(style="darkgrid")
groupvalues1=health.groupby('Place').sum().reset_index()
g = sns.barplot(groupvalues1['Place'],groupvalues1['Value'])
for index, row in groupvalues.iterrows():
    g.text(row.name,row.Value, round(row.Value,2), color='black', ha="center")
    g.set_xticklabels(g.get_xticklabels(), rotation= 90, fontsize= 25)
    g.set_xlabel("Place")

plt.show()

health['Place'].value_counts()
health['State']=health['Place'].apply(lambda x: x.split(",")).str[1]
plt.figure(figsize=(15,10))
cp=sns.countplot(x=health['State'],data=health,order = health['State'].value_counts().index)
cp.set_xticklabels(cp.get_xticklabels(),rotation=90,fontsize=18)
cp.set_xlabel('State',fontsize=15)
cp.set_ylabel('Count',fontsize=10)
plt.show()


plt.figure(figsize = (15,10))
health.State.value_counts().plot(kind="pie")
plt.show()

sns.set(style="darkgrid")
ax = sns.countplot(y='Indicator Category', hue="Gender",data=health)
plt.show()

plt.figure(figsize=(40,10))
sns.set(style="darkgrid")
groupvalues2=health.groupby('Indicator').sum().reset_index()
g = sns.barplot(groupvalues2['Indicator'],groupvalues2['Value'])
for index, row in groupvalues.iterrows():
    g.text(row.name,row.Value, round(row.Value,2), color='black', ha="center")
    g.set_xticklabels(g.get_xticklabels(),rotation= 90, fontsize= 25)
    g.set_xlabel("Indicator")

plt.show()


plt.figure(figsize=(25,12))
cp=sns.countplot(x=health['Indicator'],data=health,hue=health['State'],order = health['Indicator'].value_counts().index)
cp.set_xticklabels(cp.get_xticklabels(),rotation=90,fontsize=18)
cp.set_xlabel('Indicator',fontsize=15)
cp.set_ylabel('Count',fontsize=18)


health_dummies= pd.get_dummies(health)
X= health_dummies.drop(columns= "Value")
Y= health_dummies["Value"]
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
train_x, val_x, train_y, val_y= train_test_split(X,Y, test_size= 0.4, random_state= 100)
lr= LinearRegression()
lr.fit(train_x,train_y)
train_pred= lr.predict(train_x)
test_pred= lr.predict(val_x)

print("train_r2:", r2_score(train_pred,train_y))