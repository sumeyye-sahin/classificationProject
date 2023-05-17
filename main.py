import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

dataset =pd.read_csv("data1.csv")

dataset=dataset.drop(["Unnamed: 0"], axis=1)
dataset=dataset.drop(["Unnamed: 3304"],axis=1)
dataset=dataset.drop(["Unnamed: 3305"], axis=1)
dataset=dataset.drop(["Unnamed: 3306"], axis=1)
dataset=dataset.drop(["Unnamed: 3307"], axis=1)

L = dataset[dataset.HandType == "left"]
R = dataset[dataset.HandType == "right"]

dataset.HandType = [1 if i == "left" else 0 for i in dataset.HandType]

y = dataset.drop(["HandType"], axis = 1)
x = dataset.HandType.values

x = (x - np.min(x)) / (np.max(x) - np.min(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

gaus = GaussianNB()
gaus.fit(y_train, x_train)
#print(gaus.fit(y_train, x_train))


x_pred=gaus.predict(y_test)
print(classification_report(x_test,x_pred))

print("Naive Bayes score: ",gaus.score(y_test, x_test))