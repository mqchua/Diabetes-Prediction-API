import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('diabetes.csv')

df_cols = df.drop('Outcome', axis=1)
df_target = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    df_cols, df_target, test_size=0.3, random_state=2)

lr = LogisticRegression()
lr.fit(X_train, y_train)

pickle.dump(lr, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
