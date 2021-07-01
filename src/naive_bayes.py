import numpy as np
from main import load_data 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

def naive_bayes():
    X_train, X_test, y_train, y_test = load_data('./wifi_localization.txt')

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    print(f"Correct: {((y_test == y_pred).sum() / len(y_pred))*100:.2f}%")
    print(f"Incorrect: {((y_test != y_pred).sum() / len(y_pred))*100:.2f}%")

naive_bayes()