from main import load_data 
from sklearn.tree import DecisionTreeClassifier

def decision_tree():
    X_train, X_test, y_train, y_test = load_data('./wifi_localization.txt')

    dtc = DecisionTreeClassifier()
    y_pred = dtc.fit(X_train, y_train).predict(X_test)

    print(f"Correct: {((y_test == y_pred).sum() / len(y_pred))*100:.2f}%")
    print(f"Incorrect: {((y_test != y_pred).sum() / len(y_pred))*100:.2f}%")

decision_tree()