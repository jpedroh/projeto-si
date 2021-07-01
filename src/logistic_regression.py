import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from main import load_data

def logistic_regression():
    X_train, X_test, y_train, y_test = load_data('./wifi_localization.txt')

    for solver in ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']:
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for max_iter in range(0, 20):
            gnb = LogisticRegression(max_iter=max_iter, solver=solver, l1_ratio=0.5)

            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            precision_scores.append(precision_score(y_test, y_pred, average='micro'))
            recall_scores.append(recall_score(y_test, y_pred, average='micro'))
            f1_scores.append(f1_score(y_test, y_pred, average='micro'))

        plt.title('Relação Iterações vs Precisão (' + solver + ')')
        plt.xlabel('Total de iteracoes')
        plt.ylabel('Precisão')
        plt.plot(range(0,20), precision_scores, marker="*")
        plt.savefig(solver + '.png')
        plt.figure()

logistic_regression()
