from main import load_data
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

def decision_tree():
    X_train, X_test, y_train, y_test = load_data('./wifi_localization.txt')

    for criterion in ["gini", "entropy"]:
        precision_scores = []
        for max_depth in range(1, 15):
            dtc = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
            y_pred = dtc.fit(X_train, y_train).predict(X_test)
            precision_scores.append(precision_score(y_test, y_pred, average='micro'))

        plt.title("[" + criterion + "] Relação Iterações vs Precisão (max_depth)")
        plt.xlabel('Total de iteracoes')
        plt.ylabel('Precisão')
        plt.xticks(range(1, 15))
        plt.plot(range(1, 15), precision_scores, marker="*")
        plt.savefig(criterion + "-" + "d" + '.png')
        plt.figure()
        
        precision_scores = []
        for max_leaf_nodes in range(2, 20):
            dtc = DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=max_leaf_nodes, random_state=42)
            y_pred = dtc.fit(X_train, y_train).predict(X_test)
            precision_scores.append(precision_score(y_test, y_pred, average='micro'))

        plt.title("[" + criterion + "] Relação Iterações vs Precisão (max_leaf_nodes)")
        plt.xlabel('Total de iteracoes')
        plt.ylabel('Precisão')
        plt.xticks(range(2, 20))
        plt.plot(range(2, 20), precision_scores, marker="*")
        plt.savefig(criterion + "-" + "l" + '.png')
        plt.figure()

    dtc = DecisionTreeClassifier(max_depth=1, random_state=42)
    y_pred = dtc.fit(X_train, y_train).predict(X_test)
    print(dtc.get_n_leaves())

decision_tree()
