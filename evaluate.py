from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 评价函数
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[1]
    )
    return {
        "Accuracy": acc,
        "Precision(Depressed)": p[0],
        "Recall(Depressed)": r[0],
        "F1(Depressed)": f1[0]
    }