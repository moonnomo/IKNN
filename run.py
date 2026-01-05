from data import load_txt_dataset
from sklearn.model_selection import train_test_split
from classic_KNN import classic_KNN
from multinomial_NB import multinomial_NB
from SVM import SVM
from I_KNN import I_KNN_predict
from I_KNN import *

dep_txt = "Dataset/depression.txt"
non_dep_txt = "Dataset/non_depression.txt"

def draw_pic(classic_KNN, multinomal_NB, svm, iknn):
    import matplotlib.pyplot as plt
    import numpy as np

    # ===== 1. 收集结果 =====
    models = ["KNN", "Naive Bayes", "SVM", "I-KNN (Stack)"]

    acc = [
        classic_KNN["Accuracy"],
        multinomal_NB["Accuracy"],
        svm["Accuracy"],
        iknn["Accuracy"]
    ]

    precision = [
        classic_KNN["Precision(Depressed)"],
        multinomal_NB["Precision(Depressed)"],
        svm["Precision(Depressed)"],
        iknn["Precision(Depressed)"]
    ]

    recall = [
        classic_KNN["Recall(Depressed)"],
        multinomal_NB["Recall(Depressed)"],
        svm["Recall(Depressed)"],
        iknn["Recall(Depressed)"]
    ]

    f1 = [
        classic_KNN["F1(Depressed)"],
        multinomal_NB["F1(Depressed)"],
        svm["F1(Depressed)"],
        iknn["F1(Depressed)"]
    ]

    # ===== 2. 画并列柱状图 =====
    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(10, 6))

    plt.bar(x - 1.5 * width, acc, width, label="Accuracy")
    plt.bar(x - 0.5 * width, precision, width, label="Precision")
    plt.bar(x + 0.5 * width, recall, width, label="Recall")
    plt.bar(x + 1.5 * width, f1, width, label="F1")

    plt.xticks(x, models, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Comparison of the performance of different algorithms")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    X_text, y = load_txt_dataset(dep_txt, non_dep_txt)
    y = np.asarray(y).ravel()

    # 划分：train_full / test
    X_train_full_text, X_test_text, y_train_full, y_test = train_test_split(
        X_text, y, test_size=0.2, stratify=y, random_state=42)

    # 从 train_full 再划分出 subtrain / val 用于调参和堆叠训练
    X_subtrain_text, X_val_text, y_subtrain, y_val = train_test_split(
        X_train_full_text, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

    # 对照组
    classic_KNN = classic_KNN(X_train_full_text, X_test_text, y_train_full, y_test)
    multinomal_NB = multinomial_NB(X_train_full_text, X_test_text, y_train_full, y_test)
    svm = SVM(X_train_full_text, X_test_text, y_train_full, y_test)


    # iknn
    iknn = I_KNN_predict(X_subtrain_text, X_val_text, X_test_text, y_subtrain, y_test, y_val)
    
    draw_pic(classic_KNN, multinomal_NB, svm, iknn)