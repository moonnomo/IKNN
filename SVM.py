from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from evaluate import evaluate_model

def SVM(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=5
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )

    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    svm_result = evaluate_model(y_test, y_pred)
    print("SVM:", svm_result)
    return svm_result
