from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from evaluate import evaluate_model

def classic_KNN(X_train_text, X_test_text, y_train, y_test):
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    knn = KNeighborsClassifier(
        n_neighbors=10,
        metric="cosine"
    )

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    knn_result = evaluate_model(y_test, y_pred)
    print("KNN:", knn_result)
    return knn_result
