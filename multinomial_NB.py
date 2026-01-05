from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluate import evaluate_model

def multinomial_NB(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=5
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)

    nb_result = evaluate_model(y_test, y_pred)
    print("Naive Bayes:", nb_result)
    return nb_result
