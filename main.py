from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from data import train_x, train_y

classifier = svm.SVC(kernel='linear')

def vectorize_sentences(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return X, vectorizer

if __name__ == "__main__":
    X, vectorizer = vectorize_sentences(train_x)
    print("Feature names:", vectorizer.get_feature_names_out())
    print("Vectorized shape:", X.shape)
    print("Vectorized data:\n", X.toarray())

    classifier.fit(X, train_y)

    text_x = vectorizer.transform([
        "The new JavaScript framework simplifies frontend development",
        "The movie won several awards for its direction and screenplay"
    ])

    predictions = classifier.predict(text_x)

    print("Predictions:", predictions)