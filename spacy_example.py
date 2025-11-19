import spacy
from data import train_x, train_y
from sklearn import svm

svm_cls = svm.SVC(kernel='linear')
nlp = spacy.load("en_core_web_md")

docs = [nlp(text) for text in train_x]
vectors = [doc.vector for doc in docs]
svm_cls.fit(vectors, train_y)

text_x = [
    "The new JavaScript framework simplifies frontend development",
    "The movie won several awards for its direction and screenplay"
]
text_docs = [nlp(text) for text in text_x]
text_vectors = [doc.vector for doc in text_docs]
predictions = svm_cls.predict(text_vectors)

print("Predictions:", predictions)

