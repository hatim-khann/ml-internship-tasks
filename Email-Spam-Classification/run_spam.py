import pickle
import sys

model = pickle.load(open("spam_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

text = " ".join(sys.argv[1:])
features = vectorizer.transform([text])
prediction = model.predict(features)[0]

print("Spam" if prediction==1 else "Not Spam")