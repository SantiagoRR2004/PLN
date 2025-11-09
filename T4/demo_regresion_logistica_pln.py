import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

docs = [
    "I loved this movie , it was fantastic !",
    "Absolutely terrible , I hated it",
    "What a great experience , so good",
    "Worst film ever , complete waste of time",
    "Really enjoyed it , highly recommend",
    "Awful acting and boring story",
]
y = np.array([1, 0, 1, 0, 1, 0])

# Try scrambling
positive_words = {"love", "loved", "fantastic", "great", "good", "enjoyed", "recommend"}
negative_words = {"terrible", "hated", "worst", "awful", "boring", "waste"}


def extract_two_features(docs, pos_set, neg_set):
    X = np.zeros((len(docs), 2))
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        X[i, 0] = (
            sum(w in pos_set for w in words) + np.random.rand() * 0.1
        )  # positive count with noise
        X[i, 1] = (
            sum(w in neg_set for w in words) + np.random.rand() * 0.1
        )  # negative count with noise
    return X


X = extract_two_features(docs, positive_words, negative_words)

model = LogisticRegression()
model.fit(X, y)

# Decision boundary grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k")

plt.xlabel("Count of positive words")
plt.ylabel("Count of negative words")
plt.title("Logistic Regression on Reviews (2 features)")
plt.show()
