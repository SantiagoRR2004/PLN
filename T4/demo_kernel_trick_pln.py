import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Toy dataset
docs = [
    "I loved this movie , it was fantastic !",
    "Absolutely terrible , I hated it",
    "What a great experience , so good",
    "Worst film ever , complete waste of time",
    "Fantastic movie ! I loved the great acting , and had few boring parts",  # mixed
    "Boring movie , overall . Fantastic cast , loved it . Great start and good ending , but not worth it",  # mixed
    "Awful acting and boring story",
    "Loved some parts of it , which were really good , but hated most of them",  # mixed
    "Bad pacing , hated the cast . Overall a great movie , nonetheless"  # mixed
]
# Labels
labels = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1])  # 1 = positive, 0 = negative

# Define sentiment lexicons
positive_words = {"love", "loved", "fantastic", "great", "good", "enjoyed", "recommend"}
negative_words = {"terrible", "hated", "worst", "awful", "boring", "waste", "bad"}

# Extract just 2 features: positive count, negative count
def extract_two_features(docs, pos_set, neg_set):
    X = np.zeros((len(docs), 2))
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        X[i, 0] = sum(w in pos_set for w in words) + np.random.rand() * 0.1  # positive count with noise
        X[i, 1] = sum(w in neg_set for w in words) + np.random.rand() * 0.1  # positive count with noise
    return X

X = extract_two_features(docs, positive_words, negative_words)

# Train SVM with RBF kernel
clf = SVC(kernel="linear")
# clf = SVC(kernel="rbf", gamma=2.0)  # gamma -> focus or inverse influence radius
clf.fit(X, labels)

# Decision boundary grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.RdYlBu, edgecolors="k")

plt.xlabel("Count of positive words")
plt.ylabel("Count of negative words")
plt.title("SVM with RBF Kernel on Reviews (2 handcrafted features)")
plt.show()
