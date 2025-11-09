import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

texts = [
    "I love natural language processing",
    "Linear regression is simple",
    "This sentence is short",
    "Scikit learn makes things easier",
    "Deep learning models need lots of data",
    "NLP is fun",
    "Bag of words is a classic method",
    "Transformers changed the field",
    "Text is just data",
    "Machine learning is everywhere"
]
y = np.array([len(t.split()) for t in texts])

# Try counting spaces
X = np.array([[len(t)] for t in texts])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color="blue", alpha=0.7)
plt.plot(X, y_pred, color="red", linewidth=2)
plt.xlabel("Character count")
plt.ylabel("Word count")
plt.show()