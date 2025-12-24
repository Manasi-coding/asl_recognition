import numpy as np
import joblib
from sklearn.svm import SVC

# load dataset
X = np.load("X.npy")
y = np.load("y.npy")

print("Dataset loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# create SVM model
model = SVC(
    kernel="linear",      # simple + works well for landmarks
)

# train
model.fit(X, y)

# save model
joblib.dump(model, "model.pkl")

print("SVM model trained and saved as: model.pkl")