import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# load the features and labels
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# encode the labels (A, B, C, ... to 0, 1, 2, ...)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

# save classes for realtime later
np.save("classes.npy", le.classes_)
# Later, in real-time prediction, the model will output something like 3; it will have to be converted back to, say, "D"

# scale features (important for SVM and logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# try SVM
model = SVC(kernel="rbf", C=1, gamma="scale")
model.fit(X_train_scaled, y_train_encoded)

val_predictions = model.predict(X_val_scaled)

print("SVM Accuracy:", accuracy_score(y_val_encoded, val_predictions))
print(classification_report(y_val_encoded, val_predictions))

# Logistic Regression
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train_scaled, y_train_encoded)

log_preds = log_model.predict(X_val_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_val_encoded, log_preds))

# Random Forest (no scaling needed)
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train_encoded)

rf_preds = rf_model.predict(X_val)
print("Random Forest Accuracy:", accuracy_score(y_val_encoded, rf_preds))


# SVM Accuracy: 0.9114459137220897
# Logistic Regression Accuracy: 0.8795794493821808
# Random Forest Accuracy: 0.9514415781487102

# choosing rf 
best_model = rf_model

print("---- TEST EVALUATION ----")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

y_test_encoded = le.transform(y_test)

test_predictions = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test_encoded, test_predictions))
print(classification_report(y_test_encoded, test_predictions))

import joblib

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")  # needed if using SVM or Logistic