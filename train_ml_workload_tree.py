import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from __global_paths import *

print("Loading files")
with open(TMP_DIR + 'X_train_wl.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open(TMP_DIR + 'X_test_wl.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open(TMP_DIR + 'y_train_wl.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open(TMP_DIR + 'y_test_wl.pkl', 'rb') as f:
    y_test = pickle.load(f)
print("Files loaded")

print(f"Using Decision Tree Regressor")
model = DecisionTreeRegressor(max_depth=10, random_state=42)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_loss = mean_squared_error(y_train, y_train_pred)
test_loss = mean_squared_error(y_test, y_test_pred)

print(f"Train Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save the model
with open(TMP_DIR + 'decision_tree_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {TMP_DIR}decision_tree_model.pkl")
