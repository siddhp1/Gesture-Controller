from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Load processed data
X = np.load('data/processed/X.npy')
y = np.load('data/processed/y.npy')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
model = svm.SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Save the trained model
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f'Accuracy: {accuracy}')