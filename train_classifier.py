import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict.get('data', [])  # Get data from dictionary or initialize as empty list
labels = data_dict.get('labels', [])  # Get labels from dictionary or initialize as empty list

if not data or not labels:
    print("Error: No data or labels available for training.")
    exit(1)

# Check if there are enough samples for splitting
if len(data) == 0 or len(labels) == 0:
    print("Error: No samples available for training.")
    exit(1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

if len(x_train) == 0 or len(y_train) == 0:
    print("Error: Training set is empty after splitting.")
    exit(1)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate accuracy score
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)