import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 100 samples with temperature, voltage, current; binary target
x = np.random.rand(100, 3)  
y = np.random.randint(2, size=100)

print(x)
print(y)

# # Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# # Initialize the Random Forest model 
model = RandomForestClassifier()

# # Train the model
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
