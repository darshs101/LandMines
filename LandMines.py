# Step 1: import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression

# Step 2: load data
data = pd.read_csv('Mine_Dataset.csv')

#sets mines to 0 if no mines are present, 1 if any mine is present
for i in range(len(data)):
    if data.loc[i,'M'] == 1:
        data.loc[i,'M'] = 0
    else:
        data.loc[i,'M'] = 1

X = data[['V','H','S']]
y = data['M']

print(X)
print(y)
# Step 3: split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: create and fit the model
model = LogisticRegression(class_weight={0:2.3, 1:1},max_iter=10000)
model.fit(X_train, y_train)

# Step 5: make predictions
y_pred = model.predict(X_test)

y_pred_proba = model.predict_proba(X_test)
print("Probabilities")
print (y_pred_proba)

print()
y_pred_binary = (y_pred_proba[:,1] > 0.5).astype(int)
print("Binary Predictions (no mines -> 0, one or more mines -> 1)")
print (y_pred_binary)

# Step 6: evaluate the model
print()
report = classification_report(y_test, y_pred)
print('classification report:')
print(report)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Mines', 'Mines'])
disp.plot()
plt.show()