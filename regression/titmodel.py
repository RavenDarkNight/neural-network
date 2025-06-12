import pandas as pd 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt



# Load and preprocess training data
train = pd.read_csv('titanic/train.csv')
train.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True)
train.dropna(inplace=True)
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Load and preprocess test data
test = pd.read_csv('titanic/test.csv')
test_ids = test['PassengerId'].copy()  # Save PassengerId for submission
test.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Fill missing values in test data
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Prepare features and target
X = train[['Pclass', 'Sex', 'Age', 'Fare']]
y = train['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1
)
model.fit(X_train, y_train)

# Make predictions on test data
X_test = test[['Pclass', 'Sex', 'Age', 'Fare']]
predictions = model.predict(X_test)

# Plot feature importances
plt.figure(figsize=(10, 5))
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importances = feature_importances.sort_values('importance', ascending=False)

plt.bar(feature_importances['feature'], feature_importances['importance'])
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create submission DataFrame
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': predictions
})

# Save predictions to CSV
submission.to_csv('submission.csv', index=False)

# Print validation metrics
val_predictions = model.predict(X_val)
quantity = len(val_predictions)
count = sum(val_predictions)
print(f'Validation set predictions: {count}/{quantity}')