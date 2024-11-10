from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [200],          # Number of trees
    'max_depth': [None],          # Maximum depth of each tree
    'min_samples_split': [5],          # Minimum samples required to split a node
    'min_samples_leaf': [1],            # Minimum samples required at each leaf node
    'max_features': ['sqrt']  # Number of features considered for splitting
}
#Best Parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}

# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='accuracy', verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and train the classifier with them
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the model with the best parameters found
best_rf_classifier = RandomForestClassifier(**best_params, random_state=42)
best_rf_classifier.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = best_rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy after Grid Search: {accuracy:.2f}")
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", classification_rep)
