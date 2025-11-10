import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib

# Load the CSV file
file_path = r'C:\Users\Chloe\git\IndependentStudy\phishingDetection\PhishingDetection\files\Training_Files\Text_Classification\email_classification.csv'
df = pd.read_csv(file_path)

# Feature and target variables
X = df[['email']]
y_type = df['email_type']
y_noreply = df['no_reply']

# Split the data into training and validation sets
X_train, X_val, y_type_train, y_type_val = train_test_split(X, y_type, test_size=0.2, random_state=42)
_, _, y_noreply_train, y_noreply_val = train_test_split(X, y_noreply, test_size=0.2, random_state=42)

# Balance the dataset using oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_type_train)

# Define the pipeline with TF-IDF vectorizer and regularization
pipeline_type = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(penalty='l2', max_iter=1000))
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(pipeline_type, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled['email'], y_resampled)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Train the final model with the best parameters
pipeline_type.set_params(**grid_search.best_params_)
pipeline_type.fit(X_resampled['email'], y_resampled)

# Save the model
joblib.dump(pipeline_type, 'email_type_model.pkl')

# Load the model
pipeline_type = joblib.load('email_type_model.pkl')

# Predictions on the validation set
y_type_pred_val = pipeline_type.predict(X_val['email'])

# Evaluate the performance
print("Email Type - Validation Set")
print(classification_report(y_type_val, y_type_pred_val))
