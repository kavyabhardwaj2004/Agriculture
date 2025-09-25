import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_excel("C:\\Users\\HP\\OneDrive\\Desktop\\augmented_odisha_agriculture.xlsx")

# Calculate Yield
df['Yield'] = df['Production'] / df['Crop_Field_in_hectares']

# Features and target (removed Fertilizer, Pesticide, Temperature)
X = df[['Crop_Year', 'Season', 'State', 'Annual_Rainfall', 'Area']]
y = df['Yield']

# Define categorical and numeric features
categorical_features = ['Season', 'State']
numeric_features = ['Crop_Year', 'Annual_Rainfall', 'Area']

# Preprocessor: OneHotEncode categorical, passthrough numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('crop', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Build pipeline with RandomForest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Sample farmer input
sample_input = pd.DataFrame({
    'Crop_Year': [2022],
    'Season': ['Kharif'],
    'State': ['Odisha'],
    'Annual_Rainfall': [1200],
    'Area': [3.5]
})

pred_yield = model.predict(sample_input)
print("Predicted Yield (tons/hectare):", pred_yield[0])

#..................Crop Type prediction......................#

feature = df[['Crop_Year','Season','State','Area','Production','Annual_Rainfall','Fertilizer','Pesticide']]
target = df['Crop']

alpha_features = ['Season','State']
num_features = ['Crop_Year','Area','Production','Annual_Rainfall','Fertilizer','Pesticide']
preprocessor = ColumnTransformer(
    transformers=[
        ('crop_cat',OneHotEncoder(handle_unknown='ignore'),alpha_features)
    ],
    remainder='passthrough'
)
model_crop = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',RandomForestClassifier(n_estimators=100,random_state=42))
])
X_train, X_test,y_train,y_test = train_test_split(feature,target,test_size=0.15,random_state=42)
model_crop.fit(X_train,y_train)

y_predict_crop = model_crop.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_predict_crop))
print("\nClassification Report:\n", classification_report(y_test, y_predict_crop))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_predict_crop))

# -----------------Sample Prediction----------------- #
sample_input_2 = pd.DataFrame({
    'Crop_Year': [1997],
    'Season': ['Autumn'],
    'State': ['Odisha'],
    'Annual_Rainfall': [1200],
    'Area': [25621],
    'Production': [6487],
    'Fertilizer': [2438350.57],
    'Pesticide': [7942.51]
})

predict_crop_type = model_crop.predict(sample_input_2)
print("\nPredicted Crop Type:", predict_crop_type)