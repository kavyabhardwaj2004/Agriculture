import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


#finding suitable crop based on season,crop_type,state
df1 = pd.read_csv("C:\\Users\\HP\\Downloads\\crop_production.csv\\crop_production.csv")
print(df1.head())


# Features and target
X = df1[['State_Name', 'District_Name', 'Crop_Year', 'Season']]
y = df1['Crop']

# Encode target (Crop)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# OneHotEncoder for categorical features
features = ['State_Name', 'District_Name', 'Season']
preprocessor = ColumnTransformer(
    transformers=[
        ('crop', OneHotEncoder(handle_unknown='ignore'), features)
    ],
    remainder='passthrough'
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=0.85)
print(model.fit(X_train, y_train))
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

# Example prediction
y_pred = model.predict(X_test)
print("Encoded predictions:", y_pred)
print("Crop predictions:", le.inverse_transform(y_pred))
print(le.inverse_transform(y_pred))

#testing model
farmer_input = pd.DataFrame({
    'State_Name': ["Andaman and Nicobar Islands"],
    'District_Name': ["NICOBARS"],
    'Crop_Year': [2001],
    'Season': ["Kharif"]
})

encoded_prediction = model.predict(farmer_input)
crop_name = le.inverse_transform(encoded_prediction)[0]
print("Encoded prediction:", encoded_prediction)
print("Recommended Crop:", crop_name)

df = pd.read_excel("C:\\Users\\HP\\OneDrive\\Desktop\\augmented_odisha_agriculture.xlsx")
print(df.head())

production = df['Production']
area = df['Area']
cultivated_field = df['Crop_Field_in_hectares']

def yield_val(production,area,cultivated_field):
    crop_yield = (production/area)*cultivated_field
    return crop_yield
print(yield_val(6487,25621,19.04))
print(yield_val(1,2,0.5))

#irrigation prediction we can use if else

# approximate fertilizers and pesticides required to get maximum yield
# farmer will submit the report of soil...from which we get pH,potash,phosphate,nitrogen
# right now