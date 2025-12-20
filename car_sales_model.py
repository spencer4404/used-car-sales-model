# %%
# imports
import os
import numpy as np
import pandas as pd
import kagglehub
from IPython.display import display
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# %%
# Download latest version
path = kagglehub.dataset_download("austinreese/craigslist-carstrucks-data")

# print("Path to dataset files:", path)
# print(os.listdir(path))

# df = pd.read_csv(os.path.join(path, "vehicles.csv"))

# display(df.head(5))

# %%
# read the dataframe
df_read = pd.read_csv(os.path.join(path, "vehicles.csv"))
# print(df_read.dtypes)

# %%
# trim and preprocess
df = df_read[['price',
                   'year',
                   'manufacturer',
                   'model',
                   'condition',
                   'fuel',
                   'odometer',
                   'drive',
                   'type',
                   'paint_color',
                   'state',
                   'lat',
                   'long']].dropna()

# trim low price and mileage
df = df[df['price'] >= 1000]
df = df[df['price'] <= 40000]
df = df[df["odometer"] >= 5000]
df = df[df["odometer"] <= 300000]

# replace year with age
current_year = 2025
df["age"] = current_year - df["year"]
df.drop('year', axis=1, inplace=True)

# split into model and trim, fill empties with unknown
split = df["model"].str.split()
df["model"] = split.str[:2].str.join(" ")
df["trim"] = split.str[2:].str.join(" ").replace("", "Unknown")

# Define features and target
features_X = df[['age',
                   'manufacturer',
                   'model',
                   'trim',
                   'condition',
                   'fuel',
                   'odometer',
                   'drive',
                   'type',
                   'paint_color',
                   'state',
                   'lat',
                   'long']]
target_y = df['price']
# print(features_X.columns)
# print(df_read.columns)

# %%
# print(df.dtypes)

# %%
#split into train and test
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features_X, target_y, test_size=0.2)

# %%
# define categorical and numerical columns
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

# %%
# one-hot encode the categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('numeric','passthrough', num_cols)
    ]
)

# %%
# # build pipeline with the linear regression model
# model = Pipeline(steps=[
#     ('preprocess', preprocessor),
#     ('regressor', LinearRegression())
# ])

# %%
# build pipeline with the random forest model
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1))
])

# %%
# Try log values of y
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)
# fit the model
print("Training...")
model.fit(X_train, y_train_log)

# %%
# predict and report error
y_pred_log = model.predict(X_test)
y_pred = np.exp(y_pred_log)
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)
# current MAE: 2200

# %%
# Compare error
comparison = pd.DataFrame()
comparison["true_price"] = y_test.values
comparison["predicted_price"] = y_pred
comparison["error"] = comparison["predicted_price"] - comparison["true_price"]
comparison["absolute % error"] = abs(comparison["error"] / comparison["true_price"] * 100)

print(comparison.describe())

# %%
my_car = pd.DataFrame([{
    'age': 20.0,
                   'manufacturer': 'jeep',
                   'model': 'grand cherokee',
                   'trim': 'laredo',
                   'condition': 'good',
                   'fuel': 'gas',
                   'odometer': 170000,
                   'drive': '4wd',
                   'type': 'SUV',
                   'paint_color': 'red',
                   'state': 'ct'
}])

norah_car = pd.DataFrame([{
    'age': 13.0,
                   'manufacturer': 'honda',
                   'model': 'crv',
                   'trim': "",
                   'condition': 'fair',
                   'fuel': 'gas',
                   'odometer': 264000,
                   'drive': 'fwd',
                   'type': 'SUV',
                   'paint_color': 'white',
                   'state': 'tx'
}])

jodi = pd.DataFrame([{
    'age': 4.0,
                   'manufacturer': 'hyundai',
                   'model': 'palisade',
                   'trim': 'calligraphy',
                   'condition': 'excellent',
                   'fuel': 'gas',
                   'odometer': 70000,
                   'drive': '4wd',
                   'type': 'SUV',
                   'paint_color': 'black',
                   'state': 'ct',
                   'lat': 41,
                   'long': 72
}])

tesla = pd.DataFrame([{
    'age': 3,
                   'manufacturer': 'tesla',
                   'model': 'model y',
                   'trim': '',
                   'condition': 'excellent',
                   'fuel': 'electric',
                   'odometer': 30000,
                   'drive': '4wd',
                   'type': 'SUV',
                   'paint_color': 'red',
                   'state': 'ct'
}])

my_car_value = np.exp(model.predict(jodi))

print(f"Estimated value of car: ${my_car_value[0]:.2f}")

print(f"Conditions: {df["condition"].unique()}")
print(f"Drives: {df["drive"].unique()}")
print(f"States: {df["state"].unique()}")
print(f"Colors: {df["paint_color"].unique()}")
print(f"Types: {df["type"].unique()}")
print(f"Fuels: {df["fuel"].unique()}")



# %%
joblib.dump(model, "car_price_model.joblib")