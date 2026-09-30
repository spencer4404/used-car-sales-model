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

# %%
# read the dataframe
# df_read = pd.read_csv(os.path.join(path, "vehicles.csv"))
df_read = pd.read_csv("/Users/siannantuono/Documents/GitHub/used-car-value-model/training_data/car_price_training.csv")
# %%

print(df_read.columns)

# trim and preprocess
df = df_read[['price_usd_2023',
                    'price',
                    'source_dataset',
                   'age',
                   'manufacturer',
                   'model',
                   'condition',
                   'fuel',
                   'odometer',
                   'drive',
                   'type',
                   'paint_color',
                   'specs_imputed']]

# trim low price and mileage
df = df[df['price'] >= 1000]
df = df[df['price'] <= 100000]
df = df[df["odometer"] >= 5000]
df = df[df["odometer"] <= 300000]

# replace year with age
# current_year = 2026
# df["age"] = current_year - df["year"]
# df.drop('year', axis=1, inplace=True)

# split into model and trim, fill empties with unknown
split = df["model"].str.split()
df["model"] = split.str[:2].str.join(" ")
df["trim"] = split.str[2:].str.join(" ").replace("", "Unknown")

# normalize cases
for col in ["manufacturer", "model", "condition", "fuel", "drive", "type", "paint_color", "trim"]:
    df[col] = df[col].astype(str).str.strip().str.lower()

# combine truck and pickup
df['type'] = df["type"].replace("truck", "pickup")

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
                   'specs_imputed']]
target_y = df['price_usd_2023']

# %% [markdown]
# print(len(df))
# df.to_csv("trimmed_data_test.csv")
# %%
#split into train and test
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features_X, target_y, test_size=0.2)

# # carry the source labels through the split
# X_train, X_test, y_train, y_test, s_train, s_test = sklearn.model_selection.train_test_split(
#     features_X, target_y, df['source_dataset'], test_size=0.2, random_state=42)


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
        n_estimators=125,
        max_depth=None,
        min_samples_leaf=2,    # light regularization + faster training
        random_state=42,
        n_jobs=-1))
])

# %%
# Try log values of y
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)
# fit the model
model.fit(X_train, y_train_log)

# %%
# predict and report error
y_pred_log = model.predict(X_test)

# smearing correction: compute on TRAIN predictions (no leakage)
train_pred_log = model.predict(X_train)
smear = np.exp(y_train_log - train_pred_log).mean()
print(f"Smear: {smear}")

y_pred = np.exp(y_pred_log) * smear
mae = mean_absolute_error(y_test, y_pred)

print("\nMAE: ", mae)
# current MAE: 3346, median error 10.67%

# %%
# Compare error
comparison = pd.DataFrame()
comparison["true_price"] = y_test.values
comparison["predicted_price"] = y_pred
comparison["error"] = comparison["predicted_price"] - comparison["true_price"]
comparison["absolute % error"] = abs(comparison["error"] / comparison["true_price"] * 100)

print(comparison.describe())

# dump to joblib
joblib.dump(model, "backend/car_price_model.joblib")

# for src in s_test.unique():
#     sub = comparison[s_test.values == src]
#     print(src, "MAE:", round(mean_absolute_error(sub['true_price'], sub['predicted_price'])),
#           "median APE:", round(sub['absolute % error'].median(), 2))