import os
import pandas as pd
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sandeep1080/used-car-sales")

df = pd.read_csv(os.path.join(path, "used_car_sales.csv"))