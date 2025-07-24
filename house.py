import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏡 House Price Prediction ")

# --- 1. Define Default Feature Set ---
default_values = {
    'MSSubClass': 60, 'MSZoning': 'RL', 'LotFrontage': 65, 'LotArea': 8450, 'Street': 'Pave',
    'Alley': 'NA', 'LotShape': 'Reg', 'LandContour': 'Lvl', 'Utilities': 'AllPub', 'LotConfig': 'Inside',
    'LandSlope': 'Gtl', 'Neighborhood': 'CollgCr', 'Condition1': 'Norm', 'Condition2': 'Norm',
    'BldgType': '1Fam', 'HouseStyle': '2Story', 'OverallQual': 7, 'OverallCond': 5, 'YearBuilt': 2003,
    'YearRemodAdd': 2003, 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd', 'MasVnrType': 'BrkFace', 'MasVnrArea': 196, 'ExterQual': 'Gd',
    'ExterCond': 'TA', 'Foundation': 'PConc', 'BsmtQual': 'Gd', 'BsmtCond': 'TA', 'BsmtExposure': 'No',
    'BsmtFinType1': 'GLQ', 'BsmtFinSF1': 706, 'BsmtFinType2': 'Unf', 'BsmtFinSF2': 0, 'BsmtUnfSF': 150,
    'TotalBsmtSF': 856, 'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y', 'Electrical': 'SBrkr',
    '1stFlrSF': 856, '2ndFlrSF': 854, 'LowQualFinSF': 0, 'GrLivArea': 1710, 'BsmtFullBath': 1,
    'BsmtHalfBath': 0, 'FullBath': 2, 'HalfBath': 1, 'Bedroom': 3, 'Kitchen': 1, 'KitchenQual': 'Gd',
    'TotRmsAbvGrd': 8, 'Functional': 'Typ', 'Fireplaces': 0, 'FireplaceQu': 'NA', 'GarageType': 'Attchd',
    'GarageYrBlt': 2003, 'GarageFinish': 'RFn', 'GarageCars': 2, 'GarageArea': 548, 'GarageQual': 'TA',
    'GarageCond': 'TA', 'PavedDrive': 'Y', 'WoodDeckSF': 0, 'OpenPorchSF': 61, 'EnclosedPorch': 0,
    '3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': 'NA', 'Fence': 'NA',
    'MiscFeature': 'NA', 'MiscVal': 0, 'MoSold': 2, 'YrSold': 2008, 'SaleType': 'WD',
    'SaleCondition': 'Normal'
}

categorical_feats = [k for k, v in default_values.items() if isinstance(v, str)]
numerical_feats = [k for k, v in default_values.items() if isinstance(v, (int, float))]

# --- 2. Dummy Dataset Generator ---
def generate_dummy_data(n=200):
    np.random.seed(0)
    df = pd.DataFrame()

    for feat in numerical_feats:
        mean = default_values[feat]
        std = mean * 0.25 if mean != 0 else 1
        df[feat] = np.random.normal(loc=mean, scale=std, size=n).clip(min=0).round().astype(int)

    for feat in categorical_feats:
        options = [default_values[feat], "Option1", "Option2"]
        df[feat] = np.random.choice(options, size=n)

    y = (
        df['OverallQual'] * 20000 +
        df['GrLivArea'] * 40 +
        df['GarageCars'] * 10000 +
        df['TotalBsmtSF'] * 25 +
        df['FullBath'] * 7000 +
        (df['YearBuilt'] - 1900) * 120 +
        df['LotArea'] * 2 +
        np.random.normal(0, 15000, n)
    )
    return df, y

# --- 3. Label Encoding ---
def simple_label_encoder(series, mapping=None):
    if mapping is None:
        uniques = series.dropna().unique()
        mapping = {k: i for i, k in enumerate(uniques)}
    return series.map(mapping).fillna(-1).astype(int), mapping

# --- 4. Generate Training Data ---
df_train, y_train = generate_dummy_data()
encoders = {}

for feat in categorical_feats:
    df_train[feat], encoders[feat] = simple_label_encoder(df_train[feat])

# --- 5. Model Training ---
model = GradientBoostingRegressor()
model.fit(df_train, y_train)
train_cols = df_train.columns.tolist()  # save exact column order

# --- 6. UI Inputs in Table Format ---
st.markdown("### 🔢 Input Features")

input_data = {}
cols = st.columns(4)
for i, feat in enumerate(default_values):
    default = default_values[feat]
    if feat in numerical_feats:
        input_data[feat] = cols[i % 4].number_input(feat, value=default)
    else:
        input_data[feat] = cols[i % 4].text_input(feat, value=default)

# --- 7. Prepare Input Row ---
input_df = pd.DataFrame([input_data])
for feat in categorical_feats:
    input_df[feat], _ = simple_label_encoder(input_df[feat], mapping=encoders[feat])

for feat in numerical_feats:
    input_df[feat] = pd.to_numeric(input_df[feat], errors='coerce').fillna(0)

# Align column order to match training
input_df = input_df[train_cols]

# --- 8. Predict and Show ---
if st.button("🎯 Predict House Price"):
    pred = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: **${pred:,.2f}**")

# --- 9. Show Final Input Table ---
st.markdown("---")
st.subheader("📋 Input Summary")
st.dataframe(pd.DataFrame([input_data]))


