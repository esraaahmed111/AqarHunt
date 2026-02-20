# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# Load dataset (from local directory)
df = pd.read_csv('property_finder_sale.csv')
df

# Set display options for pandas
pd.set_option("display.max_columns", None)

# Check dataset info and summary statistics
print(df.info())
print(df.describe())

# Check for missing values
total_missing_values = df.isnull().sum()
print(total_missing_values)

# Handle missing values
df = df[df['bathrooms'] != 'None']
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce').astype('Int64')
df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce').astype('Int64')
df = df.dropna(subset=['bedrooms', 'bathrooms'])
df['down_payment_price'] = df['down_payment_price'].fillna(0)
df['district/compound'] = df['district/compound'].fillna('unknown')
df['furnished'] = df['furnished'].astype(str).str.strip().str.capitalize()

# Update 'completion_status' based on 'furnished' values
df.loc[df['completion_status'].isna() & df['furnished'].isin(['Yes', 'Partly']), 'completion_status'] = 'Completed'
df.loc[df['completion_status'].isna() & df['furnished'].isin(['No']), 'completion_status'] = 'Unknown'

print(df.info())

# Check for missing values after handling
total_missing_values = df.isnull().sum()
print(total_missing_values)

# Convert 'listed_date' to datetime
df['listed_date'] = pd.to_datetime(df['listed_date'], format="%Y-%m-%dT%H:%M:%SZ")

# Remove duplicate rows
duplicates = df.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")
df = df.drop_duplicates()
df.reset_index(drop=True, inplace=True)

# Handle categorical column consistency
categorical_columns = df.select_dtypes(include='object').columns
for column in categorical_columns:
    df[column] = df[column].str.capitalize()

# Remove irrelevant columns
df.drop(columns=['id', 'url+X13A1:V13', 'location_full_name', 'has_view_360', 'amenity_names', 'payment_method', 'listed_date'], inplace=True)

# Apply IQR method to remove outliers from 'price', 'size', 'bedrooms', and 'bathrooms'
for col in ['price', 'size', 'bedrooms', 'bathrooms']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("Shape of DataFrame after outlier removal:", df.shape)

# Visualize data after outlier removal
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.boxplot(y=df['price'])
plt.title('Box Plot of Price (After Outlier Removal)')
plt.ylabel('Price')

plt.subplot(2, 2, 2)
sns.boxplot(y=df['size'])
plt.title('Box Plot of Size (After Outlier Removal)')
plt.ylabel('Size')

plt.subplot(2, 2, 3)
sns.boxplot(y=df['bedrooms'])
plt.title('Box Plot of Bedrooms (After Outlier Removal)')
plt.ylabel('Bedroom Count')

plt.subplot(2, 2, 4)
sns.boxplot(y=df['bathrooms'])
plt.title('Box Plot of Bathrooms (After Outlier Removal)')
plt.ylabel('Bathroom Count')

plt.tight_layout()
plt.show()

# Correlation heatmap for numerical columns
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Prepare for Ensemble Model
dataset = df.copy()
categorical_cols = ['property_type', 'city', 'town', 'district/compound', 'completion_status', 'offering_type', 'furnished']
numerical_cols = ['lat', 'lon', 'bedrooms', 'bathrooms', 'size', 'down_payment_price']  # Removed 'price' from numerical_cols

# Preprocessing steps for numerical and categorical features
numerical_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a ColumnTransformer to apply different transformations to different columns
column_trans = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'  # Keep other columns (like lat, lon, size, down_payment_price)
)

X = df.drop(columns='price')
y = df['price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='price'), df['price'], test_size=0.2, random_state=42)

# Define base estimator
base_estimator = DecisionTreeRegressor(max_depth=None, min_samples_leaf=2, random_state=42)

# Bagging Regressor
bag = BaggingRegressor(
        estimator=base_estimator,
        n_estimators=500,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
)
model = Pipeline([('prep', column_trans),  # Use column_trans for preprocessing
                  ('bag', bag)])

# Fit the model
model.fit(X_train, y_train)
print('OOB R²:', model[-1].oob_score_)  # Quick sanity check

X_train_trans = column_trans.fit_transform(X_train)
X_test_trans = column_trans.transform(X_test)

# Convert to dense arrays
X_train_dense = X_train_trans.toarray()
X_test_dense = X_test_trans.toarray()

# Train HistGradientBoostingRegressor model
boost_model = HistGradientBoostingRegressor(
    learning_rate=0.1,
    max_iter=1000,
    max_depth=10,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

boost_model.fit(X_train_dense, y_train)

# Evaluate the model
y_train_pred = boost_model.predict(X_train_dense)
y_test_pred = boost_model.predict(X_test_dense)

print(f"R² on train: {r2_score(y_train, y_train_pred):.4f}")
print(f"R² on test : {r2_score(y_test, y_test_pred):.4f}")
print('MAE on test:', mean_absolute_error(y_test, y_test_pred))
print('MSE on test:', mean_squared_error(y_test, y_test_pred))
print('RMSE on test:', np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Display results
results = pd.DataFrame({
    'Actual Price (EGP)': y_test.values,
    'Predicted Price (EGP)': y_test_pred
})

display(
    results.head(20).style.format({
        'Actual Price (EGP)': '{:,.0f}',
        'Predicted Price (EGP)': '{:,.0f}'
    })
)

# Plot Actual vs Predicted Prices
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.xlabel("Actual Price (EGP)")
plt.ylabel("Predicted Price (EGP)")
plt.title("Actual vs Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
