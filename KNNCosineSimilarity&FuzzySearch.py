# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import scipy.stats as stats

# Load dataset (from local directory)
df = pd.read_csv('property_finder_sale.csv')  # Update with the correct file path

# Set display options for pandas
pd.set_option("display.max_columns", None)

# Check dataset info and summary statistics
print(df.info())
print(df.describe())

# Handle missing values
total_missing_values = df.isnull().sum()
df = df[df['bathrooms'] != 'None']
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce').astype('Int64')
df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce').astype('Int64')
df = df.dropna(subset=['bedrooms', 'bathrooms'])
df['down_payment_price'] = df['down_payment_price'].fillna(0)
df['district/compound'] = df['district/compound'].fillna('unknown')
df['furnished'] = df['furnished'].astype(str).str.strip().str.capitalize()

# Update completion status based on furnished column
df.loc[df['completion_status'].isna() & df['furnished'].isin(['Yes', 'Partly']), 'completion_status'] = 'Completed'
df.loc[df['completion_status'].isna() & df['furnished'].isin(['No']), 'completion_status'] = 'Unknown'

# Convert listed_date to datetime
df['listed_date'] = pd.to_datetime(df['listed_date'], format="%Y-%m-%dT%H:%M:%SZ")

# Remove duplicate rows
df = df.drop_duplicates()
df.reset_index(drop=True, inplace=True)

# Handle categorical column consistency
categorical_columns = df.select_dtypes(include='object').columns
for column in categorical_columns:
    df[column] = df[column].str.capitalize()

# Remove irrelevant columns
df.drop(columns=['id', 'url+X13A1:V13', 'location_full_name', 'has_view_360', 'amenity_names', 'payment_method', 'listed_date'], inplace=True)

# Visualizing price distribution
filtered_data = df[df['price'] < df['price'].quantile(0.95)]
plt.figure(figsize=(12, 8))
ax = sns.histplot(filtered_data['price'], bins=30, kde=True, color='blue', edgecolor='black', linewidth=1.5)
plt.title('Property Prices Distribution', fontsize=16)
plt.xlabel('Price (EGP)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(filtered_data['price'].mean(), color='red', linestyle='--', label=f'Mean Price: {filtered_data["price"].mean():,.2f} EGP')
plt.legend()

# Visualizing property types
plt.figure(figsize=(12, 5))
ax_type = sns.countplot(data=df, x='property_type', palette='viridis')
plt.title('Distribution of Property Types')
plt.xlabel("Property Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Median price by apartment type
median_prices = df.groupby('property_type')['price'].median()
plt.figure(figsize=(12, 6))
ax_bar = sns.barplot(x=median_prices.index, y=median_prices.values, palette='viridis')
plt.title('Price Distribution by Apartment Type')
plt.xlabel("Apartment Type")
plt.ylabel("Median Price (EGP)")
plt.xticks(rotation=45)
for i, median in enumerate(median_prices.values):
    ax_bar.annotate(f'{median:.0f}', (i, median), ha='center', va='center', fontsize=12, color='black')
plt.show()

# Map visualization of property locations
fig = px.scatter_mapbox(df, lat='lat', lon='lon', color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=7,
                        hover_data={'price': True, 'property_type': True, 'furnished': True, 'completion_status': True})
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()

# Correlation heatmap for numerical columns
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Encoding categorical columns
categorical_cols = ['property_type', 'city', 'town', 'district/compound', 'offering_type', 'furnished']
numerical_cols = ['price', 'lat', 'lon', 'bedrooms', 'bathrooms', 'size', 'down_payment_price']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Normalize numerical columns
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Prepare the data for recommendation model
prep_data = np.hstack([df[numerical_cols].to_numpy(), df[categorical_cols].to_numpy()])
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(prep_data)

# Function to recommend properties based on index
def recommendEncoded(index, n_neighbors=6):
    input_vector = prep_data[index].reshape(1, -1)
    distances, indices = model.kneighbors(input_vector, n_neighbors=n_neighbors)
    return df.iloc[indices[0]]

recommendations = recommendEncoded(index=7)
recommendations[['property_type', 'city', 'price', 'size', 'bedrooms', 'bathrooms']]

# Function to dynamically filter properties based on user input
def dynamic_filter(df):
    print("🔍 Real Estate Search - You can skip any field by pressing Enter")
    cities = df['city'].dropna().unique().tolist()
    towns = df['town'].dropna().unique().tolist()
    compounds = df['district/compound'].dropna().unique().tolist()

    city = input(f"\nSelect a City ({len(cities)} options) or press Enter to skip: ").strip()
    town = input(f"\nSelect a Town ({len(towns)} options) or press Enter to skip: ").strip()
    compound = input(f"\nSelect a Compound/District ({len(compounds)} options) or press Enter to skip: ").strip()

    try:
        min_price = int(input("\nMinimum Price (or press Enter to skip): ").strip() or 0)
    except:
        min_price = None

    try:
        max_price = int(input("Maximum Price (or press Enter to skip): ").strip() or 0)
    except:
        max_price = None

    filtered = df.copy()

    if city:
        filtered = filtered[filtered['city'] == city]

    if town:
        filtered = filtered[filtered['town'] == town]

    if compound:
        filtered = filtered[filtered['district/compound'] == compound]

    if min_price:
        filtered = filtered[filtered['price'] >= min_price]

    if max_price:
        filtered = filtered[filtered['price'] <= max_price]

    if filtered.empty:
        print("\n No properties match the selected criteria.")
        return pd.DataFrame()  
    else:
        print(f"\n Found {len(filtered)} matching properties.")
        return filtered

filtered_data = dynamic_filter(df)

if not filtered_data.empty:
    print("\nYou can now apply KNN on the filtered data.")
filtered_data
