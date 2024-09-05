import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

all_shops_file = pd.read_csv('/Users/brycen.izu/Downloads/archive/all_shops.csv')
kpop_groups = pd.read_csv('/Users/brycen.izu/Downloads/archive/kpop_groups.csv')

# the different vendors
jyp_shop_file = pd.read_csv('/Users/brycen.izu/Downloads/archive/jyp_shop.csv')
kpopalbums_shop_file = pd.read_csv('/Users/brycen.izu/Downloads/archive/kpopalbums_shop.csv')
kpopstoreinusa_file = pd.read_csv('/Users/brycen.izu/Downloads/archive/kpopstoreinusa.csv')
musicplaza_shop_file = pd.read_csv('/Users/brycen.izu/Downloads/archive/musicplaza_shop.csv')
smglobalshop_all_file = pd.read_csv('/Users/brycen.izu/Downloads/archive/smglobalshop_all.csv')

def drop_all_shops_file_labels(df):
    df.drop(['url', 'is_autograph', 'avg_review_value', 'number_of_questions', 'number_of_reviews', 'presale'], axis=1, inplace=True, errors='ignore')

# Create a dictionary for quick lookup
description_to_artist = dict(zip(kpop_groups['Name'], kpop_groups['Korean Name']))

# Drop unnecessary labels from both files
kpop_groups.drop(['Fanclub Name'], axis=1, inplace=True)
drop_all_shops_file_labels(all_shops_file)

# change the value in the table to the correct one
kpop_groups.loc[kpop_groups['Name'] == 'tripleS', 'Members'] = 24

# Iterate through DataFrame and update 'artist' column
for index, row in all_shops_file.iterrows():
    item = str(row['item'])  # Ensure 'item' is a string
    if pd.isna(row['artist']):
        for key, value in description_to_artist.items():
            # Ensure key and value are strings and check if they are in 'item'
            if key in item or value in item:
                all_shops_file.at[index, 'artist'] = key
                break  # Exit the loop once a match is found

# Display the resulting DataFrame
print(all_shops_file)
print(all_shops_file[['item','artist']])

# fill the discounted rows with the original price if not discounted
all_shops_file['discount_price'] = all_shops_file.apply(lambda row: row['price'] if pd.isna(row['discount_price']) else row['discount_price'], axis=1)

# check for non-null values to see how well the columns were adjusted
print(all_shops_file.info())

# Print rows with groups with more than 7 members
print(kpop_groups.loc[kpop_groups['Members'] > 7, ['Name', 'Members']])

# means of all the prices of each group
by_artist = all_shops_file.groupby('artist')[['discount_price']].mean()
print(by_artist.sort_values(by='discount_price'))

# how many albums from each group that were being sold in 2022
print(all_shops_file['artist'].value_counts())

# print mean number of members from each company
print(kpop_groups.groupby('Company')[['Members']].mean())

# print mean price of items in ascending order by vendor
print(all_shops_file.groupby('vendor').agg({'discount_price': 'mean'}).sort_values(by='discount_price'))

# percent change with the discount
all_shops_file['price_pct_change'] = all_shops_file['discount_price'] / all_shops_file['price'] * 100
print(all_shops_file[['item', 'price_pct_change']])

# describe both data frame
print(all_shops_file.describe())
print(kpop_groups.describe())

plt.figure(figsize=(10, 6))
plt.hist(all_shops_file['discount_price'], bins=30, edgecolor='k')
plt.title('Distribution of Discounted Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(kpop_groups['Members'], bins=30, edgecolor='k')
plt.title('Distribution of Members')
plt.xlabel('Members')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
correlation_matrix = all_shops_file[['discount_price','price']].corr()
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Correlation matrix
correlation_matrix = kpop_groups[['Members','Orig Memb']].corr()
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Count of items per vendor
vendor_counts = all_shops_file['vendor'].value_counts()
print(vendor_counts)

# Visualize item counts per vendor
vendor_counts.plot(kind='bar', figsize=(10, 6), edgecolor='k')
plt.title('Number of Items Sold by Vendor')
plt.xlabel('Vendor')
plt.ylabel('Number of Items')
plt.show()

# Boxplot of prices by vendor
plt.figure(figsize=(12, 8))
sns.boxplot(x='vendor', y='discount_price', data=all_shops_file)
plt.title('Boxplot of Discount Prices by Vendor')
plt.xticks(rotation=45)
plt.show()

# Boxplot of prices by artist
top_artists = all_shops_file['artist'].value_counts().head(10).index
plt.figure(figsize=(12, 8))
sns.boxplot(x='artist', y='discount_price', data=all_shops_file[all_shops_file['artist'].isin(top_artists)])
plt.title('Boxplot of Discount Prices by Top 10 Artists')
plt.xticks(rotation=45)
plt.show()

# Boxplot of members by gender
plt.figure(figsize=(4, 8))
sns.boxplot(x='Gender', y='Members', data=kpop_groups)
plt.title('Boxplot of Members by Gender')
plt.show()

# Convert 'Debut' column to datetime
kpop_groups['Debut'] = pd.to_datetime(kpop_groups['Debut'], format='%d/%m/%Y')

# Group by year and count the number of debuts
debut_counts = kpop_groups['Debut'].groupby(kpop_groups['Debut'].dt.to_period('Y')).count()

# Calculate cumulative total of debuts
cumulative_debut_counts = debut_counts.cumsum()

# Convert the PeriodIndex to a datetime index for plotting
debut_counts.index = debut_counts.index.to_timestamp()
cumulative_debut_counts.index = cumulative_debut_counts.index.to_timestamp()

# Plot the trend
plt.figure(figsize=(12, 6))
plt.plot(debut_counts, label='Debuts per Year', color='blue', marker='o')
plt.plot(cumulative_debut_counts, label='Cumulative Total Debuts', color='green', marker='o')
plt.title('Number of K-pop Groups Debuted Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Groups')
plt.grid(True)  # Optional
plt.legend()  # Add legend to differentiate the lines
plt.show()

# Prepare features and target variable
X = all_shops_file[['price', 'vendor', 'artist']]
X = pd.get_dummies(X, drop_first=True)
y = all_shops_file['discount_price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.to_numpy())

# Predict and evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Random Forest): {mse}")

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)  # Diagonal line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Discount Prices')
plt.show()
