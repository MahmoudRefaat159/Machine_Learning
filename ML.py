import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('BMW_Car_Sales_Classification.csv')

le = preprocessing.LabelEncoder()

RegressorData=pd.read_excel('Regressor.xlsx')
encoders = {}
categorical_cols = data.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
def show_all_label_mappings(encoders):
    for col, le in encoders.items():
        print(f"\n Models: {col}")
        for i, cls in enumerate(le.classes_):
            print(f"  {i} => {cls}")


def remove_outliers(df):
    outlier_indices = set()
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.update(outliers)
    return df.drop(index=outlier_indices)

data = remove_outliers(data)

for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col].astype(str))

x = data[['Model', 'Engine_Size_L', 'Mileage_KM', 'Year', 'Sales_Volume', 'Sales_Classification']]
y = data['Price_USD']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(x_train, y_train)

sample_input = RegressorData[['Model', 'Engine_Size_L', 'Mileage_KM', 'Year', 'Sales_Volume', 'Sales_Classification']].copy()

for col in categorical_cols:
    if col in sample_input.columns:
        sample_input[col] = encoders[col].transform(sample_input[col].astype(str))


predictions = model.predict(sample_input)
RegressorData['Predicted_Price_USD'] = predictions


output_filename = 'Predictions_Results.xlsx'
RegressorData.to_excel(output_filename, index=False, engine='openpyxl')

print("\nPredicted Prices:")
for i, price in enumerate(predictions):
    print(f"Record {i+1}: ${price:,.2f}")

