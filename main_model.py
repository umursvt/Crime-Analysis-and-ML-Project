from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,f1_score,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd

dataset = pd.read_csv('crime_dataset.csv')
dataset = dataset.drop(['Crm Cd 2','Crm Cd 3','Crm Cd 2','Crm Cd 4','Cross Street','Mocodes','AREA','DR_NO'],axis=1)

# Cleaning and Transform
dataset['Premis Cd'] = dataset['Premis Cd'].fillna(0)  # Eksik mekan türleri 0 olarak doldur.
dataset['Vict Age'] = dataset['Vict Age'].fillna(dataset['Vict Age'].mean())
dataset['Weapon Used Cd'] =dataset['Weapon Used Cd'].fillna(0)

# Fill Unknown
dataset['Vict Sex'] = dataset['Vict Sex'].fillna('Unknown')
dataset['Vict Descent'] = dataset['Vict Descent'].fillna('Unknown')
dataset['Crm Cd 1'] = dataset['Crm Cd 1'].fillna(dataset['Crm Cd 1'].mean())
dataset['Status'] = dataset['Status'].fillna('Unknown')

# LAbel Encode
dataset['Crm Cd Desc'] = dataset['Crm Cd Desc'].astype('category').cat.codes
dataset['Premis Cd'] = dataset['Premis Cd'].astype('category').cat.codes
dataset['AREA NAME'] = dataset['AREA NAME'].astype('category').cat.codes
dataset['Vict Sex'] = dataset['Vict Sex'].astype('category').cat.codes
dataset['Vict Descent'] = dataset['Vict Descent'].astype('category').cat.codes
dataset['Weapon Used Cd'] = dataset['Weapon Used Cd'].astype('category').cat.codes
# dataset['Status'] = dataset['Status'].astype('category').cat.codes

# Tarihi işle ve yıl ay çıkar
dataset['DATE OCC'] = pd.to_datetime(dataset['DATE OCC'],errors='coerce')
dataset['YEAR OCC'] = dataset['DATE OCC'].dt.year
dataset['MONTH OCC'] = dataset['DATE OCC'].dt.month

# LONG ve LAT eksik doldur ve birleştir
dataset['LAT'] = dataset['LAT'].fillna(0)
dataset['LON'] = dataset['LON'].fillna(0)
dataset['GEOCODE'] = list(zip(dataset['LAT'],dataset['LON']))
dataset = dataset.drop(['LAT','LON'],axis=1)

# Last Check for columns
dataset = dataset.drop(['Date Rptd','Premis Desc','Weapon Desc','LOCATION','Part 1-2','Rpt Dist No','Crm Cd Desc','Status Desc',],axis=1)

# Features
X = dataset.drop(['Crm Cd', 'DATE OCC', 'GEOCODE'],axis=1)
Y = dataset['Crm Cd']

# Label Encode
Le = LabelEncoder()
X['Status'] = Le.fit_transform(X['Status'])


# Split dataset train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Model Training
model_random_forest = RandomForestClassifier(n_estimators=100,random_state=42)
model_random_forest.fit(X_train,y_train)

# Model prediction
model_random_forest_prediction = model_random_forest.predict(X_test)

results_forest = pd.DataFrame({
    'Actual': y_test,
    'Prediction': model_random_forest_prediction
})

print(results_forest)
print(classification_report(y_test,model_random_forest_prediction))





from sklearn.neighbors import KNeighborsClassifier
# KNN modelini
model_knn = KNeighborsClassifier(n_neighbors=5)
# Modeli eğit
model_knn.fit(X_train, y_train)
# Test setinde tahmin yap
model_knn_prediction = model_knn.predict(X_test)

result_knn = pd.DataFrame({
    'Actual': y_test,
    'Prediction':model_knn_prediction
})
# Sonuçları değerlendirmek için metrikler
print(classification_report(y_test, model_knn_prediction))


# Feature and Target Sets
X = dataset[['Crm Cd', 'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Cd', 'TIME OCC', 'YEAR OCC', 'MONTH OCC']]
y = dataset['Weapon Used Cd']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#  Model train
model_logistic = LogisticRegression(random_state=42)
model_logistic.fit(X_train, y_train)

# Test setinde tahmin yap
model_logistic_prediction = model_logistic.predict(X_test)

result_logistic = pd.DataFrame({
    'Actual': y_test,
    'Prediction':model_knn_prediction
})

print(result_logistic)


