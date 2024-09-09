import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap


dataset = pd.read_csv("crime_dataset.csv")


# Boş satır kontrolü
(dataset.isnull().sum())

dataset = dataset.drop('TIME OCC',axis=1)

# suçun bildirilmesi ve suçun işlenmesi ile ilgili
# Tarih sütunlarını belirli bir formatta datetime formatına çevirme
dataset['Date Rptd'] = dataset['Date Rptd'].str.split(' ').str[0]
dataset['Date Rptd'] = pd.to_datetime(dataset['Date Rptd'], format='%m/%d/%Y')

dataset['DATE OCC'] = dataset['DATE OCC'].str.split(' ').str[0]
dataset['DATE OCC'] =  pd.to_datetime(dataset['DATE OCC'], format='%m/%d/%Y')

fark_date = (dataset['Date Rptd'] - dataset['DATE OCC']).dt.days

crime_and_time = pd.DataFrame({
    'Crime': dataset['Crm Cd Desc'],
    'Time Diff' : fark_date
})
crime_and_time_filtered = crime_and_time[crime_and_time['Time Diff'] != 0]

print(crime_and_time_filtered.sort_values(by='Time Diff',ascending=False))



# Zaman farkı için histogram
fig = px.histogram(crime_and_time_filtered, x="Time Diff", nbins=30, title="Suçun Bildirilmesi ile Gerçekleşmesi Arasındaki Gün Sayısı Dağılımı")
fig.update_layout(xaxis_title="Gün Sayısı", yaxis_title="Olay Sayısı")
fig.show()


fig = px.bar(crime_and_time_filtered, x='Crime', y='Time Diff', title="Suç Türlerine Göre Ortalama Bildirme Zamanı")
fig.update_layout(xaxis_title="Suç Türü", yaxis_title="Ortalama Gün Sayısı", xaxis_tickangle=-45)
fig.show()

# Suç türü ve zaman farkı için scatter plot
fig = px.scatter(crime_and_time_filtered.head(1000), x='Crime', y='Time Diff', title="Suç Türleri ve Zaman Farkı İlişkisi")
fig.update_layout(xaxis_title="Suç Türü", yaxis_title="Gün Sayısı", xaxis_tickangle=-45)
fig.show()


# Suç türlerinin sayımını al
crime_counts = dataset['Crm Cd Desc'].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=crime_counts.index, y=crime_counts.values, palette='Blues_d')
plt.title('Suç Türlerinin Dağılımı')
plt.xlabel('Suç Türü')
plt.ylabel('Olay Sayısı')
plt.xticks(rotation=30)
plt.show()


# Area Crime Counts
area_crime_counts = dataset['AREA NAME'].value_counts().head(20)

plt.figure(figsize=(8,12))
sns.barplot(x=area_crime_counts.index, y=area_crime_counts.values, palette='Blues_d')
plt.title('Bölgelere Göre Suç Dağılımı (TOP 20)')
plt.xlabel('Bölge Adı')
plt.ylabel('Suç Sayısı')
plt.xticks(rotation=30)
plt.show()

# En çok tekrarlanan ilk 20 suç

top_20_crimes = dataset['Crm Cd Desc'].value_counts().head(20).index
top_20_data = dataset['Crm Cd Desc'].isin(top_20_crimes)




dataset['Year OCC'] = dataset['DATE OCC'].dt.year


crime_by_area_year = dataset.groupby(['AREA NAME', 'Year OCC']).size().reset_index(name='Crime Count')

# Grafik için bir pivot tablosu
crime_by_area_year_pivot = crime_by_area_year.pivot(index="Year OCC", columns="AREA NAME", values="Crime Count")


# Isı haritası (heatmap)
plt.figure(figsize=(12,8))
sns.heatmap(crime_by_area_year_pivot, cmap="YlGnBu", linewidths=0.5)
plt.title('Bölgelere Göre Suçların Yıllara Dağılımı')
plt.xlabel('Bölge')
plt.ylabel('Yıl')
plt.xticks(rotation=45)
plt.show()

print(dataset['Crm Cd Desc'])

dataset['Crm Cd Desc'] = dataset['Crm Cd Desc'].astype('category').cat.codes
dataset['Weapon Used Cd'] = dataset['Weapon Used Cd'].astype('category').cat.codes
dataset['Vict Sex'] = dataset['Vict Sex'].astype('category').cat.codes

corr_data = dataset[['AREA','Vict Age', 'Vict Sex' ,'Weapon Used Cd', 'Crm Cd Desc']]

corr_matrix = corr_data.corr()

# Kolerasyon ısı haritası

plt.figure(figsize=(8,12))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',linewidths=0.2)
plt.title('Kolerassyon matrisi')
plt.show()




# Latitude ve Longitude verileri
locations = dataset[['LAT', 'LON']].dropna()
# Folium haritası
map = folium.Map(location=[34.05, -118.25], zoom_start=10)  # Los Angeles merkez noktası
# HeatMap eklentisini kullanarak suç yoğunluğu haritası
HeatMap(locations.values, radius=10).add_to(map)
map.save('los_angeles_crime_heat_map.html')


# Kurban Yaş Dağılımı Histogramı
plt.figure(figsize=(10,6))
sns.histplot(dataset['Vict Age'], bins=30, kde=True, color='purple')
plt.title('Kurban Yaş Dağılımı')
plt.xlabel('Yaş')
plt.ylabel('Olay Sayısı')
plt.show()

weapon_counts = dataset['Weapon Used Cd'].value_counts()

plt.figure(figsize=(10,6))
sns.barplot(x=weapon_counts.index, y=weapon_counts.values, palette='Oranges_d')
plt.title('Silah Kullanımı Durumuna Göre Suç Dağılımı')
plt.xlabel('Silah Kullanımı Kodu')
plt.ylabel('Olay Sayısı')
plt.show()

# Suç Türlerine Göre  (Pie Chart)
crime_counts = dataset['Crm Cd Desc'].value_counts().head(10)  
plt.figure(figsize=(8,8))
plt.pie(crime_counts.values, labels=crime_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Suç Türlerine Göre Suç Sayısı Dağılımı (İlk 10)')
plt.axis('equal')  # Daire şeklinde tutar
plt.show()

# Cinsiyete Göre (Box Plot)
plt.figure(figsize=(10,6))
sns.boxplot(x='Vict Sex', y='Vict Age', data=dataset, palette='Set3')
plt.title('Cinsiyete Göre Suç Kurbanı Yaşı Dağılımı')
plt.xlabel('Cinsiyet')
plt.ylabel('Yaş')
plt.xticks(ticks=[0, 1], labels=['Erkek', 'Kadın'])
plt.show()

