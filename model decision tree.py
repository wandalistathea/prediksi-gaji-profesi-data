import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv("Data Gabungan dari Linkedin dan Jobstreet (3 Profesi).csv")
data.tail()

data.info()

def summarize_features(df):
    # first column will be data types of each feature
    summary = pd.DataFrame(df.dtypes, columns = ['dtypes'])
    summary = summary.reset_index()
    # how many missing values in each feature
    summary['Missing'] = df.isnull().sum().values
    # how many unique values in each feature (cardinality indicator)
    summary['Uniques'] = df.nunique().values

    return summary

summarize_features(data)

len(data)

data.drop(["company", "gaji", "sumber"], axis = 1, inplace = True)
data.columns

# subset -> Define in which columns to look for missing values
data.dropna(subset = ["median_gaji"], inplace = True)

data.reset_index(drop = True, inplace = True)

len(data)

data.groupby(["job_name"]).size()

summarize_features(data)


data_awal = data.copy()
len(data_awal) # "data_awal" yang disimpan aja, "data" dipakai di bawah2nya

data_awal.rename(columns = {"job_name": "Nama Profesi Data",
                       "lokasi": "Lokasi Perusahaan", 
                       "tingkat_job": "Tingkat Pekerjaan", 
                       "lama_pengalaman": "Lama Pengalaman",
                       "jenis_job": "Jenis Pekerjaan", 
                       "ukuran_company": "Ukuran Perusahaan", 
                       "industri": "Industri", 
                       "median_gaji": "Gaji"}, inplace=True)


#!pip install dython

### KORELASI ------------------------------------
from dython import nominal
nominal.associations(data_awal, 
                     nominal_columns = ["Nama Profesi Data", "Lokasi Perusahaan", "Tingkat Pekerjaan", "Jenis Pekerjaan", "Ukuran Perusahaan", "Industri"], 
                     numerical_columns = ["Lama Pengalaman", "Gaji"],
                     figsize = (13, 12), 
                     title = "Korelasi Antar Variabel",
                     filename = "Hasil Korelasi Dython")

# Coba ambil yang korelasinya >0,25 dan yang masuk akal
# Coba variabel "jenis_job" dihapus saja

data.drop(["jenis_job"], axis = 1, inplace = True)

summarize_features(data)


### HANDLE MISSING VALUE ------------------------------------

# ### 1. Menangani Missing Value di "ukuran_company"

# Berdasarkan variabel -> "job_name"
import statistics

count_ukuran_company1 = data.groupby(["job_name", "ukuran_company"])["ukuran_company"].count()
mode_ukuran_company1 = data[data["ukuran_company"].notna()].groupby(["job_name"])["ukuran_company"].apply(statistics.mode)
len(mode_ukuran_company1)

mode_ukuran_company1.values


dict_group1a = {}

for indeks in mode_ukuran_company1.index:    
    group1a = {indeks: mode_ukuran_company1[indeks]}
    
    dict_group1a.update(group1a)


missing_ukuran_company1 = data[pd.isna(data["ukuran_company"])]

for i in missing_ukuran_company1.index:
    for j in dict_group1a.keys():
        data_asli = (data["job_name"][i])
        
        if data_asli == j:
            data["ukuran_company"][i] = dict_group1a[j]


pd.isna(data).sum()


# ### 2. Menangani Missing Value di "industri"

# Berdasarkan variabel -> "job_name"

count_industri1 = data.groupby(["job_name", "industri"])["industri"].count() # Hasilnya adalah banyaknya data yang tidak missing
mode_industri1 = data[data["industri"].notna()].groupby(["job_name"])["industri"].apply(statistics.mode) # Pakainya modus karena data kategorik
len(mode_industri1)

mode_industri1.values
# Sama semua sehingga ada yang akan diambil nilai modus kedua
# "data analyst" pakai yang industri "konsultasi"

# Coba impute menggunakan modus yang lain
impute_industri = {"data analyst": "konsultasi", 
                   "data engineer": "teknologi informasi dan komunikasi", 
                   "data scientist": "teknologi informasi dan komunikasi"}

missing_industri1 = data[pd.isna(data["industri"])]

for i in missing_industri1.index:
    for j in impute_industri.keys():
        data_asli = (data["job_name"][i])
        
        if data_asli == j:
            data["industri"][i] = impute_industri[j]

pd.isna(data).sum()


# ### 3. Menangani Missing Value di "lama_pengalaman"

data["lama_pengalaman"].value_counts()

# Dilihat terlebih dahulu distribusi datanya
filtered_lama_pengalaman = data["lama_pengalaman"][~np.isnan(data["lama_pengalaman"])]
plt.boxplot(filtered_lama_pengalaman, meanline = True, showmeans = True)
plt.title("Boxplot Lama Pengalaman")
plt.ylabel("Lama Pengalaman (Tahun)")
#plt.savefig("1. Boxplot Lama Pengalaman (Untuk Handling Missing Value) REVISI.png")
plt.show()

sns.distplot(filtered_lama_pengalaman, hist = False, kde = True, axlabel = "Lama Pengalaman (Tahun)")
# plt.savefig("1. Distplot Lama Pengalaman (Untuk Handling Missing Value) REVISI.png")
plt.show()
# Distribusi cenderung Normal sehingga penggantinya akan menggunakan nilai mean


# Berdasarkan variabel -> "job_name"

count_lama_pengalaman1 = data.groupby(["job_name"])["lama_pengalaman"].count() # Hasilnya adalah banyaknya data yang tidak missing
mean_lama_pengalaman1 = data[data["lama_pengalaman"].notna()].groupby(["job_name"])["lama_pengalaman"].mean()

len(mean_lama_pengalaman1)

mean_lama_pengalaman1.values


dict_group4a = {}

for indeks in mean_lama_pengalaman1.index:      
    group4a = {indeks: round(mean_lama_pengalaman1[indeks], 1)}

    dict_group4a.update(group4a)


missing_lama_pengalaman1 = data[pd.isna(data["lama_pengalaman"])]

for i in missing_lama_pengalaman1.index:
    for j in dict_group4a.keys():
        data_asli = (data["job_name"][i])
        
        if data_asli == j:
            data["lama_pengalaman"][i] = dict_group4a[j]


pd.isna(data).sum()

data.info()

data["job_name"].value_counts()[0]

data.groupby(["job_name"]).size()


### HANDLE CATEGORICAL DATA ------------------------------------

data_oke = data.copy()
len(data_oke)

summarize_features(data_oke)


# NOMINAL -> "job_name", "lokasi", "industri" (pakai get_dummies)
# ORDINAL -> "tingkat_job", "ukuran_company" (pakai OrdinalEncoder)


# NOMINAL

# "lokasi", "industri"
nominal_cols = data_oke[["lokasi", "industri"]]
encoded_nominal = pd.get_dummies(data = nominal_cols)

# Gabungkan ke data asli
data_oke = pd.concat(objs = [encoded_nominal, data_oke], axis = 1)
data_oke.drop(nominal_cols, axis = 1, inplace = True)
data_oke.head(2)


#!pip install category_encoders

print(data_oke["tingkat_job"].unique())
print()
print(data_oke["ukuran_company"].unique())


# ORDINAL
import category_encoders as ce

# 1. "tingkat_job"
encoder_tingkat_job = ce.OrdinalEncoder(cols = ["tingkat_job"], return_df = True, 
                                        mapping = [{"col": "tingkat_job", 
                                                    "mapping": {"magang": 0, "tingkat pemula": 1, "asosiasi": 2, 
                                                                "senior tingkat menengah": 3, "direktur": 4, "eksekutif": 5}}])

data_oke["tingkat_job"] = encoder_tingkat_job.fit_transform(data_oke["tingkat_job"])

# 2. "ukuran_company"
encoder_ukuran_company = ce.OrdinalEncoder(cols = ["ukuran_company"], return_df = True, 
                                           mapping = [{"col": "ukuran_company", 
                                                       "mapping": {"1-50 pekerja": 0, "51-200 pekerja": 1, "201-500 pekerja": 2,
                                                                   "501-1.000 pekerja": 3, "1.001-5.000 pekerja": 4, ">5.000 pekerja": 5}}])

data_oke["ukuran_company"] = encoder_ukuran_company.fit_transform(data_oke["ukuran_company"])


### Resampling SMOTE (Disamakan Banyak Data untuk Masing2 "Job_Name") ------------------------------------

# Coba dinaikkan jumlah datanya menggunakan SMOTE -> Coba 87
# Karena mau pakai SMOTE, yang jadi y itu nama profesi data dulu
X = data_oke.drop("job_name", axis = 1)
y = data_oke["job_name"]
y.value_counts()


from imblearn.over_sampling import SMOTE
strategy = {"data analyst": 87, "data engineer": 87, "data scientist": 87}
oversample = SMOTE(random_state = 0, sampling_strategy = strategy)
X_smote, y_smote = oversample.fit_resample(X, y)
y_smote.value_counts()


data_smote = pd.concat([y_smote, X_smote], axis = 1)


# NOMINAL
# "job_name"
nominal_cols = data_smote[["job_name"]]
encoded_nominal = pd.get_dummies(data = nominal_cols)
# Gabungkan ke data asli
data_smote = pd.concat(objs = [encoded_nominal, data_smote], axis = 1)
data_smote.drop(nominal_cols, axis = 1, inplace = True)
data_smote.head(2)

len(data_smote)


### Target Engineering ------------------------------------

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


len(data_smote)

# Data Asli
sns.distplot(data_smote["median_gaji"], hist = False, kde = True, axlabel = "Gaji")

#plt.savefig("11. Distplot Gaji (261 Data) REVISI.png")

plt.show()


# Data Hasil Transformasi

#perform Box-Cox transformation on original data
transformed_data, best_lambda = stats.boxcox(data_smote["median_gaji"]) 

#plot the distribution of the transformed data values
sns.distplot(transformed_data, hist = False, kde = True, axlabel = "Gaji Hasil Transformasi")

#plt.savefig("12. Distplot Gaji Setelah Transformasi (261 Data) REVISI.png")

plt.show()


nilai_lambda = best_lambda
nilai_lambda


print("Data Asli:", list(data_smote["median_gaji"][0:5]))
print("Data Hasil Transformasi:", transformed_data[0:5])


# Transformasikan
data_smote["median_gaji"] = transformed_data

data_smote["median_gaji"][0:5]


### K-Fold Cross Validation ------------------------------------

# Kalau pakai k-fold cross validation berarti langsung pakai semua data
X = data_smote.drop("median_gaji", axis = 1)
y = data_smote["median_gaji"]

len(y)


### MODELLING ------------------------------------

# Model Decision Tree

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state = 0)

# PAKAI CROSS VAL
from sklearn.model_selection import cross_val_score
r2_tree = cross_val_score(estimator = tree, X = X, y = y, cv = 5)
r2_tree.mean()


from sklearn.model_selection import GridSearchCV

tree_x = DecisionTreeRegressor(random_state = 0)
param_grid = {"max_depth": list(range(21)), 
              "criterion": ["mse", "friedman_mse", "mae"], 
              "max_features": ["auto", "sqrt", "log2"], 
              "min_samples_split": list(range(6))}

grid_search_x = GridSearchCV(tree_x, param_grid, n_jobs = 2, verbose = 1, cv = 5)

grid_search_x.fit(X, y)

grid_search_x.best_params_

# Ini nilai score test (R2)-nya
grid_search_x.best_score_ # Mean cross-validated score of the best_estimator


# COBA BUAT MODEL BERDASARKAN "best_params_"
tree_x1 = DecisionTreeRegressor(random_state = 0, 
                                criterion = 'mse',
                                max_depth = 10,
                                max_features = 'sqrt',
                                min_samples_split = 3)
tree_x1.fit(X, y)


# PAKAI CROSS VAL
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute

r2_tree_x1 = cross_val_score(estimator = tree_x1, X = X, y = y, cv = 5, scoring  ='r2')
nrmse_tree_x1 = cross_val_score(estimator = tree_x1, X = X, y = y, cv = 5, scoring  ='neg_root_mean_squared_error')
nmape_tree_x1 = cross_val_score(estimator = tree_x1, X = X, y = y, cv = 5, scoring  ='neg_mean_absolute_percentage_error')

r2_tree_x1 = mean(r2_tree_x1)
rmse_tree_x1 = mean(absolute(nrmse_tree_x1))
mape_tree_x1 = mean(absolute(nmape_tree_x1))

print("R2 TREE:", r2_tree_x1)
print("RMSE TREE:", rmse_tree_x1)
print("MAPE TREE:", mape_tree_x1)


# grid_search ini itu sebagai modelnya
# Simpan model

import pickle
with open("model_decisiontree_smote.pkl", "wb") as f:
    pickle.dump(tree_x1, f)


###------------------------------------


# Prediksi

with open("model_decisiontree_smote.pkl", "rb") as f:
    model = pickle.load(f)


kolom = ["job_name_data analyst", "job_name_data engineer", "job_name_data scientist", 
         "lokasi_bali", "lokasi_banten", "lokasi_diy", "lokasi_dki jakarta", 
         "lokasi_jawa barat", "lokasi_jawa timur", "lokasi_kalimantan barat", 
         "industri_acara", "industri_ekonomi", "industri_human resources", "industri_keuangan", 
         "industri_konstruksi", "industri_konsultasi", "industri_media", 
         "industri_pemasaran & periklanan", "industri_teknologi informasi dan komunikasi", "industri_transportasi",
         "tingkat_job", "lama_pengalaman", "ukuran_company"]

nilai = [[1, 0, 0, # "job_name" = "data analyst"
         1, 0, 0, 0, 0, 0, 0, # "lokasi" = "bali"
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, # "industri" = "acara"
         1, 0, 0]] # "tingkat_job" = "tingkat pemula", "lama_pengalaman" = 0, "ukuran_company" = "1-50 pekerja"


coba_test = pd.DataFrame(nilai, columns = kolom)
coba_test


# Predict-nya harus bentuk data frame (masih dalam bentuk transformasi Box-Cox)
hasil_pred = float(model.predict(coba_test))
hasil_pred

nilai_lambda = 0.3624440881025335

# Kembalikan ke nilai asli
nilai_asli = round(((hasil_pred * nilai_lambda) + 1) ** (1/float(nilai_lambda)))
nilai_asli