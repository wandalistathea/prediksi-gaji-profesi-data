import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("Data Gabungan dari Linkedin dan Jobstreet.csv")
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


size_job_name = data.groupby(["job_name"]).size()
size_job_name
# "data architect" dan "database administrator" sangat kecil jumlah datanya


x = size_job_name.index
y = size_job_name.values

plt.bar(x, y)
plt.title("Frekuensi Data untuk Setiap Job Name (DATA GABUNGAN)")
plt.xlabel("Job Name")
plt.ylabel("Frekuensi")
plt.xticks(x, rotation = 25)
plt.show()


# Data "data architect" dan "database administrator" dihapus saja
data.drop(data.loc[data["job_name"] == "data architect"].index, inplace = True)
data.drop(data.loc[data["job_name"] == "database administrator"].index, inplace = True)


data["job_name"].unique() # Sudah terhapus

data = data.reset_index(drop = True)
data.index


len(data)

pd.isna(data).sum()


# Mengisi missing value di variabel "lokasi"
data[pd.isna(data["lokasi"])] # Lokasinya dibuat "jakarta"
# Sumber: https://www.lokerindo.id/lowongan-kerja/e70fc39a62f3476a/telesales-officer-wfh-based-in-indonesia-tri7-solutions-inc

data["lokasi"][86] = "dki jakarta"

data[pd.isna(data["lokasi"])] # Sudah tidak ada

pd.isna(data).sum()


data_awal = data.copy()


data.drop(["gaji"], axis = 1, inplace = True)

data.columns


# subset -> Define in which columns to look for missing values
data.dropna(subset = ["median_gaji"], inplace = True)

len(data)

data.groupby(["job_name"]).size()

summarize_features(data)


#!pip install dython

### KORELASI ------------------------------------
from dython import nominal
nominal.associations(data, 
                     nominal_columns = ["company", "job_name", "lokasi", "tingkat_job", "jenis_job", "ukuran_company", "industri"], 
                     numerical_columns = ["lama_pengalaman", "median_gaji"],
                     figsize = (10, 10))

# Ambil yang korelasinya >= 25%
# Variabel "jenis_job" dihapus saja


data.drop(["jenis_job"], axis = 1, inplace = True)

summarize_features(data)


### HANDLE MISSING VALUE ------------------------------------

# ### 1. Menangani Missing Value di "ukuran_company"

# Berdasarkan variabel -> "job_name"
import statistics

count_ukuran_company1 = data.groupby(["job_name"])["ukuran_company"].count()
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

count_industri1 = data.groupby(["job_name"])["industri"].count() # Hasilnya adalah banyaknya data yang tidak missing
mode_industri1 = data[data["industri"].notna()].groupby(["job_name"])["industri"].apply(statistics.mode) # Pakainya modus karena data kategorik
len(mode_industri1)

mode_industri1.values


dict_group2a = {}

for indeks in mode_industri1.index:
    group2a = {indeks: mode_industri1[indeks]}

    dict_group2a.update(group2a)

dict_group2a.values() # Sudah tidak ada yang nan


missing_industri1 = data[pd.isna(data["industri"])]

for i in missing_industri1.index:
    for j in dict_group2a.keys():
        data_asli = (data["job_name"][i])
        
        if data_asli == j:
            data["industri"][i] = dict_group2a[j]


pd.isna(data).sum()


# ### 3. Menangani Missing Value di "lama_pengalaman"

# Berdasarkan variabel -> "job_name"

count_lama_pengalaman1 = data.groupby(["job_name"])["lama_pengalaman"].count() # Hasilnya adalah banyaknya data yang tidak missing
median_lama_pengalaman1 = data[data["lama_pengalaman"].notna()].groupby(["job_name"])["lama_pengalaman"].median() # Pakainya median karena data numerik
#mode_median_gaji1 = df3.groupby(["job_name", "lokasi", "jenis_job"])["median_gaji"].apply(statistics.mode)

len(median_lama_pengalaman1)

median_lama_pengalaman1.values


dict_group4a = {}

for indeks in median_lama_pengalaman1.index:      
    group4a = {indeks: median_lama_pengalaman1[indeks]}

    dict_group4a.update(group4a)


missing_lama_pengalaman1 = data[pd.isna(data["lama_pengalaman"])]

for i in missing_lama_pengalaman1.index:
    for j in dict_group4a.keys():
        data_asli = (data["job_name"][i])
        
        if data_asli == j:
            data["lama_pengalaman"][i] = dict_group4a[j]


pd.isna(data).sum() 

data.drop(["company"], axis = 1, inplace = True)

summarize_features(data)


### Resampling (Disamakan Banyak Data untuk Masing2 "Job_Name") ------------------------------------

data.info()

data.groupby(["job_name"]).size()


# Coba di oversampling (dinaikkan jumlah datanya) -> Coba 100
s1 = data[data["job_name"] == "data analyst"].sample(100, replace = True, random_state = 0)
s2 = data[data["job_name"] == "data engineer"].sample(100, replace = True, random_state = 0)
s3 = data[data["job_name"] == "data scientist"].sample(100, replace = True, random_state = 0)

data_resampling = pd.concat([s1, s2, s3])

data_resampling.groupby(["job_name"]).size()

data_resampling = data_resampling.reset_index(drop = True)

data_resampling.index


### HANDLE CATEGORICAL DATA ------------------------------------

data_oke = data_resampling.copy()
len(data_oke)

summarize_features(data_oke)


# NOMINAL -> "job_name", "lokasi", "industri" (pakai get_dummies)
# ORDINAL -> "tingkat_job", "ukuran_company" (pakai OrdinalEncoder)


# NOMINAL

# "job_name", "lokasi", "industri"
nominal_cols = data_oke[["job_name", "lokasi", "industri"]]
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
                                                    "mapping": {"magang": 0, "tingkat pemula": 1, "asosiasi": 2, "senior tingkat menengah": 3}}])

data_oke["tingkat_job"] = encoder_tingkat_job.fit_transform(data_oke["tingkat_job"])

# 2. "ukuran_company"
encoder_ukuran_company = ce.OrdinalEncoder(cols = ["ukuran_company"], return_df = True, 
                                           mapping = [{"col": "ukuran_company", 
                                                       "mapping": {"1-50 pekerja": 0, "51-200 pekerja": 1, "201-500 pekerja": 2,
                                                                   "501-1.000 pekerja": 3, "1.001-5.000 pekerja": 4, ">5.000 pekerja": 5}}])

data_oke["ukuran_company"] = encoder_ukuran_company.fit_transform(data_oke["ukuran_company"])


### Target Engineering ------------------------------------

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# Data Asli
sns.distplot(data_oke["median_gaji"], hist = False, kde = True)


# Data Hasil Transformasi

#perform Box-Cox transformation on original data
transformed_data, best_lambda = stats.boxcox(data_oke["median_gaji"]) 

#plot the distribution of the transformed data values
sns.distplot(transformed_data, hist = False, kde = True)


nilai_lambda = best_lambda
nilai_lambda


print("Data Asli:", list(data_oke["median_gaji"][0:5]))
print("Data Hasil Transformasi:", transformed_data[0:5])


# Transformasikan
data_oke["median_gaji"] = transformed_data

data_oke["median_gaji"][0:5]


### K-Fold Cross Validation ------------------------------------


# Kalau pakai k-fold cross validation berarti langsung pakai semua data
X = data_oke.drop("median_gaji", axis = 1)
y = data_oke["median_gaji"]

len(y)


### MODELLING ------------------------------------

# Model Random Forest

from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(random_state = 0)


# PAKAI CROSS VAL
from sklearn.model_selection import cross_val_score

r2_rf = cross_val_score(estimator = random_forest, X = X, y = y, cv = 5)
r2_rf.mean()


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

random_forest_x = RandomForestRegressor(random_state = 0)
param_grid = {"n_estimators": list(range(100, 131)),
              "criterion": ["squared_error", "absolute_error"],
              "max_depth": list(range(16)), 
              "max_features": ["auto", "sqrt", "log2"], 
              "min_samples_split": list(range(2, 16))}

# grid_search_x1 = GridSearchCV(random_forest_x, param_grid, n_jobs = 2, verbose = 1, cv = 5)
# grid_search_x1.fit(X, y)

random_search_x = RandomizedSearchCV(random_forest_x, param_grid, cv = 5, n_jobs = 2, verbose = 1, 
                                     n_iter = 1000, random_state = 0)
# Harusnya kandidatnya ada banyak banget, tapi n_iter = 1000 jadi cuma pilih 1000 aja
random_search_x.fit(X, y)

random_search_x.best_params_

random_search_x.best_score_


# COBA BUAT MODEL BERDASARKAN "best_params_"
random_forest_x1 = RandomForestRegressor(random_state = 0, 
                                         n_estimators = 104,
                                         min_samples_split = 2,
                                         max_features = 'auto',
                                         max_depth = 10,
                                         criterion = 'squared_error')
random_forest_x1.fit(X, y)


# PAKAI CROSS VAL
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute

r2_rf_x1 = cross_val_score(estimator = random_forest_x1, X = X, y = y, cv = 5, scoring  ='r2')
nrmse_rf_x1 = cross_val_score(estimator = random_forest_x1, X = X, y = y, cv = 5, scoring  ='neg_root_mean_squared_error')
nmape_rf_x1 = cross_val_score(estimator = random_forest_x1, X = X, y = y, cv = 5, scoring  ='neg_mean_absolute_percentage_error')

r2_rf_x1 = mean(r2_rf_x1)
rmse_rf_x1 = mean(absolute(nrmse_rf_x1))
mape_rf_x1 = mean(absolute(nmape_rf_x1))

print("R2 RANDOM FOREST:", r2_rf_x1)
print("RMSE RANDOM FOREST:", rmse_rf_x1)
print("MAPE RANDOM FOREST:", mape_rf_x1)


# random_search ini itu sebagi modelnya
# Simpan model

with open("model_randomforest.pkl", "wb") as f:
    pickle.dump(random_forest_x1, f)


###------------------------------------


# Prediksi

with open("model_randomforest.pkl", "rb") as f:
    model = pickle.load(f)


kolom = ["job_name_data analyst", "job_name_data engineer", "job_name_data scientist", 
         "lokasi_bali", "lokasi_banten", "lokasi_diy", "lokasi_dki jakarta", 
         "lokasi_jawa barat", "lokasi_jawa timur", "lokasi_kalimantan barat", 
         "industri_acara", "industri_ekonomi", "industri_human resources", "industri_keuangan", 
         "industri_konstruksi", "industri_konsultasi", "industri_media", 
         "industri_pemasaran & periklanan", "industri_teknologi informasi dan komunikasi", 
         "tingkat_job", "lama_pengalaman", "ukuran_company"]

nilai = [[1, 0, 0, # "job_name" = "data analyst"
         1, 0, 0, 0, 0, 0, 0, # "lokasi" = "bali"
         1, 0, 0, 0, 0, 0, 0, 0, 0, # "industri" = "acara"
         1, 0, 0]] # "tingkat_job" = "tingkat pemula", "lama_pengalaman" = 0, "ukuran_company" = "1-50 pekerja"


coba_test = pd.DataFrame(nilai, columns = kolom)
coba_test


# Predict-nya harus bentuk data frame (masih dalam bentuk transformasi Box-Cox)
hasil_pred = float(model.predict(coba_test))
hasil_pred


# Kembalikan lagi ke nilai asli
nilai_asli = round(((hasil_pred * nilai_lambda) + 1) ** (1/float(nilai_lambda)))
nilai_asli