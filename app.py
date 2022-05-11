from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

model = pickle.load(open("model_decisiontree_smote.pkl", "rb")) # Ini masukkan file pickle-nya

app = Flask(__name__)


@app.route("/")
def home():
	return render_template("index.html")


@app.route("/hasil", methods = ["POST"])
def hasil():
	# Aksesnya pakai atribut "name" dalam tag
	job_name = request.form["job name"]
	lokasi = request.form["lokasi"]
	industri = request.form["industri"]
	tingkat_job = request.form["tingkat job"]
	lama_pengalaman = request.form["lama pengalaman"]
	ukuran_perusahaan = request.form["ukuran perusahaan"]


	# Job Name
	job_name1 = [0] * 3
	if job_name == "data analyst":
		job_name1[0] = 1
	elif job_name == "data engineer":
		job_name1[1] = 1
	elif job_name == "data scientist":
		job_name1[2] = 1


	# Lokasi
	lokasi1 = [0] * 7
	if lokasi == "bali":
		lokasi1[0] = 1
	elif lokasi == "banten":
		lokasi1[1] = 1
	elif lokasi == "diy":
		lokasi1[2] = 1
	elif lokasi == "dki jakarta":
		lokasi1[3] = 1
	elif lokasi == "jawa barat":
		lokasi1[4] = 1
	elif lokasi == "jawa timur":
		lokasi1[5] = 1
	elif lokasi == "kalimantan barat":
		lokasi1[6] = 1


	# Industri
	industri1 = [0] * 10
	if industri == "acara":
		industri1[0] = 1
	elif industri == "ekonomi":
		industri1[1] = 1
	elif industri == "human resources":
		industri1[2] = 1
	elif industri == "keuangan":
		industri1[3] = 1
	elif industri == "konstruksi":
		industri1[4] = 1
	elif industri == "konsultasi":
		industri1[5] = 1
	elif industri == "media":
		industri1[6] = 1
	elif industri == "pemasaran & periklanan":
		industri1[7] = 1
	elif industri == "teknologi informasi dan komunikasi":
		industri1[8] = 1
	elif industri == "transportasi":
		industri1[9] = 1


	gabung = [job_name1, lokasi1, industri1, tingkat_job, lama_pengalaman, ukuran_perusahaan]

	list_gabung = []
	for i in gabung:
	    if isinstance(i, list):
	        for x in i:
	        	list_gabung.append(int(x))
	    else:
	        list_gabung.append(float(i))


	kolom = [
	"job_name_data analyst", "job_name_data engineer", "job_name_data scientist", 
	"lokasi_bali", "lokasi_banten", "lokasi_diy", "lokasi_dki jakarta", 
	"lokasi_jawa barat", "lokasi_jawa timur", "lokasi_kalimantan barat", 
	"industri_acara", "industri_ekonomi", "industri_human resources", "industri_keuangan", 
	"industri_konstruksi", "industri_konsultasi", "industri_media", 
	"industri_pemasaran & periklanan", "industri_teknologi informasi dan komunikasi", "industri_transportasi",
	"tingkat_job", "lama_pengalaman", "ukuran_company"
	]


	df_test = pd.DataFrame([list_gabung], columns = kolom)
	

    # Predict-nya harus bentuk data frame karena training-nya juga dalam bentuk data frame 
    # (masih dalam bentuk transformasi Box-Cox)
	hasil_pred = float(model.predict(df_test))

	# Nilai lambda dari proses training (target engineering)
	nilai_lambda = 0.3624440881025335

	# Kembalikan ke nilai asli
	nilai_asli = round(((hasil_pred * nilai_lambda) + 1) ** (1/float(nilai_lambda)))


	return render_template("hasil.html", data = nilai_asli)


if __name__ == "__main__":
	app.run(debug = True)