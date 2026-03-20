# 🌍 Air Quality Data Analysis Dashboard

## 📌 Deskripsi Proyek

Proyek ini merupakan proyek yang menganalisis kualitas udara berbasis Python yang dirancang untuk mengeksplorasi, memvisualisasikan, dan menganalisis dataset [PRSA_Data_20130301–20170228](https://drive.google.com/file/d/1RhU3gJlkteaAQfyn9XOVAz7a5o1-etgr/view). Dataset tersebut berisi data pengukuran polusi udara dari berbagai stasiun pemantauan yang tersebar di Beijing, Tiongkok, dengan rentang waktu pengamatan dari 1 Maret 2013 hingga 28 Februari 2017.

Proyek ini dikembangkan sebagai bagian dari tugas proyek analisis data pada kelas Belajar Fundamental Analisis Data yang diselenggarakan oleh Dicoding.

Terdapat 2 output utama dari proyek ini yaitu file notebook dan dashboard simple berbasis streamlit. 

### 📓 1. Notebook Analisis
Notebook proyek ini dikembangkan dengan alur kerja analisis data end-to-end, mulai dari perumusan masalah hingga penyajian insight. 
Alur kerjanya adalah sebagai berikut:

#### 1️⃣ Business Understanding
* Menentukan fokus analisis: tren tahunan, perbandingan antar stasiun, variasi musiman, pengaruh cuaca, korelasi suhu dengan polutan, serta pola harian.

#### 2️⃣ Data Preparation
* Menggabungkan seluruh file dataset menjadi satu DataFrame
* Melakukan assessment awal (struktur data, missing values, outlier)
* Data cleaning (imputasi, penanganan outlier, pembentukan datetime, dan finalisasi dataset bersih)

#### 3️⃣ Exploratory Data Analysis
* Melakukan agregasi dan analisis tren tahunan, bulanan, musiman, serta korelasi antar variabel untuk memahami pola polusi.

#### 4️⃣ Visualisasi & Insight
* Membuat grafik dan analisis statistik untuk menjawab pertanyaan bisnis, kemudian merangkum temuan utama terkait karakteristik polusi, variasi spasial/temporal, dan peran faktor meteorologi.

#### 5️⃣ Analisis Lanjutan & Peta Interaktif
* Melakukan klasifikasi kualitas udara, analisis top kasus ekstrem, serta visualisasi spasial menggunakan peta interaktif.

### 📊 2. Dashboard

[dashboard.py](https://github.com/GaryFaldi/Fundamental-Data-Analysis-Project/blob/main/Main_Project/app/dashboard.py) adalah aplikasi Streamlit yang mengubah seluruh proses analisis di notebook menjadi dashboard interaktif. Navigasi dilakukan melalui sidebar dengan empat halaman utama:

#### 🏠 1. Home
* Preview dataset, informasi kolom (dtype, missing values), dan statistik deskriptif untuk memberikan gambaran awal data.

#### 🧹 2. Data Cleaning
* Penjelasan proses preprocessing (konversi datetime, penanganan missing values, outlier) serta preview data hasil pembersihan.

#### 📈 3. Data Exploration
* Eksplorasi interaktif berdasarkan stasiun:
* Tren tahunan dan perbandingan polutan
* Pola meteorologi
* Variasi bulanan & musiman
* Analisis korelasi
* Pola harian jam sibuk

#### 🔬 4. Analisis Lanjutan
* Top 5 konsentrasi tertinggi, klasifikasi kualitas udara, heatmap korelasi, dan peta interaktif PM2.5 & PM10.

Dashboard ini merepresentasikan alur analisis lengkap dalam bentuk visual dan interaktif untuk memudahkan pemahaman insight kualitas udara.

## 🛠️ 3. Teknologi yang Digunakan

- Python 3.9
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Folium
- Streamlit-Folium
- Joblib


## 📂 4. Struktur Folder

```
Main_Project/
├── app/
│   ├── data/
│   │   ├── cleaned_air_quality_data.csv
│   │   ├── merged_air_quality_data.csv
│   │   └── treshold_air_quality_data.csv
│   ├── Proyek_Analisis_Data.ipynb
│   └── dashboard.py
├── dataset/
├── requirements.txt
├── .gitignore
└── url.txt
```

## 🚀 5. Cara Menjalankan

### Setup Environment

Pilih salah satu metode berikut sesuai preferensi yang diinginkan:

#### Menggunakan Anaconda

```bash
conda create --name air-quality-ds python=3.9
conda activate air-quality-ds
pip install -r requirements.txt
```

#### Menggunakan Shell/Terminal (pipenv)

```bash
mkdir proyek_analisis_data
cd proyek_analisis_data
pipenv install
pipenv shell
pip install -r requirements.txt
```

### Langkah 1: Analisis pada Notebook

Anda bisa melakukan analisis langsung pada file notebook yang tersedia [Proyek_Analisis_Data.ipynb](https://github.com/GaryFaldi/Fundamental-Data-Analysis-Project/blob/main/Main_Project/app/Proyek_Analisis_Data.ipynb). Jika visualisasi peta menggunakan folium tidak tampil, ini dikarenakan Visualisasi peta pada notebook ini dibuat menggunakan library folium, yang menghasilkan output berbasis HTML dan JavaScript (Leaflet.js).

Perlu diketahui bahwa:
Preview notebook di GitHub, Vs code, atau beberapa code editor lainnya hanya menampilkan konten statis.
Hal ini tidak mengeksekusi JavaScript demi alasan keamanan.
Akibatnya, visualisasi peta interaktif tidak akan dirender dengan sempurna pada tampilan preview notebook.

Notebook ini telah dijalankan sepenuhnya dan seluruh output telah tersimpan sebelum diunggah. Untuk melihat visualisasi peta secara utuh dan interaktif, silakan buka notebook melalui nbviewer pada tautan berikut:

🔗 [https://nbviewer.org/github/GaryFaldi/Fundamental-Data-Analysis-Project/blob/main/Main_Project/app/Proyek_Analisis_Data.ipynb?flush_cache=true]

nbviewer akan merender notebook dengan lebih lengkap karena mendukung tampilan output HTML yang dihasilkan oleh folium.

Dengan demikian, apabila visualisasi peta tidak tampil pada preview GitHub, hal tersebut bukan merupakan kesalahan kode, melainkan keterbatasan dari sistem rendering terhadap konten berbasis JavaScript.

### Langkah 2: Menjalankan Dashboard

Ada dua opsi untuk menjalankan dashboard, yaitu lewat localhost atau versi deploy. Berikut cara menjalankan keduanya:

#### 1. Jalankan di Browser yang sudah di deploy

Untuk run web yang sudah di deploy, tinggal buka link berikut: [prsa-air-pollution-data-analysis.streamlit.app](https://prsa-air-pollution-data-analysis.streamlit.app)

#### 2. Jalankan di Browser localhost

Pastikan sudah mengaktifkan virtual environment (lihat **Setup Environment** di atas), lalu jalankan perintah berikut:

```bash
streamlit run Main_Project/app/dashboard.py
```

Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`). (Bisa berbeda tiap device).

-----

**Created for Data Analytics Project - Dicoding**