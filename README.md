# Analisis Sentimen Pelanggan Amazon Digital Ebook
**Laporan Tugas Besar Big Data**
Program Studi S1 Informatika — Fakultas Informatika — Universitas Telkom Bandung 2026

---

## Deskripsi Proyek

Proyek ini membangun sistem klasifikasi sentimen ulasan pelanggan terhadap produk buku digital (ebook) di platform Amazon. Pemrosesan data dilakukan menggunakan **Apache Spark** dengan pendekatan **Natural Language Processing (NLP)** berbasis TF-IDF dan algoritma **Regresi Logistik**, menghasilkan akurasi model sebesar **91,88%** dan nilai AUC sebesar **0,9462**.

---

## Anggota Kelompok

| No | Nama | NIM |
|----|------|-----|
| 1 | Muhammad Ilyas Alyasa | 1301213121 |
| 2 | Dinda Cantika Putri Baruara | 103012300135 |
| 3 | Bianka Rosa | 103012300335 |
| 4 | Adam Teguh Maulaanaa | 1301213218 |

---

## Dataset

- **Nama Dataset:** Amazon US Customer Reviews Dataset — Digital Ebook Purchase
- **Sumber:** [Kaggle — Cynthia Rempel](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset)
- **Nama Berkas:** `amazon_reviews_us_Digital_Ebook_Purchase_v1_01.tsv`
- **Ukuran:** 3,22 GB (format Tab-Separated Values / TSV)

**Atribut yang Digunakan:**

| Atribut | Keterangan |
|---------|------------|
| `review_body` | Teks ulasan yang ditulis oleh pelanggan, digunakan sebagai masukan utama model NLP |
| `star_rating` | Penilaian bintang dalam skala 1 hingga 5 |
| `verified_purchase` | Status verifikasi pembelian pelanggan (Y/N) |
| `helpful_votes` | Jumlah suara yang menilai ulasan tersebut bermanfaat |

---

## Persyaratan Sistem

Proyek ini dijalankan pada lingkungan **Google Colaboratory**. Pastikan library berikut tersedia sebelum menjalankan notebook:

```
pyspark
kaggle
pandas
matplotlib
seaborn
```

Instalasi dapat dilakukan melalui perintah berikut:

```bash
pip install pyspark kaggle
```

---

## Cara Menjalankan Proyek

### Langkah 1 — Persiapan Kaggle API Token

1. Masuk ke akun [Kaggle](https://www.kaggle.com)
2. Buka menu **Account → API → Create New Token**
3. Salin token yang diperoleh
4. Masukkan token tersebut pada variabel `KAGGLE_API_TOKEN` di dalam notebook:

```python
os.environ['KAGGLE_API_TOKEN'] = "isi_dengan_token_kaggle_anda"
```

### Langkah 2 — Membuka Notebook

1. Unggah berkas `BigdataTUBES_.ipynb` ke Google Colaboratory
2. Disarankan menggunakan runtime **GPU (T4)** untuk mempercepat proses komputasi melalui menu **Runtime → Change runtime type → T4 GPU**

### Langkah 3 — Menjalankan Notebook

Jalankan seluruh sel secara berurutan melalui **Runtime → Run all**, atau secara manual sesuai urutan berikut:

| Tahap | Keterangan |
|-------|------------|
| 1. Setup & Pemuatan Data | Instalasi library, inisialisasi Spark, pemuatan dataset TSV dari Kaggle |
| 2. Pembersihan Data | Penghapusan nilai kosong, konversi tipe data, pelabelan sentimen biner |
| 3. Rekayasa Fitur | Tokenisasi, penghapusan stop words, ekstraksi fitur TF-IDF |
| 4. Pelatihan Model | Perangkaian pipeline, pembagian data 80/20, pelatihan Regresi Logistik |
| 5. Evaluasi Model | Pengukuran AUC, Akurasi, F1-Score, Weighted Precision, dan Weighted Recall |
| 6. Analisis & Visualisasi | Pembuatan empat grafik analisis perilaku pelanggan |

---

## Arsitektur Pipeline NLP

```
review_body
    │
    ▼
Tokenizer           →  Memecah teks ulasan menjadi token kata
    │
    ▼
StopWordsRemover    →  Menghapus kata-kata umum yang tidak informatif
    │
    ▼
HashingTF           →  Merepresentasikan frekuensi kata dalam vektor (10.000 fitur)
    │
    ▼
IDF                 →  Menormalisasi bobot kata berdasarkan kepentingannya
    │
    ▼
LogisticRegression  →  Mengklasifikasikan sentimen (Positif / Negatif)
```

---

## Konfigurasi Apache Spark

```python
spark = SparkSession.builder \
    .appName("AmazonEbookBigDataAI") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
```

---

## Hasil Evaluasi Model

| Metrik | Nilai |
|--------|-------|
| Area Under ROC (AUC) | 0,9462 |
| Akurasi | 91,88% |
| F1-Score | 0,8984 |
| Weighted Precision | 0,9168 |
| Weighted Recall | 0,9188 |

---

## Hasil Analisis Perilaku Pelanggan

Empat dimensi analisis yang dihasilkan berdasarkan prediksi model:

1. **Pola Sentimen berdasarkan Status Verifikasi Pembelian** — Pelanggan dengan status terverifikasi cenderung memberikan ulasan yang lebih positif dibandingkan pelanggan tidak terverifikasi.
2. **Dampak Sentimen terhadap Helpful Votes** — Ulasan bersentimen negatif memperoleh rata-rata helpful votes yang lebih tinggi (~7,2) dibandingkan ulasan positif (~1,4).
3. **Korelasi Panjang Teks Ulasan dengan Sentimen** — Ulasan negatif rata-rata dua kali lebih panjang (~1.000 karakter) dibandingkan ulasan positif (~450 karakter).
4. **Distribusi Rating Bintang** — Dataset didominasi oleh rating bintang 5 (~3.000.000 ulasan), mengindikasikan ketidakseimbangan kelas yang signifikan.

---

## Catatan Penting

- Proses pengunduhan dataset memerlukan koneksi internet yang stabil karena ukuran berkas mencapai 3,22 GB.
- Proses pelatihan model memerlukan beberapa menit tergantung pada ketersediaan sumber daya komputasi Google Colaboratory.
- Ulasan dengan rating bintang 3 (netral) dieliminasi dari proses pelatihan model, namun tetap diikutsertakan dalam visualisasi distribusi rating bintang.
