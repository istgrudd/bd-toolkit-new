# Big Data Toolkit 2

Big Data Toolkit 2 adalah aplikasi Streamlit untuk workflow end-to-end machine learning tanpa coding. Aplikasi ini dibuat untuk kegiatan study group divisi Big Data, terutama untuk membantu peserta menjalankan alur dari upload CSV sampai membuat submission CSV.

## Fitur Utama

- Upload dataset CSV
- Exploratory Data Analysis
- Train/test split
- Data cleansing
- Data preprocessing
- Model validation
- Model training
- Evaluation summary
- Submission CSV generation

## Quick Start

1. Jalankan aplikasi dengan `streamlit run app.py`.
2. Upload train CSV di halaman Data.
3. Pilih target column dan task type.
4. Lakukan EDA untuk memahami data.
5. Split dataset agar model bisa dievaluasi dengan data hold-out.
6. Lakukan cleansing bila perlu.
7. Lakukan preprocessing untuk scaling, encoding, atau sampling.
8. Jalankan validation untuk membandingkan model.
9. Train model final.
10. Upload test CSV di Submission.
11. Generate dan download submission CSV.

## Batasan Dataset untuk Study Group

- Train CSV maksimal 10 MB
- Test CSV maksimal 5 MB
- Maksimal 100.000 rows
- Maksimal 200 columns
- One-hot encoding dibatasi untuk mencegah ledakan kolom/RAM
- Cross-validation maksimal 5 folds
- RandomForest maksimal 100 estimators
- Permutation importance nonaktif secara default
- Upload model `.pkl` / `.joblib` dari user dinonaktifkan untuk keamanan

## Temporary Session Recovery

Aplikasi menampilkan Session ID anonim di sidebar, misalnya `BDT-8F2A9C`.
Gunakan tombol **Save Progress Now** untuk menyimpan progress sementara.
Jika halaman refresh atau koneksi terputus, masukkan Session ID yang sama di
sidebar lalu klik **Load Progress**.

Progress tersimpan di server selama 2 jam sejak aktivitas terakhir pada Session
ID tersebut, lalu otomatis kedaluwarsa dan dibersihkan. Session ID bukan
akun/login dan tidak memberi proteksi privasi; siapa pun yang mengetahui
Session ID dapat mencoba memuat progress sementara selama belum kedaluwarsa.

## Untuk Peserta

Lihat panduan lengkap di [docs/PANDUAN_PESERTA.md](docs/PANDUAN_PESERTA.md).

## Untuk Mentor / Asisten Lab

Lihat panduan singkat sesi di [docs/PANDUAN_MENTOR.md](docs/PANDUAN_MENTOR.md).

## Untuk Developer

Report hardening dan implementasi bertahap tersedia di folder [reports/](reports/):

- Stage 1: safety & resource limits
- Stage 2: session state cleanup & invalidation
- Stage 3: ML correctness fixes
- Stage 4: participant documentation

## Menjalankan Aplikasi

Disarankan memakai Python environment terpisah.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Untuk workshop, jalankan aplikasi di lingkungan internal/terkontrol dan gunakan dataset kecil sesuai batasan di atas.
