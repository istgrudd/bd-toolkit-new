# Panduan Mentor - Big Data Toolkit 2

## Tujuan

Panduan ini membantu mentor/asisten lab menjalankan sesi study group Big Data Toolkit 2 secara lancar. Fokus sesi adalah membantu peserta memahami workflow machine learning tanpa coding, bukan mengejar tuning model yang terlalu rumit.

## Sebelum Sesi

- Pastikan app bisa dijalankan dengan `streamlit run app.py`.
- Pastikan dataset train dan test sudah tersedia untuk peserta.
- Pastikan peserta tahu target column dan jenis task.
- Jelaskan batas upload: train 10 MB, test 5 MB, maksimal 100.000 rows dan 200 columns.
- Ingatkan bahwa upload model `.pkl` / `.joblib` dari peserta dinonaktifkan.
- Minta peserta mencatat Session ID di sidebar sebelum mulai praktik.
- Siapkan contoh jawaban untuk error umum: upload ditolak, one-hot ditolak, dan submission missing feature.

## Saat Sesi

- Arahkan peserta mengikuti urutan halaman: Data, EDA, Split Dataset, Cleansing, Preprocessing, Validation, Training, Evaluation Summary, Submission.
- Tekankan pentingnya split sebelum menilai performa model.
- Ingatkan peserta agar memilih encoding sesuai tipe kolom.
- Bantu peserta membaca metric sesuai task.
- Untuk classification, gunakan confusion matrix untuk menjelaskan jenis error.
- Untuk regression, jelaskan MAE/RMSE sebagai ukuran besar error.
- Sarankan peserta mencatat preprocessing dan model yang menghasilkan metric terbaik.
- Setelah milestone penting, ingatkan peserta klik **Save Progress Now** di sidebar.

## Temporary Session Recovery

Gunakan bagian **Temporary Session Recovery** saat peserta mengalami refresh, reconnect, atau tab tertutup.

Langkah bantu cepat:

1. Minta peserta membuka sidebar.
2. Minta peserta memasukkan Session ID lama, misalnya `BDT-8F2A9C`.
3. Klik **Load Progress**.
4. Jika berhasil, cek halaman Data/Split/Training untuk memastikan workflow kembali.
5. Jika gagal karena kedaluwarsa, jelaskan bahwa progress sementara hanya berlaku 2 jam sejak aktivitas terakhir.

Peringatan untuk mentor: Session ID bukan akun/login dan bukan auth. Jangan minta peserta membagikan Session ID di grup besar. Siapa pun yang mengetahui Session ID dapat mencoba memuat progress sementara selama TTL masih aktif.

## Common Issues

### Upload gagal

Cek ukuran file, format CSV, jumlah row, dan jumlah column.

### One-hot ditolak

Kolom punya terlalu banyak kategori unik. Sarankan peserta memilih kolom lain, memakai label/ordinal encoding, atau drop kolom jika memang tidak informatif.

### Submission missing feature

Nama atau struktur kolom test tidak sesuai dengan fitur saat training. Minta peserta membandingkan kolom train/test dan menghindari rename/drop kolom penting.

### Model terlalu lambat

Kurangi fitur, turunkan CV folds ke 3, gunakan estimator RandomForest 50-100, dan hindari one-hot untuk high-cardinality categorical.

### Metric tidak sesuai ekspektasi

Cek target, missing value, imbalance, preprocessing, dan apakah peserta memilih task type yang benar.

## Recommended Defaults

- Split: 80:20
- Classification: stratify jika tersedia
- CV folds: 3 untuk cepat, 5 untuk evaluasi lebih stabil
- RandomForest estimators: 50-100
- One-hot: hanya untuk kategori sedikit
- Permutation importance: biarkan off kecuali dataset kecil dan peserta butuh interpretasi fitur

## Batasan yang Perlu Dijelaskan

- Aplikasi cocok untuk workshop/study group terkontrol, bukan public production.
- Data dan model disimpan di session browser dan bisa disimpan sementara lewat Session ID anonim.
- Session ID bukan login/auth, tidak menyediakan isolasi privasi seperti akun sungguhan.
- CV leakage belum sepenuhnya diselesaikan karena pipeline belum full sklearn Pipeline.
- Feature engineering tertentu belum bisa direplay penuh saat submission.
- Untuk tahap belajar, gunakan workflow sederhana dan hindari eksperimen yang terlalu berat.
