# Panduan Peserta - Big Data Toolkit 2

## 1. Tujuan Aplikasi

Big Data Toolkit 2 membantu kamu menjalankan workflow machine learning tanpa coding. Di aplikasi ini kamu bisa upload data, melihat EDA, membagi data, membersihkan data, melakukan preprocessing, mencoba model, training model final, lalu membuat file submission.

Gunakan aplikasi ini sebagai alat bantu belajar. Fokus utamanya adalah memahami urutan kerja machine learning dan alasan di balik setiap langkah.

## 2. Sebelum Mulai

Checklist singkat:

- Siapkan train CSV.
- Siapkan test CSV untuk submission.
- Pastikan file berbentuk `.csv`.
- Cari tahu nama target column dari soal.
- Jangan upload file model `.pkl` atau `.joblib`.
- Pastikan nama kolom train dan test konsisten.
- Catat Session ID yang muncul di sidebar, misalnya `BDT-8F2A9C`.

## 3. Format Dataset

Train CSV harus punya target column, yaitu kolom yang ingin diprediksi. Contoh target: `Survived`, `price`, `label`, atau `target`.

Test CSV biasanya tidak punya target column. Test CSV dipakai di halaman Submission untuk membuat prediksi akhir.

Hal yang sebaiknya diperhatikan:

- Nama kolom train dan test sebaiknya sama untuk fitur yang dipakai model.
- Kolom ID boleh ada, tetapi biasanya tidak dipakai sebagai fitur.
- Missing value boleh ada, nanti bisa ditangani di halaman Cleansing.
- Categorical column boleh ada, nanti bisa diencoding di halaman Preprocessing.
- Hindari mengubah nama kolom secara manual di luar aplikasi setelah training, karena submission butuh struktur kolom yang konsisten.

## 4. Alur Cepat

1. Data: upload train CSV, pilih target column, pilih task type.
2. EDA: pahami isi data, missing value, distribusi target, dan pola awal.
3. Split Dataset: bagi data menjadi train/test internal.
4. Cleansing: tangani missing value, duplicate, outlier, atau kolom yang tidak perlu.
5. Preprocessing: scaling numeric feature, encoding categorical feature, sampling/resampling bila perlu.
6. Validation: bandingkan beberapa model dengan data training.
7. Training: latih model final.
8. Evaluation Summary: baca ringkasan performa model.
9. Submission: upload test CSV, generate prediction, download submission CSV.

## 5. Penjelasan Setiap Halaman

### Data

Gunakan halaman Data untuk upload train CSV. Setelah file terbaca, pilih target column dan task type.

Pilih Classification jika target berupa kelas/kategori, misalnya `yes/no`, `0/1`, atau nama kelas. Pilih Regression jika target berupa angka kontinu, misalnya harga, durasi, jumlah, atau skor.

### EDA

Gunakan EDA untuk mengenal data sebelum mengubahnya. Cek ukuran dataset, tipe kolom, missing value, distribusi target, dan korelasi jika datanya numerik.

EDA membantu kamu menentukan langkah berikutnya, misalnya kolom mana yang perlu dibersihkan atau diencoding.

### Split Dataset

Split Dataset membagi train CSV menjadi data training dan data hold-out internal. Data hold-out dipakai untuk mengecek performa model pada data yang belum dilihat saat training.

Rekomendasi sederhana: gunakan split 80:20. Untuk classification, gunakan stratify jika tersedia agar proporsi kelas lebih seimbang.

### Cleansing

Gunakan Cleansing untuk memperbaiki masalah kualitas data:

- handle missing value
- hapus duplicate
- treatment outlier
- drop column yang tidak perlu
- ubah tipe data jika perlu

Setelah cleansing, hasil validation/training lama akan dibersihkan karena data sudah berubah.

### Preprocessing

Gunakan Preprocessing untuk menyiapkan fitur agar bisa dipakai model:

- scaling untuk fitur numerik
- encoding untuk fitur kategorikal
- feature engineering sederhana
- value mapping
- sampling/resampling jika data imbalance

One-hot encoding cocok untuk kolom kategorikal dengan jumlah kategori sedikit. Jika kategori terlalu banyak, aplikasi akan menolak one-hot agar jumlah kolom tidak meledak.

### Validation

Validation dipakai untuk membandingkan beberapa model sebelum memilih model final. Gunakan CV folds secukupnya, biasanya 3 atau 5.

Permutation importance nonaktif secara default karena cukup berat. Aktifkan hanya jika dataset kecil dan kamu memang ingin melihat estimasi pentingnya fitur.

### Training

Training melatih model final berdasarkan pilihan model, fitur, dan dataset training. Setelah training, aplikasi menyimpan model di session browser saat ini.

Untuk classification, cek accuracy, precision, recall, F1-score, dan confusion matrix. Untuk regression, cek MAE, RMSE, dan R2.

### Evaluation Summary

Evaluation Summary menampilkan ringkasan hasil validation dan training. Gunakan halaman ini untuk membandingkan performa dan memastikan hasil model masuk akal sebelum membuat submission.

### Submission

Gunakan Submission untuk upload test CSV dan membuat prediksi akhir. Aplikasi akan memakai model yang dilatih dari session saat ini atau saved bundle server-side yang tersedia.

Test CSV harus punya kolom fitur yang sesuai dengan fitur saat training. Extra column aman karena akan diabaikan jika tidak dibutuhkan, tetapi missing required feature akan ditolak.

## 6. Tips Pemilihan Preprocessing

- Numeric feature: gunakan scaling jika model sensitif skala, seperti Logistic Regression, SVM, KNN, atau model linear.
- Tree-based model: RandomForest biasanya tidak wajib scaling.
- Categorical dengan sedikit kategori: gunakan one-hot encoding.
- Categorical dengan urutan jelas: gunakan ordinal encoding.
- Categorical biner: label encoding boleh dipakai.
- High-cardinality categorical: hindari one-hot; pertimbangkan drop kolom, mapping manual, atau encoding lain.
- Missing value numerik: coba mean/median sesuai distribusi.
- Missing value kategorikal: coba mode atau isi dengan kategori seperti `Unknown`.

## 7. Tips Membaca Metric

Classification:

- Accuracy: persentase prediksi benar. Bagus untuk data yang kelasnya cukup seimbang.
- Precision: dari semua prediksi positif, berapa yang benar-benar positif.
- Recall: dari semua data positif asli, berapa yang berhasil ditemukan model.
- F1-score: gabungan precision dan recall. Berguna saat data tidak seimbang.
- Confusion matrix: tabel yang menunjukkan prediksi benar/salah untuk tiap kelas.

Regression:

- MAE: rata-rata besar error. Lebih mudah dibaca karena satuannya sama dengan target.
- MSE/RMSE: memberi penalti lebih besar untuk error besar. RMSE satuannya sama dengan target.
- R2: seberapa baik model menjelaskan variasi target. Makin mendekati 1 biasanya makin baik.

Tidak ada metric yang selalu paling benar untuk semua kasus. Pilih metric sesuai tujuan soal.

## 8. Batasan Aplikasi

Batasan untuk sesi study group:

- Train CSV maksimal 10 MB.
- Test CSV maksimal 5 MB.
- Maksimal 100.000 rows.
- Maksimal 200 columns.
- One-hot maksimal 50 kategori unik per kolom.
- One-hot maksimal 500 output column baru.
- Cross-validation maksimal 5 folds.
- RandomForest maksimal 100 estimators.
- Permutation importance nonaktif secara default.
- Upload `.pkl` / `.joblib` dari user dinonaktifkan untuk keamanan.
- Feature engineering tertentu mungkin belum bisa direplay saat Submission. Jika replay gagal, aplikasi akan menampilkan error dan prediction tidak dilanjutkan.
- Progress sementara melalui Session ID kedaluwarsa setelah 2 jam sejak aktivitas terakhir.

Batasan ini sengaja dipasang supaya aplikasi tetap stabil saat dipakai banyak peserta secara bersamaan.

## 9. Temporary Session Recovery

Di sidebar ada bagian **Temporary Session Recovery**.

Yang perlu kamu ingat:

- "Session ID Anda: BDT-XXXXXX" adalah kode sementara untuk browser/session kamu.
- Simpan Session ID ini. Jika halaman refresh atau koneksi terputus, masukkan Session ID yang sama lalu klik **Load Progress**.
- Klik **Save Progress Now** setelah langkah penting, misalnya setelah confirm dataset, split, preprocessing, training, atau membuat submission.
- Progress disimpan sementara selama 2 jam sejak aktivitas terakhir. Setelah itu, progress akan otomatis dihapus.
- **Clear My Progress** menghapus progress tersimpan untuk Session ID kamu.
- Session ID bukan akun/login. Jangan bagikan Session ID ke peserta lain.

Catatan penting: siapa pun yang tahu Session ID dapat mencoba memuat progress sementara selama belum kedaluwarsa. Jangan simpan data rahasia di aplikasi workshop ini.

## 10. Troubleshooting

### File saya gagal di-upload

Cek ukuran file, format CSV, jumlah row, dan jumlah column. Train CSV maksimal 10 MB, test CSV maksimal 5 MB, maksimal 100.000 rows, dan maksimal 200 columns.

### CSV terbaca aneh atau kolomnya tidak sesuai

Pastikan file memakai format CSV standar dengan header di baris pertama. Jika file memakai delimiter selain koma, rapikan dulu file CSV sebelum upload.

### One-hot encoding ditolak

Kolom mungkin punya terlalu banyak kategori unik. One-hot cocok untuk kategori sedikit. Untuk kategori yang sangat banyak, coba drop kolom, mapping manual, label/ordinal encoding, atau pilih fitur lain.

### Validation atau training terasa lambat

Kurangi jumlah fitur, hindari one-hot untuk kolom high-cardinality, pakai CV folds 3, dan gunakan RandomForest estimator yang tidak terlalu besar.

### Metric saya jelek

Cek missing value, encoding, imbalance, pemilihan fitur, dan split data. Coba model lain atau preprocessing yang lebih sesuai.

### Submission gagal karena missing feature

Kolom test tidak sesuai dengan fitur saat training. Pastikan nama kolom test sama dengan train untuk fitur yang dipakai. Jangan hapus atau rename kolom yang dibutuhkan model.

### Prediction gagal setelah feature engineering

Beberapa feature engineering belum bisa direplay otomatis saat submission. Jika ini terjadi, gunakan fitur asli yang sudah ada di test CSV atau ulang workflow dengan preprocessing yang bisa direplay.

### Halaman refresh dan progress hilang

Buka bagian **Temporary Session Recovery** di sidebar, masukkan Session ID yang kamu catat, lalu klik **Load Progress**. Jika belum pernah klik **Save Progress Now** atau sudah lebih dari 2 jam sejak aktivitas terakhir, progress mungkin tidak bisa dipulihkan.

## 11. Rekomendasi Workflow untuk Penugasan

Workflow sederhana yang aman untuk mulai:

1. Upload train CSV.
2. Pilih target column dan task type.
3. Cek EDA.
4. Split 80:20.
5. Handle missing value.
6. Encode categorical feature.
7. Scale numeric feature jika model membutuhkannya.
8. Validate 2-3 model.
9. Train model terbaik.
10. Upload test CSV.
11. Generate dan download submission CSV.

Catat eksperimen terbaikmu: model apa, preprocessing apa, dan metric apa yang menjadi alasan kamu memilih model tersebut.
