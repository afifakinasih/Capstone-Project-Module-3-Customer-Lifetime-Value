# Customer Lifetime Value Prediction

Proyek ini bertujuan untuk membangun model prediksi Customer Lifetime Value (CLV) pada data pelanggan asuransi menggunakan machine learning. Model yang dihasilkan dapat membantu perusahaan dalam melakukan segmentasi pelanggan, strategi retensi, dan pengambilan keputusan bisnis berbasis data.

## 1. Deskripsi Proyek

- **Business Problem:** Memprediksi nilai CLV pelanggan berdasarkan data demografi, produk asuransi, dan perilaku transaksi.
- **Dataset:** Data pelanggan asuransi dengan fitur numerik dan kategorikal, seperti Vehicle Class, Employment Status, Marital Status, Coverage, Education, Monthly Premium Auto, Total Claim Amount, Income, dan Number of Policies.

## 2. Proses Data Science

### A. Data Cleaning & Outlier Handling
- Outlier pada fitur numerik diidentifikasi dan diatasi menggunakan metode IQR, namun data pelanggan bernilai tinggi (VIP) tetap dipertahankan agar model dapat belajar dari pola mereka.
- Data dibersihkan dan disimpan dalam file `data_customer_lifetime_value_clean.csv`.

### B. Feature Engineering & Preprocessing
- Encoding fitur kategorikal menggunakan OneHotEncoder dan OrdinalEncoder sesuai karakteristik data.
- Scaling fitur numerik diuji dengan beberapa metode (StandardScaler, MinMaxScaler, RobustScaler).
- Seluruh proses preprocessing digabung dalam `ColumnTransformer` untuk pipeline yang konsisten.

### C. Modeling & Evaluation
- Berbagai algoritma regresi diuji: Linear Regression, Lasso, Ridge, KNN, Decision Tree, Random Forest, AdaBoost, XGBoost, Gradient Boosting, dan Stacking.
- Evaluasi awal menggunakan cross-validation pada data latih dengan metrik RMSE, MAE, MAPE, dan R².
- Model terbaik: **Gradient Boosting Regressor** dengan RobustScaler, setelah hyperparameter tuning menggunakan GridSearchCV.
- Performa akhir pada data uji: R² ≈ 0.68, RMSE ≈ 3.858, MAE ≈ 1.722, MAPE ≈ 13.8%.

### D. Interpretasi Model
- Fitur paling berpengaruh: **Number of Policies** dan **Monthly Premium Auto**.
- Interpretasi global dan lokal menggunakan Feature Importance dan SHAP (SHapley Additive exPlanations).
- Model cenderung underpredict pada pelanggan dengan CLV sangat tinggi, namun tetap cukup akurat untuk segmen pelanggan lain.

### E. Deployment
- Model dan pipeline preprocessing disimpan menggunakan pickle (`model_gradient_boosting_tuned.pkl`).
- Contoh prediksi untuk pelanggan baru disediakan.
- File terpisah untuk deployment dengan Streamlit: `CLV_prediction.py`.

## 3. Insight & Rekomendasi

- **Segmentasi pelanggan** berdasarkan prediksi CLV untuk strategi pemasaran dan retensi yang lebih efektif.
- **Cross-selling** dan optimasi premi bulanan dapat meningkatkan CLV.
- **Monitoring model** secara berkala untuk menjaga akurasi seiring perubahan data.
- **Interpretasi model** menggunakan SHAP untuk transparansi dan pengambilan keputusan bisnis.

## 4. File Terkait

- `data_customer_lifetime_value_clean.csv` : Dataset hasil pembersihan.
- `model_gradient_boosting_tuned.pkl` : Model Gradient Boosting yang sudah di-tuning.
- `CLV_prediction.py` : Script Streamlit untuk deployment prediksi CLV.

---

**Catatan:**  
Model ini hanya valid untuk data dengan skema dan rentang fitur yang sama seperti pada proses pelatihan. Input di luar rentang/isi fitur yang telah ditentukan dapat menghasilkan prediksi yang tidak valid.