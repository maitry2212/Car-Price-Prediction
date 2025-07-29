# 🚗 Car Price Prediction App

This is a **Streamlit web app** that predicts the **selling price of a used car** using a machine learning model that includes both structured data and Natural Language Processing (NLP) on the car name.

---

### 📊 Features

* Predicts car price based on:

  * Car name (NLP-based input)
  * Year of manufacture
  * Present price
  * Kilometers driven
  * Fuel type
  * Selling type
  * Transmission type
  * Number of previous owners
  * Interactive Streamlit UI with sliders, inputs, and dropdowns
  * Uses **TF-IDF Vectorizer** for text processing and **XGBoost Regressor** for prediction

---

### 📁 Project Structure

```
car-price-prediction-app/
├── app.py                 # Streamlit application
├── car data.csv           # Dataset used for model training
├── README.md              # Project documentation
```

---

### 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* TfidfVectorizer

---

### ▶️ How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/car-price-prediction-app.git
cd car-price-prediction-app
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas scikit-learn xgboost
```

3. **Run the App**

```bash
streamlit run app.py
```

### 📌 Notes

* The model is trained every time the app is run (demo purpose). For production, save the trained model using `joblib` and load it.
* You can customize the dataset or model parameters in `app.py`.

---

### 📄 License

This project is open source under the [MIT License](LICENSE).

---


