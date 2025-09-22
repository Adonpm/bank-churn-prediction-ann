# Bank Churn Prediction using Artificial Neural Network

A deep learning project that predicts customer churn for banks using an Artificial Neural Network (ANN) built with TensorFlow/Keras and deployed with Streamlit.

## 🎯 Project Overview

Customer churn is a critical business metric for banks, representing the percentage of customers who stop using the bank's services during a certain timeframe. This project uses machine learning to predict which customers are likely to churn, enabling proactive retention strategies.

## 📁 Project Structure

```
bank-churn-prediction-ann/
├── data/
│   └── Churn_Modelling.csv          # Dataset for training and testing
├── logs/
│   └── fit/
│       └── 20250920-193008/         # TensorBoard logs
│           ├── train/               # Training logs
│           └── validation/          # Validation logs
├── models/
│   ├── ann/
│   │   └── churn_ann_model.h5       # Trained ANN model
│   └── encoders/
│       ├── label_encoder_gender.pkl # Gender label encoder
│       ├── onehot_encoder_geo.pkl   # Geography one-hot encoder
│       └── scaler.pkl               # Feature scaler
├── notebooks/
│   └── experiments.ipynb           # Data exploration and model development
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

## 🚀 Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Neural Network Model**: Deep learning model built with TensorFlow/Keras
- **Web Interface**: Interactive Streamlit application for real-time predictions
- **Model Persistence**: Saved model and encoders for deployment
- **Visualization**: TensorBoard integration for training monitoring

## 📊 Dataset Features

The model uses the following customer features for prediction:

- **CreditScore**: Customer's credit score (350-850)
- **Geography**: Customer's country (France, Germany, Spain)
- **Gender**: Customer's gender (Male/Female)
- **Age**: Customer's age (18-92 years)
- **Tenure**: Number of years with the bank (0-10)
- **Balance**: Account balance
- **NumOfProducts**: Number of bank products used (1-4)
- **HasCrCard**: Whether customer has a credit card (0/1)
- **IsActiveMember**: Whether customer is active (0/1)
- **EstimatedSalary**: Customer's estimated salary

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Adonpm/bank-churn-prediction-ann.git
   cd bank-churn-prediction-ann
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Requirements

```
streamlit
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
plotly
```

## 🏃‍♂️ Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The web application will open in your browser, allowing you to:
- Input customer information through interactive widgets
- Get real-time churn probability predictions
- View whether the customer is likely to churn or not

### Training the Model

To retrain the model with new data:

1. Open `notebooks/experiments.ipynb`
2. Load your dataset in the same format as `Churn_Modelling.csv`
3. Run through the data preprocessing steps
4. Train the neural network
5. Save the model and encoders

## 🧠 Model Architecture

The ANN model consists of:
- Input layer matching the number of features
- Hidden layers with ReLU activation
- Dropout layers for regularization
- Output layer with sigmoid activation for binary classification
- Binary crossentropy loss function
- Adam optimizer

## 📈 Model Performance

The model's performance can be monitored through:
- **TensorBoard logs**: Located in `logs/fit/20250920-193008/`
- **Training/Validation metrics**: Accuracy, loss, and other metrics
- **Real-time predictions**: Through the Streamlit interface

To view TensorBoard:
```bash
tensorboard --logdir logs/fit
```

## 🔄 Data Preprocessing

The preprocessing pipeline includes:

1. **Label Encoding**: Gender feature converted to numerical values
2. **One-Hot Encoding**: Geography feature expanded into binary columns
3. **Feature Scaling**: StandardScaler applied to numerical features
4. **Data Splitting**: Train/validation/test splits for model evaluation

## 🌐 Deployment

The application is ready for deployment on various platforms:

- **Streamlit Cloud**: Direct deployment from GitHub

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset source: [Kaggle - Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- TensorFlow/Keras for deep learning framework
- Streamlit for web application framework
- Scikit-learn for preprocessing utilities

## 📞 Contact

For questions or suggestions, please open an issue on GitHub or contact [adon.pmpm@gmail.com]

## 🔮 Future Enhancements

- [ ] Add more sophisticated feature engineering
- [ ] Implement ensemble methods
- [ ] Add model explainability features (SHAP, LIME)
- [ ] Create API endpoints for model serving
- [ ] Add automated model retraining pipeline
- [ ] Implement A/B testing framework
- [ ] Add customer segmentation analysis

---

**Made with ❤️ using TensorFlow, Streamlit, and Python**