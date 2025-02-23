import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

def preprocess(input_data):
    # Encoding the categorical data
    encoder = LabelEncoder()
    for col in input_data.columns:
        if input_data[col].dtype == 'object':
            input_data[col] = encoder.fit_transform(input_data[col])

    # Scaling the input data
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    return input_data

def predict_data(input_data, model):
    # Predicting the data
    prediction = model.predict(input_data)
    return prediction

def load_model():
    # Load the trained model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def count_outliers(data):
    outlier_counts = {}
    for column in data.columns:
        if data[column].dtype != 'int64' and data[column].dtype != 'float64':
            continue
        feature_data = data[column]
        Q1 = np.percentile(feature_data, 10)
        Q3 = np.percentile(feature_data, 90)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
        outlier_counts[column] = len(outliers)
    return outlier_counts

def main():
    st.title('Raisin Class Prediction')
    st.write('The model used here is an XGB Classifier')
    About_work, Prediction = st.tabs(["About the Work", "Prediction"])
    with About_work:
        st.title('Data Information')
        st.write('The dataset should contain the following columns:')
        data = pd.read_csv('raw_data.csv')
        columns = data.columns.tolist()
        st.write(columns)
        st.write('The target variable is "Class" that contains Two Classes:')
        st.write(data['Class'].unique().tolist())
        st.markdown('### **About Data Quality :**')
        
        overview, missValues, outliers, imbalancedData = st.tabs(["Overview", "Missing Values", "Outliers", "Imbalanced Data"])
        with overview:
            st.write("the data overall seem to be in good shape, it contains around 900 records with 7 columns including the target variable, all the input features are numerical")
            st.write('the data does not seen to suffer from data quality issues, which makes the process of building a model easier and reliable')
            st.write('here is a view of the first 5 records of the data:')
            st.write(data.head())
            
        with missValues:
            st.write("the data does not contain any missing values, here is the count of missing values per column:")
            st.write(data.isnull().sum())
        with outliers:
            st.write("the data does not contain any outliers, we need to mention that the method we used to check for outliers is the soft IQR method since we are dealing with small data, here is the count of outliers per column:")
            st.write(count_outliers(data))
            
        with imbalancedData:
            st.write("The data does not seem to be imbalanced, here is the distribution of the target variable:")
            st.write(data['Class'].value_counts())
            
        st.markdown('### **Data Pre-processing :**')
        st.write('since the data is in good shape, we will only encode the categorical data and scale the input features')
        st.write('the data will be preprocessed using the following steps:')
        st.write('1. Encoding the categorical data')
        st.write('2. Scaling the input data')
        st.write('the input data should be in the following format: ')
        st.write(preprocess(data.drop('Class', axis=1).head()))
        
    with Prediction:
        st.title('Prediction')
        st.write('To make a prediction, please upload a file containing the data to be predicted.')
        st.markdown('### The submitted data should be in the following format:')
        st.write(data.drop('Class', axis=1).head())
        
        uploaded_file = st.file_uploader("Upload a file (CSV or Excel)", type=["csv", "xlsx"])

        if uploaded_file is not None:
            # Read the file into a DataFrame
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            st.success('The dataset was submitted successfully')
            st.write('Preview of the dataset:')
            st.dataframe(df.head())
            try:
                processed_data = preprocess(df)
                st.success('Data preprocessing completed successfully.')

                # Load the model
                model = load_model()
                st.success('Model loaded successfully.')

                # Making predictions
                prediction = predict_data(processed_data, model)
                st.write('Predictions:')
                st.write(prediction)

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
        else:
            st.error('Error: No file was uploaded.')
            st.warning('Please upload a file to make a prediction.')

if __name__ == '__main__':
    main()