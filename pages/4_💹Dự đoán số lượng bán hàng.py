import streamlit as st
import random
import pandas as pd
import requests
import time
import datetime
import smtplib
from bs4 import BeautifulSoup
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from streamlit_image_coordinates import streamlit_image_coordinates
import plotly.express as px
import altair as alt   
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

original_price = st.number_input("Giá gốc")
discount_rate = st.number_input("Phần trăm giảm giá")
rating_average = st.number_input("Đánh giá trung bình", step=0.1) # Sử dụng step=0.1 để cho phép nhập số thập phân
review_count = st.number_input("Số lượng đánh giá")
if st.button("Dự đoán số lượng bán hàng"):
  
    df_product = pd.read_csv('data.csv')
    df_product = df_product.dropna()
    X = df_product[['Giá gốc', 'Tỷ lệ giảm giá', 'Đánh giá trung bình', 'Số lượng đánh giá']]
    y = df_product['Số lượng bán']
    
    # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    st.markdown("<h3 style='text-align: center;color: red'>Biểu đồ hồi quy tuyến tính</h3>", unsafe_allow_html=True)
    # st.header(":red[Biểu đồ hồi quy tuyến tính]")
 
    X_test_values = X_test.values
    for i in range(X_test.shape[1]):
                            plt.figure(figsize=(12, 6))
                            plt.scatter(X_test_values[:, i], y_test, color ='b', label='Actual')
                            plt.scatter(X_test_values[:, i], y_pred, color ='r', label='Predicted')
                            plt.xlabel('Feature {}'.format(i))
                            plt.ylabel('Output')
                            plt.legend()
                            plt.show()

    st.pyplot(plt)
    st.markdown(f"<h6 style='text-align: center;color: green'>Độ chính xác mô hình là:{accuracy}</h6>", unsafe_allow_html=True)


    input_data = pd.DataFrame({
        'Giá gốc': [original_price],
        'Tỷ lệ giảm giá': [discount_rate],
        'Đánh giá trung bình': [rating_average],
        'Số lượng đánh giá': [review_count]
    })

    predicted_price = model.predict(input_data)
    predicted_sales = float(predicted_price[0])
    st.write(f"Số lượng sản phẩm bán ra: {round(predicted_sales)}")


    