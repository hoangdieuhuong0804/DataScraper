import streamlit as st
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

df_product = pd.read_csv('data.csv')
product_list = df_product['Tên'].values
ph = st.empty()

selected = ph.selectbox("Chọn sản phẩm bạn yêu thích", product_list)

if st.button("Gợi ý sản phẩm"):
  
    def combineFeature(row):
        return str(row['Giá']) + ',' + str(row['Mô tả ngắn'])

    df_product['combined_feature'] = df_product.apply(combineFeature, axis=1)

    tf = TfidfVectorizer()
    tfMatrix = tf.fit_transform(df_product['combined_feature'])
    # Tính toán ma trận tương đồng cosine
    similarity_matrix = cosine_similarity(tfMatrix)

    def recommend(product):
        index = df_product[df_product['Tên'] == product].index[0]
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda vector: vector[1])
        recommend_products = []
        for i in distances[1:6]:
            recommend_products.append(df_product.iloc[i[0]])
        return recommend_products

    product_recommendations = recommend(selected)
    st.subheader("Các sản phẩm tương tự là:")
    for product in product_recommendations:
            st.write("Tên sản phẩm:", product['Tên'])
            st.write("Link:", product['Đường dẫn ngắn'])
            st.write("Giá:", product['Giá'])
            st.write("Điểm đánh giá:", product['Đánh giá trung bình'])
            st.write("---")

with st.sidebar.expander("Theo dõi sản phẩm"):


    st.title("Theo dõi sản phẩm bạn yêu thích")

    url = st.text_input("Nhập link sản phẩm")
    desired_price = st.number_input("Nhập giá mong muốn")
    recipient_email = st.text_input("Nhập địa chỉ email của bạn")


    def track_product(url, desired_price, recipient_email):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
        }
        page = requests.get(url, headers=headers)
        soup1 = BeautifulSoup(page.content, "html.parser")
        soup2 = BeautifulSoup(soup1.prettify(), "html.parser")

        title = soup2.find(class_='Title__TitledStyled-sc-c64ni5-0 iXccQY').get_text()
        price_str = soup2.find(class_='product-price__current-price').get_text()

        # Extract the numeric part of the price string
        price_str = price_str.replace('₫', '').replace('.', '')
        price = float(price_str) / 100  # Assuming the price is in Vietnamese Dong (VND)

        title = title.strip()
        today = datetime.date.today()

        df = pd.read_csv('Tiki.csv')

        header = ['Title', 'Price', 'Date']
        data = [title, price, today]

        df_new = pd.DataFrame(data=[data], columns=header)
        df = pd.concat([df, df_new], ignore_index=True)

        df.to_csv('Tiki.csv', index=False)
        # st.write(df)

        if price < desired_price:
            send_mail(recipient_email)

    def send_mail(recipient_email):
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login('hoangdieuhuong08042003@gmail.com', 'ndjd jtoq jmtf miuw')

        subject = f"The product you want is below the {desired_price} ! Now is your chance to buy!"
        body = f"Hello {recipient_email},\n\nThis is the moment we have been waiting for. Now is your chance to pick up the product you want at a great price. Don't miss out!\n\nLink: {url}"

        msg = f"Subject: {subject}\n\n{body}"

        server.sendmail('hoangdieuhuong08042003@gmail.com', recipient_email, msg)



    if st.button("Xác nhận"):
        track_product(url, desired_price, recipient_email)

