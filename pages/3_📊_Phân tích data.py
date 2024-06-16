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
import plotly.graph_objects as go


df_product = pd.read_csv('data.csv')

# Sắp xếp dữ liệu theo điểm đánh giá giảm dần và giá tăng dần
sorted_df = df_product.sort_values(by=['Đánh giá trung bình', 'Giá'], ascending=[False, True])

# Lấy thông tin của sản phẩm tốt nhất (sản phẩm ở hàng đầu tiên)
best_product = sorted_df.iloc[0]

# In ra thông
# tin của sản phẩm tốt nhất
st.markdown(f"<h2 style='text-align:center; font-weight:bold;'>SẢN PHẨM ĐÁNG MUA NHẤT:</h2>", unsafe_allow_html=True)

st.write("Tên sản phẩm:", best_product['Tên'])
st.write("Link",best_product['Đường dẫn ngắn'])
st.write("Giá:", best_product['Giá'])
st.write("Điểm đánh giá:",  best_product['Đánh giá trung bình'])

st.markdown(f"<h2 style='text-align:center; font-weight:bold;'>MỘT SỐ BIỂU ĐỒ TRỰC QUAN</h2>", unsafe_allow_html=True)
chart1, chart2=st.columns(2)
with chart1:
    brand_counts = df_product.groupby('Tên thương hiệu').size().reset_index(name='Số lượng')

    fig = go.Figure(data=[go.Pie(labels=brand_counts['Tên thương hiệu'], values=brand_counts['Số lượng'])])
    fig.update_layout(
        title='Phân bố Thương hiệu'
    )

    st.plotly_chart(fig, use_container_width=True)
with chart2:


    # Tính toán tỷ lệ giảm giá trung bình của mỗi thương hiệu
    df_brand = df_product.groupby('Tên thương hiệu').agg({'ID': 'count', 'Tỷ lệ giảm giá': 'mean'}).reset_index()

    # Tính tổng tỷ lệ giảm giá
    total_discount = sum(df_brand['Tỷ lệ giảm giá'])

    # Tính tỷ lệ phần trăm giảm giá của mỗi thương hiệu
    discount_percentages = [(discount / total_discount) * 100 for discount in df_brand['Tỷ lệ giảm giá']]

    # Tạo biểu đồ tròn
    fig = go.Figure(data=[go.Pie(labels=df_brand['Tên thương hiệu'], values=discount_percentages)])
    fig.update_layout(title="Tỷ Lệ Phân Chia Giảm Giá Theo Thương Hiệu")

    # Hiển thị biểu đồ bằng Streamlit
    st.plotly_chart(fig, use_container_width=True)
chart3, chart4=st.columns(2)
with chart3:
    # Lấy cột điểm đánh giá trung bình (rating_average) từ DataFrame
    ratings = df_product['Đánh giá trung bình']

    # Đếm số lượng đánh giá cho mỗi điểm đánh giá trung bình
    rating_counts = ratings.value_counts()

    # Sắp xếp lại theo thứ tự tăng dần của điểm đánh giá
    rating_counts = rating_counts.sort_index()

    # Tạo biểu đồ cột bằng Plotly
    fig = go.Figure(data=[go.Bar(x=rating_counts.index, y=rating_counts.values)])
    fig.update_layout(title="Phân bố điểm đánh giá trung bình",
                    xaxis_title="Đánh giá trung bình",
                    yaxis_title="Số lượng đánh giá")

    # Hiển thị biểu đồ bằng Streamlit
    st.plotly_chart(fig, use_container_width=True)

with chart4:

    # Lấy cột giá (price) từ DataFrame
    prices = df_product['Giá']

    # Tạo biểu đồ histogram bằng Plotly
    fig = go.Figure(data=[go.Histogram(y=prices)])
    fig.update_layout(title="Phân bố giá sản phẩm",
                    yaxis_title="Giá",
                    xaxis_title="Số lượng sản phẩm")

    # Hiển thị biểu đồ bằng Streamlit
    st.plotly_chart(fig, use_container_width=True)

chart5, chart6=st.columns(2)
with chart5:
    # Tạo DataFrame rate_main_cat từ nhóm công việc
    rate_main_cat = df_product.groupby(['Tên thương hiệu', 'Đánh giá trung bình']).size().reset_index(name='Amount')

    # Tạo biểu đồ Box Plot tương tác
    fig = px.box(rate_main_cat, y='Đánh giá trung bình',x='Tên thương hiệu')
    fig.update_layout(
        title='Phân bổ xếp hạng theo sản phẩm các nhãn hiệu',
        yaxis_title='Rating',
        xaxis_title='Nhãn hiệu',
    )

    # Hiển thị biểu đồ bằng Streamlit
    st.plotly_chart(fig)

with chart6:
    # Biểu đồ hộp
   
    fig = px.box(df_product, x='Tên thương hiệu', y='Giá')
    fig.update_layout(
        title="Phân bố giá theo nhãn hiệu"
    )
    st.plotly_chart(fig)

with st.sidebar.form(key="filter_form"):
    df_product = pd.read_csv('data.csv')

    selected_brands = st.multiselect('Chọn thương hiệu', df_product['Tên thương hiệu'].unique())
    price_range = st.slider('Chọn khoảng giá', float(df_product['Giá'].min()), float(df_product['Giá'].max()), (float(df_product['Giá'].min()), float(df_product['Giá'].max())))
    min_rating = st.slider("Đánh giá tối thiểu", 1.0, 5.0, 4.0, 0.1)
    min_reviews = st.slider("Số lượng đánh giá tối thiểu", 0, df_product['Số lượng đánh giá'].max(), 10)
    submit_button = st.form_submit_button(label="Lọc")
if submit_button:
        
    if not df_product.empty:
            filtered_df = df_product[(df_product['Tên thương hiệu'].isin(selected_brands)) &
                                    (df_product['Giá'] >= price_range[0]) & (df_product['Giá'] <= price_range[1]) &
                                    (df_product['Đánh giá trung bình'] >= min_rating) &
                                    (df_product['Số lượng đánh giá'] >= min_reviews)]
            if not filtered_df.empty:
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.write("No products found with the selected filters.")
    else:
        st.write("No data available. Please check the data file.")


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

