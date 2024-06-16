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

st.markdown('__📅Phân tích sản phẩm tiki__ ')

# Hiển thị/collapse trường đầu vào cho giá trị của headers
with st.expander("Headers"):
    referer = st.text_input("Referer")

# Hiển thị/collapse trường đầu vào cho giá trị của params
with st.expander("Params"):
    trackity_id = st.text_input("trackity_id")
    q = st.text_input("q")

# Hiển thị/collapse trường đầu vào cho giá trị của params2
with st.expander("Params2"):
    spid = st.text_input("spid")
    version = st.text_input("version")

# Sử dụng st.columns() để tạo hai cột
col1, col2 = st.columns(2)

# Button 1 trong cột 1
button1 = col1.button("🛒Lấy sản phẩm")

# Button 2 trong cột 2
button2 = col2.button("📟Xem sản phẩm")

# Tạo nút "Lưu sản phẩm"
if button1:
  

    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language':'en-US,en;q=0.9,vi;q=0.8',
        'Referer': referer,
        'x-guest-token': 'OrALSigQ1ofYvNdzewknbh5DKTtm3HWj',
    }
    # Định dạng lại giá trị params
    params = {
        'limit': '40',
        'include': 'advertisement',
        'aggregations': '2',
        'trackity_id': trackity_id,
        'q': q,
    }



    product_id = []

    for i in range(1, 3):
        params['page'] = i
        response = requests.get('https://tiki.vn/api/v2/products', headers=headers, params=params)
        if response.status_code == 200:
            print('request success!!!')
            for record in response.json().get('data'):
                product_id.append({'id': record.get('id')})
        time.sleep(random.randrange(3, 10))

    df = pd.DataFrame(product_id)
    df.to_csv('product_id_ncds.csv', index=False)
        # Xem sản phẩm
    headers2 = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language':'en-US,en;q=0.9,vi;q=0.8',
        'Referer': 'https://tiki.vn/apple-iphone-11-p184036446.html?spid=32033721',
        'x-guest-token': 'OrALSigQ1ofYvNdzewknbh5DKTtm3HWj',

    }

    params2 = (
        ('platform','web'),
        ('spid', spid),
        ('version', version),

    )
    def parser_product(json):
        d = dict()
        d['id'] = json.get('id')
        d['name'] = json.get('name')
        d['short_url'] = json.get('short_url')
        d['original_price'] = json.get('original_price')
        d['price'] = json.get('price')
        d['discount_rate'] = json.get('discount_rate')
        d['quantity_sold'] = json.get('quantity_sold', {}).get('value', 0)
        d['rating_average'] = json.get('rating_average')
        d['review_count'] = json.get('review_count')
        d['brand_name'] = json.get('brand').get('name')
        d['short_description'] = json.get('short_description')
        return d


    df_id = pd.read_csv('product_id_ncds.csv')
    p_ids = df_id.id.to_list()
    print(p_ids)
    result = []
    for pid in tqdm(p_ids, total=len(p_ids)):
        response = requests.get('https://tiki.vn/api/v2/products/{}'.format(pid), headers=headers2, params=params2)
        if response.status_code == 200:
            print('Crawl data {} success !!!'.format(pid))
            result.append(parser_product(response.json()))
            # time.sleep(random.randrange(3, 5))
    df_product = pd.DataFrame(result)
    df_product.columns = ['ID', 'Tên', 'Đường dẫn ngắn', 'Giá gốc', 'Giá', 'Tỷ lệ giảm giá', 'Số lượng bán', 'Đánh giá trung bình', 'Số lượng đánh giá', 'Tên thương hiệu', 'Mô tả ngắn']
    df_product.to_csv('data.csv', index=False, encoding='utf-8-sig')
    st.write('Đã lấy data thành công')
pd.set_option('display.max_colwidth', None)
# # Xem sản phẩm
if button2:

    st.subheader("Các sản phẩm từ sàn tiki")

    df_product = pd.read_csv('data.csv')
    st.dataframe(df_product, use_container_width=True)


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
