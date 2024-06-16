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

st.set_page_config(
    page_title="Data Scraper",
    page_icon="📅",
)

# Add custom CSS to center align the title
st.markdown(
    """
    <style>
    .css-1l02zno h1 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the centered title
st.title("Data Scraper")

header_placeholder = st.empty()
step1_placeholder = st.empty()
image1_placeholder = st.empty()
step2_placeholder = st.empty()
image2_placeholder = st.empty()
step3_placeholder = st.empty()
step4_placeholder = st.empty()
step5_placeholder = st.empty()
step6_placeholder = st.empty()


# Display the elements initially
header_placeholder.subheader('Cách lấy headers, params')
step1_placeholder.write('Bước 1: Vào tiki và nhập sản phẩm bạn muốn tìm kiếm')
image1_placeholder.image('images/anh1.png', use_column_width=True)
step2_placeholder.write('Bước 2:')
image2_placeholder.image('images/anh2.png', use_column_width=True)
step3_placeholder.write('Bước 3: Vào Network(Mạng), Vào Tiêu đề để lấy giá trị cho Headers, vào Dung lượng để lấy giá trị cho params')
step4_placeholder.write('Bước 4: Nhấn chọn "Lưu id sản phẩm"')
step5_placeholder.write('Bước 5: Chọn 1 sản phẩm bất kì và lặp lại B2, B3 để lấy giá trị cho params2')
step6_placeholder.write('Bước 6: Nhấn chọn "Xem sản phẩm" để lấy thông tin sản phẩm')
