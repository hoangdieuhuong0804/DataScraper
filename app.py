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

csv_file = 'crawled_data.csv'
# Sidebar
st.sidebar.markdown('__Phân tích sản phẩm tiki__ ')
st.sidebar.markdown('')

# Hiển thị/collapse trường đầu vào cho giá trị của headers
with st.sidebar.expander("Headers"):
    referer = st.text_input("Referer")

# Hiển thị/collapse trường đầu vào cho giá trị của params
with st.sidebar.expander("Params"):
    trackity_id = st.text_input("trackity_id")
    q = st.text_input("q")

# Hiển thị/collapse trường đầu vào cho giá trị của params2
with st.sidebar.expander("Params2"):
    spid = st.text_input("spid")
    version = st.text_input("version")

# Tạo nút "Lưu sản phẩm"
if st.sidebar.button("Lưu id sản phẩm"):
    header_placeholder.empty()
    step1_placeholder.empty()
    image1_placeholder.empty()
    step2_placeholder.empty()
    image2_placeholder.empty()
    step3_placeholder.empty()
    step4_placeholder.empty()
    step5_placeholder.empty()
    step6_placeholder.empty()

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
        d['short_url']=json.get('short_url')
        d['price'] = json.get('price')
        d['discount'] = json.get('discount')
        d['discount_rate'] = json.get('discount_rate')
        d['rating_average'] = json.get('rating_average')
        d['review_count'] = json.get('review_count')
        d['brand_id'] = json.get('brand').get('id')
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
    df_product.to_csv('crawled_data.csv', index=False, encoding='utf-8-sig')
    st.write('Đã lấy data thành công')
pd.set_option('display.max_colwidth', None)
# # Xem sản phẩm
if st.sidebar.button("Xem sản phẩm"):
    
    header_placeholder.empty()
    step1_placeholder.empty()
    image1_placeholder.empty()
    step2_placeholder.empty()
    image2_placeholder.empty()
    step3_placeholder.empty()
    step4_placeholder.empty()
    step5_placeholder.empty()
    step6_placeholder.empty()
    st.subheader("Các sản phẩm từ sàn tiki")

    df_product = pd.read_csv('crawled_data.csv')
    st.dataframe(df_product, use_container_width=True)
    df_product = pd.read_csv('crawled_data.csv')

    # Sắp xếp dữ liệu theo điểm đánh giá giảm dần và giá tăng dần
    sorted_df = df_product.sort_values(by=['rating_average', 'price'], ascending=[False, True])

    # Lấy thông tin của sản phẩm tốt nhất (sản phẩm ở hàng đầu tiên)
    best_product = sorted_df.iloc[0]

    # In ra thông
    # tin của sản phẩm tốt nhất
    st.subheader("Sản phẩm đáng mua nhất là:")
    st.write("Tên sản phẩm:", best_product['name'])
    st.write("Link",best_product['short_url'])
    st.write("Giá:", best_product['price'])
    st.write("Điểm đánh giá:", best_product['rating_average'])


    chart1, chart2=st.columns(2)
    with chart1:
        # Sự phổ biến của từng brand
        # Tạo biểu đồ pie
        st.subheader("Biểu đồ sự phổ biến của các brand")

        # Tạo DataFrame df_brand từ nhóm công việc
        df_brand = df_product.groupby('brand_name').agg({'id': 'count'}).reset_index()

        # Xác định ngưỡng cho các phần tử lớn
        threshold = 2

        # Tạo danh sách nhãn chỉ cho các phần tử lớn hơn ngưỡng
        labels = df_brand.loc[df_brand['id'] > threshold, 'brand_name']

        # Tạo danh sách giá trị chỉ cho các phần tử lớn hơn ngưỡng
        sizes = df_brand.loc[df_brand['id'] > threshold, 'id']

        # Tạo biểu đồ donut với kích thước lớn hơn
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))

        # Tùy chỉnh định dạng chữ
        ax.axis('equal')
        ax.set_title('Sự phổ biến của từng nhãn hiệu', fontsize=12)

        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)
    with chart2:
        #Phân phối phần trăm giảm giá của các nhãn hiệu
        st.subheader("Phân phối phần trăm giảm giá của các nhãn hiệu")

        # Tạo DataFrame df_brand từ nhóm công việc
        df_brand = df_product.groupby('brand_name').agg({'id': 'count', 'discount': 'mean'}).reset_index()

        # Sắp xếp DataFrame theo cột discount giảm dần
        df_brand_sorted = df_brand.sort_values('discount', ascending=False)

        # Chỉ lấy các nhãn hiệu có tỷ lệ giảm giá cao nhất (ví dụ: top 5 nhãn hiệu)
        top_brands = df_brand_sorted.head(5)

        # Tính toán tỷ lệ phần trăm giảm giá
        total_discount = sum(top_brands['discount'])
        discount_percentages = [(discount / total_discount) * 100 for discount in top_brands['discount']]

        # Tạo biểu đồ tròn
        fig, ax = plt.subplots()
        ax.pie(discount_percentages, labels=top_brands['brand_name'], autopct='%1.1f%%')
        ax.set_title("Tỉ lệ phân chia giảm giá của các nhãn hiệu có tỷ lệ cao nhất")
        ax.axis('equal')  # Đảm bảo biểu đồ tròn không bị méo

        # Hiển thị biểu đồ bằng Streamlit
        st.pyplot(fig)
    chart3, chart4=st.columns(2)
    with chart3:
        st.subheader("Tần suất số lượng đánh giá theo điểm đánh giá trung bình")

        # Lấy cột điểm đánh giá trung bình (rating_average) từ DataFrame
        ratings = df_product['rating_average']

        # Đếm số lượng đánh giá cho mỗi điểm đánh giá trung bình
        rating_counts = ratings.value_counts()

        # Sắp xếp lại theo thứ tự tăng dần của điểm đánh giá
        rating_counts = rating_counts.sort_index()

        # Tạo biểu đồ tần suất (bar chart)
        fig, ax = plt.subplots()
        ax.bar(rating_counts.index, rating_counts.values)

        # Đặt tên cho trục x và trục y
        ax.set_xlabel('Điểm đánh giá trung bình')
        ax.set_ylabel('Số lượng đánh giá')

        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)

    with chart4:
        st.subheader('Top 10 nhãn hiệu nhiều sản phẩm nhất')

        # Tạo DataFrame df_brand_counts từ nhóm công việc
        df_brand_counts = df_product['brand_name'].value_counts().reset_index()
        df_brand_counts.columns = ['brand_name', 'product_count']

        # Sắp xếp DataFrame theo số lượng sản phẩm giảm dần
        df_brand_counts_sorted = df_brand_counts.sort_values('product_count', ascending=False)

        # Giới hạn DataFrame chỉ lấy top 10 nhãn hiệu nhiều nhất
        df_top_10_brands = df_brand_counts_sorted.head(10)

        # Tạo biểu đồ cột
        fig, ax = plt.subplots()
        bars = ax.bar(df_top_10_brands['brand_name'], df_top_10_brands['product_count'])
        ax.set_xlabel('Tên nhãn hiệu')
        ax.set_ylabel('Số lượng sản phẩm')

        # Xoá các khoảng trống giữa các cột
        plt.tight_layout()

        # Xoay tên nhãn hiệu xiên
        plt.xticks(rotation=45, ha='right')

        # Hiển thị biểu đồ bằng Streamlit
        st.pyplot(fig)
    rate_main_cat = df_product.groupby(['brand_name', 'rating_average']).agg('count').iloc[:, 1].rename_axis().reset_index(
        name='Amount')
    fig, ax = plt.subplots(figsize=(15, 8))

    sns.boxplot(ax=ax, data=rate_main_cat, x='rating_average', y='brand_name')

    ax.set_title('Phân bổ xếp hạng theo sản phẩm các nhãn hiệu', fontweight='heavy', size='xx-large', y=1.03)
    ax.set_xlabel('Rating', fontweight='bold')
    ax.set_ylabel('Nhãn hiệu', fontweight='bold')
    st.pyplot(plt)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    fig.suptitle('Giá thực tế & Giá chiết khấu', fontweight='heavy', size='xx-large')

    sns.histplot(ax=ax[0], data=df_product, x='price', bins=10, kde=True, color='red')
    sns.histplot(ax=ax[1], data=df_product, x='discount', bins=10, kde=True, color='purple')

    ax[0].set_xlabel('Giá thực tế', fontweight='bold')
    ax[1].set_xlabel('Giá chiết khấu', fontweight='bold')

    ax[0].set_ylabel('Count', fontweight='bold')
    ax[1].set_ylabel('Count', fontweight='bold')

    ax[0].set_title('Dự đoán giá thực tế', fontweight='bold')
    ax[1].set_title('Dự đoán giá chiết khấu', fontweight='bold')

    st.pyplot(plt)
price = st.sidebar.number_input("Price")
discount_rate = st.sidebar.number_input("Discount Rate")
rating_average = st.sidebar.number_input("Rating Average")
review_count = st.sidebar.number_input("Review Count")
if st.sidebar.button("Dự đoán giá giảm"):
    header_placeholder.empty()
    step1_placeholder.empty()
    image1_placeholder.empty()
    step2_placeholder.empty()
    image2_placeholder.empty()
    step3_placeholder.empty()
    step4_placeholder.empty()
    step5_placeholder.empty()
    step6_placeholder.empty()    
    df_product = pd.read_csv('crawled_data.csv')
    df_product = df_product.dropna()
    X = df_product[['price', 'discount_rate', 'rating_average', 'review_count']]
    y = df_product['discount']
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
        'price': [price],
        'discount_rate': [discount_rate],
        'rating_average': [rating_average],
        'review_count': [review_count]
    })

    predicted_price = model.predict(input_data)

    st.write("Mức giảm giá dự đoán:", predicted_price)
    

with st.sidebar.form(key="filter_form"):
    df_product = pd.read_csv('crawled_data.csv')

    selected_brands = st.multiselect('Chọn thương hiệu', df_product['brand_name'].unique())
    price_range = st.slider('Chọn khoảng giá', float(df_product['price'].min()), float(df_product['price'].max()), (float(df_product['price'].min()), float(df_product['price'].max())))
    min_rating = st.slider("Đánh giá tối thiểu", 1.0, 5.0, 4.0, 0.1)
    min_reviews = st.slider("Số lượng đánh giá tối thiểu", 0, df_product['review_count'].max(), 10)
    submit_button = st.form_submit_button(label="Lọc")
if submit_button:
        
    header_placeholder.empty()
    step1_placeholder.empty()
    image1_placeholder.empty()
    step2_placeholder.empty()
    image2_placeholder.empty()
    step3_placeholder.empty()
    step4_placeholder.empty()
    step5_placeholder.empty()
    step6_placeholder.empty()
    if not df_product.empty:
            filtered_df = df_product[(df_product['brand_name'].isin(selected_brands)) &
                                    (df_product['price'] >= price_range[0]) & (df_product['price'] <= price_range[1]) &
                                    (df_product['rating_average'] >= min_rating) &
                                    (df_product['review_count'] >= min_reviews)]
            if not filtered_df.empty:
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.write("No products found with the selected filters.")
    else:
        st.write("No data available. Please check the data file.")


df_product = pd.read_csv('crawled_data.csv')
product_list = df_product['name'].values
ph = st.sidebar.empty()

selected = ph.selectbox("Chọn sản phẩm bạn yêu thích", product_list)

if st.sidebar.button("Gợi ý sản phẩm"):
    header_placeholder.empty()
    step1_placeholder.empty()
    image1_placeholder.empty()
    step2_placeholder.empty()
    image2_placeholder.empty()
    step3_placeholder.empty()
    step4_placeholder.empty()
    step5_placeholder.empty()
    step6_placeholder.empty()
    st.header("Gợi ý sản phẩm tương tự")

    def combineFeature(row):
        return str(row['price']) + ',' + str(row['short_description'])

    df_product['combined_feature'] = df_product.apply(combineFeature, axis=1)

    tf = TfidfVectorizer()
    tfMatrix = tf.fit_transform(df_product['combined_feature'])
    similarity_matrix = cosine_similarity(tfMatrix)

    def recommend(product):
        index = df_product[df_product['name'] == product].index[0]
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda vector: vector[1])
        recommend_products = []
        for i in distances[1:6]:
            recommend_products.append(df_product.iloc[i[0]])
        return recommend_products

    product_recommendations = recommend(selected)
    st.subheader("Các sản phẩm tương tự là:")
    for product in product_recommendations:
            st.write("Tên sản phẩm:", product['name'])
            st.write("Link:", product['short_url'])
            st.write("Giá:", product['price'])
            st.write("Điểm đánh giá:", product['rating_average'])
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
        price = soup2.find(class_='product-price__current-price').get_text()

        price = price.replace('₫', '')
        title = title.strip()
        price = price.strip()
        today = datetime.date.today()

        df = pd.read_csv('Tiki.csv')

        header = ['Title', 'Price', 'Date']
        data = [title, price, today]

        df_new = pd.DataFrame(data=[data], columns=header)
        df = pd.concat([df, df_new], ignore_index=True)

        df.to_csv('Tiki.csv', index=False)
        # st.write(df)

        if float(price) < desired_price:
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
        header_placeholder.empty()
        step1_placeholder.empty()
        image1_placeholder.empty()
        step2_placeholder.empty()
        image2_placeholder.empty()
        step3_placeholder.empty()
        step4_placeholder.empty()
        step5_placeholder.empty()
        step6_placeholder.empty()
        track_product(url, desired_price, recipient_email)