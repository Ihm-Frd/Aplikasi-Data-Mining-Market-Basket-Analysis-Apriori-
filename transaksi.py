import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

# Function to display Home Page
def display_home_page():
    st.title("Analisis Asosiasi Produk Menggunakan Algoritma Apriori") 

    st.markdown("---")

    st.header("Nama Pembuat")
    st.write("Nama lu bre")

    st.header("Tujuan Aplikasi")
    st.write("Aplikasi ini bertujuan untuk memberikan rekomendasi item yang mungkin diminati oleh pelanggan berdasarkan pola pembelian di keranjang belanja.Serta mengidentifikasi produk yang sering dibeli bersama dan korelasinya. Datatransaksi dianalisis untuk mengetahui hubungan antar barang yang dibeli bersama, dengan tujuan meningkatkanpenempatan produk agar lebih menarik bagi konsumen dan memicu pembelian impulsif")

    st.header("Algoritma Apriori")
    st.write("Algoritma Apriori merupakan algoritma dalam data mining yang digunakan untuk menemukan pola asosiasi dalam data transaksional seperti keranjang belanja. Algoritma ini mencari itemset yang sering muncul bersama dalam transaksi dan menggunakan metrik seperti support, confidence, dan lift untuk menemukan aturan asosiasi yang kuat antara item-item tersebut.")

    st.markdown("---")

# Function to display the Application
def display_application():
   #APLIKASI 
    df = pd.read_csv("bread basket.csv")
    df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")

    df["month"] = df['date_time'].dt.month
    df["day"] = df['date_time'].dt.weekday

    df["month"].replace([i for i in range(1, 12 + 1)], ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], inplace=True)
    df["day"].replace([i for i in range(6 + 1)], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], inplace=True)

    st.title("Analisis Asosiasi Produk 🛒") 

    def get_data(period_day = '',  weekday_weekend = '', month = '', day = ''):
        data = df.copy()
        filtered = data.loc[
            (data["period_day"].str.contains(period_day)) &
            (data["weekday_weekend"].str.contains(weekday_weekend)) &
            (data["month"].str.contains(month.title())) &
            (data["day"].str.contains(day.title())) 
        ]
        return filtered if filtered.shape[0] else "Ngga Ada Nih Data Nya 🤷🏽‍♂️!"

    def user_input_features():
        item = st.selectbox("item", ['Bread', 'Scandinavian', 'Hot chocolate', 'Jam', 'Cookies', 'Muffin', 'Coffee', 'Pastry', 'Medialuna', 'Tea', 'Tartine', 'Basket', 'Mineral water', 'Farm House', 'Fudge', 'Juice', "Ella's Kitchen Pouches", 'Victorian Sponge', 'Frittata', 'Hearty & Seasonal', 'Soup', 'Pick and Mix Bowls', 'Smoothies', 'Cake', 'Mighty Protein', 'Chicken sand', 'Coke', 'My-5 Fruit Shoot', 'Focaccia', 'Sandwich', 'Alfajores', 'Eggs', 'Brownie', 'Dulce de Leche', 'Honey', 'The BART', 'Granola', 'Fairy Doors', 'Empanadas', 'Keeping It Local', 'Art Tray', 'Bowl Nic Pitt', 'Bread Pudding', 'Adjustment', 'Truffles', 'Chimichurri Oil', 'Bacon', 'Spread', 'Kids biscuit', 'Siblings', 'Caramel bites', 'Jammie Dodgers', 'Tiffin', 'Olum & polenta', 'Polenta', 'The Nomad', 'Hack the stack', 'Bakewell', 'Lemon and coconut', 'Toast', 'Scone', 'Crepes', 'Vegan mincepie', 'Bare Popcorn', 'Muesli', 'Crisps', 'Pintxos', 'Gingerbread syrup', 'Panatone', 'Brioche and salami', 'Afternoon with the baker', 'Salad', 'Chicken Stew', 'Spanish Brunch', 'Raspberry shortbread sandwich', 'Extra Salami or Feta', 'Duck egg', 'Baguette', "Valentine's card", 'Tshirt', 'Vegan Feast', 'Postcard', 'Nomad bag', 'Chocolates', 'Coffee granules ', 'Drinking chocolate spoons ', 'Christmas common', 'Argentina Night', 'Half slice Monster ', 'Gift voucher', 'Cherry me Dried fruit', 'Mortimer', 'Raw bars', 'Tacos/Fajita'])
        period_day = st.selectbox('Period day', ['Morning', 'Afternoon', 'Evening', 'Night',])
        weekend_weekday = st.selectbox('Type of day', ['Weekend', 'Weekday'])
        month = st.select_slider("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        day = st.select_slider("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], value="Sat")

        return period_day, weekend_weekday, month, day, item

    period_day, weekend_weekday, month, day, item = user_input_features()

    data = get_data(period_day.lower(), weekend_weekday.lower(), month, day)

    def encode(x):
        if x <=0:
            return 0
        elif x >= 1:
            return 1

    if type(data) != type ("Tidak Ada Rekomendasi Karna Data Tidak Ada 🤷🏽‍♂️"):
        item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
        item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
        item_count_pivot = item_count_pivot.applymap(encode)

        
        support = 0.01
        frequent_items = apriori(item_count_pivot, min_support= support, use_colnames=True) 

        metric = "lift"
        min_threshold = 1

        rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents","consequents","support","confidence","lift"]]
        rules.sort_values('confidence', ascending=False,inplace=True)

    def parse_list(x):
        x = list(x)
        if len(x) ==1:
            return x[0]
        elif len(x) > 1:
            return ", ".join(x)


    def return_item_df(item_antecedents):
        data = rules[["antecedents", "consequents"]].copy()

        data["antecedents"] = data["antecedents"].apply(parse_list)
        data["consequents"] = data["consequents"].apply(parse_list)

        filtered_data = data.loc[data["antecedents"] == item_antecedents]

        if not filtered_data.empty:
            return list(filtered_data.iloc[0, :])
        else:
            return None  


    def display_recommendations(item, recommendations):
        highlighted_item = f"<span style='background-color: rgb(14, 17, 23); color: #FFFFFF; padding: 5px; border-radius: 5px; font-weight: bold;'>{item}</span>"
        highlighted_rec = f"<span style='background-color: rgb(14, 17, 23); color: #FFFFFF; padding: 5px; border-radius: 5px; font-weight: bold;'>{recommendations[0]}</span>"
        
        st.markdown(f"""
        <div style='background-color: #000; padding: 10px; border-radius: 5px;'>
            <p>Jika Anda membeli {highlighted_item}, Pelanggan kami juga sering membeli:</p>
            <ol>
                <li>{highlighted_item} dan</li>
                <li>{highlighted_rec}</li>
            </ol>
            <p>Secara Bersamaan.</p>
        </div>
        """, unsafe_allow_html=True)

    if type(data) != type("Tidak Ada Rekomendasi Karna Data Tidak Ada 🤷🏽‍♂️"):
        item_recommendation = return_item_df(item)
        if item_recommendation:
            item, *recommendations = item_recommendation
            display_recommendations(item, recommendations)
        else:
            st.warning("Maaf, tidak ada rekomendasi karena data tidak tersedia 🤷🏽‍♂️")
    else:
        st.warning("Data tidak tersedia 🤷🏽‍♂️, tidak ada rekomendasi")


# Sidebar to select between Home Page and Application
selected_page = st.sidebar.radio("Pilih Halaman", ["Home Page", "Aplikasi"])

# Display the selected page based on user choice
if selected_page == "Home Page":
    display_home_page()
else:
    display_application()





