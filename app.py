import streamlit as st
import plotly.express as px
from backend import load_data, filter_data

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Food Delivery Dashboard",
    page_icon="🍔",
    layout="wide",
)

# --- LOAD DATA ---
df = load_data()

# --- TITLE ---
st.title("🍔 Food Delivery Insights Dashboard")
st.markdown("Analyze your food delivery performance across areas, restaurants, and ratings.")

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filters")

# Delivery Area Filter
delivery_areas = sorted(df['Delivery_Area'].unique())
selected_areas = st.sidebar.multiselect(
    "Select Delivery Areas",
    options=delivery_areas,
    default=delivery_areas
)

# Date Range Filter
min_date = df['OrderDate'].min().date()
max_date = df['OrderDate'].max().date()
selected_dates = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# --- FILTER DATA ---
filtered_df = filter_data(df, delivery_area=selected_areas, date_range=selected_dates)

# --- KPIs ---
st.subheader("📊 Key Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", len(filtered_df))
col2.metric("Average Rating ⭐", round(filtered_df['Customer_Rating'].mean(), 2))
col3.metric("Total Revenue 💰", f"₹{round(filtered_df['Order_Amount'].sum(), 2)}")

# --- CHARTS ---
st.divider()

# Orders by Restaurant
st.subheader("📦 Orders by Restaurant")
orders_by_restaurant = filtered_df.groupby('Restaurant').size().reset_index(name='Orders')
fig1 = px.bar(orders_by_restaurant, x='Restaurant', y='Orders', color='Restaurant', text='Orders')
st.plotly_chart(fig1, use_container_width=True)

# Average Delivery Time
st.subheader("⏱️ Average Delivery Time by Restaurant")
avg_delivery = filtered_df.groupby('Restaurant')['Delivery_Time'].mean().reset_index()
fig2 = px.bar(avg_delivery, x='Restaurant', y='Delivery_Time', color='Restaurant', text=avg_delivery['Delivery_Time'].round(1))
st.plotly_chart(fig2, use_container_width=True)

# Revenue by Area
st.subheader("🏙️ Revenue by Delivery Area")
revenue_by_area = filtered_df.groupby('Delivery_Area')['Order_Amount'].sum().reset_index()
fig3 = px.pie(revenue_by_area, names='Delivery_Area', values='Order_Amount', hole=0.4)
st.plotly_chart(fig3, use_container_width=True)

st.success("✅ Dashboard loaded successfully!")
