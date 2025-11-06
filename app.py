# app.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ---- helper: try import backend but allow fallback ----
try:
    from backend import load_data as backend_load_data, filter_data as backend_filter_data
    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False
    backend_load_data = None
    backend_filter_data = None

# --- PAGE CONFIG + DARK MODE ---
st.set_page_config(page_title="Food Delivery Dashboard", page_icon="🍔", layout="wide")

dark_css = """
<style>
body { background-color:#0b0b0b !important; color:#e6e6e6 !important; }
section[data-testid="stAppViewContainer"] { background-color:#0b0b0b; color:#e6e6e6; }
header, footer { background:#0b0b0b !important; color:#e6e6e6 !important; }
.stButton>button { background-color:#1f2937 !important; color:#fff !important; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)


# ---- robust loader that falls back to local CSV ----
@st.cache_data
def load_data():
    if BACKEND_AVAILABLE:
        try:
            df = backend_load_data()
            if isinstance(df, pd.DataFrame):
                return df.copy()
        except Exception:
            pass

    # fallback path — adjust if needed
    fallback_path = "./Food_Delivery_Data_25PLUS_100ROWS.csv"
    try:
        df = pd.read_csv(fallback_path)
    except FileNotFoundError:
        st.error(f"Couldn't find backend.load_data() nor fallback CSV at {fallback_path}. Please provide data.")
        return pd.DataFrame()

    return df


# ---- helper to standardize and detect column names ----
def detect_col(df, candidates):
    """Return first matching column name (from candidates variants) or None."""
    cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None

def normalize_datetime_column(df):
    # try multiple common names
    cand = detect_col(df, ['OrderDate','Order_Date','order_date','orderdate','Date','date','order date'])
    if cand is None:
        return df, None
    try:
        df[cand] = pd.to_datetime(df[cand], errors='coerce')
    except Exception:
        pass
    return df, cand

def try_filter(df, delivery_area=None, date_range=None):
    """If backend_filter_data exists, use it; otherwise do local filtering."""
    if BACKEND_AVAILABLE and backend_filter_data is not None:
        try:
            return backend_filter_data(df, delivery_area=delivery_area, date_range=date_range)
        except Exception:
            pass

    # local simple filter implementation
    out = df.copy()
    if delivery_area:
        area_col = detect_col(out, ['Delivery_Area','delivery_area','DeliveryArea','area','Area'])
        if area_col:
            out = out[out[area_col].isin(delivery_area)]
    if date_range:
        out, dt_col = normalize_datetime_column(out)
        if dt_col:
            start, end = date_range
            out = out[(out[dt_col].dt.date >= start) & (out[dt_col].dt.date <= end)]
    return out


# ---- load df ----
df = load_data()
if df.empty:
    st.stop()

# Normalize column names (strip whitespace)
df.columns = [c.strip() for c in df.columns]

# Attempt to find columns (provide a large set of variants)
col_orderdate = detect_col(df, ['OrderDate','Order_Date','order_date','orderdate','Date'])
col_restaurant = detect_col(df, ['Restaurant','restaurant','restaurant_name'])
col_delivery_area = detect_col(df, ['Delivery_Area','delivery_area','area','DeliveryArea'])
col_order_amount = detect_col(df, ['Order_Amount','order_amount','Amount','orderamount','order value'])
col_customer_rating = detect_col(df, ['Customer_Rating','customer_rating','Rating','customer_rating'])
col_delivery_time = detect_col(df, ['Delivery_Time','delivery_time','Time','deliverytime'])
col_distance = detect_col(df, ['Distance_km','distance_km','distance','km','Distance'])
col_payment_mode = detect_col(df, ['Payment_Mode','payment_mode','PaymentMode','payment'])
col_rider_rating = detect_col(df, ['Rider_Rating','rider_rating','RiderRating'])
col_vehicle_type = detect_col(df, ['Vehicle_Type','vehicle_type','vehicle'])
col_promo = detect_col(df, ['Promo_Discount_Perc','promo_discount_perc','promo_discount','discount'])
col_food_category = detect_col(df, ['Food_Category','food_category','category','FoodCategory'])
col_tip = detect_col(df, ['Tip_Amount','tip_amount','tip'])

# convert OrderDate to datetime if present
if col_orderdate:
    try:
        df[col_orderdate] = pd.to_datetime(df[col_orderdate], errors='coerce')
    except Exception:
        pass

# Sidebar filters
st.sidebar.header("🔍 Filters")

# delivery areas
if col_delivery_area:
    delivery_areas = sorted(df[col_delivery_area].dropna().unique().tolist())
else:
    delivery_areas = []
selected_areas = st.sidebar.multiselect("Select Delivery Areas", options=delivery_areas, default=delivery_areas)

# date range input
if col_orderdate:
    min_date = df[col_orderdate].dropna().min().date()
    max_date = df[col_orderdate].dropna().max().date()
    selected_dates = st.sidebar.date_input("Select Date Range", [min_date, max_date])
else:
    selected_dates = None

# apply filter (use backend.filter_data if available)
filtered_df = try_filter(df, delivery_area=selected_areas, date_range=selected_dates)

# Download button
st.sidebar.download_button(
    "⬇ Download Filtered CSV",
    filtered_df.to_csv(index=False),
    "filtered_data.csv",
    "text/csv"
)

# --- PAGE HEADER & KPIs ---
st.title("🍔 Food Delivery Insights Dashboard")
st.markdown("Gain data-driven insights into restaurant performance, delivery operations, and customer satisfaction.")

# KPIs (safe retrieval with fallbacks)
def safe_mean(col):
    if col and col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
        return round(filtered_df[col].mean(), 2)
    return "N/A"

def safe_sum(col):
    if col and col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
        return round(filtered_df[col].sum(), 2)
    return "N/A"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", len(filtered_df))
col2.metric("Average Rating ⭐", safe_mean(col_customer_rating))
col3.metric("Total Revenue 💰", f"₹{safe_sum(col_order_amount)}")
col4.metric("Avg Delivery Time ⏱️", safe_mean(col_delivery_time))

st.divider()

# ---------------- CORE CHARTS ----------------

# Orders by Restaurant (bar)
if col_restaurant:
    st.subheader("📦 Orders by Restaurant")
    orders_by_rest = filtered_df.groupby(col_restaurant).size().reset_index(name='Orders').sort_values('Orders', ascending=False)
    fig1 = px.bar(orders_by_rest, x=col_restaurant, y='Orders', text='Orders')
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("Column for Restaurant not found — Orders by Restaurant chart skipped.")

# Average Delivery Time by Restaurant
if col_restaurant and col_delivery_time:
    st.subheader("⏱️ Average Delivery Time by Restaurant")
    avg_delivery = filtered_df.groupby(col_restaurant)[col_delivery_time].mean().reset_index()
    fig2 = px.bar(avg_delivery, x=col_restaurant, y=col_delivery_time, text=col_delivery_time)
    st.plotly_chart(fig2, use_container_width=True)

# Revenue by Delivery Area (pie)
if col_delivery_area and col_order_amount:
    st.subheader("🏙️ Revenue by Delivery Area")
    revenue_by_area = filtered_df.groupby(col_delivery_area)[col_order_amount].sum().reset_index()
    fig3 = px.pie(revenue_by_area, names=col_delivery_area, values=col_order_amount, hole=0.4)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------- ADVANCED CHARTS ----------------

# Delivery Time vs Distance (scatter + trend)
if col_distance and col_delivery_time:
    st.subheader("🚴 Delivery Time vs Distance (km)")
    fig4 = px.scatter(filtered_df, x=col_distance, y=col_delivery_time,
                      color=col_restaurant if col_restaurant else None,
                      hover_data=[col_order_amount] if col_order_amount else None,
                      trendline="ols")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Distance or Delivery Time column missing — scatter omitted.")

# Payment Mode Distribution (pie)
if col_payment_mode:
    st.subheader("💳 Payment Mode Distribution")
    pm = filtered_df[col_payment_mode].fillna("Unknown").value_counts().reset_index()
    pm.columns = [col_payment_mode, "Count"]
    fig5 = px.pie(pm, names=col_payment_mode, values="Count", hole=0.3)
    st.plotly_chart(fig5, use_container_width=True)

# Rider Rating by Area (bar)
if col_delivery_area and col_rider_rating:
    st.subheader("🛵 Rider Rating by Delivery Area")
    rating_by_area = filtered_df.groupby(col_delivery_area)[col_rider_rating].mean().reset_index()
    fig6 = px.bar(rating_by_area, x=col_delivery_area, y=col_rider_rating, text=col_rider_rating)
    st.plotly_chart(fig6, use_container_width=True)

# Delivery Time Trend by Date (line)
if col_orderdate and col_delivery_time:
    st.subheader("📈 Delivery Time Trend by Date")
    trend = filtered_df.groupby(col_orderdate)[col_delivery_time].mean().reset_index()
    fig7 = px.line(trend, x=col_orderdate, y=col_delivery_time, markers=True)
    st.plotly_chart(fig7, use_container_width=True)

st.divider()

# ---------------- UNIQUE INSIGHTS ----------------

# Traffic vs Weather impact boxplot (if these columns exist)
col_traffic = detect_col(filtered_df, ['Traffic_Level','traffic_level','Traffic'])
col_weather = detect_col(filtered_df, ['Weather','weather'])
if col_traffic and col_weather and col_delivery_time:
    st.subheader("🚦 Delivery Time vs Traffic & Weather")
    fig8 = px.box(filtered_df, x=col_traffic, y=col_delivery_time, color=col_weather,
                  title="Impact of Traffic and Weather on Delivery Time")
    st.plotly_chart(fig8, use_container_width=True)

# Top Food Categories by Revenue
if col_food_category and col_order_amount:
    st.subheader("🍽️ Top Food Categories by Revenue")
    top_food = filtered_df.groupby(col_food_category)[col_order_amount].sum().reset_index().sort_values(col_order_amount, ascending=False)
    fig9 = px.bar(top_food.head(10), x=col_food_category, y=col_order_amount, text=col_order_amount)
    st.plotly_chart(fig9, use_container_width=True)

# Discount (promo) impact scatter
if col_promo and col_order_amount:
    st.subheader("💸 Discount Impact on Order Value")
    fig10 = px.scatter(filtered_df, x=col_promo, y=col_order_amount,
                       color=col_payment_mode if col_payment_mode else None,
                       size=col_tip if col_tip else None,
                       trendline="ols")
    st.plotly_chart(fig10, use_container_width=True)

# Rider Rating vs Delivery Distance
if col_distance and col_rider_rating:
    st.subheader("🛵 Rider Rating vs Delivery Distance")
    fig11 = px.scatter(filtered_df, x=col_distance, y=col_rider_rating,
                       color=col_vehicle_type if col_vehicle_type else None,
                       size=col_delivery_time if col_delivery_time else None,
                       trendline="ols")
    st.plotly_chart(fig11, use_container_width=True)

# Kitchen load heatmap (if available)
col_kitchen = detect_col(filtered_df, ['Kitchen_Load_Level','kitchen_load_level','kitchen_load'])
if col_kitchen and col_delivery_time and col_delivery_area:
    st.subheader("🔥 Kitchen Load vs Delivery Time Heatmap")
    heatmap_data = filtered_df.pivot_table(index=col_kitchen, columns=col_delivery_area, values=col_delivery_time, aggfunc='mean', fill_value=0)
    fig12 = px.imshow(heatmap_data, color_continuous_scale="YlOrRd", text_auto=".1f", aspect="auto")
    st.plotly_chart(fig12, use_container_width=True)

# Orders by Hour — create Order_Hour if OrderDate exists
if col_orderdate:
    try:
        filtered_df["_order_hour_temp"] = filtered_df[col_orderdate].dt.hour
        st.subheader("⏰ Orders by Hour of the Day")
        orders_by_hour = filtered_df.groupby("_order_hour_temp").size().reset_index(name="Orders")
        orders_by_hour.rename(columns={"_order_hour_temp": "Order_Hour"}, inplace=True)
        fig13 = px.area(orders_by_hour, x="Order_Hour", y="Orders", markers=True)
        st.plotly_chart(fig13, use_container_width=True)
        # cleanup
        filtered_df.drop(columns=["_order_hour_temp"], inplace=True, errors='ignore')
    except Exception:
        pass

# Customer Rating vs Spice Level (if spice col exists)
col_spice = detect_col(filtered_df, ['Food_Spice_Level','food_spice_level','spice_level','spice'])
if col_spice and col_customer_rating:
    st.subheader("🌶️ Customer Rating vs Food Spice Level")
    fig14 = px.box(filtered_df, x=col_spice, y=col_customer_rating, color=col_spice)
    st.plotly_chart(fig14, use_container_width=True)

# Revenue by Customer Segment
col_segment = detect_col(filtered_df, ['Customer_Segment','customer_segment','segment'])
if col_segment and col_order_amount:
    st.subheader("💰 Revenue by Customer Segment")
    segment_revenue = filtered_df.groupby(col_segment)[col_order_amount].sum().reset_index()
    fig15 = px.pie(segment_revenue, names=col_segment, values=col_order_amount, hole=0.4)
    st.plotly_chart(fig15, use_container_width=True)

# Delivery Time Distribution by Area
if col_delivery_area and col_delivery_time:
    st.subheader("📦 Delivery Time Distribution by Delivery Area")
    fig16 = px.box(filtered_df, x=col_delivery_area, y=col_delivery_time, color=col_delivery_area)
    st.plotly_chart(fig16, use_container_width=True)

st.divider()

# ---------------- CORRELATION & MATRIX PLOTS ----------------

# Build numeric dataframe for correlations
numeric_df = filtered_df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    st.subheader("🔗 Correlation Matrix (numeric features)")
    corr = numeric_df.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("🔎 Scatter Matrix (pairwise relationships)")
    # limit columns for readability: pick top 8 numeric cols by variance
    if numeric_df.shape[1] > 1:
        variances = numeric_df.var().sort_values(ascending=False)
        top_cols = variances.index[:8].tolist()
        try:
            fig_matrix = px.scatter_matrix(numeric_df[top_cols])
            fig_matrix.update_layout(height=800)
            st.plotly_chart(fig_matrix, use_container_width=True)
        except Exception:
            st.info("Scatter matrix couldn't be rendered due to data size/limits.")
else:
    st.info("No numeric columns found for correlation / scatter matrix.")

# Scatter + regression examples for a few likely pairs
if col_distance and col_delivery_time and col_order_amount:
    st.subheader("🔍 Delivery Time vs Order Amount (scatter)")
    fig_extra = px.scatter(filtered_df, x=col_distance, y=col_delivery_time, size=col_order_amount,
                           color=col_restaurant if col_restaurant else None, trendline="ols")
    st.plotly_chart(fig_extra, use_container_width=True)

# Heatmap: Restaurant vs Area (order counts)
if col_restaurant and col_delivery_area:
    st.subheader("🔥 Heatmap: Restaurant vs Delivery Area (Orders)")
    try:
        hm = filtered_df.pivot_table(index=col_restaurant, columns=col_delivery_area, values=detect_col(filtered_df, ['Order_ID','Order_Id','OrderId','order_id']), aggfunc='count', fill_value=0)
        fig_hm = px.imshow(hm, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        st.plotly_chart(fig_hm, use_container_width=True)
    except Exception:
        st.info("Heatmap (restaurant vs area) couldn't be generated — missing order identifier column.")

st.success("✅ Dashboard loaded successfully ")
