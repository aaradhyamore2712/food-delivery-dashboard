import pandas as pd

def load_data(path="Food_Delivery_Data.csv"):
    """
    Loads and cleans the food delivery dataset.
    Ensures consistent column names and correct datatypes.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at path: {path}")

    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

    # Ensure required columns exist
    required_cols = [
        'order_id', 'restaurant', 'customer_rating',
        'delivery_time', 'order_amount', 'delivery_area', 'orderdate'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Convert date
    df['orderdate'] = pd.to_datetime(df['orderdate'], errors='coerce')
    df.dropna(subset=['orderdate'], inplace=True)

    # Rename for display in app
    df.rename(columns={
        'order_id': 'Order_ID',
        'restaurant': 'Restaurant',
        'customer_rating': 'Customer_Rating',
        'delivery_time': 'Delivery_Time',
        'order_amount': 'Order_Amount',
        'delivery_area': 'Delivery_Area',
        'orderdate': 'OrderDate'
    }, inplace=True)

    return df


def filter_data(df, delivery_area=None, date_range=None):
    """
    Filters the dataset by selected delivery areas and date range.
    """
    dff = df.copy()

    if delivery_area:
        dff = dff[dff['Delivery_Area'].isin(delivery_area)]

    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range)
        dff = dff[(dff['OrderDate'] >= start) & (dff['OrderDate'] <= end)]

    return dff
