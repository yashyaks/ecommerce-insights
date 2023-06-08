import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Olist EDA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

site_df = pd.read_csv("./data/merged_data.csv")

st.sidebar.title("Olist EDA")
rad1 = st.sidebar.radio("Navigation", ["Orders", "Reviews"])

if rad1 == "Orders":
    st.title("Cities Generating the Most Orders (Top 10)")
    top_orders_cities = (
        site_df.groupby("customer_city")["order_id"]
        .count()
        .reset_index()
        .sort_values("order_id", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    orders = sns.barplot(
        x="order_id", y="customer_city", data=top_orders_cities[:10], palette="magma"
    )
    ax.set_xlabel("Number of Orders", fontsize=14)
    ax.set_ylabel("Cities", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)

    st.pyplot(fig)
