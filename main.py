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
site_df["order_purchase_timestamp"] = pd.to_datetime(
    site_df["order_purchase_timestamp"]
)

st.sidebar.title("Olist EDA")
rad = st.sidebar.radio(
    "Navigation",
    ["Orders and Revenue", "Order Time Analytics", "Category wise Sales Distribution"],
)

import streamlit as st

if rad == "Orders and Revenue":
    st.title("Order and Revenue Insights")

    top_orders_cities = (
        site_df.groupby("customer_city")["order_id"]
        .count()
        .reset_index()
        .sort_values("order_id", ascending=False)
    )
    top_revenue_cities = (
        site_df.groupby("customer_city")["payment_value"]
        .sum()
        .reset_index()
        .sort_values("payment_value", ascending=False)
    )

    col0, col1 = st.columns(2)

    with col0:
        fig, ax0 = plt.subplots(figsize=(8, 9))
        sns.barplot(
            x="order_id",
            y="customer_city",
            data=top_orders_cities[:10],
            palette="magma",
            ax=ax0,
        )
        ax0.set_xlabel("Number of Orders", fontsize=14)
        ax0.set_ylabel("Cities", fontsize=14)
        ax0.tick_params(axis="y", labelsize=12)
        st.write("Cities Generating the Most Orders (Top 10)", fontsize=15)

        st.pyplot(fig)

    with col1:
        fig, ax2 = plt.subplots(figsize=(8, 9))
        sns.barplot(
            x=site_df.seller_city.value_counts().values[:10],
            y=site_df.seller_city.value_counts().index[:10],
            palette="magma",
        )
        ax2.set_xlabel("Number of Orders", fontsize=14)
        ax2.set_ylabel("Cities", fontsize=14)
        ax2.tick_params(axis="y", labelsize=12)
        st.write("Top 10 Sellers Cities", fontsize=15)
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        x="payment_value",
        y="customer_city",
        data=top_revenue_cities[:10],
        palette="magma",
        ax=ax,
    )
    ax.set_xlabel("Total Revenue (in Millions of Brazilian Real)", fontsize=14)
    ax.set_ylabel("Cities", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    st.write("Cities Generating the Highest Revenue (Top 10)", fontsize=15)
    st.pyplot(fig)

if rad == "Order Time Analytics":
    st.title("Order Time Insights")

    clrp = sns.color_palette("hls", 1)

    orders_byHour = (
        site_df.groupby(site_df.order_purchase_timestamp.dt.hour)["order_id"]
        .nunique()
        .reset_index()
    )
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(
        x="order_purchase_timestamp", y="order_id", data=orders_byHour, palette=clrp
    )
    ax.set_xlabel("Hour of Day", fontsize=14)
    ax.set_ylabel("Number of Orders", fontsize=14)
    st.write("Orders by Hour", fontsize=20)
    st.pyplot(ax.figure)

    orders_byDays = (
        site_df.groupby(site_df.order_purchase_timestamp.dt.day_name())["order_id"]
        .nunique()
        .reset_index()
        .sort_values("order_id", ascending=False)
    )
    clrp = sns.color_palette("hls", 1)

    plt.figure(figsize=(15, 5))
    ax = sns.barplot(
        x="order_purchase_timestamp", y="order_id", data=orders_byDays, palette=clrp
    )
    ax.set_xlabel("Day of Week", fontsize=14)
    ax.set_ylabel("Number of Orders", fontsize=14)
    st.write("Orders by Day of Week", fontsize=20)
    st.pyplot(ax.figure)

if rad == "Category wise Sales Distribution":
    print("work in progress")
