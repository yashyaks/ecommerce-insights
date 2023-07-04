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


def classify_cat(x):
    if x in [
        "office_furniture",
        "furniture_decor",
        "furniture_living_room",
        "kitchen_dining_laundry_garden_furniture",
        "bed_bath_table",
        "home_comfort",
        "home_comfort_2",
        "home_construction",
        "garden_tools",
        "furniture_bedroom",
        "furniture_mattress_and_upholstery",
    ]:
        return "Furniture"

    elif x in [
        "auto",
        "computers_accessories",
        "musical_instruments",
        "consoles_games",
        "watches_gifts",
        "air_conditioning",
        "telephony",
        "electronics",
        "fixed_telephony",
        "tablets_printing_image",
        "computers",
        "small_appliances_home_oven_and_coffee",
        "small_appliances",
        "audio",
        "signaling_and_security",
        "security_and_services",
    ]:
        return "Electronics"

    elif x in [
        "fashio_female_clothing",
        "fashion_male_clothing",
        "fashion_bags_accessories",
        "fashion_shoes",
        "fashion_sport",
        "fashion_underwear_beach",
        "fashion_childrens_clothes",
        "baby",
        "cool_stuff",
    ]:
        return "Fashion"

    elif x in [
        "housewares",
        "home_confort",
        "home_appliances",
        "home_appliances_2",
        "flowers",
        "costruction_tools_garden",
        "garden_tools",
        "construction_tools_lights",
        "costruction_tools_tools",
        "luggage_accessories",
        "la_cuisine",
        "pet_shop",
        "market_place",
    ]:
        return "Home & Garden"

    elif x in [
        "sports_leisure",
        "toys",
        "cds_dvds_musicals",
        "music",
        "dvds_blu_ray",
        "cine_photo",
        "party_supplies",
        "christmas_supplies",
        "arts_and_craftmanship",
        "art",
    ]:
        return "Entertainment"

    elif x in ["health_beauty", "perfumery", "diapers_and_hygiene"]:
        return "Beauty & Health"

    elif x in ["food_drink", "drinks", "food"]:
        return "Food & Drinks"

    elif x in [
        "books_general_interest",
        "books_technical",
        "books_imported",
        "stationery",
    ]:
        return "Books & Stationery"

    elif x in [
        "construction_tools_construction",
        "construction_tools_safety",
        "industry_commerce_and_business",
        "agro_industry_and_commerce",
    ]:
        return "Industry & Construction"


st.sidebar.title("Olist EDA")
rad = st.sidebar.radio(
    "Navigation",
    ["Orders and Revenue", "Order Time Analytics", "Category wise Sales Distribution"],
)

site_df["product_category"] = site_df.product_category_name_english.apply(classify_cat)


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
    st.title("Category wise Sales Distribution")

    col0, col1 = st.columns(2)
    prodCat_TopOrders = (
        site_df.groupby(site_df["product_category_name_english"])["order_id"]
        .nunique()
        .reset_index()
        .sort_values("order_id", ascending=False)
    )
    prodCat_TopOrders = (
        site_df.groupby(site_df["product_category_name_english"])["order_id"]
        .nunique()
        .reset_index()
        .sort_values("order_id", ascending=True)
    )

    with col0:
        fig, ax0 = plt.subplots(figsize=(8, 9))
        sns.barplot(
            x="order_id",
            y="product_category_name_english",
            data=prodCat_TopOrders[:10],
            palette="magma",
            ax=ax0,
        )
        ax0.set_xlabel("Number of Orders", fontsize=14)
        ax0.set_ylabel("Product Categories", fontsize=14)
        ax0.tick_params(axis="y", labelsize=12)
        st.write("Product Categories with the Highest Orders (Top 10)")

        st.pyplot(fig)

    with col1:
        fig, ax1 = plt.subplots(figsize=(8, 9))
        sns.barplot(
            x="order_id",
            y="product_category_name_english",
            data=prodCat_TopOrders[:10],
            palette="rocket_r",
        )
        ax1.set_xlabel("Number of Orders", fontsize=14)
        ax1.set_ylabel("Product Categories", fontsize=14)
        ax1.tick_params(axis="y", labelsize=12)
        st.write("Product Categories with the Lowest Orders")

        st.pyplot(fig)

    fig = plt.figure(figsize=[5, 3])
    sns.barplot(
        x=site_df.product_category.value_counts().values,
        y=site_df.product_category.value_counts().index,
        palette="crest_r",
    )
    st.write("Number of orders per each SuperCategory")
    plt.xticks(rotation=45)
    st.pyplot(fig)
