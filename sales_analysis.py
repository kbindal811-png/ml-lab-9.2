
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("sales_dataset.csv")

# Data cleaning
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df = df.drop_duplicates()
df = df.dropna()

# Display dataset
print("Sales Dataset")
print(df.head())

print("\nDataset Information")
print(df.info())

# Basic statistics
print("\nStatistical Summary")
print(df[["Sales", "Profit", "Quantity"]].describe())

# Total sales and profit
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order_ID"].nunique()
total_customers = df["Customer_ID"].nunique()

print("\nBusiness Summary")
print("Total Sales:", total_sales)
print("Total Profit:", total_profit)
print("Total Orders:", total_orders)
print("Total Customers:", total_customers)

# Monthly sales analysis
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

monthly_sales = df.groupby("Month")[
    "Sales"
].sum().reset_index()

monthly_profit = df.groupby("Month")[
    "Profit"
].sum().reset_index()

# Monthly sales graph
plt.figure(figsize=(12, 5))

sns.lineplot(
    data=monthly_sales,
    x="Month",
    y="Sales",
    marker="o"
)

plt.title("Monthly Sales Analysis")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/monthly_sales.png")
plt.show()

# Monthly profit graph
plt.figure(figsize=(12, 5))

sns.barplot(
    data=monthly_profit,
    x="Month",
    y="Profit"
)

plt.title("Monthly Profit Analysis")
plt.xlabel("Month")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/monthly_profit.png")
plt.show()

# Product performance
product_sales = df.groupby("Product")[
    "Sales"
].sum().reset_index().sort_values(
    "Sales", ascending=False
)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=product_sales,
    x="Product",
    y="Sales"
)

plt.title("Product-wise Sales Performance")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/product_performance.png")
plt.show()

# Regional sales analysis
region_sales = df.groupby("Region")[
    "Sales"
].sum().reset_index()

plt.figure(figsize=(8, 5))

sns.barplot(
    data=region_sales,
    x="Region",
    y="Sales"
)

plt.title("Region-wise Sales Analysis")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/regional_analysis.png")
plt.show()

# Customer segmentation
customer_sales = df.groupby("Customer_ID").agg(
    Total_Sales=("Sales", "sum"),
    Total_Orders=("Order_ID", "nunique"),
    Total_Quantity=("Quantity", "sum")
).reset_index()

customer_sales["Customer_Segment"] = pd.qcut(
    customer_sales["Total_Sales"].rank(method="first"),
    q=3,
    labels=["Low Value", "Medium Value", "High Value"]
)

segment_sales = customer_sales.groupby(
    "Customer_Segment", observed=False
)["Total_Sales"].sum().reset_index()

plt.figure(figsize=(8, 5))

sns.barplot(
    data=segment_sales,
    x="Customer_Segment",
    y="Total_Sales"
)

plt.title("Customer Segmentation Analysis")
plt.xlabel("Customer Segment")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/customer_segmentation.png")
plt.show()

# Save analysis results
monthly_sales.to_csv(
    "outputs/monthly_sales.csv", index=False
)

product_sales.to_csv(
    "outputs/product_sales.csv", index=False
)

region_sales.to_csv(
    "outputs/regional_sales.csv", index=False
)

customer_sales.to_csv(
    "outputs/customer_segmentation.csv", index=False
)

# Interactive monthly sales dashboard
fig1 = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True,
    title="Interactive Monthly Sales Dashboard"
)

fig1.write_html(
    "outputs/monthly_sales_dashboard.html"
)

# Interactive product analysis
fig2 = px.bar(
    product_sales,
    x="Product",
    y="Sales",
    color="Product",
    title="Product Performance Dashboard"
)

fig2.write_html(
    "outputs/product_dashboard.html"
)

# Interactive regional analysis
fig3 = px.pie(
    region_sales,
    names="Region",
    values="Sales",
    title="Regional Sales Distribution"
)

fig3.write_html(
    "outputs/region_dashboard.html"
)

# Interactive customer segmentation
fig4 = px.bar(
    segment_sales,
    x="Customer_Segment",
    y="Total_Sales",
    color="Customer_Segment",
    title="Customer Segmentation Dashboard"
)

fig4.write_html(
    "outputs/customer_dashboard.html"
)

# Final business insights
print("\nBUSINESS INSIGHTS")

print("Best Selling Product:",
      product_sales.iloc[0]["Product"])

print("Highest Sales Region:",
      region_sales.sort_values(
          "Sales", ascending=False
      ).iloc[0]["Region"])

print("Highest Sales Month:",
      monthly_sales.sort_values(
          "Sales", ascending=False
      ).iloc[0]["Month"])

print("\nSales Analysis Completed Successfully!")
print("Check the outputs folder.")
