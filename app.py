import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# IMPORT LIBRARIES
# =========================

from flask import Flask, render_template, request
import joblib
import numpy as np

# =========================
# LOAD MODELS
# =========================

churn_model = joblib.load("churn_model.pkl")

spending_model = joblib.load("spending_model.pkl")

df = pd.read_csv("final_customer_data.csv")

# =========================
# CREATE APP
# =========================

app = Flask(__name__)

# =========================
# HOME PAGE
# =========================

# =========================
# GENERATE CHARTS
# =========================

# Segmentation Chart
plt.figure(figsize=(6,4))

plt.scatter(
    df['Recency'],
    df['Monetary'],
    c=df['Cluster']
)

plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Customer Segmentation")

plt.savefig(
    "static/images/segmentation.png",
    bbox_inches='tight'
)

plt.close()

# Churn Chart
plt.figure(figsize=(5,4))

df['Churn'].value_counts().plot(
    kind='bar'
)

plt.title("Churn Distribution")

plt.savefig(
    "static/images/churn.png",
    bbox_inches='tight'
)

plt.close()

@app.route("/")

def home():

    total_customers = len(df)

    total_revenue = round(
        df['Monetary'].sum() / 1000000,
        2
    )

    churn_rate = round(
        df['Churn'].mean() * 100,
        2
    )

    avg_spending = round(
        df['Monetary'].mean(),
        2
    )

    return render_template(
        "index.html",

        total_customers=total_customers,

        total_revenue=total_revenue,

        churn_rate=churn_rate,

        avg_spending=avg_spending
    )

# =========================
# AI INSIGHTS PAGE
# =========================

@app.route("/insights")

def insights():

    total_customers = len(df)

    total_revenue = round(
        df['Monetary'].sum(),
        2
    )

    churn_rate = round(
        df['Churn'].mean() * 100,
        2
    )

    avg_spending = round(
        df['Monetary'].mean(),
        2
    )

    return render_template(

        "insights.html",

        total_customers=total_customers,

        total_revenue=total_revenue,

        churn_rate=churn_rate,

        avg_spending=avg_spending
    )

# =========================
# PREDICTION ROUTE
# =========================

@app.route("/predict", methods=["POST"])

def predict():

    # Get user inputs
    first_order = request.form["first_order"]

    last_order = request.form["last_order"]

    frequency = float(
        request.form["frequency"]
    )

    product_diversity = float(
        request.form["product_diversity"]
    )

    total_quantity = float(
        request.form["total_quantity"]
    )

    # =========================
    # DATE CALCULATIONS
    # =========================

    today = datetime.today()

    first_order_date = datetime.strptime(
        first_order,
        "%Y-%m-%d"
    )

    last_order_date = datetime.strptime(
        last_order,
        "%Y-%m-%d"
    )

    # Recency
    recency = (
        today - last_order_date
    ).days

    # Customer Lifetime
    customer_lifetime = (
        last_order_date - first_order_date
    ).days

    # Purchase Frequency Rate
    purchase_rate = (
        frequency /
        (customer_lifetime + 1)
    )

    # =========================
    # SPENDING PREDICTION
    # =========================

    spending_features = np.array([[
        recency,
        frequency,
        product_diversity,
        purchase_rate,
        total_quantity
    ]])

    predicted_spending = spending_model.predict(
        spending_features
    )[0]

    # =========================
    # CHURN PREDICTION
    # =========================

    churn_features = np.array([[

        recency,

        frequency,

        predicted_spending,

        product_diversity,

        purchase_rate,

        total_quantity

    ]])

    churn_prediction = churn_model.predict(
        churn_features
    )[0]

    # Prediction Probability

    churn_probability = churn_model.predict_proba(
        churn_features
    )[0][1]

    churn_probability = round(
        churn_probability * 100,
        2
    )

    # =========================
    # RESULT
    # =========================

    if recency > 180 or churn_prediction == 1:
        churn_result = "High Churn Risk"
    else:
        churn_result = "Active Customer"

    # =========================
    # CUSTOMER SEGMENT
    # =========================

    if predicted_spending > 3000 and frequency > 5:

        customer_segment = "VIP"

    elif churn_prediction == 1:

        customer_segment = "At-Risk"

    elif frequency >= 3:

        customer_segment = "Loyal"

    else:

        customer_segment = "New"

    # =========================
    # AI BUSINESS INSIGHTS
    # =========================

    if customer_segment == "VIP Customer":

        insight = (
            "High-value customer detected. "
            "Recommend loyalty rewards and premium retention campaigns."
        )

    elif customer_segment == "At-Risk Customer":

        insight = (
            "Customer likely to churn. "
            "Recommend personalized discounts and re-engagement emails."
        )

    elif customer_segment == "Loyal Customer":

        insight = (
            "Loyal customer identified. "
            "Recommend upselling and cross-selling strategies."
        )

    else:

        insight = (
            "New customer detected. "
            "Recommend onboarding offers and engagement campaigns."
        )

    # =========================
    # KPI VALUES
    # =========================

    total_customers = len(df)

    total_revenue = round(
        df['Monetary'].sum(),
        2
    )

    churn_rate = round(
        df['Churn'].mean() * 100,
        2
    )

    avg_spending = round(
        df['Monetary'].mean(),
        2
    )

    # =========================
    # SEND TO FRONTEND
    # =========================

    return render_template(

        "index.html",

        scroll_to_results=True,
        
        insight=insight,

        churn_result=churn_result,

        churn_probability=churn_probability,

        customer_segment=customer_segment,

        predicted_spending=round(
            predicted_spending,
            2
        ),

        total_customers=total_customers,

        total_revenue=total_revenue,

        churn_rate=churn_rate,

        avg_spending=avg_spending
    )

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(debug=True)