import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("customer_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Customer Retention Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Retention Prediction")
st.info("""
This application predicts whether a telecom customer is likely to churn
based on customer demographics, services and billing information.
""")
st.write("Predict whether a customer is likely to churn.")

# ================= SIDEBAR =================

st.sidebar.header("Customer Information")
st.sidebar.subheader("📈 Model Performance")
st.sidebar.write("Accuracy: 80.3%")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

# ================= PREDICT =================

if st.button("🔍 Predict Churn"):

    data = pd.DataFrame(0, index=[0], columns=columns)

    data["SeniorCitizen"] = senior
    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly_charges
    data["TotalCharges"] = total_charges

    if gender == "Male":
        data["gender_Male"] = 1

    if partner == "Yes":
        data["Partner_Yes"] = 1

    if dependents == "Yes":
        data["Dependents_Yes"] = 1

    if internet == "Fiber optic":
        data["InternetService_Fiber optic"] = 1
    elif internet == "No":
        data["InternetService_No"] = 1

    if contract == "One year":
        data["Contract_One year"] = 1
    elif contract == "Two year":
        data["Contract_Two year"] = 1

    if paperless == "Yes":
        data["PaperlessBilling_Yes"] = 1

    if payment == "Credit card (automatic)":
        data["PaymentMethod_Credit card (automatic)"] = 1
    elif payment == "Electronic check":
        data["PaymentMethod_Electronic check"] = 1
    elif payment == "Mailed check":
        data["PaymentMethod_Mailed check"] = 1

    # Scaling
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)

    # Probability
    probability = model.predict_proba(data_scaled)

    churn_risk = probability[0][1] * 100

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if prediction[0] == 1:
            st.error("⚠ Customer is likely to Churn")
        else:
            st.success("✅ Customer is likely to Stay")

    with col2:
        st.metric(
            label="Churn Risk",
            value=f"{churn_risk:.2f}%"
        )

    st.progress(int(churn_risk))
    prob_df = pd.DataFrame({
    "Outcome": ["Stay", "Churn"],
    "Probability (%)": [
        round(100 - churn_risk, 2),
        round(churn_risk, 2)
    ]
    })

    st.subheader("📊 Prediction Probability")
    col1, col2 = st.columns([1,3])

    with col1:
       
       st.table(prob_df)
   


    if churn_risk < 30:
        st.success("🟢 Low Risk Customer")
    elif churn_risk < 70:
        st.warning("🟡 Medium Risk Customer")
    else:
        st.error("🔴 High Risk Customer")

    st.subheader("📋 Customer Summary")

    st.write(f"**Tenure:** {tenure} months")
    st.write(f"**Monthly Charges:** ₹{monthly_charges}")
    st.write(f"**Total Charges:** ₹{total_charges}")
    st.write(f"**Internet Service:** {internet}")
    st.write(f"**Contract:** {contract}")

    st.subheader("💡 Recommendation")

    if churn_risk > 70:
        st.write("""
        - Offer discount
        - Upgrade customer support
        - Provide loyalty benefits
        """)

    
    elif churn_risk > 30:
        st.write("""
        - Monitor customer activity
        - Send promotional offers
        """)
    else:
       st.write("""
       - Customer is stable
       - Maintain service quality
       """)
st.markdown("---")
st.caption("Developed using Python, Scikit-Learn and Streamlit")