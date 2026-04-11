import streamlit as st
import pandas as pd
import joblib
import warnings
from datetime import datetime
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Fresh Sense | Retail Decision Support",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM STYLING
# ==================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
            margin-top:20px;
            letter-spacing: 0.3px;
        }

        .sub-title {
            font-size: 1.02rem;
            color: #475569;
            margin-bottom: 1rem;
        }
        .section-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 1.15rem;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }
        .action-box {
            padding: 1.1rem;
            border-radius: 16px;
            border-left: 7px solid #2563eb;
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            margin: 1rem 0;
            box-shadow: 0 8px 22px rgba(37, 99, 235, 0.10);
        }
        .risk-low {
            padding: 0.95rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border-left: 6px solid #10b981;
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.08);
        }
        .risk-moderate {
            padding: 0.95rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
            border-left: 6px solid #f59e0b;
            box-shadow: 0 6px 16px rgba(245, 158, 11, 0.08);
        }
        .risk-high {
            padding: 0.95rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border-left: 6px solid #ef4444;
            box-shadow: 0 6px 16px rgba(239, 68, 68, 0.08);
        }
        .highlight-card {
            padding: 1rem;
            border-radius: 16px;
            color: white;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
            text-align: center;
            margin-bottom: 0.8rem;
        }
        .freshness-highlight {
            background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        }
        .discount-highlight {
            background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%);
        }
        .metric-label {
            font-size: 0.92rem;
            font-weight: 600;
            opacity: 0.95;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .metric-subtext {
            font-size: 0.9rem;
            margin-top: 0.35rem;
            opacity: 0.9;
        }
        .small-note {
            font-size: 0.9rem;
            color: #6b7280;
        }
        .summary-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.6rem 0 0.2rem 0;
        }
        .summary-chip {
            background: #f1f5f9;
            color: #0f172a;
            border: 1px solid #dbeafe;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 600;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            padding: 14px 12px;
            border-radius: 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 700;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# LOAD MODELS
# ==================================================
@st.cache_resource
def load_models():
    freshness_model = joblib.load("freshness_model.pkl")
    discount_model = joblib.load("discount_model.pkl")
    return freshness_model, discount_model

freshness_model, discount_model = load_models()


# ==================================================
# HELPERS
# ==================================================
def prepare_input_for_model(input_df, model):
    input_encoded = pd.get_dummies(input_df)

    if hasattr(model, "feature_names_in_"):
        input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    else:
        raise ValueError(
            "Model does not contain feature_names_in_. "
            "Please re-save the model with feature names or save training columns separately."
        )

    return input_encoded


def estimate_remaining_shelf_life(product_category, freshness_score, days_since_arrival):
    # HIGH: 75 to 100
    if 75 <= freshness_score <= 100:
        status = "High"

        if product_category == "Vegetables":
            if days_since_arrival <= 2:
                remaining_days = 15
            elif days_since_arrival <= 5:
                remaining_days = 12
            else:
                remaining_days = 10
        elif product_category == "Dairy":
            remaining_days = 2 if days_since_arrival <= 3 else 1
        elif product_category == "Fruits":
            remaining_days = 5 if days_since_arrival <= 4 else 4
        elif product_category == "Meat":
            remaining_days = 10 if days_since_arrival <= 2 else 8
        elif product_category == "Bakery":
            remaining_days = 4 if days_since_arrival <= 2 else 3
        else:
            remaining_days = 5

    # MEDIUM: 50 to <75
    elif 50 <= freshness_score < 75:
        status = "Medium"

        if product_category == "Vegetables":
            if days_since_arrival <= 2:
                remaining_days = 10
            elif days_since_arrival <= 5:
                remaining_days = 7
            else:
                remaining_days = 5
        elif product_category == "Dairy":
            remaining_days = 1
        elif product_category == "Fruits":
            remaining_days = 3 if days_since_arrival <= 4 else 2
        elif product_category == "Meat":
            remaining_days = 5 if days_since_arrival <= 2 else 4
        elif product_category == "Bakery":
            remaining_days = 2 if days_since_arrival <= 2 else 1
        else:
            remaining_days = 3

    # LOW: below 50
    else:
        status = "Low"

        if product_category == "Vegetables":
            remaining_days = 2 if days_since_arrival <= 3 else 1
        elif product_category == "Dairy":
            remaining_days = 0
        elif product_category == "Fruits":
            remaining_days = 1
        elif product_category == "Meat":
            remaining_days = 2 if days_since_arrival <= 2 else 1
        elif product_category == "Bakery":
            remaining_days = 1
        else:
            remaining_days = 1

    total_days = days_since_arrival + remaining_days
    return remaining_days, total_days, status


def get_expiry_risk(status, remaining_days, is_damaged):
    if remaining_days <= 0:
        risk = "Critical"
    elif remaining_days <= 2 or status == "Low":
        risk = "High"
    elif status == "Medium" or is_damaged == "yes":
        risk = "Moderate"
    else:
        risk = "Low"
    return risk


def get_stock_pressure(current_stock, daily_sales, remaining_days):
    if daily_sales <= 0:
        return "Unclear", 0

    stock_cover_days = round(current_stock / daily_sales, 2)

    if remaining_days <= 0:
        pressure = "Critical"
    elif stock_cover_days > remaining_days * 2:
        pressure = "High"
    elif stock_cover_days > remaining_days:
        pressure = "Moderate"
    else:
        pressure = "Balanced"

    return pressure, stock_cover_days


def generate_freshness_recommendation(status, remaining_days, product_category, is_damaged):
    if remaining_days <= 0:
        return {
            "priority": "Critical",
            "action": "Inspect immediately / remove from sale",
            "message": f"{product_category} is at or beyond its recommended selling window. Immediate quality inspection is required before continuing sale.",
        }

    if status == "High":
        message = f"{product_category} is in good saleable condition and can continue under normal pricing and regular shelf placement."
        if is_damaged == "yes":
            message += " Physical damage is present, so manual inspection is recommended even though freshness is high."
        return {
            "priority": "Low",
            "action": "Continue normal sale",
            "message": message,
        }

    if status == "Medium":
        return {
            "priority": "Moderate",
            "action": "Monitor closely and promote early",
            "message": f"{product_category} is still saleable, but freshness is declining. Improve visibility, monitor within 24 hours, and prepare for a mild promotional strategy.",
        }

    return {
        "priority": "High",
        "action": "Push quick-sale strategy",
        "message": f"{product_category} has low freshness and limited remaining shelf life. Prioritize faster movement, closer monitoring, and markdown planning to reduce wastage.",
    }


def generate_discount_recommendation(discount_pred, final_price, cost_price, remaining_days, stock_pressure):
    profit_after = final_price - cost_price

    if remaining_days <= 0:
        return "Product has reached its risk limit. Discount alone may not be sufficient; manual inspection or removal is recommended."

    if discount_pred <= 0:
        return "No discount is currently recommended. Product can continue with normal pricing and standard shelf display."

    if profit_after < 0:
        return "The predicted discount may lead to a loss. Review manually before applying this markdown in store operations."

    if stock_pressure == "High" and discount_pred < 10:
        return "Inventory pressure is high. Even with a small markdown, consider better shelf placement or a limited promotion campaign."

    if discount_pred <= 10:
        return "A mild discount is recommended to improve product movement while protecting margin."
    elif discount_pred <= 25:
        return "A moderate discount is recommended to balance stock clearance and profitability."
    else:
        return "A high discount is recommended for urgent inventory movement and waste reduction."


def generate_final_action(status, risk_level, discount_pred, remaining_days, stock_pressure):
    if remaining_days <= 0:
        return "Remove from sale / inspect manually"
    if risk_level == "High" and discount_pred >= 20:
        return "Urgent clearance"
    if status == "Low":
        return "Fast-track discount sale"
    if status == "Medium" and stock_pressure in ["Moderate", "High"]:
        return "Promotion with closer monitoring"
    if status == "High" and discount_pred <= 5:
        return "Normal sale"
    return "Monitor and review"


def validate_inputs(product_category, storage_type, display_type, is_damaged, original_price, cost_price, current_stock, daily_sales):
    issues = []

    if cost_price > original_price:
        issues.append("Cost price is greater than selling price. Margin may already be negative.")

    if daily_sales > current_stock and current_stock > 0:
        issues.append("Daily sales are higher than current stock. Please verify stock or sales values.")

    if product_category == "Dairy" and storage_type == "room_temp":
        issues.append("Dairy stored at room temperature may deteriorate quickly. Review storage handling.")

    if product_category == "Meat" and display_type == "open_shelf":
        issues.append("Meat displayed on an open shelf may increase spoilage risk.")

    if is_damaged == "yes":
        issues.append("Damaged products should be visually inspected before pricing decisions are applied.")

    return issues


def add_history_row(record):
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    st.session_state.prediction_history.insert(0, record)
    st.session_state.prediction_history = st.session_state.prediction_history[:10]


def reset_form():
    keys_to_clear = [
        "result_ready",
        "freshness_score",
        "freshness_status",
        "remaining_days",
        "total_days",
        "risk_level",
        "discount_pred",
        "final_price",
        "final_action",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("Fresh Sense")
st.sidebar.caption("Retail freshness and discount decision support")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Analyze Product", "Recent Predictions"]
)

st.sidebar.markdown("---")
if st.sidebar.button("Reset Current Result"):
    reset_form()
    st.sidebar.success("Current result cleared.")

if "prediction_history" in st.session_state and st.session_state.prediction_history:
    last = st.session_state.prediction_history[0]
    st.sidebar.markdown("### Last Result")
    st.sidebar.write(f"**Category:** {last['Category']}")
    st.sidebar.write(f"**Freshness:** {last['Freshness Score']}%")
    st.sidebar.write(f"**Discount:** {last['Discount %']}%")
    st.sidebar.write(f"**Action:** {last['Action']}")


# ==================================================
# OVERVIEW PAGE
# ==================================================
if page == "Overview":
    st.markdown('<div class="main-title" id="title">🥬 Fresh Sense</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Dynamic freshness-based discounting system for retail decision support</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h4>What this application does</h4>
                <ul>
                    <li>Predicts freshness score using the trained freshness model</li>
                    <li>Estimates remaining shelf life using product-specific rule logic</li>
                    <li>Predicts a recommended discount using the trained discount model</li>
                    <li>Transforms predictions into store-ready business actions</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <h4>How to use it</h4>
                <ol>
                    <li>Go to <b>Analyze Product</b></li>
                    <li>Enter product, storage, and pricing details</li>
                    <li>Click <b>Analyze Product</b></li>
                    <li>Review freshness, shelf life, risk, discount, and business action</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "This app keeps your trained models unchanged. The improvements are focused on business presentation, validation, operational guidance, and retail decision support."
        )

    with col2:
        st.success("Models loaded successfully")
        st.metric("Freshness Model", "Ready")
        st.metric("Discount Model", "Ready")
        history_count = len(st.session_state.get("prediction_history", []))
        st.metric("Recent Analyses Saved", history_count)

    st.markdown("---")
    st.caption(
        "Predictions are decision-support outputs. Final product handling should always be confirmed through visual inspection and store policy."
    )


# ==================================================
# ANALYZE PRODUCT PAGE
# ==================================================
elif page == "Analyze Product":
    st.markdown('<div class="main-title">📊 Analyze Product</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Run freshness, shelf-life, pricing, and discount analysis in one workflow</div>',
        unsafe_allow_html=True,
    )

    preset = st.selectbox(
        "Quick preset (optional)",
        [
            "Custom",
            "Healthy Vegetables Stock",
            "Dairy Near Expiry",
            "Fruit Medium Freshness",
            "Bakery Clearance Case",
        ],
    )

    preset_values = {
        "Custom": {
            "product_category": "Vegetables",
            "days_since_arrival": 3,
            "storage_type": "fridge",
            "storage_condition": "good",
            "packaging_type": "sealed",
            "display_type": "fridge",
            "is_damaged": "no",
            "weather_condition": "normal",
            "product_sensitivity": "medium",
            "demand_level": "medium",
            "original_price": 40.0,
            "cost_price": 28.0,
            "current_stock": 120,
            "daily_sales": 35,
            "season": "normal",
        },
        "Healthy Vegetables Stock": {
            "product_category": "Vegetables",
            "days_since_arrival": 2,
            "storage_type": "fridge",
            "storage_condition": "good",
            "packaging_type": "sealed",
            "display_type": "fridge",
            "is_damaged": "no",
            "weather_condition": "normal",
            "product_sensitivity": "medium",
            "demand_level": "high",
            "original_price": 50.0,
            "cost_price": 34.0,
            "current_stock": 100,
            "daily_sales": 42,
            "season": "normal",
        },
        "Dairy Near Expiry": {
            "product_category": "Dairy",
            "days_since_arrival": 5,
            "storage_type": "fridge",
            "storage_condition": "average",
            "packaging_type": "sealed",
            "display_type": "fridge",
            "is_damaged": "no",
            "weather_condition": "hot",
            "product_sensitivity": "high",
            "demand_level": "medium",
            "original_price": 65.0,
            "cost_price": 48.0,
            "current_stock": 90,
            "daily_sales": 18,
            "season": "summer",
        },
        "Fruit Medium Freshness": {
            "product_category": "Fruits",
            "days_since_arrival": 4,
            "storage_type": "fridge",
            "storage_condition": "average",
            "packaging_type": "open",
            "display_type": "open_shelf",
            "is_damaged": "no",
            "weather_condition": "normal",
            "product_sensitivity": "medium",
            "demand_level": "medium",
            "original_price": 80.0,
            "cost_price": 52.0,
            "current_stock": 160,
            "daily_sales": 28,
            "season": "normal",
        },
        "Bakery Clearance Case": {
            "product_category": "Bakery",
            "days_since_arrival": 4,
            "storage_type": "room_temp",
            "storage_condition": "average",
            "packaging_type": "open",
            "display_type": "open_shelf",
            "is_damaged": "yes",
            "weather_condition": "hot",
            "product_sensitivity": "high",
            "demand_level": "low",
            "original_price": 35.0,
            "cost_price": 24.0,
            "current_stock": 140,
            "daily_sales": 15,
            "season": "summer",
        },
    }

    defaults = preset_values[preset]

    st.markdown("### Product and Storage Details")
    col1, col2 = st.columns(2)

    with col1:
        product_category = st.selectbox(
            "Product Category",
            ["Vegetables", "Dairy", "Fruits", "Meat", "Bakery"],
            index=["Vegetables", "Dairy", "Fruits", "Meat", "Bakery"].index(defaults["product_category"]),
        )
        days_since_arrival = st.number_input(
            "Days Since Arrival",
            min_value=0,
            max_value=20,
            value=defaults["days_since_arrival"],
        )
        storage_type = st.selectbox(
            "Storage Type",
            ["fridge", "room_temp", "freezer"],
            index=["fridge", "room_temp", "freezer"].index(defaults["storage_type"]),
        )
        storage_condition = st.selectbox(
            "Storage Condition",
            ["good", "average", "poor"],
            index=["good", "average", "poor"].index(defaults["storage_condition"]),
        )
        packaging_type = st.selectbox(
            "Packaging Type",
            ["sealed", "open"],
            index=["sealed", "open"].index(defaults["packaging_type"]),
        )

    with col2:
        display_type = st.selectbox(
            "Display Type",
            ["fridge", "open_shelf", "freezer"],
            index=["fridge", "open_shelf", "freezer"].index(defaults["display_type"]),
        )
        is_damaged = st.selectbox(
            "Is Damaged",
            ["no", "yes"],
            index=["no", "yes"].index(defaults["is_damaged"]),
        )
        weather_condition = st.selectbox(
            "Weather Condition",
            ["normal", "hot", "cold"],
            index=["normal", "hot", "cold"].index(defaults["weather_condition"]),
        )
        product_sensitivity = st.selectbox(
            "Product Sensitivity",
            ["low", "medium", "high"],
            index=["low", "medium", "high"].index(defaults["product_sensitivity"]),
        )
        demand_level = st.selectbox(
            "Demand Level",
            ["low", "medium", "high"],
            index=["low", "medium", "high"].index(defaults["demand_level"]),
        )

    st.markdown("### Pricing and Inventory Details")
    col3, col4 = st.columns(2)

    with col3:
        original_price = st.number_input("Original Price", min_value=0.0, value=float(defaults["original_price"]), step=1.0)
        cost_price = st.number_input("Cost Price", min_value=0.0, value=float(defaults["cost_price"]), step=1.0)
        current_stock = st.number_input("Current Stock", min_value=0, value=int(defaults["current_stock"]), step=1)

    with col4:
        daily_sales = st.number_input("Daily Sales", min_value=0, value=int(defaults["daily_sales"]), step=1)
        season = st.selectbox(
            "Season",
            ["summer", "winter", "rainy", "normal"],
            index=["summer", "winter", "rainy", "normal"].index(defaults["season"]),
        )

    issues = validate_inputs(
        product_category,
        storage_type,
        display_type,
        is_damaged,
        original_price,
        cost_price,
        current_stock,
        daily_sales,
    )

    if issues:
        st.markdown("### Input Review")
        for issue in issues:
            st.warning(issue)

    analyze_clicked = st.button("Analyze Product", type="primary", use_container_width=True)

    if analyze_clicked:
        freshness_input = pd.DataFrame([
            {
                "product_category": product_category,
                "days_since_arrival": days_since_arrival,
                "storage_type": storage_type,
                "storage_condition": storage_condition,
                "packaging_type": packaging_type,
                "display_type": display_type,
                "is_damaged": is_damaged,
                "weather_condition": weather_condition,
                "product_sensitivity": product_sensitivity,
                "demand_level": demand_level,
            }
        ])

        try:
            freshness_ready = prepare_input_for_model(freshness_input, freshness_model)
            freshness_pred = freshness_model.predict(freshness_ready)[0]
            freshness_score = round(float(freshness_pred) * 100, 2)

            remaining_days, total_days, freshness_status = estimate_remaining_shelf_life(
                product_category,
                freshness_score,
                days_since_arrival,
            )

            risk_level = get_expiry_risk(freshness_status, remaining_days, is_damaged)
            stock_pressure, stock_cover_days = get_stock_pressure(current_stock, daily_sales, remaining_days)

            discount_input = pd.DataFrame([
                {
                    "product_category": product_category,
                    "freshness_score": freshness_score,
                    "days_since_arrival": days_since_arrival,
                    "days_to_expiry": remaining_days,
                    "original_price": original_price,
                    "cost_price": cost_price,
                    "current_stock": current_stock,
                    "daily_sales": daily_sales,
                    "demand_level": demand_level,
                    "season": season,
                }
            ])

            discount_ready = prepare_input_for_model(discount_input, discount_model)
            discount_pred = float(discount_model.predict(discount_ready)[0])
            discount_pred = max(0.0, round(discount_pred, 2))

            discount_amount = round(original_price * discount_pred / 100, 2)
            final_price = round(original_price - discount_amount, 2)
            profit_before = round(original_price - cost_price, 2)
            profit_after = round(final_price - cost_price, 2)

            freshness_reco = generate_freshness_recommendation(
                freshness_status, remaining_days, product_category, is_damaged
            )
            discount_reco = generate_discount_recommendation(
                discount_pred, final_price, cost_price, remaining_days, stock_pressure
            )
            final_action = generate_final_action(
                freshness_status, risk_level, discount_pred, remaining_days, stock_pressure
            )

            st.markdown("---")
            st.markdown("## Analysis Result")

            h1, h2 = st.columns(2)
            with h1:
                st.markdown(
                    f"""
                    <div class="highlight-card freshness-highlight">
                        <div class="metric-label">Freshness Highlight</div>
                        <div class="metric-value">{freshness_score}%</div>
                        <div class="metric-subtext">Status: {freshness_status} • Shelf life: {remaining_days} day(s)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with h2:
                st.markdown(
                    f"""
                    <div class="highlight-card discount-highlight">
                        <div class="metric-label">Discount Highlight</div>
                        <div class="metric-value">{discount_pred}%</div>
                        <div class="metric-subtext">Final price: ₹ {final_price} • Profit after: ₹ {profit_after}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Freshness Score", f"{freshness_score}%")
            m2.metric("Freshness Status", freshness_status)
            m3.metric("Remaining Shelf Life", f"{remaining_days} day(s)")
            m4.metric("Expiry Risk", risk_level)

            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Recommended Discount", f"{discount_pred}%")
            n2.metric("Final Price", f"₹ {final_price}")
            n3.metric("Profit Before Discount", f"₹ {profit_before}")
            n4.metric("Profit After Discount", f"₹ {profit_after}")

            n5, n6 = st.columns(2)
            n5.metric("Stock Pressure", stock_pressure)
            n6.metric("Stock Cover", f"{stock_cover_days} day(s)")

            risk_class = "risk-low"
            if risk_level == "Moderate":
                risk_class = "risk-moderate"
            elif risk_level in ["High", "Critical"]:
                risk_class = "risk-high"

            st.markdown(
                f"""
                <div class="{risk_class}">
                    <b>Operational Summary</b><br>
                    Product condition is <b>{freshness_status}</b>, expiry risk is <b>{risk_level}</b>,
                    stock pressure is <b>{stock_pressure}</b>, and the current recommended action is
                    <b>{final_action}</b>.
                    <div class="summary-chip-row">
                        <span class="summary-chip">Freshness: {freshness_status}</span>
                        <span class="summary-chip">Risk: {risk_level}</span>
                        <span class="summary-chip">Discount: {discount_pred}%</span>
                        <span class="summary-chip">Stock: {stock_pressure}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="action-box">
                    <h4>Recommended Business Action</h4>
                    <p><b>{final_action}</b></p>
                    <p><b>Freshness Guidance:</b> {freshness_reco['message']}</p>
                    <p><b>Discount Guidance:</b> {discount_reco}</p>
                    <p><b>Priority Level:</b> {freshness_reco['priority']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if profit_after < 0:
                st.error("Applying the predicted discount may cause a loss. Manual review is recommended before final pricing.")
            elif remaining_days <= 1:
                st.warning("This product has very limited selling time remaining. Immediate store action is recommended.")
            else:
                st.success("Analysis completed successfully. Review the business action card before applying the final decision.")

            st.markdown("### Input Snapshot")
            input_snapshot = pd.DataFrame([
                {
                    "Product Category": product_category,
                    "Days Since Arrival": days_since_arrival,
                    "Storage Type": storage_type,
                    "Storage Condition": storage_condition,
                    "Packaging Type": packaging_type,
                    "Display Type": display_type,
                    "Is Damaged": is_damaged,
                    "Weather Condition": weather_condition,
                    "Product Sensitivity": product_sensitivity,
                    "Demand Level": demand_level,
                    "Original Price": original_price,
                    "Cost Price": cost_price,
                    "Current Stock": current_stock,
                    "Daily Sales": daily_sales,
                    "Season": season,
                }
            ])
            st.dataframe(input_snapshot, use_container_width=True)

            report_df = pd.DataFrame([
                {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Category": product_category,
                    "Freshness Score": freshness_score,
                    "Freshness Status": freshness_status,
                    "Remaining Shelf Life": remaining_days,
                    "Risk Level": risk_level,
                    "Recommended Discount": discount_pred,
                    "Final Price": final_price,
                    "Profit After Discount": profit_after,
                    "Stock Pressure": stock_pressure,
                    "Final Action": final_action,
                }
            ])

            csv_data = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Product Assessment Report",
                data=csv_data,
                file_name="product_assessment_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

            add_history_row(
                {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Category": product_category,
                    "Freshness Score": freshness_score,
                    "Status": freshness_status,
                    "Risk": risk_level,
                    "Discount %": discount_pred,
                    "Final Price": final_price,
                    "Action": final_action,
                }
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Please ensure the saved models match the same preprocessing and feature format used during training.")


# ==================================================
# RECENT PREDICTIONS PAGE
# ==================================================
elif page == "Recent Predictions":
    st.markdown('<div class="main-title">🕘 Recent Predictions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">View the latest analyzed products from this session</div>',
        unsafe_allow_html=True,
    )

    history = st.session_state.get("prediction_history", [])

    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True)

        history_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Prediction History",
            history_csv,
            "prediction_history.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        st.info("No predictions have been saved in this session yet. Analyze a product to build prediction history.")

    st.caption(
        "Session history is temporary and available only while the current app session remains active."
    )
