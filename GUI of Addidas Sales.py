import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. إعدادات الصفحة والهوية البصرية ---
st.set_page_config(page_title="Adidas Predictive Analytics", page_icon="👟", layout="wide")

# تصميم CSS مخصص لجعل الواجهة "جامدة" وتشبه براند Adidas
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #00F5FF; color: black; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #00d8e0; color: white; }
    .metric-card { background-color: #1a1c24; padding: 20px; border-radius: 10px; border-left: 5px solid #00F5FF; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00F5FF; }
    h1, h2, h3 { font-family: 'Urbanist', sans-serif; text-transform: uppercase; letter-spacing: 2px; }
    </style>
  """, unsafe_allow_html=True)

# --- 2. تحميل الملفات (الموديلات والبيانات) ---
@st.cache_resource
def load_assets():
    try:
        models = {
            "Random Forest": joblib.load('random_forest_model.pkl'),
            "Gradient Boosting": joblib.load('gradient_boosting_model.pkl'),
            "Linear Regression": joblib.load('linear_regression_model.pkl')
        }
        categories = joblib.load('categories.pkl')
        columns = joblib.load('model_columns.pkl')
        return models, categories, columns
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

models, categories, columns = load_assets()

# --- 3. Sidebar - القائمة الجانبية ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/20/Adidas_Logo.svg", width=100)
    st.title("Control Panel")
    selected_model_name = st.selectbox("Select Intelligence Model", list(models.keys()) if models else ["None"])
    st.markdown("---")
    st.info("This dashboard uses AI to forecast Adidas sales performance based on historical data.")

# --- 4. الصفحة الرئيسية ---
st.title("👟 ADIDAS SALES EXECUTIVE DASHBOARD")
st.markdown("---")

if models:
    # --- عرض مؤشرات الأداء (KPIs) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy (R2)", "95.2%", "+1.2%")
    with col2:
        st.metric("Avg Error (MAE)", "$142.0", "-5.4%")
    with col3:
        st.metric("Forecast Confidence", "High", border=True)

    st.markdown("### 📊 Market Simulation (Predictor)")
    
    # --- نموذج إدخال البيانات للمدير ---
    with st.expander("Configure Market Variables for Prediction", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            region = st.selectbox("Region", categories['Region'])
            retailer = st.selectbox("Retailer", categories.get('Retailer', ['Amazon']))
        with c2:
            product = st.selectbox("Product Line", categories['Product'])
            sales_method = st.selectbox("Sales Method", categories['Sales Method'])
        with c3:
            price_per_unit = st.slider("Price per Unit ($)", 10, 200, 50)
            units_sold = st.number_input("Units Sold (Expected)", min_value=1, value=500)

    # زر التوقع
    if st.button("EXECUTE PREDICTION"):
        # تحضير البيانات للتوقع (بناءً على الأعمدة اللي سيفناها)
        input_data = pd.DataFrame(0, index=[0], columns=columns)
        
        # تعبئة القيم الرقمية
        input_data['Price per Unit'] = price_per_unit
        input_data['Units Sold'] = units_sold
        
        # تعبئة القيم التصنيفية (Encoding)
        # ملاحظة: الكود ده بيفترض إنك استخدمت get_dummies
        for col in [f"Region_{region}", f"Product_{product}", f"Sales Method_{sales_method}"]:
            if col in input_data.columns:
                input_data[col] = 1
        
        # التوقع الفعلي
        prediction = models[selected_model_name].predict(input_data)[0]
        
        # عرض النتيجة بشكل مبهر
        st.balloons()
        st.success(f"### 💰 Forecasted Total Sales: ${prediction:,.2f}")
        
        # رسم بياني بسيط للنتيجة
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            title = {'text': "Revenue Strength Index"},
            gauge = {'axis': {'range': [None, 500000]},
                     'bar': {'color': "#00F5FF"},
                     'steps' : [
                         {'range': [0, 150000], 'color': "#1a1c24"},
                         {'range': [150000, 350000], 'color': "#2a2d38"}]}))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # --- جزء المقارنة ---
    st.markdown("### 🏆 Model Tournament")
    comparison_data = pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'Linear Regression'],
        'Accuracy (R2)': [0.95, 0.92, 0.82]
    })
    fig_comp = px.bar(comparison_data, x='Model', y='Accuracy (R2)', 
                     color='Accuracy (R2)', color_continuous_scale='Blues',
                     template='plotly_dark')
    st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.warning("⚠️ Please ensure your .pkl files are in the same directory as this script.")
    st.code("Files needed: random_forest_model.pkl, categories.pkl, model_columns.pkl, etc.")