import streamlit as st
import paho.mqtt.client as mqtt
import json
import joblib
import csv
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import threading
import time

# ============== CONFIG ==============
BROKER = "broker.emqx.io"
TOPIC_SUB = "iot/so/cool/sensor"
TOPIC_PUB = "iot/so/cool/output"
MODEL_PATH = "iot_temp_model.pkl"
CSV_LOG = "feedback_log.csv"

# ============== STREAMLIT CONFIG ==============
st.set_page_config(
    page_title="IoT Temperature Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "IoT Temperature & Humidity Monitoring System with ML Prediction"
    }
)

# ============== MODERN GREEN-WHITE CSS ==============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Main background - Green gradient */
    .stApp {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 50%, #b8dcc5 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Title styling - Clean and modern */
    h1 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        color: #2e7d32 !important;
        text-align: center;
        padding: 1.5rem 0;
        letter-spacing: -0.5px;
    }
    
    h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #1b5e20 !important;
        font-weight: 600 !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Metric cards - Elegant style */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #2e7d32 !important;
        font-family: 'Poppins', sans-serif;
    }
    
    [data-testid="stMetricDelta"] {
        color: #424242 !important;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stMetricLabel"] {
        color: #424242 !important;
        font-weight: 500;
        font-size: 0.9rem;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar clean styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #c8e6c9 0%, #a5d6a7 100%);
        border-right: 3px solid #4caf50;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #424242;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #2e7d32 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }
    
    /* Button styling - Modern and clean */
    .stButton > button {
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
        border: 2px solid #4caf50;
        background: white;
        color: #2e7d32;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background: #f1f8f4;
        border-color: #2e7d32;
        transform: translateY(-1px);
    }
    
    /* Primary button (Start button) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        color: white;
        border: none;
        font-weight: 600;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #388e3c 0%, #4caf50 100%);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #c8e6c9, transparent);
    }
    
    /* Card containers */
    .element-container {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #a5d6a7;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.1);
    }
    
    /* Data table styling */
    [data-testid="stDataFrame"] {
        background: white;
        border-radius: 12px;
        border: 2px solid #e8f5e9;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Info box styling */
    .stAlert {
        background: #e8f5e9;
        border: 1px solid #4caf50;
        border-radius: 12px;
        color: #1b5e20;
        font-family: 'Inter', sans-serif;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: #4caf50;
    }
    
    /* Text adjustments */
    p, span, label {
        color: #424242 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Caption styling */
    .stCaptionContainer {
        color: #757575 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom containers */
    .info-card {
        background: white;
        border: 2px solid #e8f5e9;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(76, 175, 80, 0.1);
    }
    
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 5px;
    }
    
    .badge-hot {
        background: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    
    .badge-normal {
        background: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌿 IoT Temperature & Humidity Dashboard")
st.markdown("### Environmental Monitoring with Real-time ML Prediction")

# ============== INITIALIZE SESSION STATE ==============
if 'mqtt_running' not in st.session_state:
    st.session_state.mqtt_running = False
if 'client' not in st.session_state:
    st.session_state.client = None
if 'last_message_time' not in st.session_state:
    st.session_state.last_message_time = None
if 'stop_logging' not in st.session_state:
    # When True, on_message will not write CSV or publish to device
    st.session_state.stop_logging = False

# ============== LOAD MODEL ==============
try:
    model = joblib.load(MODEL_PATH)
except:
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

# ============== CREATE CSV IF NOT EXIST ==============
with open(CSV_LOG, mode="a", newline="") as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(["timestamp", "temperature", "humidity", "predicted_label"])

# ============== MQTT CALLBACKS ==============
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ Connected to MQTT Broker (RC: {rc})")
        # Ensure session state reflects running
        st.session_state.mqtt_running = True
        try:
            client.subscribe(TOPIC_SUB)
        except Exception as e:
            print(f"❌ Subscribe failed: {e}")
    else:
        print(f"❌ Connection failed (RC: {rc})")
        st.session_state.mqtt_running = False

def on_message(client, userdata, msg):
    # If logging is stopped, skip processing/writing
    if st.session_state.get('stop_logging', False):
        return

    try:
        data = json.loads(msg.payload.decode())
        temp = float(data.get("temp", 0))
        hum = float(data.get("hum", 0))

        # Update last message time
        st.session_state.last_message_time = datetime.now()
    except Exception as e:
        print(f"❌ Error parsing message: {e}")
        return

    # Predict dengan DataFrame
    X = pd.DataFrame([[temp, hum]], columns=["temperature", "humidity"])
    y_pred = model.predict(X)[0]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log to CSV (guarded by stop_logging check above)
    try:
        with open(CSV_LOG, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, temp, hum, y_pred])
    except Exception as e:
        print(f"❌ Unable to write CSV: {e}")

    # Trigger output to ESP32 (guarded by stop_logging already)
    try:
        if y_pred == "Panas":
            client.publish(TOPIC_PUB, "BUZZER_ON")
            print(f"🔥 {timestamp}  |  Temp:{temp}°C  Hum:{hum}%  →  STATUS: PANAS (BUZZER ON)")
        else:
            client.publish(TOPIC_PUB, "BUZZER_OFF")
            print(f"🟢 {timestamp}  |  Temp:{temp}°C  Hum:{hum}%  →  STATUS: {y_pred}")
    except Exception as e:
        print(f"❌ Publish failed: {e}")

def on_disconnect(client, userdata, rc):
    st.session_state.mqtt_running = False
    if rc != 0:
        print(f"⚠️ Unexpected disconnection (RC: {rc})")
    else:
        print("🔌 Disconnected cleanly")

def on_subscribe(client, userdata, mid, granted_qos):
    print(f"📡 Subscribed to {TOPIC_SUB}")

# ============== START MQTT THREAD ==============
def start_mqtt():
    try:
        # Reset stop flag to allow logging
        st.session_state.stop_logging = False

        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        client.on_subscribe = on_subscribe

        client.connect(BROKER, 1883, 60)
        st.session_state.client = client

        # Use loop_start so main thread can control stop/disconnect
        client.loop_start()
        # Note: on_connect will set mqtt_running = True once connected
    except Exception as e:
        print(f"❌ MQTT Error: {e}")
        st.session_state.mqtt_running = False

# ============== LOAD DATA ==============
def load_data(max_records=500):
    try:
        df = pd.read_csv(CSV_LOG)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.tail(max_records)
    except:
        return pd.DataFrame()

# ============== CREATE CHARTS ==============
def create_temp_humidity_chart(df):
    if df.empty:
        st.warning("⚠️ No data available")
        return

    fig = go.Figure()

    # Temperature line - Green theme
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temperature'],
        name='Temperature (°C)',
        mode='lines+markers',
        line=dict(color='#4caf50', width=3, shape='spline'),
        marker=dict(size=6, symbol='circle', color='#4caf50'),
        fill='tozeroy',
        fillcolor='rgba(76, 175, 80, 0.15)',
        hovertemplate='<b>Temp</b>: %{y:.1f}°C<br><b>Time</b>: %{x}<extra></extra>'
    ))

    # Humidity line - White/Gray theme
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['humidity'],
        name='Humidity (%)',
        mode='lines+markers',
        line=dict(color='#757575', width=3, shape='spline'),
        marker=dict(size=6, symbol='diamond', color='#757575'),
        fill='tozeroy',
        fillcolor='rgba(117, 117, 117, 0.1)',
        yaxis='y2',
        hovertemplate='<b>Humidity</b>: %{y:.1f}%<br><b>Time</b>: %{x}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Temperature & Humidity Over Time',
            'font': {'size': 18, 'color': '#2e7d32', 'family': 'Poppins'}
        },
        xaxis=dict(
            title='Time',
            gridcolor='rgba(76, 175, 80, 0.1)',
            showgrid=True,
            color='#424242'
        ),
        yaxis=dict(
            title='Temperature (°C)',
            gridcolor='rgba(76, 175, 80, 0.1)',
            showgrid=True,
            color='#424242'
        ),
        yaxis2=dict(
            title='Humidity (%)',
            overlaying='y',
            side='right',
            gridcolor='rgba(117, 117, 117, 0.1)',
            color='#424242'
        ),
        hovermode='x unified',
        height=450,
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#424242', size=12, family='Inter'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#e0e0e0',
            borderwidth=1,
            font=dict(color='#424242')
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def create_gauge_chart(value, title, max_value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': '#2e7d32', 'family': 'Poppins'}},
        number={'font': {'size': 36, 'color': '#2e7d32', 'family': 'Poppins'}},
        gauge={
            'axis': {'range': [None, max_value], 'tickcolor': '#424242', 'tickfont': {'color': '#424242'}},
            'bar': {'color': color},
            'bgcolor': 'rgba(255, 255, 255, 0.9)',
            'borderwidth': 2,
            'bordercolor': '#e0e0e0',
            'steps': [
                {'range': [0, max_value * 0.33], 'color': 'rgba(76, 175, 80, 0.15)'},
                {'range': [max_value * 0.33, max_value * 0.66], 'color': 'rgba(255, 193, 7, 0.15)'},
                {'range': [max_value * 0.66, max_value], 'color': 'rgba(244, 67, 54, 0.15)'}
            ],
            'threshold': {
                'line': {'color': '#f44336', 'width': 3},
                'thickness': 0.75,
                'value': max_value * 0.8
            }
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#424242', size=14, family='Inter')
    )

    return fig

# ==================== INITIAL STATES ====================
if "mqtt_running" not in st.session_state:
    st.session_state.mqtt_running = False
if "stop_logging" not in st.session_state:
    st.session_state.stop_logging = True
if "force_rerun" not in st.session_state:
    st.session_state.force_rerun = False
if "data_available" not in st.session_state:
    st.session_state.data_available = False

# AUTO RERUN HANDLER
if "force_rerun" not in st.session_state:
    st.session_state.force_rerun = False
if st.session_state.force_rerun:
    st.session_state.force_rerun = False
    st.rerun()

# ============== SIDEBAR CONTROL ==============
st.sidebar.header("⚙️ Control Panel")

st.sidebar.subheader("🔌 MQTT Connection")
st.sidebar.markdown("""
<div style='background: #e8f5e9; padding: 12px; border-radius: 10px; border: 1px solid #c8e6c9; margin-bottom: 8px;'>
<small style='color: #2e7d32;'><b>Start:</b> Receive MQTT sensor data</small><br>
<small style='color: #2e7d32;'><b>Stop:</b> Disconnect & stop CSV logging</small>
</div>
""", unsafe_allow_html=True)

col_mqtt1, col_mqtt2 = st.sidebar.columns(2)
with col_mqtt1:
    if st.button("▶ Start", use_container_width=True, type="primary", disabled=st.session_state.mqtt_running):
        st.session_state.stop_logging = False
        mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
        mqtt_thread.start()
        st.session_state.mqtt_running = True
        st.sidebar.success("MQTT Started")
        st.session_state.force_rerun = True

with col_mqtt2:
    if st.button("⏹ Stop", use_container_width=True, type="secondary", disabled=not st.session_state.mqtt_running):
        st.session_state.stop_logging = True
        try:
            st.session_state.client.disconnect()
            st.session_state.client.loop_stop()
        except:
            pass
        st.session_state.mqtt_running = False
        st.session_state.client = None
        st.sidebar.warning("MQTT Stopped")
        st.session_state.force_rerun = True

# ==================== DASHBOARD DATA CONTROL ====================
# Hentikan pembacaan CSV / pembaruan grafik bila belum aktif
if not st.session_state.mqtt_running or st.session_state.stop_logging:
    st.info("⚠ MQTT belum dijalankan. Tekan ▶ START untuk mulai menerima data.")
    st.stop()

# Dashboard Settings
st.sidebar.subheader("⚙️ Dashboard Settings")
refresh_interval = st.sidebar.slider("🔄 Refresh Interval (seconds)", 2, 30, 5)
max_records = st.sidebar.slider("📊 Show Last N Records", 10, 500, 100)

# System Info
st.sidebar.subheader("ℹ️ System Information")
st.sidebar.markdown(f"""
<div style='background: white; padding: 12px; border-radius: 10px; border: 1px solid #e0e0e0;'>
<small style='color: #424242;'><b>Broker:</b> {BROKER}</small><br>
<small style='color: #424242;'><b>Subscribe:</b> {TOPIC_SUB}</small><br>
<small style='color: #424242;'><b>Publish:</b> {TOPIC_PUB}</small><br>
<small style='color: #424242;'><b>CSV Log:</b> {CSV_LOG}</small>
</div>
""", unsafe_allow_html=True)

st.sidebar.caption("🌿 Made with care for the environment")

# ============== MAIN DASHBOARD ==============
df = load_data(max_records)

if df.empty:
    st.error("📊 No data available")
    st.info("👉 Click 'Start' button in the Control Panel to begin monitoring")
else:
    # Key Metrics
    st.markdown("### 📊 Current Readings")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_temp = df['temperature'].iloc[-1]
        temp_delta = df['temperature'].iloc[-1] - df['temperature'].iloc[-2] if len(df) > 1 else 0
        st.metric(
            "🌡️ Temperature",
            f"{current_temp:.1f}°C",
            f"{temp_delta:+.2f}°C" if len(df) > 1 else "N/A"
        )

    with col2:
        current_hum = df['humidity'].iloc[-1]
        hum_delta = df['humidity'].iloc[-1] - df['humidity'].iloc[-2] if len(df) > 1 else 0
        st.metric(
            "💧 Humidity",
            f"{current_hum:.1f}%",
            f"{hum_delta:+.2f}%" if len(df) > 1 else "N/A"
        )

    with col3:
        avg_temp = df['temperature'].mean()
        st.metric("📊 Avg Temperature", f"{avg_temp:.1f}°C")

    with col4:
        avg_hum = df['humidity'].mean()
        st.metric("📊 Avg Humidity", f"{avg_hum:.1f}%")

    # Status Distribution - Using circles and text
    st.markdown("### 🎯 Status Distribution")

    panas_count = (df['predicted_label'] == 'Panas').sum()
    normal_count = (df['predicted_label'] == 'Normal').sum()
    total_count = len(df)
    panas_pct = (panas_count / total_count * 100) if total_count > 0 else 0
    normal_pct = (normal_count / total_count * 100) if total_count > 0 else 0

    col_status1, col_status2 = st.columns(2)

    with col_status1:
        st.markdown(f"""
        <div class='info-card'>
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <div style='width: 60px; height: 60px; background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                     border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                     border: 3px solid #ef5350; margin-right: 15px;'>
                    <span style='font-size: 24px;'>🔥</span>
                </div>
                <div>
                    <h2 style='margin: 0; color: #c62828; font-family: Poppins;'>{panas_count}</h2>
                    <p style='margin: 0; color: #757575; font-size: 0.9rem;'>Hot Status Events</p>
                </div>
            </div>
            <div style='background: #ffebee; padding: 8px 12px; border-radius: 8px; border-left: 4px solid #ef5350;'>
                <small style='color: #c62828;'><b>{panas_pct:.1f}%</b> of total readings</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_status2:
        st.markdown(f"""
        <div class='info-card'>
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <div style='width: 60px; height: 60px; background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                     border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                     border: 3px solid #66bb6a; margin-right: 15px;'>
                    <span style='font-size: 24px;'>✅</span>
                </div>
                <div>
                    <h2 style='margin: 0; color: #2e7d32; font-family: Poppins;'>{normal_count}</h2>
                    <p style='margin: 0; color: #757575; font-size: 0.9rem;'>Normal Status Events</p>
                </div>
            </div>
            <div style='background: #e8f5e9; padding: 8px 12px; border-radius: 8px; border-left: 4px solid #66bb6a;'>
                <small style='color: #2e7d32;'><b>{normal_pct:.1f}%</b> of total readings</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Gauge charts
    st.markdown("### 📈 Live Monitoring")
    gauge_col1, gauge_col2 = st.columns(2)

    with gauge_col1:
        st.plotly_chart(create_gauge_chart(current_temp, "Temperature (°C)", 50, "#4caf50"), use_container_width=True)

    with gauge_col2:
        st.plotly_chart(create_gauge_chart(current_hum, "Humidity (%)", 100, "#757575"), use_container_width=True)

    # Main Chart
    st.markdown("### 📉 Historical Trends")
    create_temp_humidity_chart(df)

    # Statistics Summary
    st.markdown("### 📊 Statistical Summary")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("🔺 Max Temperature", f"{df['temperature'].max():.1f}°C")

    with stat_col2:
        st.metric("🔻 Min Temperature", f"{df['temperature'].min():.1f}°C")

    with stat_col3:
        st.metric("🔺 Max Humidity", f"{df['humidity'].max():.1f}%")

    with stat_col4:
        st.metric("🔻 Min Humidity", f"{df['humidity'].min():.1f}%")

    # Data Table
    st.markdown("### 📋 Recent Data Logs")
    display_df = df[['timestamp', 'temperature', 'humidity', 'predicted_label']].sort_values('timestamp', ascending=False).reset_index(drop=True)
    display_df.index = display_df.index + 1

    # Add status with simple text
    display_df['Status'] = display_df['predicted_label'].apply(lambda x: f"🔥 {x}" if x == "Panas" else f"✅ {x}")
    display_df = display_df.drop('predicted_label', axis=1)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        column_config={
            "timestamp": st.column_config.DatetimeColumn(
                "Timestamp",
                format="DD/MM/YYYY HH:mm:ss",
                width="medium"
            ),
            "temperature": st.column_config.NumberColumn(
                "Temperature (°C)",
                width="small",
                format="%.2f"
            ),
            "humidity": st.column_config.NumberColumn(
                "Humidity (%)",
                width="small",
                format="%.2f"
            ),
            "Status": st.column_config.TextColumn(
                "Status",
                width="small"
            )
        }
    )

# ============== FOOTER ==============
col_footer1, col_footer2 = st.columns([1, 1])
with col_footer1:
    st.caption(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col_footer2:
    st.caption(f"⏱️ Auto-refresh in {refresh_interval} seconds")

# ============== AUTO REFRESH ==============
time.sleep(refresh_interval)
st.rerun()
