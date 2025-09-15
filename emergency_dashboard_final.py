import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from wordcloud import WordCloud
import io
from twilio.rest import Client
import time
from collections import deque


# Set page config for wide mode and icon
st.set_page_config(
    page_title="Emergency Analytics Dashboard",
    page_icon="🚨",
    layout="wide"
)

# Twilio credentials
TWILIO_SID = "AC788079dd57c668349b1e12c04fede38a"
TWILIO_AUTH = "2cfeeb3c552bec4819a2cb4180271eaa"
TWILIO_FROM = "+18284922419"
TO_NUMBER = "+919845183332"

# Alert configuration
ALERT_THRESHOLD = 3  # Number of emergencies to trigger alert
ALERT_WINDOW = 30 # Time window in seconds
last_alert_time = 0
last_banner_alert_time = 0

def send_twilio_alert(message_body):
    """Send SMS alert via Twilio"""
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_FROM,
            to=TO_NUMBER
        )
        st.success(f"🚨 Alert sent: {message_body}")
        return True
    except Exception as e:
        st.error(f"Failed to send alert: {e}")
        return False

def check_emergency_pattern(df_filtered):
    """Check if emergency pattern warrants an alert"""
    global last_alert_time
    current_time = time.time()

    # Get emergency events in the last ALERT_WINDOW seconds
    recent_emergencies = df_filtered[
        (df_filtered["is_emergency"] == True) & 
        (df_filtered["timestamp"] >= current_time - ALERT_WINDOW)
    ]

    # Check if we have enough emergencies to trigger alert
    if len(recent_emergencies) >= ALERT_THRESHOLD:
        # Don't send duplicate alerts within 5 minutes
        if current_time - last_alert_time > 300:  # 5 minutes cooldown
            emergency_types = recent_emergencies["label"].tolist()
            message = f"🚨 CRITICAL: {len(recent_emergencies)} emergencies in {ALERT_WINDOW}s: {', '.join(emergency_types)}"

            if send_twilio_alert(message):
                last_alert_time = current_time
                return True

    return False

st.title("🚨 Emergency Sound Event Analytics Dashboard")

# Sidebar Filters
st.sidebar.header("🔎 Filters")
conn = sqlite3.connect("mqtt_logs.db")
df = pd.read_sql_query("SELECT * FROM logs", conn)
conn.close()

if df.empty:
    st.warning("No data available yet.")
    st.stop()

# Clean IST conversion
df["timestamp_readable"] = pd.to_datetime(df["timestamp"], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)

# Sidebar time filter
time_min = df["timestamp_readable"].min().to_pydatetime()
time_max = df["timestamp_readable"].max().to_pydatetime()

st.sidebar.write("**Time Range:**")
time_range = st.sidebar.slider("", min_value=time_min, max_value=time_max, value=(time_min, time_max))

df_filtered = df[(df["timestamp_readable"] >= time_range[0]) & (df["timestamp_readable"] <= time_range[1])]

# Emergency type filter
event_types = sorted(df_filtered["label"].unique())
chosen_types = st.sidebar.multiselect("Event Type", event_types, default=event_types)
df_filtered = df_filtered[df_filtered["label"].isin(chosen_types)]

emergencies_only = st.sidebar.checkbox("Show Emergencies Only", False)
if emergencies_only:
    df_filtered = df_filtered[df_filtered["is_emergency"] == True]
    st.sidebar.info(f"Showing only emergency alerts out of {len(df)} records.")

# Emergency Alert System Check
if len(df_filtered) > 0:
    alert_triggered = check_emergency_pattern(df_filtered)
    if alert_triggered:
        st.error("🚨 CRITICAL EMERGENCY PATTERN DETECTED - SMS ALERT SENT!")

# Add alert status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Alert System")
current_time = time.time()
recent_emergencies = df_filtered[
    (df_filtered["is_emergency"] == True) & 
    (df_filtered["timestamp"] >= current_time - ALERT_WINDOW)
]
st.sidebar.metric("Emergencies (30s)", len(recent_emergencies))
st.sidebar.metric("Alert Threshold", f"{ALERT_THRESHOLD} events")

if len(recent_emergencies) > 0:
    st.sidebar.warning(f"⚠️ {len(recent_emergencies)} recent emergencies!")
else:
    st.sidebar.success("✅ No recent emergencies")

# Summary Panels
colA, colB, colC, colD = st.columns(4)
colA.metric("Total Events", len(df_filtered))
colB.metric("Emergencies", int(df_filtered["is_emergency"].sum()))

if len(df_filtered) > 0:
    colC.metric("Most Common Event", df_filtered["label"].value_counts().index[0])
    colD.metric("Time Window", f"{(df_filtered['timestamp_readable'].max() - df_filtered['timestamp_readable'].min()).total_seconds()/3600:.1f}h")

    # Emergency rate
    rate = (df_filtered["is_emergency"].sum() / len(df_filtered) * 100)
    st.progress(rate/100)
    st.write(f"**Emergency Rate: {rate:.1f}%**")

st.markdown("---")

# Interactive Timeline
st.subheader("📈 Event Timeline (IST)")
if len(df_filtered) > 0:
    time_counts = df_filtered.set_index("timestamp_readable").resample("30S").count()["id"]
    fig, ax = plt.subplots(figsize=(12, 3))
    time_counts.plot(ax=ax, color="purple", linewidth=2)
    ax.set_ylabel("Event Count")
    ax.set_xlabel("Time (IST)")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close()

# Emergency Heatmap
st.subheader("🔥 Emergency Hourly Intensity (Heatmap)")
if len(df_filtered) > 0:
    heatmap_df = df_filtered.copy()
    heatmap_df["time_bucket"] = heatmap_df["timestamp_readable"].dt.floor("5T")  # 5-minute buckets
    heatmap_df["time_label"] = heatmap_df["time_bucket"].dt.strftime("%H:%M")  # Just hour:minute
    pivot = heatmap_df.pivot_table(index="time_label", columns="label", values="is_emergency", aggfunc="sum", fill_value=0)

    if not pivot.empty:
        plt.figure(figsize=(12,5))
        sns.heatmap(pivot, annot=True, cmap="Reds", fmt="g", cbar_kws={'label': 'Emergency Counts'})
        st.pyplot(plt.gcf())
        plt.close()

# Wordcloud of event types
st.subheader("☁️ Event Label Frequency (Wordcloud)")
if len(df_filtered) > 0:
    word_freq = df_filtered["label"].value_counts().to_dict()
    if word_freq:
        try:
            wc = WordCloud(width=800, height=300, background_color="white", colormap="tab10").generate_from_frequencies(word_freq)
            buf = io.BytesIO()
            wc.to_image().save(buf, format="PNG")
            st.image(buf.getvalue(), width=800)
        except Exception as e:
            st.info("WordCloud not available. Install with: pip install wordcloud")

# Detailed Data Table with Highlighting
st.subheader("📝 Recent Events (IST)")
if len(df_filtered) > 0:
    shown_table = df_filtered[["timestamp_readable", "label", "is_emergency", "topic"]].copy()
    shown_table.rename(columns={"timestamp_readable": "Time (IST)", "label": "Event", "is_emergency": "Emergency", "topic": "Topic"}, inplace=True)
    shown_table["Emergency"] = shown_table["Emergency"].apply(lambda x: "Yes" if x else "No")
    shown_table.sort_values(by="Time (IST)", ascending=False, inplace=True)
    st.dataframe(shown_table.head(20), use_container_width=True)

# Alert Banner (floating bar for latest emergency)
if len(df_filtered) > 0:
    latest_em = df_filtered[df_filtered["is_emergency"] == True].sort_values(by="timestamp_readable", ascending=False).head(1)
    if not latest_em.empty:
        latest_text = f"🚨 ALERT: {latest_em.iloc[0]['label']} detected at {latest_em.iloc[0]['timestamp_readable'].strftime('%Y-%m-%d %H:%M:%S')} IST!"
        st.error(latest_text)
        # Automatically send Twilio alert with cooldown to avoid spam
        current_time = time.time()
        if current_time - last_banner_alert_time > 300:  # 5 minutes cooldown
            if send_twilio_alert(latest_text):
                last_banner_alert_time = current_time
                st.success("Last suspicious sound alert sent automatically!")

st.markdown("---")

# Advanced Analytics
st.subheader("📊 Event Type Distribution")
if len(df_filtered) > 0:
    graph_data = df_filtered["label"].value_counts().reset_index()
    graph_data.columns = ["Event Type", "Count"]
    bar_fig, bar_ax = plt.subplots(figsize=(12,5))
    bar_ax.bar(graph_data["Event Type"], graph_data["Count"], color=sns.color_palette("bright", len(graph_data)))
    bar_ax.set_ylabel("Event Count")
    bar_ax.set_xlabel("Event Type")
    bar_ax.grid(alpha=0.3)
    st.pyplot(bar_fig)
    plt.close()

# Emergency Rate trend over time
st.subheader("📉 Emergency Rate Trend")
if len(df_filtered) > 0:
    df_filtered_copy = df_filtered.copy()
    df_filtered_copy["30s_bucket"] = df_filtered_copy["timestamp_readable"].dt.floor("30S")
    rate_df = df_filtered_copy.groupby("30s_bucket")["is_emergency"].mean() * 100

    if len(rate_df) > 0:
        plt.figure(figsize=(12,3))
        rate_df.plot(color="red", linewidth=2)
        if len(df_filtered) > 0:
            current_rate = (df_filtered["is_emergency"].sum() / len(df_filtered) * 100)
            plt.axhline(current_rate, color="grey", linestyle="--", alpha=0.6, label="Current Rate")
        plt.ylabel("Emergency Rate (%)")
        plt.xlabel("Time (IST)")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

st.markdown("---")

# Show data download option
st.sidebar.markdown("---")
st.sidebar.subheader("⬇️ Download Filtered Data")
if len(df_filtered) > 0:
    filtered_csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download as CSV",
        data=filtered_csv,
        file_name="emergency_events_filtered.csv",
        mime="text/csv",
    )