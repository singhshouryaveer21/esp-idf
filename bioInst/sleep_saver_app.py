import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sleep Saver | Sleep Apnea Analysis",
    page_icon="🌙",
    layout="wide"
)

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wireless_sleep_data.csv")

# --- 1. ML MODEL (AASM-Calibrated Random Forest) ---
@st.cache_resource
def load_ml_model():
    rng = np.random.default_rng(42)
    n_samples = 400

    # Class 0: Normal
    X_normal = np.column_stack([
        rng.uniform(0.75, 1.30, n_samples),
        rng.uniform(0.0, 2.0, n_samples),
        rng.uniform(0.5, 2.0, n_samples)
    ])
    y_normal = np.zeros(n_samples, dtype=int)

    # Class 1: Hypopnea
    X_hypopnea = np.column_stack([
        rng.uniform(0.15, 0.70, n_samples),
        rng.uniform(3.0, 6.0, n_samples),
        rng.uniform(1.0, 3.5, n_samples)
    ])
    y_hypopnea = np.ones(n_samples, dtype=int)

    # Class 2: OSA
    X_osa = np.column_stack([
        rng.uniform(0.0, 0.15, n_samples),
        rng.uniform(3.0, 12.0, n_samples),
        rng.uniform(3.0, 12.0, n_samples)
    ])
    y_osa = np.full(n_samples, 2, dtype=int)

    # Class 3: CSA
    X_csa = np.column_stack([
        rng.uniform(0.0, 0.15, n_samples),
        rng.uniform(3.0, 12.0, n_samples),
        rng.uniform(0.2, 1.8, n_samples)
    ])
    y_csa = np.full(n_samples, 3, dtype=int)

    X = np.vstack([X_normal, X_hypopnea, X_osa, X_csa])
    y = np.concatenate([y_normal, y_hypopnea, y_osa, y_csa])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

# --- 2. SIGNAL CONDITIONING & PIPELINE ---
def bandpass_filter(signal, fs, lowcut=0.1, highcut=0.5, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def classify_posture(ax, ay, az):
    """
    Assumes ADXL343 is mounted flat on the chest with:
      +Z pointing outward from chest, +Y pointing toward head, +X to the side.
    Adjust axis signs if your PCB is mounted differently.
    """
    abs_x, abs_y, abs_z = abs(ax), abs(ay), abs(az)
    if abs_z >= abs_x and abs_z >= abs_y:
        return "Supine (Back)" if az > 0 else "Prone (Stomach)"
    elif abs_x >= abs_y:
        return "Left Lateral" if ax > 0 else "Right Lateral"
    else:
        return "Upright"

def process_sleep_data(df, model):
    has_accel = {'Acc_X', 'Acc_Y', 'Acc_Z'}.issubset(df.columns)
    numeric_cols = ['Timestamp', 'Piezo', 'SpO2_Percent', 'HeartRate_BPM', 'Audio_RMS']
    if has_accel:
        numeric_cols += ['Acc_X', 'Acc_Y', 'Acc_Z']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Timestamp', 'Piezo', 'Audio_RMS']).reset_index(drop=True)

    df['Time_sec'] = (df['Timestamp'] - df['Timestamp'].iloc[0]) / 1000.0
    total_duration_sec = df['Time_sec'].iloc[-1]

    dt = np.median(np.diff(df['Time_sec']))
    fs = 1.0 / dt if dt > 0 else 50.0

    # Clean DFRobot MAX30102 calibration values (-1 or 0)
    df['SpO2_Clean'] = df['SpO2_Percent'].where(df['SpO2_Percent'] > 50, np.nan)
    df['SpO2_Clean'] = df['SpO2_Clean'].interpolate(method='linear').bfill().ffill().fillna(98.0)

    df['HR_Clean'] = df['HeartRate_BPM'].where(df['HeartRate_BPM'] > 30, np.nan)
    df['HR_Clean'] = df['HR_Clean'].interpolate(method='linear').bfill().ffill().fillna(70.0)

    # Accelerometer magnitude (m/s^2, ~9.81 at rest)
    if has_accel:
        df[['Acc_X', 'Acc_Y', 'Acc_Z']] = df[['Acc_X', 'Acc_Y', 'Acc_Z']].ffill().bfill().fillna(0.0)
        df['Acc_Mag'] = np.sqrt(df['Acc_X']**2 + df['Acc_Y']**2 + df['Acc_Z']**2)

    # Phase 1: Bandpass filter Piezo (0.1 Hz - 0.5 Hz)
    if len(df) > 30:
        df['Piezo_Filtered'] = bandpass_filter(df['Piezo'].values, fs)
    else:
        df['Piezo_Filtered'] = df['Piezo'] - df['Piezo'].mean()

    baseline_samples = max(int(120 * fs), 50)
    df['SpO2_Baseline'] = df['SpO2_Clean'].rolling(window=baseline_samples, min_periods=1).median()
    piezo_baseline_amp = np.percentile(np.abs(df['Piezo_Filtered']), 90) + 1e-6
    audio_baseline_rms = np.median(df['Audio_RMS']) + 1e-6

    # Phase 2: Sliding 10-second window feature extraction & ML classification
    window_sec = 10.0
    step_sec = 5.0
    motion_std_threshold = 1.2  # m/s^2 std threshold for gross body movement
    motion_windows_skipped = 0
    events = []
    class_map = {0: 'Normal', 1: 'Hypopnea', 2: 'OSA', 3: 'CSA'}

    t = 0.0
    while t + window_sec <= total_duration_sec:
        win = df[(df['Time_sec'] >= t) & (df['Time_sec'] < t + window_sec)]
        post_win = df[(df['Time_sec'] >= t + window_sec) & (df['Time_sec'] < t + window_sec + 5.0)]

        if len(win) > 5:
            # Motion Artifact Gating via ADXL343
            posture = "N/A"
            if has_accel:
                acc_std = float(win['Acc_Mag'].std())
                if acc_std > motion_std_threshold:
                    motion_windows_skipped += 1
                    t += step_sec
                    continue
                posture = classify_posture(
                    float(win['Acc_X'].mean()),
                    float(win['Acc_Y'].mean()),
                    float(win['Acc_Z'].mean())
                )

            win_piezo_amp = np.percentile(np.abs(win['Piezo_Filtered']), 90)
            piezo_amp_ratio = win_piezo_amp / piezo_baseline_amp

            spo2_check = pd.concat([win, post_win]) if len(post_win) > 0 else win
            spo2_drop = float(win['SpO2_Baseline'].median() - spo2_check['SpO2_Clean'].min())
            spo2_drop = max(spo2_drop, 0.0)

            audio_check = post_win if len(post_win) > 0 else win
            audio_ratio = float(audio_check['Audio_RMS'].max() / audio_baseline_rms)

            features = np.array([[piezo_amp_ratio, spo2_drop, audio_ratio]])
            pred_class = int(model.predict(features)[0])
            confidence = float(np.max(model.predict_proba(features)))

            if pred_class != 0:
                if events and t <= events[-1]['End_sec']:
                    events[-1]['End_sec'] = t + window_sec
                    events[-1]['Duration_sec'] = round(events[-1]['End_sec'] - events[-1]['Start_sec'], 1)
                    events[-1]['SpO2_Drop_%'] = max(events[-1]['SpO2_Drop_%'], round(spo2_drop, 1))
                else:
                    events.append({
                        'Event_Type': class_map[pred_class],
                        'Start_sec': round(t, 1),
                        'End_sec': round(t + window_sec, 1),
                        'Duration_sec': round(window_sec, 1),
                        'SpO2_Drop_%': round(spo2_drop, 1),
                        'Posture': posture,
                        'Confidence_%': round(confidence * 100, 1)
                    })
        t += step_sec

    # Phase 3: AHI Calculation
    total_hours = max(total_duration_sec / 3600.0, 1e-4)
    events_df = pd.DataFrame(events)
    total_events = len(events_df)
    ahi = total_events / total_hours

    if ahi < 5:
        severity, color = "Normal", "#2E7D32"
        advice = "Your breathing patterns appear normal with minimal respiratory disruptions."
    elif ahi < 15:
        severity, color = "Mild Sleep Apnea", "#F9A825"
        advice = "Mild sleep-disordered breathing detected. Consider sharing this report with a physician."
    elif ahi < 30:
        severity, color = "Moderate Sleep Apnea", "#EF6C00"
        advice = "Moderate sleep apnea indicators detected. A clinical follow-up and sleep study are recommended."
    else:
        severity, color = "Severe Sleep Apnea", "#C62828"
        advice = "Frequent respiratory events and oxygen drops detected. Please consult a sleep specialist promptly."

    supine_events = int((events_df['Posture'] == 'Supine (Back)').sum()) if (total_events > 0 and has_accel) else 0

    summary = {
        'ahi': round(ahi, 1),
        'severity': severity,
        'color': color,
        'advice': advice,
        'duration_min': round(total_duration_sec / 60.0, 1),
        'total_events': total_events,
        'osa_count': int((events_df['Event_Type'] == 'OSA').sum()) if total_events > 0 else 0,
        'csa_count': int((events_df['Event_Type'] == 'CSA').sum()) if total_events > 0 else 0,
        'hypopnea_count': int((events_df['Event_Type'] == 'Hypopnea').sum()) if total_events > 0 else 0,
        'avg_spo2': round(df['SpO2_Clean'].mean(), 1),
        'min_spo2': round(df['SpO2_Clean'].min(), 1),
        'avg_hr': round(df['HR_Clean'].mean(), 1),
        'has_accel': has_accel,
        'motion_windows_skipped': motion_windows_skipped,
        'supine_events': supine_events
    }
    return df, events_df, summary

# --- 3. PATIENT-FACING UI ---
st.title("🌙 Sleep Saver")
st.caption("Personal Sleep Apnea Diagnostic Dashboard")

with st.sidebar:
    st.header("Data Source")
    if st.button("🔄 Refresh Receiver Data", width="stretch"):
        st.rerun()

    uploaded_file = st.file_uploader(
        "Or upload a different CSV (optional)",
        type=["csv"]
    )
    st.markdown("---")
    st.markdown(
        "**How Sleep Saver Works**\n"
        "- **Respiratory Effort:** Filters chest/tracheal motion (0.1–0.5 Hz).\n"
        "- **Oxygenation:** Tracks blood oxygen (SpO₂) drops ≥ 3%.\n"
        "- **Acoustics:** Differentiates Obstructive (OSA) vs. Central (CSA) events.\n"
        "- **ADXL343 Motion & Posture:** Filters movement artifacts and tracks sleep position."
    )

raw_df = None
data_source_label = ""

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    data_source_label = f"Uploaded file: `{uploaded_file.name}`"
elif os.path.exists(DEFAULT_CSV_PATH):
    try:
        raw_df = pd.read_csv(DEFAULT_CSV_PATH)
        data_source_label = "Auto-loaded from receiver: `wireless_sleep_data.csv`"
    except pd.errors.EmptyDataError:
        st.warning("`wireless_sleep_data.csv` was found, but it is currently empty. Wait for `receiver.py` to log data and click **Refresh Receiver Data**.")
else:
    st.info("Waiting for `wireless_sleep_data.csv` from `receiver.py`. Start your receiver script or upload a CSV in the sidebar.")

if raw_df is not None and not raw_df.empty:
    st.caption(data_source_label)
    required_cols = {'Timestamp', 'Piezo', 'SpO2_Percent', 'HeartRate_BPM', 'Audio_RMS'}

    if not required_cols.issubset(raw_df.columns):
        st.error(f"CSV is missing required columns: {required_cols - set(raw_df.columns)}")
    elif len(raw_df) < 10:
        st.warning("Not enough samples recorded yet. Collect at least a few seconds of data and click **Refresh Receiver Data**.")
    else:
        model = load_ml_model()
        with st.spinner("Analyzing respiratory, oxygen, acoustic, and motion signals..."):
            df, events_df, summary = process_sleep_data(raw_df, model)

        # Severity Banner
        st.markdown(
            f"""
            <div style="background-color:{summary['color']}; padding:20px; border-radius:10px; color:white; margin-bottom:20px;">
                <h2 style="margin:0; color:white;">Assessment: {summary['severity']} (AHI: {summary['ahi']})</h2>
                <p style="margin:8px 0 0 0; font-size:16px;">{summary['advice']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if summary['duration_min'] < 60:
            st.info(f"Short recording detected ({summary['duration_min']} min). AHI is extrapolated to events per hour.")

        # Top KPI Metrics
        cols = st.columns(6 if summary['has_accel'] else 5)
        cols[0].metric("AHI Score", f"{summary['ahi']} / hr")
        cols[1].metric("Total Events", summary['total_events'])
        cols[2].metric("OSA / CSA / Hypopnea", f"{summary['osa_count']} / {summary['csa_count']} / {summary['hypopnea_count']}")
        cols[3].metric("Avg / Min SpO₂", f"{summary['avg_spo2']}% / {summary['min_spo2']}%")
        cols[4].metric("Avg Heart Rate", f"{summary['avg_hr']} BPM")
        if summary['has_accel']:
            cols[5].metric(
                "Supine Events / Motion Filtered",
                f"{summary['supine_events']} / {summary['motion_windows_skipped']}"
            )

        # Interactive Signal Plot
        st.subheader("Overnight Physiological Signals")
        plot_df = df.iloc[::max(1, len(df) // 4000)]

        n_rows = 4 if summary['has_accel'] else 3
        titles = [
            "Respiratory Effort (Filtered Piezo)",
            "Blood Oxygen (SpO₂ %)",
            "Tracheal Audio Energy (RMS)"
        ]
        if summary['has_accel']:
            titles.append("Body Acceleration Magnitude (ADXL343 m/s²)")

        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=tuple(titles)
        )

        fig.add_trace(go.Scatter(x=plot_df['Time_sec'], y=plot_df['Piezo_Filtered'], name="Piezo", line=dict(color="#1E88E5")), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df['Time_sec'], y=plot_df['SpO2_Clean'], name="SpO₂ %", line=dict(color="#43A047")), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_df['Time_sec'], y=plot_df['Audio_RMS'], name="Audio RMS", line=dict(color="#8E24AA")), row=3, col=1)

        if summary['has_accel']:
            fig.add_trace(go.Scatter(x=plot_df['Time_sec'], y=plot_df['Acc_Mag'], name="Accel Mag", line=dict(color="#FB8C00")), row=4, col=1)

        event_colors = {'OSA': 'rgba(229, 57, 53, 0.25)', 'CSA': 'rgba(30, 136, 229, 0.25)', 'Hypopnea': 'rgba(251, 140, 0, 0.25)'}
        if not events_df.empty:
            for _, ev in events_df.iterrows():
                fig.add_vrect(
                    x0=ev['Start_sec'], x1=ev['End_sec'],
                    fillcolor=event_colors.get(ev['Event_Type'], 'rgba(150,150,150,0.2)'),
                    opacity=0.6, layer="below", line_width=0,
                    annotation_text=ev['Event_Type'], annotation_position="top left",
                    row="all", col=1
                )

        fig.update_layout(height=800 if summary['has_accel'] else 650, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        fig.update_xaxes(title_text="Elapsed Time (seconds)", row=n_rows, col=1)
        st.plotly_chart(fig, width="stretch")

        # Event Log Table & Export
        st.subheader("Detected Respiratory Events")
        if not events_df.empty:
            st.dataframe(events_df, width="stretch")
            csv_export = events_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Event Report (CSV)",
                data=csv_export,
                file_name="sleep_saver_report.csv",
                mime="text/csv"
            )
        else:
            st.success("No apnea or hypopnea events were detected during this session.")