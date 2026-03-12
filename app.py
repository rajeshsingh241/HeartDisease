import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              roc_curve, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Session State ───────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ─── Theme ───────────────────────────────────────────────────────────────────
def get_theme():
    if st.session_state.dark_mode:
        return {
            "bg": "#0f1117", "card": "#1a1d27", "text": "#e8eaf0",
            "subtext": "#8b90a0", "accent": "#e05c6a", "accent2": "#ff8fa3",
            "border": "#2a2d3a", "input_bg": "#12151f",
            "success_bg": "#0d2b1e", "success_border": "#1a7a4a",
            "error_bg": "#2b0d12", "error_border": "#7a1a22",
            "divider": "#2a2d3a", "label": "#a0a4b0",
            "shadow": "0 4px 24px rgba(0,0,0,0.4)",
            "plot_bg": "#1a1d27", "plot_text": "#e8eaf0",
            "metric_bg": "#1a1d27",
        }
    else:
        return {
            "bg": "#f5f6fa", "card": "#ffffff", "text": "#1a1d27",
            "subtext": "#5a5f72", "accent": "#c0392b", "accent2": "#e74c3c",
            "border": "#e0e3ed", "input_bg": "#f9fafc",
            "success_bg": "#e8f8f0", "success_border": "#27ae60",
            "error_bg": "#fdf0f0", "error_border": "#e74c3c",
            "divider": "#e0e3ed", "label": "#5a5f72",
            "shadow": "0 4px 24px rgba(0,0,0,0.08)",
            "plot_bg": "#ffffff", "plot_text": "#1a1d27",
            "metric_bg": "#ffffff",
        }

T = get_theme()

def apply_css(T):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        background-color: {T['bg']} !important;
        color: {T['text']} !important;
    }}
    .stApp {{ background-color: {T['bg']} !important; }}
    #MainMenu, footer {{ visibility: hidden; }}

    section[data-testid="stSidebar"] {{
        background-color: {T['card']} !important;
        border-right: 1px solid {T['border']} !important;
    }}
    section[data-testid="stSidebar"] * {{ color: {T['text']} !important; }}

    .hero {{
        text-align: center; padding: 2.5rem 0 1.5rem 0;
    }}
    .hero-icon {{
        font-size: 3.5rem; display: block; margin-bottom: 0.75rem;
        animation: pulse 2.5s ease-in-out infinite;
    }}
    @keyframes pulse {{
        0%,100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.1); }}
    }}
    .hero h1 {{
        font-family: 'Playfair Display', serif; font-size: 2.6rem;
        font-weight: 700; color: {T['text']}; margin: 0 0 0.5rem 0;
    }}
    .hero p {{ font-size: 1.05rem; color: {T['subtext']}; margin: 0; }}

    .section-title {{
        font-size: 0.72rem; font-weight: 600; letter-spacing: 1.8px;
        text-transform: uppercase; color: {T['accent']};
        margin-bottom: 1.25rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid {T['border']};
    }}

    .metric-card {{
        background: {T['metric_bg']}; border: 1px solid {T['border']};
        border-radius: 12px; padding: 1.2rem 1rem; text-align: center;
        box-shadow: {T['shadow']};
    }}
    .metric-val {{
        font-size: 1.9rem; font-weight: 700; color: {T['accent']};
        font-family: 'Playfair Display', serif;
    }}
    .metric-label {{
        font-size: 0.72rem; color: {T['subtext']};
        text-transform: uppercase; letter-spacing: 0.8px; margin-top: 0.25rem;
    }}

    .model-row {{
        background: {T['card']}; border: 1px solid {T['border']};
        border-radius: 10px; padding: 0.85rem 1.25rem;
        margin-bottom: 0.6rem; display: flex; align-items: center;
        justify-content: space-between; box-shadow: {T['shadow']};
    }}
    .model-name {{ font-weight: 600; font-size: 0.9rem; color: {T['text']}; }}
    .model-badge {{
        font-size: 0.75rem; font-weight: 600; padding: 0.25rem 0.65rem;
        border-radius: 20px; background: {T['accent']}22; color: {T['accent']};
        border: 1px solid {T['accent']}44;
    }}
    .best-model {{
        border: 1.5px solid {T['accent']} !important;
        background: {T['accent']}08 !important;
    }}

    .result-success {{
        background: {T['success_bg']}; border: 1.5px solid {T['success_border']};
        border-radius: 14px; padding: 2rem; text-align: center;
        animation: slideUp 0.4s ease;
    }}
    .result-error {{
        background: {T['error_bg']}; border: 1.5px solid {T['error_border']};
        border-radius: 14px; padding: 2rem; text-align: center;
        animation: slideUp 0.4s ease;
    }}
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(16px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .result-title {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; }}
    .result-sub {{ font-size: 0.9rem; color: {T['subtext']}; line-height: 1.6; }}

    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, {T['accent']}, {T['accent2']}) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; padding: 0.85rem !important;
        font-size: 1rem !important; font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        box-shadow: 0 4px 16px {T['accent']}44 !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px {T['accent']}55 !important;
    }}

    .stAlert {{ display: none !important; }}

    label, .stSelectbox label, .stSlider label, .stNumberInput label {{
        font-size: 0.82rem !important; font-weight: 500 !important;
        color: {T['label']} !important; letter-spacing: 0.3px !important;
    }}
    .stSelectbox > div > div, .stNumberInput > div > div > input {{
        background-color: {T['input_bg']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 8px !important; color: {T['text']} !important;
    }}

    .insight-box {{
        background: {T['card']}; border-left: 3px solid {T['accent']};
        border-radius: 0 10px 10px 0; padding: 1rem 1.25rem;
        margin-bottom: 0.75rem; font-size: 0.88rem; color: {T['subtext']};
        line-height: 1.6;
    }}
    .insight-box strong {{ color: {T['text']}; }}

    .footer {{
        text-align: center; padding: 2rem 0 1rem 0;
        font-size: 0.78rem; color: {T['subtext']};
        border-top: 1px solid {T['divider']}; margin-top: 3rem;
    }}

    hr {{ border-color: {T['divider']} !important; margin: 1.5rem 0 !important; }}
    </style>
    """, unsafe_allow_html=True)

apply_css(T)

# ─── Matplotlib theme helper ─────────────────────────────────────────────────
def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": T["plot_bg"],
        "axes.facecolor": T["plot_bg"],
        "axes.edgecolor": T["border"],
        "axes.labelcolor": T["plot_text"],
        "xtick.color": T["subtext"],
        "ytick.color": T["subtext"],
        "text.color": T["plot_text"],
        "grid.color": T["border"],
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
    })

# ─── Data & Models (cached) ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
    df['Cholesterol'] = df['Cholesterol'].replace(0, ch_mean).round(2)
    bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
    df['RestingBP'] = df['RestingBP'].replace(0, bp_mean).round(2)
    return df

@st.cache_resource
def train_all_models():
    df = load_data()
    df_enc = pd.get_dummies(df, drop_first=True)
    X = df_enc.drop('HeartDisease', axis=1)
    y = df_enc['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(random_state=42),
    }
    results = []
    trained = {}
    for name, m in models.items():
        m.fit(Xtr, y_train)
        trained[name] = m
        yp = m.predict(Xte)
        yprob = m.predict_proba(Xte)[:, 1] if hasattr(m, 'predict_proba') else None
        results.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, yp) * 100, 1),
            'F1 Score': round(f1_score(y_test, yp) * 100, 1),
            'Precision': round(precision_score(y_test, yp) * 100, 1),
            'Recall': round(recall_score(y_test, yp) * 100, 1),
            'ROC-AUC': round(roc_auc_score(y_test, yprob) * 100, 1) if yprob is not None else None,
        })
    return pd.DataFrame(results), trained, X_test, y_test, Xte, X.columns.tolist()

@st.cache_resource
def load_knn():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    cols = joblib.load("columns.pkl")
    return model, scaler, cols

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <span style='font-size:2rem;'>❤️</span>
        <div style='font-family: Playfair Display, serif; font-size:1.2rem;
             font-weight:700; color:{T["accent"]}; margin-top:0.3rem;'>CardioAI</div>
        <div style='font-size:0.72rem; color:{T["subtext"]}; letter-spacing:1px;
             text-transform:uppercase; margin-top:0.2rem;'>Heart Disease Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 EDA", "🤖 Model Comparison", "🔍 Predict"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    icon = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(icon):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown(f"""
    <div style='margin-top:2rem; font-size:0.72rem; color:{T["subtext"]};
         text-align:center; line-height:1.6;'>
        Dataset: 918 patients<br>
        Algorithm: K-Nearest Neighbours<br>
        Accuracy: 89.1%
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(f"""
    <div class="hero">
        <span class="hero-icon">❤️</span>
        <h1>Heart Disease Prediction</h1>
        <p>AI-powered cardiovascular risk assessment · 918 patients · 89.1% accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("918", "Total Patients"),
        ("89.1%", "KNN Accuracy"),
        ("93.0%", "ROC-AUC Score"),
        ("11", "Input Features"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📌 About This Project</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight-box">
            <strong>Dataset:</strong> Heart Failure Prediction Dataset with 918 patient records
            combining 5 heart disease datasets (Cleveland, Hungarian, Switzerland, Long Beach VA, Stalog).
        </div>
        <div class="insight-box">
            <strong>Model:</strong> K-Nearest Neighbours classifier trained on one-hot encoded features
            with StandardScaler normalisation. Achieves 89.1% accuracy and 93% ROC-AUC.
        </div>
        <div class="insight-box">
            <strong>Features:</strong> Age, Sex, Chest Pain Type, Resting BP, Cholesterol,
            Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, Oldpeak, ST Slope.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">🗺️ How To Use</div>', unsafe_allow_html=True)
        steps = [
            ("📊", "EDA", "Explore feature distributions, correlations & dataset insights"),
            ("🤖", "Model Comparison", "Compare 6 ML models with metrics, ROC curves & confusion matrix"),
            ("🔍", "Predict", "Enter patient data and get an instant risk prediction"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div class="insight-box">
                <strong>{icon} {title}</strong><br>{desc}
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer">CardioAI · For informational purposes only · Not a substitute for professional medical advice</div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown(f"""
    <div style='padding: 1.5rem 0 0.5rem 0;'>
        <h2 style='font-family: Playfair Display, serif; font-size: 2rem; color: {T["text"]}; margin:0;'>
            Exploratory Data Analysis
        </h2>
        <p style='color:{T["subtext"]}; margin: 0.3rem 0 0 0;'>
            Understanding the Heart Disease dataset — 918 patients, 12 features
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    df = load_data()
    set_plot_style()
    ACCENT = T["accent"]
    ACCENT2 = "#4a90d9"

    # ── Row 1: Target + Age distribution ─────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">🎯 Target Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor(T["plot_bg"])
        ax.set_facecolor(T["plot_bg"])
        counts = df['HeartDisease'].value_counts()
        bars = ax.bar(['No Disease', 'Heart Disease'], counts.values,
                      color=[ACCENT2, ACCENT], width=0.5, edgecolor='none',
                      linewidth=0)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f'{val}\n({val/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9, color=T["text"], fontweight='600')
        ax.set_ylabel('Count', color=T["subtext"], fontsize=9)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(colors=T["subtext"])
        ax.set_ylim(0, max(counts.values) * 1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">📅 Age Distribution by Outcome</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor(T["plot_bg"])
        ax.set_facecolor(T["plot_bg"])
        for val, color, label in [(0, ACCENT2, 'No Disease'), (1, ACCENT, 'Heart Disease')]:
            data = df[df['HeartDisease'] == val]['Age']
            ax.hist(data, bins=20, alpha=0.7, color=color, label=label, edgecolor='none')
        ax.set_xlabel('Age', color=T["subtext"], fontsize=9)
        ax.set_ylabel('Count', color=T["subtext"], fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=T["subtext"])
        ax.legend(fontsize=8, facecolor=T["card"], edgecolor=T["border"],
                  labelcolor=T["text"])
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Row 2: Sex + Chest Pain ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">⚧ Sex vs Heart Disease</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor(T["plot_bg"])
        ax.set_facecolor(T["plot_bg"])
        sex_data = df.groupby(['Sex', 'HeartDisease']).size().unstack()
        x = np.arange(len(sex_data.index))
        w = 0.35
        ax.bar(x - w/2, sex_data[0], w, color=ACCENT2, label='No Disease', edgecolor='none')
        ax.bar(x + w/2, sex_data[1], w, color=ACCENT, label='Heart Disease', edgecolor='none')
        ax.set_xticks(x)
        ax.set_xticklabels(['Female', 'Male'], color=T["subtext"])
        ax.set_ylabel('Count', color=T["subtext"], fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=T["subtext"])
        ax.legend(fontsize=8, facecolor=T["card"], edgecolor=T["border"], labelcolor=T["text"])
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">💔 Chest Pain Type vs Outcome</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor(T["plot_bg"])
        ax.set_facecolor(T["plot_bg"])
        cp_data = df.groupby(['ChestPainType', 'HeartDisease']).size().unstack()
        x = np.arange(len(cp_data.index))
        ax.bar(x - w/2, cp_data[0], w, color=ACCENT2, label='No Disease', edgecolor='none')
        ax.bar(x + w/2, cp_data[1], w, color=ACCENT, label='Heart Disease', edgecolor='none')
        ax.set_xticks(x)
        ax.set_xticklabels(cp_data.index, color=T["subtext"], fontsize=8)
        ax.set_ylabel('Count', color=T["subtext"], fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=T["subtext"])
        ax.legend(fontsize=8, facecolor=T["card"], edgecolor=T["border"], labelcolor=T["text"])
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Row 3: Cholesterol Boxplot + Max HR ───────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">🩸 Cholesterol by Outcome</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor(T["plot_bg"])
        ax.set_facecolor(T["plot_bg"])
        data0 = df[df['HeartDisease'] == 0]['Cholesterol']
        data1 = df[df['HeartDisease'] == 1]['Cholesterol']
        bp = ax.boxplot([data0, data1], patch_artist=True, widths=0.4,
                        medianprops=dict(color='white', linewidth=2))
        bp['boxes'][0].set_facecolor(ACCENT2 + "99")
        bp['boxes'][1].set_facecolor(ACCENT + "99")
        for element in ['whiskers', 'caps', 'fliers']:
            for item in bp[element]:
                item.set_color(T["subtext"])
        ax.set_xticklabels(['No Disease', 'Heart Disease'], color=T["subtext"])
        ax.set_ylabel('Cholesterol (mg/dL)', color=T["subtext"], fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=T["subtext"])
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">💓 Max Heart Rate by Outcome</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor(T["plot_bg"])
        ax.set_facecolor(T["plot_bg"])
        for val, color, label in [(0, ACCENT2, 'No Disease'), (1, ACCENT, 'Heart Disease')]:
            data = df[df['HeartDisease'] == val]['MaxHR']
            ax.hist(data, bins=20, alpha=0.7, color=color, label=label, edgecolor='none')
        ax.set_xlabel('Max Heart Rate', color=T["subtext"], fontsize=9)
        ax.set_ylabel('Count', color=T["subtext"], fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=T["subtext"])
        ax.legend(fontsize=8, facecolor=T["card"], edgecolor=T["border"], labelcolor=T["text"])
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
    df_enc = pd.get_dummies(df, drop_first=True)
    corr = df_enc.corr()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(T["plot_bg"])
    ax.set_facecolor(T["plot_bg"])
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, ax=ax,
                annot_kws={"size": 7}, linewidths=0.5,
                linecolor=T["border"], cbar_kws={"shrink": 0.8})
    ax.tick_params(colors=T["subtext"], labelsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Key insights
    st.markdown('<div class="section-title">💡 Key Insights</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="insight-box"><strong>ST Slope (Flat):</strong> Strongest positive predictor (r=0.55).
        Patients with flat ST slope have significantly higher disease risk.</div>
        <div class="insight-box"><strong>Exercise Angina:</strong> Strong indicator (r=0.49).
        Chest pain during exercise is a major red flag.</div>
        <div class="insight-box"><strong>Age:</strong> Heart disease patients average 56 years old vs
        51 for healthy patients — a 5-year gap.</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="insight-box"><strong>ST Slope (Up):</strong> Strongest negative predictor (r=-0.62).
        Upward ST slope is a protective indicator.</div>
        <div class="insight-box"><strong>Max Heart Rate:</strong> Lower MaxHR correlates strongly
        with disease (r=-0.40). Reduced cardiac capacity is a key signal.</div>
        <div class="insight-box"><strong>Sex:</strong> 63% of male patients have heart disease
        vs only 26% of female patients — significant gender disparity.</div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown(f"""
    <div style='padding: 1.5rem 0 0.5rem 0;'>
        <h2 style='font-family: Playfair Display, serif; font-size: 2rem; color: {T["text"]}; margin:0;'>
            Model Comparison
        </h2>
        <p style='color:{T["subtext"]}; margin: 0.3rem 0 0 0;'>
            6 algorithms evaluated on the same 80/20 train-test split
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    with st.spinner("Training all models..."):
        results_df, trained_models, X_test, y_test, Xte, feature_cols = train_all_models()

    set_plot_style()

    # ── Metrics Table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Performance Metrics</div>', unsafe_allow_html=True)
    best_acc_idx = results_df['Accuracy'].idxmax()

    for i, row in results_df.iterrows():
        best_class = " best-model" if i == best_acc_idx else ""
        st.markdown(f"""
        <div class="model-row{best_class}">
            <div>
                <span class="model-name">{row['Model']}</span>
                {'<span style="font-size:0.7rem; color:' + T["accent"] + '; margin-left:0.5rem;">★ Best</span>' if i == best_acc_idx else ''}
            </div>
            <div style="display:flex; gap:0.75rem; align-items:center;">
                <div style="text-align:center;">
                    <div class="model-badge">Acc {row['Accuracy']}%</div>
                </div>
                <div style="text-align:center; font-size:0.75rem; color:{T['subtext']};">
                    F1: {row['F1 Score']}%
                </div>
                <div style="text-align:center; font-size:0.75rem; color:{T['subtext']};">
                    Prec: {row['Precision']}%
                </div>
                <div style="text-align:center; font-size:0.75rem; color:{T['subtext']};">
                    Recall: {row['Recall']}%
                </div>
                <div style="text-align:center; font-size:0.75rem; color:{T['subtext']};">
                    AUC: {row['ROC-AUC']}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC Curves + Confusion Matrix ─────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">📈 ROC Curves</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor(T["plot_bg"])
        ax.set_facecolor(T["plot_bg"])

        colors_roc = ['#e05c6a', '#4a90d9', '#50c878', '#f5a623', '#9b59b6', '#1abc9c']
        for (name, model), color in zip(trained_models.items(), colors_roc):
            if hasattr(model, 'predict_proba'):
                yp = model.predict_proba(Xte)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, yp)
                auc = roc_auc_score(y_test, yp)
                ax.plot(fpr, tpr, color=color, lw=1.8,
                        label=f'{name} ({auc:.2f})')

        ax.plot([0, 1], [0, 1], color=T["border"], lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate', color=T["subtext"], fontsize=9)
        ax.set_ylabel('True Positive Rate', color=T["subtext"], fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=T["subtext"])
        ax.legend(fontsize=7.5, facecolor=T["card"], edgecolor=T["border"],
                  labelcolor=T["text"], loc='lower right')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">🔲 KNN Confusion Matrix</div>', unsafe_allow_html=True)
        knn_model = trained_models['KNN']
        y_pred_knn = knn_model.predict(Xte)
        cm = confusion_matrix(y_test, y_pred_knn)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor(T["plot_bg"])
        ax.set_facecolor(T["plot_bg"])

        im = ax.imshow(cm, cmap='RdYlGn', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted\nNo Disease', 'Predicted\nDisease'],
                           color=T["subtext"], fontsize=9)
        ax.set_yticklabels(['Actual\nNo Disease', 'Actual\nDisease'],
                           color=T["subtext"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white', fontsize=20, fontweight='bold')
        ax.spines[:].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🌲 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    rf = trained_models['Random Forest']
    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(T["plot_bg"])
    ax.set_facecolor(T["plot_bg"])
    colors_bar = [T["accent"] if v >= feat_imp.quantile(0.7) else "#4a90d9" for v in feat_imp.values]
    ax.barh(feat_imp.index, feat_imp.values, color=colors_bar, edgecolor='none', height=0.6)
    ax.set_xlabel('Importance Score', color=T["subtext"], fontsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(colors=T["subtext"], labelsize=9)
    ax.grid(axis='x', alpha=0.3, color=T["border"])
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Insight
    st.markdown(f"""
    <div class="insight-box">
        <strong>Why KNN was chosen:</strong> KNN and Logistic Regression both achieve the highest accuracy (89.1%)
        and F1 Score (90.3%) on this dataset. KNN was selected as the final model as it is non-parametric,
        requires no assumptions about data distribution, and is highly interpretable for medical applications.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.markdown(f"""
    <div style='padding: 1.5rem 0 0.5rem 0;'>
        <h2 style='font-family: Playfair Display, serif; font-size: 2rem; color: {T["text"]}; margin:0;'>
            Patient Risk Assessment
        </h2>
        <p style='color:{T["subtext"]}; margin: 0.3rem 0 0 0;'>
            Enter patient details below to predict heart disease risk
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    try:
        model, scaler, expected_columns = load_knn()
    except Exception as e:
        st.markdown(f"""
        <div class="result-error">
            <div class="result-title">⚠️ Model Loading Error</div>
            <div class="result-sub">{e}</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Input form
    st.markdown('<div class="section-title">👤 Patient Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 100, 40)
    with c2:
        sex = st.selectbox("Sex", ["M", "F"])
    with c3:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

    st.divider()
    st.markdown('<div class="section-title">🩺 Vitals & Lab Results</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
    with c2:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    with c3:
        fasting_bs = st.selectbox("Fasting Blood Sugar >120", [0, 1])

    st.divider()
    st.markdown('<div class="section-title">📈 Cardiac Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    with c2:
        exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**ST Slope**")
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], label_visibility="collapsed")

    st.divider()
    predict_btn = st.button("🔍 Predict Heart Disease Risk")

    if predict_btn:
        raw_input = {
            'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
            'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }
        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1] if hasattr(model, 'predict_proba') else None

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧬 Prediction Result</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            if prediction == 1:
                prob_text = f"Risk probability: <strong>{prob*100:.1f}%</strong><br>" if prob else ""
                st.markdown(f"""
                <div class="result-error">
                    <div class="result-title" style="color:#e05c6a;">⚠️ High Risk of Heart Disease</div>
                    <div class="result-sub">
                        {prob_text}
                        The model indicates elevated cardiovascular risk.<br>
                        Please consult a cardiologist for a comprehensive evaluation.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                prob_text = f"Risk probability: <strong>{prob*100:.1f}%</strong><br>" if prob else ""
                st.markdown(f"""
                <div class="result-success">
                    <div class="result-title" style="color:#27ae60;">✅ Low Risk of Heart Disease</div>
                    <div class="result-sub">
                        {prob_text}
                        The model indicates low cardiovascular risk.<br>
                        Maintain a healthy lifestyle and continue with regular checkups.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            # Risk factors summary
            risk_factors = []
            if age > 55: risk_factors.append("Age > 55")
            if chest_pain == "ASY": risk_factors.append("Asymptomatic chest pain")
            if exercise_angina == "Y": risk_factors.append("Exercise angina")
            if st_slope in ["Flat", "Down"]: risk_factors.append(f"ST Slope: {st_slope}")
            if max_hr < 120: risk_factors.append("Low max heart rate")
            if oldpeak > 2: risk_factors.append(f"High oldpeak ({oldpeak})")

            if risk_factors:
                rf_html = "".join([f"<li style='margin-bottom:0.3rem;'>{r}</li>" for r in risk_factors])
                st.markdown(f"""
                <div class="insight-box">
                    <strong>⚠️ Risk Flags Detected</strong>
                    <ul style='margin: 0.5rem 0 0 0; padding-left: 1.2rem; color:{T["subtext"]}; font-size:0.82rem;'>
                        {rf_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="insight-box" style="border-left-color: #27ae60;">
                    <strong style="color:#27ae60;">✅ No Major Risk Flags</strong><br>
                    <span style='font-size:0.82rem;'>Patient input values appear within normal ranges.</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer">CardioAI · For informational purposes only · Not a substitute for professional medical advice</div>
    """, unsafe_allow_html=True)
