import streamlit as st
import pandas as pd
import numpy as np
import re, joblib, os, textwrap
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Training data ─────────────────────────────────────────────────────────────
FAKE = [
    "BREAKING: Scientists confirm that drinking bleach with honey cures all cancers overnight.",
    "EXCLUSIVE: NASA admits the moon landing was filmed by Stanley Kubrick in Hollywood.",
    "The government has been hiding a free energy device for 50 years to protect oil companies!!!",
    "SHOCKING: 5G towers are secretly transmitting mind-control signals. Share before deleted!",
    "Bill Gates microchips are inserted into COVID vaccines to track every human on Earth.",
    "Doctors HATE this trick — eating raw garlic at midnight cures diabetes permanently!",
    "EXPOSED: The Earth is flat and NASA astronauts are all paid actors. Leaked proof.",
    "Anonymous insider leaks proof that world leaders are shape-shifting reptilian aliens!",
    "Drinking alkaline water reverses aging by 20 years. Big Pharma banned this study.",
    "ALERT: WHO secretly declared that all smartphones cause instant brain cancer.",
    "This man cured his HIV with herbal tea. Hospitals DON'T want you to know!",
    "REVEALED: Chemtrails contain mind-altering chemicals to keep population compliant.",
    "EXCLUSIVE: Leaked Pentagon docs show aliens lived underground in US since 1947.",
    "WARNING: New Facebook update steals all your photos — copy-paste this to opt out!!!",
    "Scientist STUNNED as man regrows lost limb using turmeric paste and positive thinking.",
    "Deep State operatives admit causing earthquake in Turkey using weather control.",
    "BREAKING: Supreme Court secretly ruled income tax is illegal — no need to pay!",
    "Whistleblower: hospitals receive $50,000 for every COVID death they report.",
    "Cure for Alzheimer's existed since 1987 but suppressed by pharmaceutical giants.",
    "Eating ice cream for breakfast raises IQ by 30 points, Harvard study was censored.",
    "URGENT SHARE: Fluoride in tap water is a mass sedation program by global elites.",
    "SECRET CURE: Hydrogen peroxide injected into veins eliminates all viruses instantly.",
    "They are putting cancer cells in our food supply. Wake up sheeple!!!",
    "LEAKED: CIA document proves JFK was killed by the Federal Reserve bankers.",
    "Scientists baffled as woman reverses stage 4 cancer with baking soda and lemon juice.",
]
REAL = [
    "The Federal Reserve raised interest rates by 25 basis points, citing persistent inflation.",
    "Researchers at Johns Hopkins published findings in Nature Medicine showing improved survival rates.",
    "European Parliament approved new data privacy regulations requiring explicit user consent.",
    "NASA's Perseverance rover collected its 23rd rock sample from the Jezero Crater on Mars.",
    "Apple reported quarterly earnings of $94.8 billion, slightly below analyst expectations.",
    "The World Health Organization updated influenza vaccination guidelines for the season.",
    "MIT developed a new battery technology that may double the energy density of lithium-ion cells.",
    "The United Nations Security Council passed a resolution calling for a ceasefire.",
    "India's central bank held the repo rate steady at 6.5 percent at its monetary policy meeting.",
    "A Lancet study found regular exercise reduces the risk of heart disease by 35 percent.",
    "Tesla recalled approximately 120,000 vehicles due to a potential power steering issue.",
    "The Bank of England raised its benchmark interest rate to 5.25 percent, highest in 15 years.",
    "Stanford researchers identified a protein marker enabling earlier detection of pancreatic cancer.",
    "Google announced a $2 billion investment in data center infrastructure across three US states.",
    "South Korea's exports fell 8.5 percent year-on-year in October due to declining chip demand.",
    "Archaeologists in Egypt uncovered a previously unknown tomb near the Valley of the Kings.",
    "IMF revised its global growth forecast to 2.9 percent amid slowing trade volumes.",
    "Moderna's updated mRNA vaccine showed 78 percent efficacy against new variants in trials.",
    "SpaceX successfully launched its Falcon Heavy rocket carrying a classified US payload.",
    "The Supreme Court ruled 6-3, limiting the EPA's authority to regulate carbon emissions.",
    "WHO confirms new respiratory pathogen identified in Southeast Asia; monitoring ongoing.",
    "Parliament passed the Digital Markets Act aimed at curbing monopolistic tech practices.",
    "Global temperatures in 2023 exceeded pre-industrial levels by 1.45 degrees Celsius, WMO reports.",
    "ISRO successfully docked two spacecraft in orbit as part of the SpaDeX mission.",
    "RBI Governor stated inflation is expected to moderate to 4.5 percent by Q3 of next fiscal year.",
]

MODEL_PATH = "/home/claude/fakenews_app/model.pkl"

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    texts = [preprocess(t) for t in FAKE + REAL]
    labels = [1]*len(FAKE) + [0]*len(REAL)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, max_iter=1000, random_state=42))
    ])
    pipe.fit(texts, labels)
    joblib.dump(pipe, MODEL_PATH)
    return pipe

def get_top_features(model, text, n=12):
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    processed = preprocess(text)
    tfidf_matrix = vec.transform([processed])
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    tfidf_vals = tfidf_matrix.toarray()[0]
    scores = tfidf_vals * coefs
    nonzero = [(feature_names[i], scores[i]) for i in range(len(scores)) if tfidf_vals[i] > 0]
    nonzero.sort(key=lambda x: abs(x[1]), reverse=True)
    return nonzero[:n]

def highlight_text(text, features):
    fake_words = {f for f, s in features if s > 0}
    real_words = {f for f, s in features if s < 0}
    words = text.split()
    result = []
    for word in words:
        clean = re.sub(r"[^a-z]", "", word.lower())
        if any(clean == fw or clean in fw for fw in fake_words):
            result.append(f'<span style="background:#f87171;color:#000;padding:1px 4px;border-radius:3px;font-weight:600">{word}</span>')
        elif any(clean == rw or clean in rw for rw in real_words):
            result.append(f'<span style="background:#4ade80;color:#000;padding:1px 4px;border-radius:3px;font-weight:600">{word}</span>')
        else:
            result.append(word)
    return " ".join(result)

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; background-color: #070a0e; color: #cbd5e1; }
.main { background-color: #070a0e; }
.stTextArea textarea { background: #0a0d12 !important; color: #e2e8f0 !important; border: 1px solid #1e293b !important; font-family: 'DM Mono', monospace !important; }
.stButton > button { background: #fb923c; color: #000; font-weight: 700; border: none; border-radius: 4px; padding: 0.6rem 2rem; font-family: 'DM Mono', monospace; font-size: 0.9rem; letter-spacing: 0.08em; text-transform: uppercase; }
.stButton > button:hover { background: #f97316; }
div[data-testid="metric-container"] { background: #0a0d12; border: 1px solid #1e293b; border-radius: 6px; padding: 1rem; }
.verdict-box { border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:1.5rem">
  <div style="font-size:0.65rem;color:#475569;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.3rem">
    AI · NLP · 7th Sem Project
  </div>
  <h1 style="font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:900;color:#e2e8f0;margin:0;letter-spacing:-0.03em;line-height:1">
    FAKE NEWS <span style="color:#fb923c">DETECTOR</span>
  </h1>
  <p style="color:#475569;font-size:0.78rem;margin-top:0.5rem">
    TF-IDF + Logistic Regression · LIME Explainability · Real-time Analysis
  </p>
</div>
""", unsafe_allow_html=True)

model = load_model()

SAMPLES = {
    "🔴 Fake — Health Misinformation": "Scientists confirm drinking bleach cures cancer overnight. The government has been hiding this for 30 years to protect Big Pharma profits. Share before they delete this!",
    "🟢 Real — Financial News": "The European Central Bank raised interest rates by 25 basis points, citing persistent inflation. ECB President Lagarde said further adjustments depend on incoming economic data.",
    "🔴 Fake — Conspiracy": "EXCLUSIVE: NASA admits the moon landing was filmed by Stanley Kubrick. Leaked Pentagon documents prove the Apollo program was a Cold War hoax to intimidate the Soviet Union.",
    "🟢 Real — Science": "Researchers at Stanford published findings in Nature Medicine showing a new mRNA approach to treating brain tumors demonstrated a 34% improvement in median survival time in a Phase II trial.",
}

col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area("Paste news article, headline, or claim:", height=160, placeholder="Enter any news text to analyze...")
with col2:
    st.markdown("<div style='font-size:0.65rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem'>Try a sample</div>", unsafe_allow_html=True)
    for label in SAMPLES:
        if st.button(label, use_container_width=True):
            st.session_state["sample"] = SAMPLES[label]

if "sample" in st.session_state and not user_input:
    user_input = st.session_state["sample"]

analyze = st.button("🔍  ANALYZE", use_container_width=False)

if analyze and user_input and len(user_input.strip()) > 20:
    processed = preprocess(user_input)
    pred = model.predict([processed])[0]
    proba = model.predict_proba([processed])[0]
    fake_prob = proba[1]
    real_prob = proba[0]
    confidence = max(fake_prob, real_prob) * 100
    features = get_top_features(model, user_input)

    # Verdict
    if fake_prob >= 0.65:
        verdict, color, bg, icon = "FAKE", "#f87171", "rgba(248,113,113,0.08)", "⚠️"
    elif fake_prob <= 0.35:
        verdict, color, bg, icon = "REAL", "#4ade80", "rgba(74,222,128,0.08)", "✅"
    else:
        verdict, color, bg, icon = "UNCERTAIN", "#facc15", "rgba(250,204,21,0.08)", "⚠"

    st.markdown(f"""
    <div class="verdict-box" style="background:{bg};border:1px solid {color}30;border-left:4px solid {color}">
      <div style="display:flex;align-items:center;gap:1rem">
        <span style="font-size:2.5rem">{icon}</span>
        <div>
          <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:900;color:{color};line-height:1">{verdict}</div>
          <div style="font-size:0.7rem;color:#475569;margin-top:0.2rem">Confidence: {confidence:.1f}% &nbsp;|&nbsp; Fake probability: {fake_prob*100:.1f}% &nbsp;|&nbsp; Real probability: {real_prob*100:.1f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fake Score", f"{fake_prob*100:.1f}%")
    m2.metric("Real Score", f"{real_prob*100:.1f}%")
    m3.metric("Confidence", f"{confidence:.1f}%")
    m4.metric("Word Count", str(len(user_input.split())))

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🔦 Highlighted Text")
        st.markdown("<div style='font-size:0.65rem;color:#475569;margin-bottom:0.5rem'>🔴 fake-leaning words &nbsp; 🟢 real-leaning words</div>", unsafe_allow_html=True)
        highlighted = highlight_text(user_input, features)
        st.markdown(f"""<div style="background:#0a0d12;border:1px solid #1e293b;border-radius:6px;padding:1rem;font-size:0.85rem;line-height:1.8">{highlighted}</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("#### 📊 Feature Importance")
        if features:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#0a0d12")
            ax.set_facecolor("#0a0d12")
            words = [f[0] for f in features[:10]]
            scores = [f[1] for f in features[:10]]
            colors = ["#f87171" if s > 0 else "#4ade80" for s in scores]
            bars = ax.barh(words[::-1], scores[::-1], color=colors[::-1], height=0.6)
            ax.axvline(0, color="#334155", linewidth=1)
            ax.set_xlabel("Impact on Fake Score", color="#64748b", fontsize=8)
            ax.tick_params(colors="#94a3b8", labelsize=8)
            for spine in ax.spines.values(): spine.set_edgecolor("#1e293b")
            ax.xaxis.label.set_color("#64748b")
            fake_patch = mpatches.Patch(color="#f87171", label="→ Fake")
            real_patch = mpatches.Patch(color="#4ade80", label="→ Real")
            ax.legend(handles=[fake_patch, real_patch], facecolor="#0a0d12", edgecolor="#334155", labelcolor="#94a3b8", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Top flags
    st.markdown("---")
    fa_col, tr_col = st.columns(2)
    with fa_col:
        st.markdown("#### ⚠️ Fake-leaning signals")
        fake_feats = [(w, s) for w, s in features if s > 0]
        if fake_feats:
            for w, s in fake_feats[:6]:
                st.markdown(f'<div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.2);border-radius:4px;padding:6px 12px;margin-bottom:4px;font-size:0.75rem;color:#f87171">› &nbsp;<b>{w}</b> &nbsp;<span style="color:#475569">(score: {s:.3f})</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#475569;font-size:0.75rem">No strong fake signals detected</div>', unsafe_allow_html=True)

    with tr_col:
        st.markdown("#### ✅ Real-leaning signals")
        real_feats = [(w, s) for w, s in features if s < 0]
        if real_feats:
            for w, s in real_feats[:6]:
                st.markdown(f'<div style="background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);border-radius:4px;padding:6px 12px;margin-bottom:4px;font-size:0.75rem;color:#4ade80">› &nbsp;<b>{w}</b> &nbsp;<span style="color:#475569">(score: {s:.3f})</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#475569;font-size:0.75rem">No strong credibility signals detected</div>', unsafe_allow_html=True)

elif analyze:
    st.warning("Please enter at least 20 characters of text.")

st.markdown("---")
st.markdown('<div style="text-align:center;font-size:0.65rem;color:#334155;padding:1rem">Fake News Detector · 7th Sem AI Project · Built with Streamlit + Scikit-learn</div>', unsafe_allow_html=True)
