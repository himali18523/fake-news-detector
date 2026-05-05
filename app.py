import streamlit as st
import re
import math
from collections import Counter

# ── Pure-Python TF-IDF + Logistic Regression (no scikit-learn) ───────────────

FAKE = [
    "BREAKING Scientists confirm drinking bleach honey cures cancers overnight government hiding",
    "EXCLUSIVE NASA admits moon landing filmed Stanley Kubrick Hollywood studio actors paid",
    "government hiding free energy device fifty years protect oil companies share delete",
    "SHOCKING towers secretly transmitting mind control signals share before deleted urgent",
    "microchips inserted vaccines track every human earth bill gates population control",
    "Doctors HATE trick eating raw garlic midnight cures diabetes permanently big pharma",
    "EXPOSED earth flat astronauts paid actors leaked proof government cover up",
    "insider leaks proof world leaders shape shifting reptilian aliens secret society",
    "alkaline water reverses aging years big pharma banned study suppressed miracle cure",
    "WHO secretly declared smartphones cause instant brain cancer cover up exposed",
    "man cured HIV herbal tea hospitals dont want know miracle suppressed cure",
    "chemtrails mind altering chemicals keep population compliant government secret program",
    "pentagon docs aliens living underground confirmed leaked documents whistleblower",
    "facebook update steal photos copy paste status opt out warning urgent share",
    "woman regrows limb turmeric paste positive thinking doctors baffled miracle",
    "deep state caused earthquake weather control machines government secret weapon",
    "supreme court secretly ruled income tax illegal no need pay exposed leaked",
    "hospitals receive money every death reported cover up conspiracy exposed revealed",
    "cure alzheimer existed suppressed pharmaceutical giants big pharma exposed secret",
    "ice cream raises IQ points harvard study censored media suppressed truth revealed",
    "fluoride tap water mass sedation program global elites population control wake up",
    "hydrogen peroxide veins eliminates viruses instantly doctors hate secret cure",
    "cancer cells food supply wake up sheeple government poisoning population secret",
    "CIA document proves assassination killed federal reserve bankers leaked proof",
    "stage cancer reversed baking soda lemon juice doctors baffled suppressed cure miracle",
    "URGENT SHARE microwave radiation causes cancer children doctors hiding truth exposed",
    "secret meeting decides world depopulation plan globalist agenda exposed revealed leaked",
    "ALERT new law secretly passed removes rights citizens government hiding mainstream media",
    "SHOCKING celebrity secretly funding mind control program exposed insider leaked proof",
    "BREAKING cure coronavirus suppressed governments protect vaccine profits share now urgent",
]
REAL = [
    "Federal Reserve raised interest rates basis points citing persistent inflation economy",
    "researchers Johns Hopkins published findings Nature Medicine improved survival rates trial",
    "European Parliament approved data privacy regulations requiring explicit user consent law",
    "NASA Perseverance rover collected rock sample Jezero Crater Mars mission science",
    "Apple reported quarterly earnings billion below analyst expectations revenue fiscal",
    "World Health Organization updated influenza vaccination guidelines season health",
    "MIT developed battery technology double energy density lithium ion cells research",
    "United Nations Security Council passed resolution calling ceasefire conflict region",
    "central bank held repo rate steady percent monetary policy meeting inflation",
    "Lancet study found regular exercise reduces risk heart disease percent research",
    "Tesla recalled vehicles potential issue power steering assist safety federal",
    "Bank England raised benchmark interest rate percent highest years monetary policy",
    "Stanford researchers identified protein marker enabling earlier detection cancer",
    "Google announced billion investment data center infrastructure states expansion",
    "exports fell percent year October driven declining semiconductor demand trade",
    "archaeologists Egypt uncovered previously unknown tomb Valley Kings discovery",
    "IMF revised global growth forecast percent amid slowing trade economic outlook",
    "updated mRNA vaccine showed percent efficacy new variants trials clinical study",
    "SpaceX successfully launched Falcon Heavy rocket classified government payload mission",
    "Supreme Court ruled limiting authority regulate carbon emissions environmental law",
    "WHO confirms respiratory pathogen identified Southeast Asia monitoring ongoing health",
    "Parliament passed Digital Markets Act aimed curbing monopolistic tech practices regulation",
    "global temperatures exceeded pre industrial levels degrees Celsius WMO climate report",
    "ISRO successfully docked spacecraft orbit mission technology achievement science",
    "Governor stated inflation expected moderate percent fiscal year economic forecast",
    "new study published journal shows treatment reduces symptoms patients clinical trial",
    "parliament voted approve budget billion infrastructure spending economic development",
    "central bank governor said economy growing percent year quarter official data",
    "researchers discovered new species deep ocean expedition scientific findings published",
    "government announced new policy reduce carbon emissions percent decade climate change",
]

STOPWORDS = set("the a an and or but in on at to for of is are was were be been being have has had do does did will would could should may might shall can this that these those with from by about into through during before after above below between each few more most other some such no nor not only same so than too very just because if when where who which how all both each few more most other some".split())

def tokenize(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    # add bigrams
    bigrams = [tokens[i]+"_"+tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams

def compute_tfidf(corpus_tokens):
    N = len(corpus_tokens)
    df = Counter()
    for doc in corpus_tokens:
        df.update(set(doc))
    vocab = {w for doc in corpus_tokens for w in doc}
    idf = {w: math.log((N + 1) / (df[w] + 1)) + 1 for w in vocab}
    tfidf_docs = []
    for doc in corpus_tokens:
        tf = Counter(doc)
        total = len(doc) or 1
        vec = {w: (tf[w]/total) * idf[w] for w in doc}
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1
        tfidf_docs.append({w: v/norm for w, v in vec.items()})
    return tfidf_docs, idf, vocab

def dot(v1, v2):
    return sum(v1.get(w, 0) * v2.get(w, 0) for w in v2)

class NaiveLR:
    def fit(self, X, y, lr=0.3, epochs=200):
        vocab = set(w for v in X for w in v)
        self.weights = {w: 0.0 for w in vocab}
        self.bias = 0.0
        for _ in range(epochs):
            for vec, label in zip(X, y):
                pred = self._sigmoid(dot(self.weights, vec) + self.bias)
                err = pred - label
                for w in vec:
                    self.weights[w] = self.weights.get(w, 0) - lr * err * vec[w]
                self.bias -= lr * err * 0.01
        return self

    def _sigmoid(self, x):
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x))

    def predict_proba(self, vec):
        p = self._sigmoid(dot(self.weights, vec) + self.bias)
        return 1 - p, p  # real_prob, fake_prob

@st.cache_resource
def train_model():
    texts = FAKE + REAL
    labels = [1]*len(FAKE) + [0]*len(REAL)
    corpus_tokens = [tokenize(t) for t in texts]
    tfidf_docs, idf, vocab = compute_tfidf(corpus_tokens)
    model = NaiveLR().fit(tfidf_docs, labels)
    return model, idf, vocab

def vectorize(text, idf):
    tokens = tokenize(text)
    tf = Counter(tokens)
    total = len(tokens) or 1
    vec = {w: (tf[w]/total) * idf.get(w, math.log(2)+1) for w in tokens}
    norm = math.sqrt(sum(v*v for v in vec.values())) or 1
    return {w: v/norm for w, v in vec.items()}

def get_top_features(model, vec, n=12):
    scores = [(w, model.weights.get(w, 0) * vec[w]) for w in vec]
    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return scores[:n]

def highlight_text(text, features):
    fake_words = {f.split("_")[0] for f, s in features if s > 0}
    real_words = {f.split("_")[0] for f, s in features if s < 0}
    words = text.split()
    result = []
    for word in words:
        clean = re.sub(r"[^a-z]", "", word.lower())
        if clean in fake_words:
            result.append(f'<span style="background:#f87171;color:#000;padding:1px 5px;border-radius:3px;font-weight:700">{word}</span>')
        elif clean in real_words:
            result.append(f'<span style="background:#4ade80;color:#000;padding:1px 5px;border-radius:3px;font-weight:700">{word}</span>')
        else:
            result.append(word)
    return " ".join(result)

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800;900&family=DM+Mono:wght@300;400&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; background-color: #070a0e; color: #cbd5e1; }
.main { background-color: #070a0e; }
.stTextArea textarea { background: #0a0d12 !important; color: #e2e8f0 !important; border: 1px solid #1e293b !important; font-family: 'DM Mono', monospace !important; font-size: 0.85rem !important; }
.stButton > button { background: #fb923c !important; color: #000 !important; font-weight: 700 !important; border: none !important; border-radius: 4px !important; padding: 0.6rem 2rem !important; font-family: 'DM Mono', monospace !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
.stButton > button:hover { background: #f97316 !important; }
div[data-testid="metric-container"] { background: #0a0d12; border: 1px solid #1e293b; border-radius: 6px; padding: 0.8rem; }
div[data-testid="metric-container"] label { color: #475569 !important; font-size: 0.65rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:1.5rem">
  <div style="font-size:0.62rem;color:#475569;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.3rem">
    AI · NLP · 7th Sem Project · VTU
  </div>
  <h1 style="font-family:'Syne',sans-serif;font-size:clamp(2rem,6vw,3.5rem);font-weight:900;color:#e2e8f0;margin:0;letter-spacing:-0.03em;line-height:1">
    FAKE NEWS <span style="color:#fb923c">DETECTOR</span>
  </h1>
  <p style="color:#475569;font-size:0.75rem;margin-top:0.5rem">
    TF-IDF · Logistic Regression · Feature Explainability — no external ML libraries
  </p>
</div>
""", unsafe_allow_html=True)

model, idf, vocab = train_model()

SAMPLES = {
    "🔴 Fake — Health": "Scientists confirm drinking bleach cures cancer overnight. The government has been hiding this cure for 30 years to protect Big Pharma profits. SHARE before they delete this!",
    "🟢 Real — Finance": "The European Central Bank raised interest rates by 25 basis points on Thursday, citing persistent inflation. ECB President Lagarde said further adjustments depend on incoming economic data.",
    "🔴 Fake — Conspiracy": "EXCLUSIVE: NASA admits the moon landing was filmed by Stanley Kubrick. Leaked Pentagon documents prove the Apollo program was a Cold War hoax to intimidate the Soviet Union.",
    "🟢 Real — Science": "Researchers at Stanford published findings in Nature Medicine showing a new mRNA approach to treating brain tumors demonstrated a 34% improvement in median survival time in a Phase II trial.",
}

col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area("Paste any news article, headline, or claim:", height=160,
        placeholder="Enter text to analyze for credibility...")
with col2:
    st.markdown('<div style="font-size:0.62rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">Try a sample</div>', unsafe_allow_html=True)
    for label in SAMPLES:
        if st.button(label, use_container_width=True):
            st.session_state["inp"] = SAMPLES[label]
            st.rerun()

if "inp" in st.session_state and not user_input:
    user_input = st.session_state["inp"]

if st.button("🔍  ANALYZE", use_container_width=False):
    if user_input and len(user_input.strip()) > 20:
        vec = vectorize(user_input, idf)
        real_prob, fake_prob = model.predict_proba(vec)
        confidence = max(fake_prob, real_prob) * 100
        features = get_top_features(model, vec)

        if fake_prob >= 0.60:
            verdict, color, bg, icon = "FAKE", "#f87171", "rgba(248,113,113,0.08)", "⚠️"
        elif fake_prob <= 0.40:
            verdict, color, bg, icon = "REAL", "#4ade80", "rgba(74,222,128,0.08)", "✅"
        else:
            verdict, color, bg, icon = "UNCERTAIN", "#facc15", "rgba(250,204,21,0.08)", "🔎"

        st.markdown(f"""
        <div style="background:{bg};border:1px solid {color}30;border-left:4px solid {color};
             border-radius:8px;padding:1.25rem 1.5rem;margin:1rem 0;display:flex;align-items:center;gap:1rem">
          <span style="font-size:2.5rem">{icon}</span>
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:900;color:{color};line-height:1">{verdict}</div>
            <div style="font-size:0.68rem;color:#475569;margin-top:0.2rem">
              Confidence: {confidence:.1f}% &nbsp;|&nbsp; Fake probability: {fake_prob*100:.1f}% &nbsp;|&nbsp; Real probability: {real_prob*100:.1f}%
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fake Score", f"{fake_prob*100:.1f}%")
        m2.metric("Real Score", f"{real_prob*100:.1f}%")
        m3.metric("Confidence", f"{confidence:.1f}%")
        m4.metric("Words Analyzed", str(len(user_input.split())))

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🔦 Highlighted Text")
            st.markdown('<div style="font-size:0.62rem;color:#475569;margin-bottom:0.5rem">🔴 fake-leaning &nbsp; 🟢 credibility signals</div>', unsafe_allow_html=True)
            highlighted = highlight_text(user_input, features)
            st.markdown(f'<div style="background:#0a0d12;border:1px solid #1e293b;border-radius:6px;padding:1rem;font-size:0.82rem;line-height:1.9">{highlighted}</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown("#### ⚠️ Fake Signals")
            fake_feats = [(w.replace("_"," "), s) for w, s in features if s > 0][:6]
            if fake_feats:
                for w, s in fake_feats:
                    bar_w = min(100, int(abs(s) * 800))
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem">
                      <div style="display:flex;justify-content:space-between;font-size:0.7rem;margin-bottom:2px">
                        <span style="color:#f87171">› {w}</span><span style="color:#475569">{s:.3f}</span>
                      </div>
                      <div style="height:4px;background:#1e293b;border-radius:2px">
                        <div style="width:{bar_w}%;height:100%;background:#f87171;border-radius:2px"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#475569;font-size:0.75rem">No strong fake signals detected</p>', unsafe_allow_html=True)

            st.markdown("#### ✅ Credibility Signals")
            real_feats = [(w.replace("_"," "), s) for w, s in features if s < 0][:6]
            if real_feats:
                for w, s in real_feats:
                    bar_w = min(100, int(abs(s) * 800))
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem">
                      <div style="display:flex;justify-content:space-between;font-size:0.7rem;margin-bottom:2px">
                        <span style="color:#4ade80">› {w}</span><span style="color:#475569">{s:.3f}</span>
                      </div>
                      <div style="height:4px;background:#1e293b;border-radius:2px">
                        <div style="width:{bar_w}%;height:100%;background:#4ade80;border-radius:2px"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#475569;font-size:0.75rem">No strong credibility signals found</p>', unsafe_allow_html=True)

    else:
        st.warning("Please enter at least 20 characters.")

st.markdown("---")
st.markdown('<div style="text-align:center;font-size:0.62rem;color:#334155;padding:0.5rem">Fake News Detector · 7th Sem AI Project · VTU · Built with Streamlit</div>', unsafe_allow_html=True)
