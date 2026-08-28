"""
 WildFlix — AI Film Recommendation Platform
==============================================
Data: Real TMDB API (5,500 movies)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WildFlix ",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── TMDB CONFIG ───────────────────────────────────────────────────────────────

TMDB_IMG     = "https://image.tmdb.org/t/p/w500"


# ── TRANSLATIONS ──────────────────────────────────────────────────────────────

T = {
    "English": {
        "tagline": "Your Personal Cinema, Powered by AI",
        "search_label": " Enter a film you love:",
        "search_ph": "e.g. Star Wars, Inception, Forrest Gump...",
        "get_recs": " Get Recommendations",
        "n_recs": "Number of recommendations",
        "algo": "Algorithm",
        "filters": " Filters",
        "genre_f": "Genre filter",
        "min_year": "Min release year",
        "max_rt": "Max runtime (min)",
        "min_rat": "Min rating",
        "results_for": "Recommendations for",
        "not_found": "Film not found. Try: Star Wars, Forrest Gump, Inception...",
        "login": "Login", "signup": "Sign Up",
        "username": "Username", "password": "Password",
        "welcome": "Welcome", "logout": "Logout",
        "favorites": "❤️ Favorites",
        "add_fav": "🤍 Add to Favorites",
        "in_fav": "❤️ In Favorites",
        "catalog": " Catalog",
        "stats": "Statistics",
        "about": "About",
        "recs": " Recommendations",
        "top_rated": " Top Rated",
        "rating": "Rating", "year": "Year",
        "runtime": "Runtime", "genre": "Genre",
        "director": "Director", "similarity": "Similarity",
        "search_catalog": "Search catalog...",
        "sort_by": "Sort by",
        "browse": "Browse",
        "total_films": "Total Films",
        "avg_rating": "Avg Rating",
        "avg_runtime": "Avg Runtime",
        "genres": "Genres",
    },
    "Français": {
        "tagline": "Votre Cinéma Personnel, Propulsé par l'IA",
        "search_label": " Entrez un film que vous aimez :",
        "search_ph": "ex. Star Wars, Inception, Forrest Gump...",
        "get_recs": " Obtenir des recommandations",
        "n_recs": "Nombre de recommandations",
        "algo": "Algorithme",
        "filters": " Filtres",
        "genre_f": "Filtre par genre",
        "min_year": "Année min",
        "max_rt": "Durée max (min)",
        "min_rat": "Note minimale",
        "results_for": "Recommandations pour",
        "not_found": "Film introuvable. Essayez : Star Wars, Forrest Gump, Inception...",
        "login": "Connexion", "signup": "Inscription",
        "username": "Nom d'utilisateur", "password": "Mot de passe",
        "welcome": "Bienvenue", "logout": "Déconnexion",
        "favorites": "❤️ Favoris",
        "add_fav": "🤍 Ajouter aux favoris",
        "in_fav": "❤️ Dans les favoris",
        "catalog": " Catalogue",
        "stats": " Statistiques",
        "about": "À propos",
        "recs": " Recommandations",
        "top_rated": " Mieux notés",
        "rating": "Note", "year": "Année",
        "runtime": "Durée", "genre": "Genre",
        "director": "Réalisateur", "similarity": "Similarité",
        "search_catalog": "Rechercher dans le catalogue...",
        "sort_by": "Trier par",
        "browse": "Parcourir",
        "total_films": "Total Films",
        "avg_rating": "Note Moy.",
        "avg_runtime": "Durée Moy.",
        "genres": "Genres",
    },
    "Español": {
        "tagline": "Tu Cine Personal, Impulsado por IA",
        "search_label": " Escribe una película que te guste:",
        "search_ph": "ej. Star Wars, Inception, Forrest Gump...",
        "get_recs": " Obtener recomendaciones",
        "n_recs": "Número de recomendaciones",
        "algo": "Algoritmo",
        "filters": " Filtros",
        "genre_f": "Filtro por género",
        "min_year": "Año mínimo",
        "max_rt": "Duración máx (min)",
        "min_rat": "Puntuación mínima",
        "results_for": "Recomendaciones para",
        "not_found": "Película no encontrada. Prueba: Star Wars, Forrest Gump, Inception...",
        "login": "Iniciar sesión", "signup": "Registrarse",
        "username": "Usuario", "password": "Contraseña",
        "welcome": "Bienvenido", "logout": "Cerrar sesión",
        "favorites": "❤️ Favoritos",
        "add_fav": "🤍 Añadir a favoritos",
        "in_fav": "❤️ En favoritos",
        "catalog": " Catálogo",
        "stats": " Estadísticas",
        "about": "Acerca de",
        "recs": " Recomendaciones",
        "top_rated": "Mejor valoradas",
        "rating": "Puntuación", "year": "Año",
        "runtime": "Duración", "genre": "Género",
        "director": "Director", "similarity": "Similitud",
        "search_catalog": "Buscar en el catálogo...",
        "sort_by": "Ordenar por",
        "browse": "Explorar",
        "total_films": "Total Películas",
        "avg_rating": "Puntuación Media",
        "avg_runtime": "Duración Media",
        "genres": "Géneros",
    }
}


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    .stApp{background:linear-gradient(135deg,#0a0a0f 0%,#0d1117 100%);}
    .wf-header{background:linear-gradient(90deg,#e50914,#b20710);padding:2rem;
      border-radius:12px;margin-bottom:1.5rem;text-align:center;
      box-shadow:0 8px 32px rgba(229,9,20,0.3);}
    .wf-logo{font-family:'Bebas Neue',cursive;font-size:4rem;color:white;
      letter-spacing:0.1em;text-shadow:2px 2px 8px rgba(0,0,0,0.5);margin:0;}
    .wf-sub{font-size:1.1rem;color:rgba(255,255,255,0.85);font-weight:300;margin-top:0.3rem;}
    .movie-card{background:linear-gradient(145deg,#1a1a2e,#16213e);
      border:1px solid rgba(229,9,20,0.2);border-radius:12px;padding:1.2rem;
      margin-bottom:0.8rem;transition:all 0.2s;}
    .movie-card:hover{transform:translateY(-3px);box-shadow:0 8px 24px rgba(229,9,20,0.25);
      border-color:rgba(229,9,20,0.5);}
    .movie-title{font-size:0.9rem;font-weight:700;color:#fff;margin-bottom:0.3rem;}
    .movie-meta{font-size:0.85rem;color:rgba(255,255,255,0.6);}
    .badge-red{display:inline-block;background:#e50914;color:white;font-weight:700;
      font-size:0.9rem;padding:0.2rem 0.6rem;border-radius:20px;margin-right:0.4rem;}
    .badge-genre{display:inline-block;background:rgba(229,9,20,0.15);
      border:1px solid rgba(229,9,20,0.4);color:#ff6b7a;font-size:0.75rem;
      padding:0.15rem 0.5rem;border-radius:12px;margin:0.1rem;}
    .sim-bar{height:4px;border-radius:2px;margin-top:0.5rem;
      background:linear-gradient(90deg,#e50914,#ff6b7a);}
    .kpi-box{background:linear-gradient(145deg,#1a1a2e,#16213e);
      border:1px solid rgba(229,9,20,0.3);border-radius:10px;
      padding:1rem;text-align:center;min-height:130px;display:flex;flex-direction:column;align-items:center;justify-content:center;}
    .kpi-val{font-size:1.5rem;font-weight:700;color:#e50914;white-space:nowrap;}
    .kpi-lbl{font-size:0.8rem;color:rgba(255,255,255,0.6);
      text-transform:uppercase;letter-spacing:0.05em;}
    .stButton>button{background:linear-gradient(90deg,#e50914,#b20710);
      color:white;border:none;border-radius:8px;font-weight:600;
      padding:0.6rem 1.5rem;width:100%;transition:all 0.2s;}
    .stButton>button:hover{background:linear-gradient(90deg,#ff1a27,#c00810);
      transform:translateY(-1px);box-shadow:0 4px 12px rgba(229,9,20,0.4);}
    div[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d0d1a,#0a0a0f);
      border-right:1px solid rgba(229,9,20,0.2);}
    .section-hdr{font-size:1.4rem;font-weight:700;color:#fff;
      border-left:4px solid #e50914;padding-left:0.8rem;margin:1.5rem 0 1rem;}
    </style>
    """, unsafe_allow_html=True)


# ── INTRO SOUND ───────────────────────────────────────────────────────────────

def play_intro_sound():
    if "sound_played" not in st.session_state:
        st.session_state.sound_played = True
        import numpy as np, io, wave
        sr = 44100
        def tone(f, start, dur):
            t = np.linspace(0, dur, int(sr*dur))
            return np.sin(2*np.pi*f*t) * 0.3
    
        sound = np.zeros(int(sr*1.5))
        for f, s, d in [(130,0,0.5),(196,0.3,0.4),(392,0.6,0.9),(523,0.8,0.7)]:
            start = int(s*sr)
            sig = tone(f, s, d)
            sound[start:start+len(sig)] += sig
        
        sound = (sound * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(sound.tobytes())
        buf.seek(0)
        st.audio(buf, format='audio/wav', autoplay=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    urls = [
        "https://raw.githubusercontent.com/thibdrv/Recommendation_System/refs/heads/main/BDD/powerbi_movies.csv",
        "https://raw.githubusercontent.com/thibdrv/Recommendation_System/refs/heads/main/BDD/tmdb_movies_raw.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url, low_memory=False)
        except Exception:
            continue

        # ensure required derived columns
        if "primary_genre" not in df.columns:
            df["primary_genre"] = df["genres"].apply(
                lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "Unknown")
        if "weighted_rating" not in df.columns:
            df["vote_count"]   = pd.to_numeric(df.get("vote_count", 0), errors="coerce").fillna(0)
            df["vote_average"] = pd.to_numeric(df.get("vote_average", 0), errors="coerce").fillna(0)
            m = df["vote_count"].quantile(0.75)
            C = df["vote_average"].mean()
            df["weighted_rating"] = ((df["vote_count"] / (df["vote_count"] + m)) * df["vote_average"] +
                                      (m / (df["vote_count"] + m)) * C).round(3)
        df = df[df.get("status", pd.Series(["Released"] * len(df))) == "Released"] if "status" in df.columns else df
        df = df[df.get("adult", pd.Series([False] * len(df))) == False] if "adult" in df.columns else df
        df = df.reset_index(drop=True)
        return df

    st.error("Data files not found or unreachable. Check the URLs or your internet connection.")
    st.stop()

@st.cache_resource(show_spinner=False)
def build_models(_df):
    df = _df.copy()

    def clean(t):
        return re.sub(r"[^a-zA-Z0-9 ]", " ", str(t).lower()).strip() if pd.notna(t) else ""

    df["genres_c"]   = df["genres"].apply(lambda x: " ".join([g.strip().replace(" ","") for g in str(x).split(",")]) if pd.notna(x) else "")
    df["dir_c"]      = df["director"].apply(lambda x: str(x).replace(" ","").lower() if pd.notna(x) else "")
    df["cast_c"]     = df["cast"].apply(lambda x: " ".join([a.strip().replace(" ","").lower() for a in str(x).split(",")[:5]]) if pd.notna(x) else "")
    df["overview_c"] = df["overview"].apply(clean)
    df["tagline_c"]  = df.get("tagline", pd.Series([""]*(len(df)))).apply(clean)

    df["soup"] = (df["genres_c"]*4 + " " + df["dir_c"]*3 + " " +
                  df["cast_c"]*2  + " " + df["overview_c"]*2 + " " + df["tagline_c"])

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2),
                             min_df=2, max_features=20000, sublinear_tf=True)
    mat = tfidf.fit_transform(df["soup"].fillna(""))
    df = df.reset_index(drop=True)
    t2i = pd.Series(df.index, index=df["title"].str.lower())
    return mat, t2i, df

def fuzzy(title, t2i):
    tl = title.lower().strip()
    if tl in t2i.index: return tl
    matches = [t for t in t2i.index if tl in t or t in tl]
    if matches: return sorted(matches, key=lambda x: abs(len(x)-len(tl)))[0]
    for tok in tl.split():
        if len(tok) > 3:
            matches = [t for t in t2i.index if tok in t]
            if matches: return sorted(matches, key=lambda x: abs(len(x)-len(tl)))[0]
    return None

def get_recs(title, df, mat, t2i, n=5, algo="Content-Based (TF-IDF)",
              genre_f=None, min_year=None, max_rt=None, min_rat=None,
              content_w=0.65):
    matched = fuzzy(title, t2i)
    if not matched: return None, None
    idx = t2i[matched]
    if isinstance(idx, pd.Series): idx = idx.iloc[0]

    if algo in ("Content-Based (TF-IDF)", "Hybrid (TF-IDF + Popularity)"):
        sim = cosine_similarity(mat[idx], mat).flatten()
        if algo == "Hybrid (TF-IDF + Popularity)":
            pop = df["popularity"].fillna(0).values
            pop_norm = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
            sim = content_w * sim + (1 - content_w) * pop_norm
    elif algo == "Highest Rated":
        sim = df["weighted_rating"].fillna(0).values
    else:  # Most Popular
        sim = df["popularity"].fillna(0).values

    scored = df.copy()
    scored["_score"] = sim
    scored = scored[scored.index != idx]

    if genre_f and genre_f != "All":
        scored = scored[scored["genres"].str.contains(genre_f, case=False, na=False)]
    if min_year: scored = scored[scored["release_year"] >= min_year]
    if max_rt:   scored = scored[scored["runtime"] <= max_rt]
    if min_rat:  scored = scored[scored["vote_average"] >= min_rat]

    results = scored.nlargest(n, "_score")[[
        "title","release_year","primary_genre","genres","vote_average",
        "vote_count","runtime","director","cast","_score","poster_path",
        "overview","weighted_rating","imdb_id"
    ]].reset_index(drop=True)
    return results, df.iloc[idx]


# ── AUTH ──────────────────────────────────────────────────────────────────────

def init_auth():
    if "users_db" not in st.session_state:
        st.session_state.users_db = {
            "admin": {"password":"admin123","role":"admin","favorites":[]},
            "demo":  {"password":"demo",    "role":"user", "favorites":[]},
        }
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.current_user = None

def auth_sidebar(t):
    init_auth()
    if st.session_state.logged_in:
        u = st.session_state.current_user
        role = st.session_state.users_db[u]["role"]
        st.sidebar.success(f" {t['welcome']}, **{u}**" + ( " 👑"if role=="admin" else ""))
        if st.sidebar.button(t["logout"]):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.rerun()
        return True
    else:
        tab1, tab2 = st.sidebar.tabs([t["login"], t["signup"]])
        with tab1:
            u = st.text_input(t["username"], key="lu")
            p = st.text_input(t["password"], type="password", key="lp")
            if st.button(t["login"], key="btn_l"):
                db = st.session_state.users_db
                if u in db and db[u]["password"] == p:
                    st.session_state.logged_in = True
                    st.session_state.current_user = u
                    st.rerun()
                else:
                    st.error(" Invalid. Try demo/demo")
        with tab2:
            nu = st.text_input(t["username"], key="ru")
            np_ = st.text_input(t["password"], type="password", key="rp")
            if st.button(t["signup"], key="btn_s"):
                if nu and np_:
                    if nu in st.session_state.users_db:
                        st.error("Username taken!")
                    else:
                        st.session_state.users_db[nu] = {"password":np_,"role":"user","favorites":[]}
                        st.session_state.logged_in = True
                        st.session_state.current_user = nu
                        st.rerun()
        return False

    
# ── MOVIE CARD ────────────────────────────────────────────────────────────────

def movie_card(row, rank, t, show_score=True):
    genres_html = "".join([
        f'<span class="badge-genre">{g.strip()}</span>'
        for g in str(row.get("genres", row.get("primary_genre",""))).split(",")[:3]
    ])
    score = float(row.get("_score", row.get("weighted_rating", 0)))
    score_pct = min(score * 100, 100) if score <= 1 else min(score / 10 * 100, 100)
    sim_bar = f'<div class="sim-bar" style="width:{score_pct:.0f}%"></div>' if show_score else ""
    sim_txt = f'<div style="font-size:0.73rem;color:rgba(255,255,255,0.38);margin-top:0.2rem">{t["similarity"]}: {score:.4f}</div>' if show_score and score <= 1 else ""

    st.markdown(f"""
    <div class="movie-card">
      <div class="movie-title">#{rank} {row['title']}</div>
      <div class="movie-meta">
        <span class="badge-red"> {float(row.get('vote_average',0)):.1f}</span>
        {int(row.get('release_year',0))} · {int(row.get('runtime',0))} min
        {f"· {row['director']}" if pd.notna(row.get('director','')) else ''}
      </div>
      <div style="margin-top:0.4rem">{genres_html}</div>
      {sim_bar}{sim_txt}
    </div>""", unsafe_allow_html=True)

    
# ── STATS PAGE ────────────────────────────────────────────────────────────────

def show_stats(df, t):
    st.markdown('<div class="section-hdr"> Catalog Statistics — Real TMDB Data</div>', unsafe_allow_html=True)

    all_genres = []
    for g in df["genres"].dropna():
        all_genres.extend([x.strip() for x in str(g).split(",") if x.strip()])
    gc = pd.Series(all_genres).value_counts()

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4],[
        (f"{len(df):,}", t["total_films"]),
        (f"{df['vote_average'].mean():.2f}/10", t["avg_rating"]),
        (f"{df['runtime'].mean():.0f} min", t["avg_runtime"]),
        (f"{df['primary_genre'].nunique()}", t["genres"]),
    ]):
        col.markdown(f'<div class="kpi-box"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    import plotly.express as px
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(x=gc.values[:12], y=gc.index[:12], orientation="h",
                     title="Genre Distribution — TMDB",
                     color=gc.values[:12], color_continuous_scale="Reds",
                     labels={"x":"Films","y":"Genre"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(df, x="vote_average", nbins=50, title="Rating Distribution — TMDB",
                             color_discrete_sequence=["#e50914"],
                             labels={"vote_average":"Rating","count":"Films"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        yc = df.groupby("release_year").size().reset_index(name="count")
        yc = yc[yc["release_year"] >= 1960]
        fig3 = px.area(yc, x="release_year", y="count", title="Films per Year — TMDB",
                        color_discrete_sequence=["#e50914"],
                        labels={"release_year":"Year","count":"Films"})
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        top10 = df.nlargest(10,"weighted_rating")[["title","vote_average","weighted_rating"]].reset_index(drop=True)
        top10.index += 1
        top10.columns = ["Title","TMDB Rating","Weighted Rating"]
        st.markdown("** Top 10 Films (Weighted Rating)**")
        st.dataframe(top10, use_container_width=True, height=330)


 #        ── CATALOG PAGE ──────────────────────────────────────────────────────────────

def show_catalog(df, t):
    st.markdown('<div class="section-hdr"> Browse Real TMDB Catalog</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        all_pgs = sorted(df["primary_genre"].dropna().unique().tolist())
        genre_sel = st.selectbox(t["genre_f"], ["All"] + all_pgs)
    with c2:
        sort_sel = st.selectbox(t["sort_by"], ["Weighted Rating","TMDB Rating","Release Year","Popularity"])
    with c3:
        search_term = st.text_input(t["search_catalog"])

    filtered = df.copy()
    if genre_sel != "All":
        filtered = filtered[filtered["primary_genre"] == genre_sel]
    if search_term:
        filtered = filtered[filtered["title"].str.contains(search_term, case=False, na=False)]

    sort_col = {"Weighted Rating":"weighted_rating","TMDB Rating":"vote_average",
                "Release Year":"release_year","Popularity":"popularity"}[sort_sel]
    filtered = filtered.nlargest(60, sort_col)
    st.markdown(f"**Showing {len(filtered)} films from TMDB API**")

    cols = st.columns(3)
    for i, (_, row) in enumerate(filtered.head(30).iterrows()):
        with cols[i % 3]:
            poster = str(row.get("poster_path",""))
            if poster and poster.startswith("http"):
                st.image(poster, width=130)
            st.markdown(f"""
<div class="movie-card">
  <div class="movie-title">{row['title']} ({int(row.get('release_year',0))})</div>
  <div class="movie-meta">
     {float(row.get('vote_average',0)):.1f} · {int(row.get('runtime',0))} min<br>
     {row.get('primary_genre','?')} ·  {str(row.get('director','?'))[:25]}
  </div>
</div>
""", unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # Language
    lang = st.sidebar.selectbox(" Language / Langue / Idioma", ["English","Français","Español"])
    t = T[lang]

    # Intro sound
#    play_intro_sound()

    # Auth
    logged_in = auth_sidebar(t)

    # Load
    with st.spinner(" Loading WildFlix..."):
        df = load_data()
        mat, t2i, df = build_models(df)

    # Header
    st.markdown(f"""
    <div class="wf-header">
      <div class="wf-logo"> WildFlix</div>
      <div class="wf-sub">{t['tagline']}</div>
      <div style="font-size:0.75rem;color:rgba(255,255,255,0.5);margin-top:0.5rem">
        Data : TMDB API · {len(df):,} real movies · API Key :
      </div>
    </div>""", unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.radio("Navigate", [t["recs"], t["catalog"], t["stats"], t["about"]])

    
    # ── RECOMMENDATIONS PAGE ─────────────────────────────────────────────────

    if page == t["recs"]:

        # Quick-pick from popular films
        popular = df.nlargest(30, "vote_count")["title"].tolist()
        quick = st.selectbox(" Quick pick:", ["(type below)"] + popular)

        movie_input = st.text_input(
            t["search_label"],
            value=quick if quick != "(type below)" else "",
            placeholder=t["search_ph"]
        )

        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown(f"**{t['filters']}**")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                all_pgs = sorted(df["primary_genre"].dropna().unique().tolist())
                genre_f = st.selectbox(t["genre_f"], ["All"] + all_pgs)
            with fc2:
                min_year = st.slider(t["min_year"], 1960, 2024, 1980)
            with fc3:
                max_rt = st.slider(t["max_rt"], 60, 300, 300)

        with c2:
            n_recs = st.slider(t["n_recs"], 3, 10, 5)
            algo   = st.selectbox(t["algo"], [
                "Content-Based (TF-IDF)",
                "Hybrid (TF-IDF + Popularity)",
                "Highest Rated",
                "Most Popular"
            ])
            min_rat = st.slider(t["min_rat"], 0.0, 9.0, 5.0, 0.5)
            if algo == "Hybrid (TF-IDF + Popularity)":
                c_w = st.slider("Content ↔ Popularity weight", 0.0, 1.0, 0.65, 0.05)
            else:
                c_w = 0.65

        if st.button(t["get_recs"], use_container_width=True):
            if not movie_input.strip():
                st.warning(t["search_ph"])
            else:
                with st.spinner(" Finding best matches..."):
                    gf = None if genre_f == "All" else genre_f
                    recs, inp_movie = get_recs(
                        movie_input, df, mat, t2i, n=n_recs, algo=algo,
                        genre_f=gf, min_year=min_year, max_rt=max_rt,
                        min_rat=min_rat, content_w=c_w
                    )

                if recs is None or len(recs) == 0:
                    st.error(t["not_found"])
                else:
                    # Input movie info
                    st.markdown(f'<div class="section-hdr"> Because you liked: <em>{inp_movie["title"]}</em></div>', unsafe_allow_html=True)
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric(" TMDB Rating", f"{float(inp_movie.get('vote_average',0)):.1f}")
                    m2.metric(" Year", str(int(inp_movie.get("release_year",0))))
                    m3.metric(" Runtime", f"{int(inp_movie.get('runtime',0))} min")
                    m4.metric(" Genre", str(inp_movie.get("primary_genre","?")))

                    st.markdown(f'<div class="section-hdr">{t["results_for"]} "{movie_input}"</div>', unsafe_allow_html=True)

                    show_score = algo not in ("Highest Rated","Most Popular")

                    for i, (_, row) in enumerate(recs.iterrows()):
                        col_img, col_info = st.columns([1, 4])

                        with col_img:
                            poster = str(row.get("poster_path",""))
                            if poster and poster.startswith("http"):
                                st.image(poster, width=110)
                            else:
                                emoji = {"Action":"💥","Adventure":"🗺️","Animation":"🎨",
                                         "Comedy":"😂","Crime":"🔫","Drama":"🎭",
                                         "Fantasy":"✨","Horror":"👻","Romance":"❤️",
                                         "Science Fiction":"🚀","Thriller":"😱",
                                         "Documentary":"📽️","Mystery":"🔍"}.get(
                                    str(row.get("primary_genre","")).strip(), "🎬")
                                st.markdown(f'<div style="font-size:3rem;text-align:center;padding:1rem">{emoji}</div>', unsafe_allow_html=True)

                        with col_info:
                            movie_card(row, i+1, t, show_score=show_score)
                            # Overview
                            if pd.notna(row.get("overview","")) and str(row.get("overview","")).strip():
                                with st.expander("Plot"):
                                    st.write(row["overview"])
                            # Favorite
                            if logged_in:
                                current_role = st.session_state.users_db[st.session_state.current_user].get("role","user")
                                is_admin = (current_role == "admin")
                                user_favs = st.session_state.users_db[st.session_state.current_user]["favorites"]
                                is_fav = row["title"] in user_favs
                                if st.button(t["in_fav"] if is_fav else t["add_fav"], key=f"fav_{i}_{row['title'][:20]}"):
                                    if is_fav:
                                        user_favs.remove(row["title"])
                                    else:
                                        user_favs.append(row["title"])
                                    st.rerun()

                    # Favorites list
                    if logged_in:
                        user_favs = st.session_state.users_db[st.session_state.current_user]["favorites"]
                        if user_favs:
                            st.markdown(f'<div class="section-hdr">{t["favorites"]}</div>', unsafe_allow_html=True)
                            for fav in user_favs:
                                st.write(f"• {fav}")


    # ── CATALOG ──────────────────────────────────────────────────────────────
    elif page == t["catalog"]:
        show_catalog(df, t)


    # ── STATS ────────────────────────────────────────────────────────────────
    elif page == t["stats"]:
        show_stats(df, t)
        

    # ── ABOUT ────────────────────────────────────────────────────────────────
    elif page == t["about"]:
        st.markdown("""
##  About WildFlix 
                    
WildFlix is an AI-powered film recommendation platform built for a local cinema in **Creuse, France**.

###  Real Data from TMDB API
- **Source:** `https://api.themoviedb.org/3`
- **Endpoints used:** `/movie/popular`, `/movie/top_rated`, `/discover/movie`, `/movie/{id}`, `/movie/{id}/credits`
- **Dataset:** 5,500 real movies with posters, cast, budgets, ratings

###  Recommendation Algorithms

| Algorithm | Description |
|-----------|-------------|
| **Content-Based (TF-IDF)** | Genres × 4 + Director × 3 + Cast × 2 + Overview × 2 + Tagline, Cosine Similarity |
| **Hybrid** | TF-IDF content + TMDB popularity score, weighted blend |
| **Highest Rated** | IMDb-style weighted rating formula |
| **Most Popular** | TMDB live popularity score |

###  Power BI Data Model
```
FACT:  powerbi_ratings.csv    (user interactions)
DIM:   powerbi_movies.csv     (all 5,261 cleaned movies)
DIM:   powerbi_genres.csv     (genre-exploded)
AGG:   powerbi_genre_kpi.csv  (per-genre stats)
AGG:   powerbi_directors.csv  (director analytics)
AGG:   powerbi_yearly_trends.csv
AGG:   powerbi_countries.csv
AGG:   powerbi_languages.csv
```

###  Run Locally
```bash
pip install -r requirements.txt
streamlit run app/wildflix_app.py
```

###  Bonus Features Implemented
- TF-IDF + Cosine, Hybrid, Popularity algorithms
- User authentication (Login / Sign Up / Admin)
- English 🇬🇧 / Français 🇫🇷 / Español 🇪🇸
- Filters: genre, year, runtime, rating
- Real TMDB poster images
- WildFlix intro sound
- Favorites system
- Streamlit Cloud deployment ready

---
*WildFlix | Wild Code School Projet 2 | TMDB API | Python + scikit-learn + Streamlit*
        """)
        
    # Footer
    st.markdown("""
    <div style="text-align:center;padding:2rem;color:rgba(255,255,255,0.25);
                font-size:0.78rem;margin-top:3rem;
                border-top:1px solid rgba(229,9,20,0.15);">
       WildFlix · Wild Code School Projet 2 · TMDB API · 5,500 Real Movies
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
