import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Football Predictor",
    page_icon="⚽",
    layout="wide"
)

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "football_predictor.pkl"
RANKINGS_PATH = BASE_DIR / "rankings.csv"
RESULTS_PATH = BASE_DIR / "results.csv"

STADIUM = BASE_DIR / "assets" / "stadium.jpg"

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(MODEL_PATH)

# ==========================================
# LOAD DATA
# ==========================================

rankings = pd.read_csv(RANKINGS_PATH)
results = pd.read_csv(RESULTS_PATH)

rankings["date"] = pd.to_datetime(
    rankings["date"],
    format="mixed",
    errors="coerce"
)

results["date"] = pd.to_datetime(
    results["date"],
    format="mixed",
    errors="coerce"
)

rankings = rankings.sort_values("date")
latest = rankings.groupby("team").tail(1)

teams = sorted(latest["team"].unique())
FLAG_CODES = {
    "Argentina":"ar",
    "Australia":"au",
    "Austria":"at",
    "Belgium":"be",
    "Brazil":"br",
    "Canada":"ca",
    "Chile":"cl",
    "China PR":"cn",
    "Colombia":"co",
    "Croatia":"hr",
    "Czech Republic":"cz",
    "Denmark":"dk",
    "Ecuador":"ec",
    "Egypt":"eg",
    "England":"gb-eng",
    "France":"fr",
    "Germany":"de",
    "Ghana":"gh",
    "Iran":"ir",
    "Iraq":"iq",
    "Italy":"it",
    "Japan":"jp",
    "Mexico":"mx",
    "Morocco":"ma",
    "Netherlands":"nl",
    "New Zealand":"nz",
    "Norway":"no",
    "Paraguay":"py",
    "Peru":"pe",
    "Poland":"pl",
    "Portugal":"pt",
    "Romania":"ro",
    "Scotland":"gb-sct",
    "Saudi Arabia":"sa",
    "Senegal":"sn",
    "Serbia":"rs",
    "South Africa":"za",
    "South Korea":"kr",
    "Spain":"es",
    "Sweden":"se",
    "Switzerland":"ch",
    "Tunisia":"tn",
    "Turkey":"tr",
    "Ukraine":"ua",
    "United States":"us",
    "Uruguay":"uy",
    "Wales":"gb-wls"
}

def get_flag(team):
    code = FLAG_CODES.get(team)

    if code:
        return f"https://flagcdn.com/w320/{code}.png"

    return "https://flagcdn.com/w320/un.png"

# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

header{
visibility:hidden;
}

footer{
visibility:hidden;
}

.stApp{
background:#08111d;
}

.block-container{
max-width:1450px;
padding-top:0rem;
}

.hero{

padding:0;

overflow:hidden;

border-radius:28px;

box-shadow:0 12px 35px rgba(0,0,0,.45);

margin-bottom:40px;

}

.hero img{

width:100%;

border-radius:28px;

}

.section-title{

font-size:26px;

font-weight:700;

color:white;

margin-bottom:15px;

}

.card{

background:#141f33;

border-radius:22px;

padding:25px;

border:1px solid #24344d;

box-shadow:0 10px 25px rgba(0,0,0,.35);

}

.stButton>button{

width:100%;

height:58px;

border-radius:15px;

border:none;

font-size:22px;

font-weight:700;

background:#00d26a;

color:black;

}

.stButton>button:hover{

background:#00b85d;

}

</style>
""", unsafe_allow_html=True)
# ==========================================
# RECENT FORM
# ==========================================

def get_recent_form(team, n=5):

    matches = results[
        (results["home_team"] == team) |
        (results["away_team"] == team)
    ].sort_values("date").tail(n)

    points = 0

    for _, row in matches.iterrows():

        if row["home_team"] == team:

            if row["home_score"] > row["away_score"]:
                points += 3

            elif row["home_score"] == row["away_score"]:
                points += 1

        else:

            if row["away_score"] > row["home_score"]:
                points += 3

            elif row["away_score"] == row["home_score"]:
                points += 1

    return points


# ==========================================
# HEAD TO HEAD
# ==========================================

def get_h2h(team1, team2):

    matches = results[
        (
            ((results["home_team"] == team1) &
             (results["away_team"] == team2))
            |
            ((results["home_team"] == team2) &
             (results["away_team"] == team1))
        )
    ]

    team1_wins = 0
    team2_wins = 0
    draws = 0

    for _, row in matches.iterrows():

        if row["home_score"] > row["away_score"]:

            if row["home_team"] == team1:
                team1_wins += 1
            else:
                team2_wins += 1

        elif row["away_score"] > row["home_score"]:

            if row["away_team"] == team1:
                team1_wins += 1
            else:
                team2_wins += 1

        else:
            draws += 1

    return team1_wins, team2_wins, draws

# ==========================================
# HERO
# ==========================================

if STADIUM.exists():
    st.image(str(STADIUM), use_container_width=True)
else:
    st.warning("Place stadium.jpg inside assets folder.")

st.markdown(
    "<h1 style='text-align:center;color:white;'>AI Football Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:#cfd8dc;'>Machine Learning • FIFA Rankings • Match Analytics</p>",
    unsafe_allow_html=True
)

st.divider()
st.markdown(
    "<h2 class='section-title'>Choose Teams</h2>",
    unsafe_allow_html=True
)

left, middle, right = st.columns([4,1,4])
with left:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    team1 = st.selectbox(
        "Home Team",
        teams,
        key="home"
    )

    st.image(
        get_flag(team1),
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
with middle:

    st.markdown(
        """
        <div style="
        text-align:center;
        margin-top:160px;
        font-size:70px;
        color:white;
        font-weight:900;
        ">
        VS
        </div>
        """,
        unsafe_allow_html=True
    )
with right:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    team2 = st.selectbox(
        "Away Team",
        teams,
        index=1,
        key="away"
    )

    st.image(
        get_flag(team2),
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
predict = st.button("⚽ Predict Match")

if predict:

    if team1 == team2:
        st.error("Choose two different teams.")
        st.stop()

    t1 = latest[latest["team"] == team1].iloc[0]
    t2 = latest[latest["team"] == team2].iloc[0]

    home_rank = t1["rank"]
    away_rank = t2["rank"]

    home_points = t1["total.points"]
    away_points = t2["total.points"]

    rank_diff = home_rank - away_rank
    points_diff = home_points - away_points

    home_recent = get_recent_form(team1)
    away_recent = get_recent_form(team2)

    home_h2h, away_h2h, draws = get_h2h(team1, team2)

    feature_names = [
        "home_rank",
        "away_rank",
        "home_points",
        "away_points",
        "rank_diff",
        "points_diff",
        "home_recent_points",
        "away_recent_points",
        "home_advantage",
        "tournament_encoded",
        "home_h2h_wins",
        "away_h2h_wins",
        "h2h_draws"
    ]

    X = pd.DataFrame([[
        home_rank,
        away_rank,
        home_points,
        away_points,
        rank_diff,
        points_diff,
        home_recent,
        away_recent,
        0,
        1,
        home_h2h,
        away_h2h,
        draws
    ]], columns=feature_names)

    probs = model.predict_proba(X)[0]
    st.markdown("<br>", unsafe_allow_html=True)

    winner = max(
        [
            (team1, probs[0]),
            ("Draw", probs[2]),
            (team2, probs[1])
        ],
        key=lambda x: x[1]
    )

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#0f172a,#1e293b);
        padding:30px;
        border-radius:25px;
        border:1px solid #334155;
        text-align:center;
        margin-top:20px;
        margin-bottom:30px;
    ">
        <h3 style="color:#94a3b8;">Predicted Winner</h3>
        <h1 style="color:#22c55e;font-size:52px;">
            🏆 {winner[0]}
        </h1>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Match Probability")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            team1,
            f"{probs[0]*100:.2f}%"
        )

    with c2:

        st.metric(
            "Draw",
            f"{probs[2]*100:.2f}%"
        )

    with c3:

        st.metric(
            team2,
            f"{probs[1]*100:.2f}%"
        )
        st.markdown("---")

    st.subheader("Winning Probability")

    st.write(team1)
    st.progress(float(probs[0]))

    st.write("Draw")
    st.progress(float(probs[2]))

    st.write(team2)
    st.progress(float(probs[1]))