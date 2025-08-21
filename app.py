import requests
import streamlit as st
import numpy as np
from datetime import date

# 🔑 API-avain
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# 🔗 Base URL
BASE_URL = "https://api.api-tennis.com/tennis/"

st.title("🎾 Tennis Analyzer App")

# ===============================
# 1️⃣ Turvallinen JSON-pyyntö
# ===============================
def fetch_api(method, extra_params=""):
    url = f"{BASE_URL}?method={method}&APIkey={API_KEY}{extra_params}"
    try:
        r = requests.get(url)
        st.write(f"API status ({method}):", r.status_code)
        if r.status_code != 200:
            st.error(f"{method} API virhe: {r.status_code}")
            return []
        try:
            data = r.json()
            return data.get("result", [])
        except:
            st.error(f"{method} ei palauttanut JSONia.")
            return []
    except Exception as e:
        st.error(f"{method} HTTP virhe: {e}")
        return []

# ===============================
# 2️⃣ Hae turnaukset
# ===============================
st.subheader("Turnaukset")
tournaments = fetch_api("get_tournaments")
tournament_options = {t.get("name", "N/A"): t for t in tournaments}
selected_tournament = st.selectbox("Valitse turnaus", list(tournament_options.keys()))

# ===============================
# 3️⃣ Hae ottelut valitusta turnauksesta
# ===============================
st.subheader("Ottelut")
if selected_tournament:
    tourn_id = tournament_options[selected_tournament].get("id", 0)
    matches = fetch_api("get_events", f"&tournament_id={tourn_id}")
    match_options = {f"{m.get('home','?')} vs {m.get('away','?')}": m for m in matches}
    selected_match = st.selectbox("Valitse ottelu", list(match_options.keys()))

# ===============================
# 4️⃣ Näytä ottelun tiedot
# ===============================
if selected_match:
    match = match_options[selected_match]
    home, away = match.get("home","?"), match.get("away","?")
    st.write(f"**{home} vs {away}**")
    st.write("Ottelun ID:", match.get("id","N/A"))
    st.write("Päivämäärä:", match.get("event_date","N/A"))

    # Placeholder-statistiikka simulaatiota varten
    home_serve = float(match.get("homeServeWon", 65)) if "homeServeWon" in match else 65
    away_serve = float(match.get("awayServeWon", 65)) if "awayServeWon" in match else 65
    st.write(f"{home} syöttöpisteiden voitto-%: {home_serve}")
    st.write(f"{away} syöttöpisteiden voitto-%: {away_serve}")

    # ===============================
    # 5️⃣ Simuloi ottelu
    # ===============================
    def simulate_match(a_serve, b_serve, n_sim=5000):
        a_wins = 0
        for _ in range(n_sim):
            a_sets, b_sets = 0, 0
            while a_sets < 2 and b_sets < 2:
                a_games, b_games = 0, 0
                while a_games < 6 and b_games < 6:
                    if np.random.rand() < a_serve/(a_serve+b_serve):
                        a_games += 1
                    else:
                        b_games += 1
                if a_games > b_games:
                    a_sets += 1
                else:
                    b_sets += 1
            if a_sets > b_sets:
                a_wins += 1
        return a_wins/n_sim

    home_prob = simulate_match(home_serve, away_serve)
    away_prob = 1 - home_prob

    st.subheader("🔮 Ennusteet")
    st.write(f"{home} voittotodennäköisyys: **{home_prob:.2%}**")
    st.write(f"{away} voittotodennäköisyys: **{away_prob:.2%}**")

    # ===============================
    # 6️⃣ Vedonlyönti
    # ===============================
    st.subheader("Vedonlyönti")
    home_odds = st.number_input(f"{home} kerroin", value=2.10, step=0.01)
    away_odds = st.number_input(f"{away} kerroin", value=1.75, step=0.01)

    home_value = home_prob * home_odds
    away_value = away_prob * away_odds

    st.write(f"{home}: odotusarvo = {home_value:.2f} (>{1} = value!)")
    st.write(f"{away}: odotusarvo = {away_value:.2f} (>{1} = value!)")

# ===============================
# 7️⃣ Reaaliaikainen live-score (valinnainen)
# ===============================
st.subheader("Live-score")
today = date.today().isoformat()
livescores = fetch_api("get_livescore", f"&date_start={today}&date_stop={today}")
for live in livescores[:5]:  # näytä max 5
    st.write(f"{live.get('home')} vs {live.get('away')} | Score: {live.get('score','N/A')}")
