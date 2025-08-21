import requests
import streamlit as st
import numpy as np
from datetime import date, timedelta

# 🔑 API-avain
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
BASE_URL = "https://api.api-tennis.com/tennis/"

st.title("🎾 Tennis Analyzer – Todellinen data")

# ===============================
# 1️⃣ Yleinen funktio API-kutsuille
# ===============================
def fetch_api(method, extra_params=""):
    url = f"{BASE_URL}?method={method}&APIkey={API_KEY}{extra_params}"
    try:
        r = requests.get(url)
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
# 2️⃣ Hae viimeisimmät ottelut automaattisesti
# ===============================
st.subheader("Viimeisimmät ottelut ja ennusteet")

# Haetaan viimeiset 3 päivän ottelut
today = date.today()
start = (today - timedelta(days=3)).isoformat()
end = today.isoformat()

fixtures = fetch_api("get_fixtures", f"&date_start={start}&date_stop={end}")
if not fixtures:
    st.warning("Ei otteludataa saatavilla viimeisiltä päiviltä.")
else:
    for match in fixtures[:5]:  # näytä max 5
        home, away = match.get("home","?"), match.get("away","?")
        st.write(f"**{home} vs {away}**")
        st.write("Päivämäärä:", match.get("event_date","N/A"))

        # ===============================
        # 3️⃣ Hae pelaajan todelliset tilastot, jos saatavilla
        # ===============================
        # API ei välttämättä anna täydellisiä syöttö/ranking-dataa, joten katsotaan resultista
        home_stats = match.get("homeStats", {})
        away_stats = match.get("awayStats", {})

        home_serve = float(home_stats.get("serveWinPercent", 65))  # default placeholder
        away_serve = float(away_stats.get("serveWinPercent", 65))  # default placeholder

        st.write(f"{home} syöttöpisteiden voitto-%: {home_serve}")
        st.write(f"{away} syöttöpisteiden voitto-%: {away_serve}")

        # ===============================
        # 4️⃣ Simuloi ottelu Monte Carlo -menetelmällä
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

        st.write(f"{home} voittotodennäköisyys: **{home_prob:.2%}**")
        st.write(f"{away} voittotodennäköisyys: **{away_prob:.2%}**")

        # ===============================
        # 5️⃣ Vedonlyöntiarvot (esimerkki)
        # ===============================
        home_odds, away_odds = 2.10, 1.75
        home_value = home_prob * home_odds
        away_value = away_prob * away_odds
        st.write(f"{home}: odotusarvo = {home_value:.2f}")
        st.write(f"{away}: odotusarvo = {away_value:.2f}")
        st.write("---")

# ===============================
# 6️⃣ Näytä live-scoret automaattisesti
# ===============================
st.subheader("Live-scoret tänään")
livescores = fetch_api("get_livescore", f"&date_start={today}&date_stop={today}")
if livescores:
    for live in livescores[:5]:
        st.write(f"{live.get('home')} vs {live.get('away')} | Score: {live.get('score','N/A')}")
else:
    st.write("Ei live-otteluita tänään.")
