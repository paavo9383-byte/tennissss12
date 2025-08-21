import requests
import streamlit as st
import numpy as np
from datetime import date, timedelta

# 🔑 API-avain
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
BASE_URL = "https://api.api-tennis.com/tennis/"

st.title("🎾 Tennis Analyzer – Ammattimainen automaatti")

# ===============================
# 1️⃣ Funktio API-kutsuille
# ===============================
def fetch_api(method, extra_params=""):
    url = f"{BASE_URL}?method={method}&APIkey={API_KEY}{extra_params}"
    try:
        r = requests.get(url)
        if r.status_code != 200:
            st.error(f"{method} API error: {r.status_code}")
            return []
        data = r.json()
        return data.get("result", [])
    except Exception as e:
        st.error(f"{method} error: {e}")
        return []

# ===============================
# 2️⃣ Hae tämän päivän ottelut ja turnaukset
# ===============================
today = date.today().isoformat()
fixtures = fetch_api("get_fixtures", f"&date_start={today}&date_stop={today}")
tournaments = fetch_api("get_tournaments")

if not fixtures:
    st.warning("Ei otteluita tänään.")
else:
    st.subheader(f"Tänään pelattavat ottelut ({today})")

    for match in fixtures[:10]:  # näytä max 10 ottelua
        home = match.get("home", "N/A")
        away = match.get("away", "N/A")
        tournament = match.get("tournament", "Ei turnausta")
        st.write(f"**{home} vs {away}** | Turnaus: {tournament}")

        # ===============================
        # 3️⃣ Hae pelaajien tilastot ja rankingit
        # ===============================
        home_stats = match.get("homeStats", {})
        away_stats = match.get("awayStats", {})

        # Oletetaan placeholder-arvot jos API ei anna
        home_serve = float(home_stats.get("serveWinPercent", 65))
        away_serve = float(away_stats.get("serveWinPercent", 65))
        home_rank = int(home_stats.get("rank", 100))
        away_rank = int(away_stats.get("rank", 100))

        st.write(f"{home} - Syöttövoitto%: {home_serve}, Ranking: {home_rank}")
        st.write(f"{away} - Syöttövoitto%: {away_serve}, Ranking: {away_rank}")

        # ===============================
        # 4️⃣ Ammattimainen Monte Carlo -simulaatio
        # Painotetaan pelaajien rankingia ja syöttöprosenttia
        # ===============================
        def simulate_match(a_serve, b_serve, a_rank, b_rank, n_sim=5000):
            # Ranking painottaa todennäköisyyksiä
            rank_factor = b_rank / (a_rank + b_rank)  # Mitä pienempi rank, sitä parempi
            a_wins = 0
            for _ in range(n_sim):
                a_sets, b_sets = 0, 0
                while a_sets < 2 and b_sets < 2:
                    a_games, b_games = 0, 0
                    while a_games < 6 and b_games < 6:
                        # Todennäköisyys yhdistää syöttö ja ranking
                        prob = (a_serve/(a_serve+b_serve)) * 0.7 + rank_factor * 0.3
                        if np.random.rand() < prob:
                            a_games += 1
                        else:
                            b_games += 1
                    if a_games > b_games:
                        a_sets += 1
                    else:
                        b_sets += 1
                if a_sets > b_sets:
                    a_wins += 1
            return a_wins / n_sim

        home_prob = simulate_match(home_serve, away_serve, home_rank, away_rank)
        away_prob = 1 - home_prob

        st.write(f"{home} voittotodennäköisyys: **{home_prob:.2%}**")
        st.write(f"{away} voittotodennäköisyys: **{away_prob:.2%}**")

        # ===============================
        # 5️⃣ Vedonlyöntiarvot
        # ===============================
        home_odds, away_odds = 2.10, 1.75
        home_value = home_prob * home_odds
        away_value = away_prob * away_odds
        st.write(f"{home}: odotusarvo = {home_value:.2f}")
        st.write(f"{away}: odotusarvo = {away_value:.2f}")
        st.write("---")

# ===============================
# 6️⃣ Live-scoret
# ===============================
st.subheader("Live-scoret tänään")
livescores = fetch_api("get_livescore", f"&date_start={today}&date_stop={today}")
if livescores:
    for live in livescores[:10]:
        st.write(f"{live.get('home')} vs {live.get('away')} | Score: {live.get('score','N/A')}")
else:
    st.write("Ei live-otteluita tänään.")
