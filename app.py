import requests
import streamlit as st
import numpy as np
from datetime import date

# API-avain
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
BASE_URL = "https://api.api-tennis.com/tennis/"

st.title("🎾 Tennis Analyzer – Ammattimainen automaatti")

# Hae tämän päivän ottelut
today = date.today().isoformat()
url = f"{BASE_URL}?method=get_fixtures&APIkey={API_KEY}&date_start={today}&date_stop={today}"
response = requests.get(url)
data = response.json()

if data.get("success") == 1:
    fixtures = data.get("result", [])
else:
    st.error("Virhe haettaessa otteluita.")
    fixtures = []

if not fixtures:
    st.warning("Ei otteluita tänään.")
else:
    st.subheader(f"Tänään pelattavat ottelut ({today})")

    for match in fixtures[:10]:  # näytä max 10 ottelua
        home = match.get("event_first_player", "N/A")
        away = match.get("event_second_player", "N/A")
        tournament = match.get("event_type_type", "Ei turnausta")
        st.write(f"**{home} vs {away}** | Turnaus: {tournament}")

        # Hae pelaajien tilastot
        home_stats = match.get("homeStats", {})
        away_stats = match.get("awayStats", {})

        # Oletetaan placeholder-arvot jos API ei anna
        home_serve = float(home_stats.get("serveWinPercent", 65))
        away_serve = float(away_stats.get("serveWinPercent", 65))
        home_rank = int(home_stats.get("rank", 100))
        away_rank = int(away_stats.get("rank", 100))

        st.write(f"{home} - Syöttövoitto%: {home_serve}, Ranking: {home_rank}")
        st.write(f"{away} - Syöttövoitto%: {away_serve}, Ranking: {away_rank}")

        # Monte Carlo -simulaatio
        def simulate_match(a_serve, b_serve, a_rank, b_rank, n_sim=5000):
            rank_factor = b_rank / (a_rank + b_rank)
            a_wins = 0
            for _ in range(n_sim):
                a_sets, b_sets = 0, 0
                while a_sets < 2 and b_sets < 2:
                    a_games, b_games = 0, 0
                    while a_games < 6 and b_games < 6:
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

        # Vedonlyöntiarvot
        home_odds, away_odds = 2.10, 1.75
        home_value = home_prob * home_odds
        away_value = away_prob * away_odds
        st.write(f"{home}: odotusarvo = {home_value:.2f}")
        st.write(f"{away}: odotusarvo = {away_value:.2f}")
        st.write("---")

# Live-scoret
st.subheader("Live-scoret tänään")
url = f"{BASE_URL}?method=get_livescore&APIkey={API_KEY}&date_start={today}&date_stop={today}"
response = requests.get(url)
data = response.json()

if data.get("success") == 1:
    livescores = data.get("result", [])
else:
    st.error("Virhe haettaessa live-scoreja.")
    livescores = []

if livescores:
    for live in livescores[:10]:
        st.write(f"{live.get('event_first_player')} vs {live.get('event_second_player')} | Score: {live.get('event_game_result','N/A')}")
else:
    st.write("Ei live-otteluita tänään.")
