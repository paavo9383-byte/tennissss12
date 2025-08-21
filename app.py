import requests
import streamlit as st
import numpy as np
import pandas as pd
import time

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
BASE_URL = "https://api.api-tennis.com/tennis/"

st.set_page_config(page_title="🎾 Tennis Analyzer Live", layout="wide")
st.title("🎾 Tennis Analyzer Pro – Reaaliaikainen Live-analyysi")

def fetch_live_matches():
    url = f"{BASE_URL}?method=get_livescore&APIkey={API_KEY}"
    data = requests.get(url).json()
    return data.get("result", [])

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

st.subheader("Live-ottelut")

live_matches = fetch_live_matches()
if not live_matches:
    st.warning("Ei käynnissä olevia otteluita juuri nyt.")
    st.stop()

selected_match = st.selectbox("Valitse live-ottelu", [
    f"{m.get('event_first_player')} vs {m.get('event_second_player')}" for m in live_matches
])
match_data = live_matches[[f"{m.get('event_first_player')} vs {m.get('event_second_player')}" 
                           for m in live_matches].index(selected_match)]

home = match_data.get("event_first_player")
away = match_data.get("event_second_player")

home_stats = match_data.get("homeStats", {})
away_stats = match_data.get("awayStats", {})

home_serve = float(home_stats.get("serveWinPercent", 65))
away_serve = float(away_stats.get("serveWinPercent", 65))
home_rank = int(home_stats.get("rank", 100))
away_rank = int(away_stats.get("rank", 100))

st.write(f"{home} - Syöttövoitto%: {home_serve}, Ranking: {home_rank}")
st.write(f"{away} - Syöttövoitto%: {away_serve}, Ranking: {away_rank}")

analysis_type = st.radio("Valitse analyysi", ["Simuloi ottelu", "Analysoi value-vetoja"])

# --- Reaaliaikainen päivitys ---
if st.button("Päivitä analyysi"):
    with st.spinner("Lasketaan simulointia ja odotusarvoja..."):
        home_prob = simulate_match(home_serve, away_serve, home_rank, away_rank)
        away_prob = 1 - home_prob

        if analysis_type == "Simuloi ottelu":
            st.subheader("Simulaatio tulokset")
            st.write(f"{home} voittotodennäköisyys: **{home_prob:.2%}**")
            st.write(f"{away} voittotodennäköisyys: **{away_prob:.2%}**")
        else:
            st.subheader("Value-vetojen analyysi")
            # Placeholder-kertoimet, voi hakea vedonlyönti-APIsta
            home_odds, away_odds = 2.10, 1.75
            home_value = home_prob * home_odds
            away_value = away_prob * away_odds

            df = pd.DataFrame({
                "Pelaaja": [home, away],
                "Voittotodennäköisyys": [home_prob, away_prob],
                "Kertoimet": [home_odds, away_odds],
                "Odotusarvo": [home_value, away_value]
            })

            st.dataframe(df.style.format({"Voittotodennäköisyys": "{:.2%}", "Odotusarvo": "{:.2f}"}).apply(
                lambda x: ['background-color: lightgreen' if v > 1 else '' for v in x['Odotusarvo']], axis=1))

st.write("💡 Vihreällä korostetut odotusarvot > 1 tarkoittavat mahdollisia value-vetoja.")
