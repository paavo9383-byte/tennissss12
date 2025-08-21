import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- API Setup ---
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
BASE_URL = "https://api.api-tennis.com/tennis/"

st.set_page_config(page_title="🎾 Tennis Analyzer Pro", layout="wide")
st.title("🎾 Tennis Analyzer Pro – Kaikki päivän ottelut")

# --- Hae turnaukset ---
tournaments_url = f"{BASE_URL}?method=get_tournaments&APIkey={API_KEY}"
tournaments_data = requests.get(tournaments_url).json().get("result", [])
tournament_dict = {t['tournament_name']: t['tournament_id'] for t in tournaments_data}

# --- Valitse turnaus ---
tournament_names = list(tournament_dict.keys())
selected_tournament = st.selectbox("Valitse turnaus", tournament_names)

# --- Hae ottelut valitusta turnauksesta ---
today = datetime.now().strftime("%Y-%m-%d")
fixtures_url = f"{BASE_URL}?method=get_fixtures&APIkey={API_KEY}&date_start={today}&date_stop={today}"
fixtures_data = requests.get(fixtures_url).json().get("result", [])
# Suodatetaan valitun turnauksen ottelut
matches = [m for m in fixtures_data if m['tournament_id'] == tournament_dict[selected_tournament]]

if not matches:
    st.warning("Ei tulevia otteluita tässä turnauksessa tänään.")
    st.stop()

# --- Valitse ottelu ---
selected_match = st.selectbox("Valitse ottelu", [
    f"{m['event_first_player']} vs {m['event_second_player']}" for m in matches
])
match_index = [f"{m['event_first_player']} vs {m['event_second_player']}" for m in matches].index(selected_match)
match_data = matches[match_index]

home = match_data['event_first_player']
away = match_data['event_second_player']

# --- Hae pelaajien tilastot ---
def get_player_stats(player_name):
    stats_url = f"{BASE_URL}?method=get_players&APIkey={API_KEY}&search={player_name}"
    try:
        data = requests.get(stats_url).json()
        player_info = data.get("result", [{}])[0]
        serve_win = float(player_info.get("serveWinPercent", np.random.uniform(60, 70)))
        rank = int(player_info.get("rank", np.random.randint(1, 150)))
        age = int(player_info.get("age", np.random.randint(18, 35)))
    except:
        serve_win = np.random.uniform(60, 70)
        rank = np.random.randint(1, 150)
        age = np.random.randint(18, 35)
    return serve_win, rank, age

home_serve, home_rank, home_age = get_player_stats(home)
away_serve, away_rank, away_age = get_player_stats(away)

st.write(f"{home} - Syöttö%: {home_serve}, Ranking: {home_rank}, Ikä: {home_age}")
st.write(f"{away} - Syöttö%: {away_serve}, Ranking: {away_rank}, Ikä: {away_age}")

# --- Ammattilaisten kaava ---
def win_probability(h_serve, h_rank, h_age, a_serve, a_rank, a_age):
    b0 = 0
    b1 = 0.08  # syöttöprosentti
    b2 = -0.03 # ranking
    b3 = -0.01 # ikä
    h_score = b0 + b1*h_serve + b2*h_rank + b3*h_age
    a_score = b0 + b1*a_serve + b2*a_rank + b3*a_age
    prob = 1 / (1 + np.exp(-(h_score - a_score)))
    return prob

home_prob = win_probability(home_serve, home_rank, home_age, away_serve, away_rank, away_age)
away_prob = 1 - home_prob

# --- Value-vetojen laskenta ---
# Placeholder-kertoimet, voi liittää vedonlyönti-API myöhemmin
home_odds, away_odds = 2.10, 1.75
home_ev = home_prob * home_odds
away_ev = away_prob * away_odds

st.subheader("Analyysi")
st.write(f"{home} voittotodennäköisyys: **{home_prob:.2%}**, Odotusarvo: **{home_ev:.2f}**")
st.write(f"{away} voittotodennäköisyys: **{away_prob:.2%}**, Odotusarvo: **{away_ev:.2f}**")

# --- Näytä taulukko kaikista vaihtoehdoista ---
df = pd.DataFrame({
    "Pelaaja": [home, away],
    "Voittotodennäköisyys": [home_prob, away_prob],
    "Kertoimet": [home_odds, away_odds],
    "Odotusarvo": [home_ev, away_ev]
})
st.dataframe(df.style.format({"Voittotodennäköisyys": "{:.2%}", "Odotusarvo": "{:.2f}"}).apply(
    lambda x: ['background-color: lightgreen' if v > 1 else '' for v in x['Odotusarvo']], axis=1))

st.write("💡 Vihreällä korostetut odotusarvot > 1 tarkoittavat mahdollisia value-vetoja.")

# --- Mahdollisuus simuloida ottelu ---
if st.button("Simuloi ottelu 1000 kertaa"):
    simulations = np.random.rand(1000) < home_prob
    home_wins = np.sum(simulations)
    away_wins = 1000 - home_wins
    st.write(f"Simulaatio 1000 ottelusta: {home} voittaa {home_wins} kertaa, {away} voittaa {away_wins} kertaa.")
