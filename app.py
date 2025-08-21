import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
BASE_URL = "https://api.api-tennis.com/tennis/"

st.set_page_config(page_title="🎾 Tennis Analyzer Pro", layout="wide")
st.title("🎾 Tennis Analyzer Pro – Kaikki päivän ottelut")

# --- Hae päivän ottelut ---
today = datetime.now().strftime("%Y-%m-%d")
fixtures_url = f"{BASE_URL}?method=get_fixtures&APIkey={API_KEY}&date_start={today}&date_stop={today}"
fixtures_data = requests.get(fixtures_url).json()
matches = fixtures_data.get("result", [])

if not matches:
    st.warning("Ei tulevia otteluita tänään.")
    st.stop()

# --- Valitse ottelu ---
selected_match = st.selectbox("Valitse ottelu", [
    f"{m['event_first_player']} vs {m['event_second_player']}" for m in matches
])
match_data = matches[[f"{m['event_first_player']} vs {m['event_second_player']}" 
                      for m in matches].index(selected_match)]

home = match_data['event_first_player']
away = match_data['event_second_player']

# --- Hae pelaajien tilastot ---
def get_player_stats(player_name):
    # Jos API ei anna suoraan kaikkia tilastoja, käytetään placeholder-arvoja
    stats_url = f"{BASE_URL}?method=get_players&APIkey={API_KEY}&search={player_name}"
    try:
        data = requests.get(stats_url).json()
        player_info = data.get("result", [{}])[0]
        serve_win = float(player_info.get("serveWinPercent", 65))
        rank = int(player_info.get("rank", 100))
        age = int(player_info.get("age", 25))
    except:
        serve_win, rank, age = 65, 100, 25
    return serve_win, rank, age

home_serve, home_rank, home_age = get_player_stats(home)
away_serve, away_rank, away_age = get_player_stats(away)

st.write(f"{home} - Syöttö%: {home_serve}, Ranking: {home_rank}, Ikä: {home_age}")
st.write(f"{away} - Syöttö%: {away_serve}, Ranking: {away_rank}, Ikä: {away_age}")

# --- Ammattilaisten kaava (yksinkertaistettu logistinen regressio) ---
def win_probability(h_serve, h_rank, h_age, a_serve, a_rank, a_age):
    # kertoimet arbitrarily valittu simulaatioon, voi hienosäätää
    b0 = 0
    b1 = 0.03  # syöttöprosentti
    b2 = -0.02 # ranking
    b3 = -0.01 # ikä
    h_score = b0 + b1*h_serve + b2*h_rank + b3*h_age
    a_score = b0 + b1*a_serve + b2*a_rank + b3*a_age
    prob = 1 / (1 + np.exp(-(h_score - a_score)))
    return prob

home_prob = win_probability(home_serve, home_rank, home_age, away_serve, away_rank, away_age)
away_prob = 1 - home_prob

# --- Value-vetojen laskenta ---
# Placeholder-kertoimet, voidaan liittää vedonlyönti-API
home_odds, away_odds = 2.10, 1.75
home_ev = home_prob * home_odds
away_ev = away_prob * away_odds

st.subheader("Analyysi")
st.write(f"{home} voittotodennäköisyys: **{home_prob:.2%}**, Odotusarvo: **{home_ev:.2f}**")
st.write(f"{away} voittotodennäköisyys: **{away_prob:.2%}**, Odotusarvo: **{away_ev:.2f}**")

# --- Näytä taulukko
df = pd.DataFrame({
    "Pelaaja": [home, away],
    "Voittotodennäköisyys": [home_prob, away_prob],
    "Kertoimet": [home_odds, away_odds],
    "Odotusarvo": [home_ev, away_ev]
})
st.dataframe(df.style.format({"Voittotodennäköisyys": "{:.2%}", "Odotusarvo": "{:.2f}"}).apply(
    lambda x: ['background-color: lightgreen' if v > 1 else '' for v in x['Odotusarvo']], axis=1))

st.write("💡 Vihreällä korostetut odotusarvot > 1 tarkoittavat mahdollisia value-vetoja.")
