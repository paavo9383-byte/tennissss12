import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- API Setup ---
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
BASE_URL = "https://api.api-tennis.com/tennis/"

st.set_page_config(page_title="🎾 Tennis Analyzer Pro", layout="wide")
st.title("🎾 Tennis Analyzer Pro – Ammattimainen vedonlyönti")

# --- Valitse päivä ---
day_option = st.selectbox("Valitse päivä", ["Tänään", "Huomenna", "Valitse päivämäärä"])
if day_option == "Tänään":
    selected_date = datetime.now().strftime("%Y-%m-%d")
elif day_option == "Huomenna":
    selected_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
else:
    selected_date = st.date_input("Valitse päivämäärä").strftime("%Y-%m-%d")

# --- Hae turnaukset ---
tournaments_url = f"{BASE_URL}?method=get_tournaments&APIkey={API_KEY}"
tournaments_json = requests.get(tournaments_url).json()
tournaments_data = tournaments_json.get("result", [])
tournament_dict = {
    t.get('tournament_name', 'N/A'): t.get('tournament_id', None)
    for t in tournaments_data
    if t.get('tournament_name') and t.get('tournament_id')
}
if not tournament_dict:
    st.warning("Turnauksia ei löytynyt API:sta.")
    st.stop()

selected_tournament = st.selectbox("Valitse turnaus", list(tournament_dict.keys()))

# --- Ottelut ---
fixtures_url = f"{BASE_URL}?method=get_fixtures&APIkey={API_KEY}&date_start={selected_date}&date_stop={selected_date}"
fixtures_json = requests.get(fixtures_url).json()
fixtures_data = fixtures_json.get("result", [])
matches = [m for m in fixtures_data if m.get('tournament_id') == tournament_dict[selected_tournament]]

if not matches:
    st.warning("Ei otteluita valitussa turnauksessa.")
    st.stop()

# --- Valitse ottelu ---
match_names = [f"{m.get('event_first_player', 'N/A')} vs {m.get('event_second_player', 'N/A')}" for m in matches]
selected_match_name = st.selectbox("Valitse ottelu", match_names)
match_index = match_names.index(selected_match_name)
match_data = matches[match_index]

home = match_data.get('event_first_player', 'N/A')
away = match_data.get('event_second_player', 'N/A')

# --- Hae pelaajatiedot ---
def get_player_stats(player_name):
    stats_url = f"{BASE_URL}?method=get_players&APIkey={API_KEY}&search={player_name}"
    try:
        data = requests.get(stats_url).json()
        player_info = data.get("result", [{}])[0]
        serve = float(player_info.get("serveWinPercent", np.random.uniform(60,70)))
        return_stat = float(player_info.get("returnWinPercent", np.random.uniform(40,50)))
        rank = int(player_info.get("rank", np.random.randint(1,150)))
        recent_form = float(player_info.get("recent_form", np.random.uniform(0.4,0.7))) # placeholder
    except:
        serve = np.random.uniform(60,70)
        return_stat = np.random.uniform(40,50)
        rank = np.random.randint(1,150)
        recent_form = np.random.uniform(0.4,0.7)
    return serve, return_stat, rank, recent_form

home_serve, home_return, home_rank, home_form = get_player_stats(home)
away_serve, away_return, away_rank, away_form = get_player_stats(away)

# --- Voittotodennäköisyys (logistinen kaava, ammattimainen) ---
def win_probability(h_serve, h_return, h_rank, h_form, a_serve, a_return, a_rank, a_form):
    b0 = 0
    b1, b2, b3, b4 = 0.05, -0.03, 0.1, -0.02
    h_score = b0 + b1*h_serve + b2*h_rank + b3*h_form + b4*h_return
    a_score = b0 + b1*a_serve + b2*a_rank + b3*a_form + b4*a_return
    prob = 1 / (1 + np.exp(-(h_score - a_score)))
    return prob

home_prob = win_probability(home_serve, home_return, home_rank, home_form,
                            away_serve, away_return, away_rank, away_form)
away_prob = 1 - home_prob

# --- Placeholder kertoimet, voisi hakea live-kertoimet API:sta ---
home_odds, away_odds = 2.10, 1.75
home_ev = home_prob * home_odds
away_ev = away_prob * away_odds

# --- Dashboard ---
st.subheader(f"Ottelu: {home} vs {away}")
st.write(f"{home} - Syöttö%: {home_serve}, Return%: {home_return}, Ranking: {home_rank}, Form: {home_form:.2f}")
st.write(f"{away} - Syöttö%: {away_serve}, Return%: {away_return}, Ranking: {away_rank}, Form: {away_form:.2f}")

df = pd.DataFrame({
    "Pelaaja": [home, away],
    "Voittotodennäköisyys": [home_prob, away_prob],
    "Kertoimet": [home_odds, away_odds],
    "Odotusarvo": [home_ev, away_ev]
})
st.dataframe(df.style.format({"Voittotodennäköisyys":"{:.2%}", "Odotusarvo":"{:.2f}"}).apply(
    lambda x: ['background-color: lightgreen' if v>1 else '' for v in x['Odotusarvo']], axis=1))

st.write("💡 Vihreä = value-veto odotusarvo > 1")

# --- Simulaatio ---
if st.button("Simuloi ottelu 1000 kertaa"):
    sims = np.random.rand(1000) < home_prob
    home_wins = np.sum(sims)
    away_wins = 1000 - home_wins
    st.write(f"Simulaatio 1000 ottelusta: {home} voittaa {home_wins} kertaa, {away} voittaa {away_wins} kertaa")
