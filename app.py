import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# ------------------ API ------------------
def get_json(url):
    try:
        res = requests.get(url, timeout=15)
        return res.json()
    except:
        return {}

def fetch_fixtures(date_str):
    url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    data = get_json(url)
    if data.get("success") == 1:
        return data.get("result", [])
    return []

def fetch_player(player_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_players&APIkey={API_KEY}&player_key={player_key}"
    data = get_json(url)
    if data.get("success") == 1:
        result = data.get("result", [])
        if isinstance(result, list) and result:
            return result[0]
    return None

def fetch_h2h(p1_key, p2_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_H2H&APIkey={API_KEY}&first_player_key={p1_key}&second_player_key={p2_key}"
    data = get_json(url)
    if data.get("success") == 1:
        return data.get("result", {})
    return {}

# ------------------ Laskenta ------------------
def calculate_probabilities(p1_key, p2_key):
    """Laskee voittotodennäköisyydet H2H:stä tai palauttaa 50-50"""
    h2h = fetch_h2h(p1_key, p2_key)
    games = h2h.get("H2H", [])
    if games:
        p1_wins = sum(1 for g in games if g.get("event_winner") == "First Player")
        p2_wins = sum(1 for g in games if g.get("event_winner") == "Second Player")
        total = p1_wins + p2_wins
        if total > 0:
            return p1_wins/total, p2_wins/total
    return 0.5, 0.5

# ------------------ UI ------------------
st.title("🎾 Tennis-ottelut")

# Sivupalkin valikko
today = date.today()
sel_date = st.sidebar.date_input("Päivämäärä", value=today)
match_type = st.sidebar.selectbox("Ottelutyyppi", ["Kaikki", "Live", "Tulevat"])
show_predictions = st.sidebar.checkbox("Näytä ennusteet", value=False)
simulate = st.sidebar.checkbox("Simuloi 1000 ottelua", value=False)

fixtures = fetch_fixtures(sel_date.isoformat())

# Suodata ottelutyypin mukaan
if match_type == "Live":
    fixtures = [m for m in fixtures if m.get("event_live") == "1"]
elif match_type == "Tulevat":
    fixtures = [m for m in fixtures if m.get("event_live") == "0" and not m.get("event_status")]

# Suodata turnauksen mukaan
tournaments = sorted({m.get("tournament_name", "-") for m in fixtures})
sel_tournaments = st.sidebar.multiselect("Turnaus", options=tournaments, default=tournaments)
if sel_tournaments:
    fixtures = [m for m in fixtures if m.get("tournament_name") in sel_tournaments]

# Näytä ottelut
rows = []
for match in fixtures:
    start_time = match.get("event_date") or ""
    try:
        # format "2025-08-21 14:00" → 14:00
        dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        start_str = dt.strftime("%H:%M")
    except:
        start_str = start_time

    p1 = match.get("event_first_player", "-")
    p2 = match.get("event_second_player", "-")
    tournament = match.get("tournament_name", "-")

    if show_predictions:
        prob1, prob2 = calculate_probabilities(match["first_player_key"], match["second_player_key"])
        prob1, prob2 = round(prob1*100, 1), round(prob2*100, 1)
    else:
        prob1, prob2 = "-", "-"

    rows.append([start_str, p1, p2, tournament, prob1, prob2])

df = pd.DataFrame(rows, columns=["Aika", "Pelaaja 1", "Pelaaja 2", "Turnaus", "P(1)%", "P(2)%"])
st.table(df)

# Simulaatio
if simulate and show_predictions and not df.empty:
    st.header("Simulaatiot (1000 ottelua)")
    sim_results = []
    for match in fixtures:
        prob1, prob2 = calculate_probabilities(match["first_player_key"], match["second_player_key"])
        wins1 = np.random.binomial(1000, prob1)
        wins2 = 1000 - wins1
        sim_results.append([f"{match['event_first_player']} vs {match['event_second_player']}", wins1, wins2])
    sim_df = pd.DataFrame(sim_results, columns=["Ottelu", "P1 voitot", "P2 voitot"])
    st.table(sim_df)
