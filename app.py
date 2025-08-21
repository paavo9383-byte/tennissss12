import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# 1. Välimuistitetut funktiot TennisAPI-kutsuille
@st.cache_data
def fetch_fixtures(date_str):
    url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    res = requests.get(url); data = res.json()
    if data["success"] == 1: return data["result"]
    else: return []

@st.cache_data
def fetch_player(player_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_players&APIkey={API_KEY}&player_key={player_key}"
    res = requests.get(url); data = res.json()
    if data["success"] == 1 and data["result"]:
        return data["result"][0]
    else:
        return None

@st.cache_data
def fetch_h2h(p1_key, p2_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_H2H&APIkey={API_KEY}&first_player_key={p1_key}&second_player_key={p2_key}"
    res = requests.get(url); data = res.json()
    if data["success"] == 1:
        return data["result"]
    else:
        return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}

@st.cache_data
def fetch_odds(match_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_odds&APIkey={API_KEY}&match_key={match_key}"
    res = requests.get(url); data = res.json()
    if data["success"] == 1:
        return data["result"].get(str(match_key), {})
    else:
        return {}

def calculate_probabilities(p1_key, p2_key):
    # Yritä H2H-pelejä
    h2h = fetch_h2h(p1_key, p2_key)
    games = h2h.get("H2H", [])
    if games:
        p1_wins = sum(1 for g in games if g.get("event_winner") == "First Player")
        p2_wins = sum(1 for g in games if g.get("event_winner") == "Second Player")
        total = p1_wins + p2_wins
        if total > 0:
            return p1_wins/total, p2_wins/total
    # Palautetaan kausitilastoihin perustuvat arvot
    player1 = fetch_player(p1_key)
    player2 = fetch_player(p2_key)
    def get_ratio(player):
        stats = player.get("stats") if player else None
        if stats:
            stats_sorted = sorted(stats, key=lambda x: x.get("season", ""), reverse=True)
            latest = stats_sorted[0]
            wins = int(latest.get("matches_won") or 0)
            losses = int(latest.get("matches_lost") or 0)
            if wins + losses > 0:
                return wins / (wins + losses)
        return 0.5
    r1 = get_ratio(player1)
    r2 = get_ratio(player2)
    if r1 + r2 > 0:
        return r1/(r1+r2), r2/(r1+r2)
    else:
        return 0.5, 0.5

# Käyttöliittymä
st.title("Tennis-ennusteet ja kertoimet")

# Suodattimet
today = date.today()
sel_date = st.sidebar.date_input("Päivämäärä", value=today)
match_type = st.sidebar.selectbox("Ottelutyyppi", ["Kaikki", "Live", "Tulevat"])

fixtures = fetch_fixtures(sel_date.isoformat())

if match_type == "Live":
    fixtures = [m for m in fixtures if m.get("event_live") == "1"]
elif match_type == "Tulevat":
    fixtures = [m for m in fixtures if m.get("event_live") == "0" and not m.get("event_status")]

tournaments = sorted(set(m.get("tournament_name") for m in fixtures))
sel_tournaments = st.sidebar.multiselect("Turnaus", options=tournaments, default=tournaments)
if sel_tournaments:
    fixtures = [m for m in fixtures if m.get("tournament_name") in sel_tournaments]

# Otteluiden ja kertoimien näyttö
st.header("Ottelut ja kertoimet")
header_cols = st.columns([1,2,1,2,2,1,1])
headers = ["", "Pelaaja 1", "", "Pelaaja 2", "Turnaus", "Kerroin 1", "Kerroin 2"]
for i, head in enumerate(headers):
    header_cols[i].write(f"**{head}**")

for match in fixtures:
    # Pelaajien kuvat
    player1 = fetch_player(match["first_player_key"])
    player2 = fetch_player(match["second_player_key"])
    img1 = player1["player_logo"] if player1 else None
    img2 = player2["player_logo"] if player2 else None
    
    # Todennäköisyydet ja kertoimet
    prob1, prob2 = calculate_probabilities(match["first_player_key"], match["second_player_key"])
    odds_data = fetch_odds(match["event_key"])
    home_odds = []
    away_odds = []
    if "Home/Away" in odds_data:
        home_vals = odds_data["Home/Away"].get("Home", {})
        away_vals = odds_data["Home/Away"].get("Away", {})
        home_odds = [float(v) for v in home_vals.values() if v]
        away_odds = [float(v) for v in away_vals.values() if v]
    max_home = max(home_odds) if home_odds else None
    max_away = max(away_odds) if away_odds else None

    cols = st.columns([1,2,1,2,2,1,1])
    cols[0].image(img1, width=75)
    cols[1].write(match["event_first_player"])
    cols[2].image(img2, width=75)
    cols[3].write(match["event_second_player"])
    cols[4].write(match["tournament_name"])
    cols[5].write(f"{max_home:.2f}" if max_home else "-")
    cols[6].write(f"{max_away:.2f}" if max_away else "-")

# Simulointipainike
if st.button("Simuloi 1000 ottelua"):
    st.header("Simulaatiotulokset (1000 ottelua)")
    sim_data = []
    for match in fixtures:
        prob1, prob2 = calculate_probabilities(match["first_player_key"], match["second_player_key"])
        wins1 = np.random.binomial(1000, prob1)
        wins2 = 1000 - wins1
        sim_data.append({
            "Ottelu": f"{match['event_first_player']} vs {match['event_second_player']}",
            "Pelaaja 1 voitot": wins1,
            "Pelaaja 2 voitot": wins2
        })
    sim_df = pd.DataFrame(sim_data)
    st.table(sim_df)
