import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# --- API kutsut ---
@st.cache
def fetch_fixtures(date_str):
    url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    res = requests.get(url); data = res.json()
    if data.get("success") == 1: 
        return data["result"]
    return []

@st.cache
def fetch_player(player_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_players&APIkey={API_KEY}&player_key={player_key}"
    res = requests.get(url); data = res.json()
    if data.get("success") == 1 and data.get("result"):
        return data["result"][0]
    return None

@st.cache
def fetch_h2h(p1_key, p2_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_H2H&APIkey={API_KEY}&first_player_key={p1_key}&second_player_key={p2_key}"
    res = requests.get(url); data = res.json()
    if data.get("success") == 1:
        return data["result"]
    return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}

@st.cache
def fetch_odds(match_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_odds&APIkey={API_KEY}&match_key={match_key}"
    res = requests.get(url); data = res.json()
    if data.get("success") == 1:
        return data.get("result", {}).get(str(match_key), {})
    return {}

# --- Laskenta ---
def run_simulation(player1, player2, n=1000):
    """Simuloi ottelut pelaajastatistiikoiden pohjalta"""
    def get_ratio(player):
        if not player: return 0.5
        stats = player.get("stats") or []
        if stats:
            latest = sorted(stats, key=lambda x: x.get("season", ""), reverse=True)[0]
            w = int(latest.get("matches_won") or 0)
            l = int(latest.get("matches_lost") or 0)
            if w + l > 0:
                return w / (w + l)
        return 0.5
    
    r1 = get_ratio(player1)
    r2 = get_ratio(player2)
    if r1 + r2 == 0:
        return n//2, n//2
    
    p1_prob = r1/(r1+r2)
    wins1 = np.random.binomial(n, p1_prob)
    wins2 = n - wins1
    return wins1, wins2

# --- Käyttöliittymä ---
st.title("🎾 Tennis-ennusteet ja kertoimet")

today = date.today()
sel_date = st.sidebar.date_input("Päivämäärä", value=today)
match_type = st.sidebar.selectbox("Ottelutyyppi", ["Kaikki", "Live", "Tulevat"])

fixtures = fetch_fixtures(sel_date.isoformat())

if match_type == "Live":
    fixtures = [m for m in fixtures if m.get("event_live") == "1"]
elif match_type == "Tulevat":
    fixtures = [m for m in fixtures if m.get("event_live") == "0" and not m.get("event_status")]

tournaments = sorted(set(m.get("tournament_name") for m in fixtures))
if tournaments:
    sel_tournaments = st.sidebar.multiselect("Turnaus", options=tournaments, default=tournaments)
    fixtures = [m for m in fixtures if m.get("tournament_name") in sel_tournaments]
else:
    st.sidebar.info("Ei turnauksia valitulla päivällä.")

# --- Näytä ottelut ---
st.header("Ottelut ja kertoimet")

for match in fixtures:
    cols = st.columns([2,2,2,1,1])
    cols[0].write(match["event_first_player"])
    cols[1].write("vs")
    cols[2].write(match["event_second_player"])
    cols[3].write(match["tournament_name"])
    
    odds_data = fetch_odds(match["event_key"])
    home_odds, away_odds = "-", "-"
    if "Home/Away" in odds_data:
        home = odds_data["Home/Away"].get("Home", {})
        away = odds_data["Home/Away"].get("Away", {})
        if home: home_odds = max(float(v) for v in home.values() if v)
        if away: away_odds = max(float(v) for v in away.values() if v)
    cols[4].write(f"{home_odds}" if home_odds != "-" else "-")
    cols[4].write(f"{away_odds}" if away_odds != "-" else "-")

    # Expander: tarkempi analyysi vain jos käyttäjä avaa
    with st.expander(f"Analyysi: {match['event_first_player']} vs {match['event_second_player']}"):
        player1 = fetch_player(match["first_player_key"])
        player2 = fetch_player(match["second_player_key"])

        if player1 and player2:
            st.write(f"**{player1['player_name']}** - ranking {player1.get('player_ranking', 'N/A')}")
            st.write(f"**{player2['player_name']}** - ranking {player2.get('player_ranking', 'N/A')}")

            n_sims = st.slider("Simulaatioiden määrä", 500, 5000, 1000, 500, key=match["event_key"])
            wins1, wins2 = run_simulation(player1, player2, n_sims)

            st.metric(f"{match['event_first_player']}", f"{wins1/n_sims:.1%}")
            st.metric(f"{match['event_second_player']}", f"{wins2/n_sims:.1%}")
        else:
            st.warning("Ei riittävästi pelaajadataa analyysiin.")
