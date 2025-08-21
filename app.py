# import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# ==========================
# API FUNKTIOT
# ==========================
@st.cache_data
def fetch_fixtures(date_str):
    url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    res = requests.get(url); data = res.json()
    return data["result"] if data.get("success") == 1 else []

@st.cache_data
def fetch_player(player_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_players&APIkey={API_KEY}&player_key={player_key}"
    res = requests.get(url); data = res.json()
    if data.get("success") == 1 and data.get("result"):
        return data["result"][0]
    return None

@st.cache_data
def fetch_odds(match_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_odds&APIkey={API_KEY}&match_key={match_key}"
    res = requests.get(url); data = res.json()
    return data.get("result", {}).get(str(match_key), {}) if data.get("success") == 1 else {}

# ==========================
# ANALYYSI & SIMULAATIO
# ==========================
def get_player_strength(player):
    """Laskee pelaajan voimaluvun statsien perusteella."""
    if not player: return 0.5
    stats = player.get("stats")
    if not stats: return 0.5

    latest = sorted(stats, key=lambda x: x.get("season", ""), reverse=True)[0]
    wins = int(latest.get("matches_won") or 0)
    losses = int(latest.get("matches_lost") or 0)
    serve_points = float(latest.get("serve_points_won") or 50)
    return_points = float(latest.get("return_points_won") or 50)

    total = wins + losses
    winrate = wins / total if total > 0 else 0.5

    strength = 0.5 * winrate + 0.25 * (serve_points / 100) + 0.25 * (return_points / 100)
    return strength

def run_simulation(player1, player2, n=1000):
    s1, s2 = get_player_strength(player1), get_player_strength(player2)
    p1 = s1 / (s1 + s2)
    p2 = 1 - p1
    wins1 = np.random.binomial(n, p1)
    return wins1, n - wins1

def calculate_value(prob, odds):
    return (prob * odds) - 1 if odds else -1

# ==========================
# STREAMLIT UI
# ==========================
st.title("🎾 Tennis-ennusteet ja value bets")

today = date.today()
sel_date = st.sidebar.date_input("Päivämäärä", value=today)
fixtures = fetch_fixtures(sel_date.isoformat())

# Turnausfiltteri
tournaments = sorted(set(m.get("tournament_name") for m in fixtures))
sel_tournaments = st.sidebar.multiselect("Turnaus", tournaments, default=tournaments)
if sel_tournaments:
    fixtures = [m for m in fixtures if m.get("tournament_name") in sel_tournaments]

# ==========================
# PÄIVÄN OTTELUT
# ==========================
st.header("Päivän ottelut")
for match in fixtures:
    cols = st.columns([2,2,2,2])
    cols[0].write(match["event_first_player"])
    cols[1].write("vs")
    cols[2].write(match["event_second_player"])
    cols[3].write(match["tournament_name"])

    odds_data = fetch_odds(match["event_key"])
    home_odds, away_odds = None, None
    if "Home/Away" in odds_data:
        home = odds_data["Home/Away"].get("Home", {})
        away = odds_data["Home/Away"].get("Away", {})
        if home: home_odds = max(float(v) for v in home.values() if v)
        if away: away_odds = max(float(v) for v in away.values() if v)
    cols[0].write(f"Kerroin {home_odds:.2f}" if home_odds else "-")
    cols[2].write(f"Kerroin {away_odds:.2f}" if away_odds else "-")

    # Expander: analyysi
    with st.expander(f"Analyysi: {match['event_first_player']} vs {match['event_second_player']}"):
        player1 = fetch_player(match["first_player_key"])
        player2 = fetch_player(match["second_player_key"])
        if not player1 or not player2:
            st.warning("Ei tarpeeksi dataa analyysiin.")
            continue

        n_sims = st.slider("Simulaatioiden määrä", 500, 5000, 1000, 500, key=match["event_key"])
        wins1, wins2 = run_simulation(player1, player2, n_sims)

        prob1, prob2 = wins1/n_sims, wins2/n_sims
        st.metric(match["event_first_player"], f"{prob1:.1%}")
        st.metric(match["event_second_player"], f"{prob2:.1%}")

# ==========================
# VALUE BETS - TOP 10
# ==========================
if st.checkbox("Näytä päivän Top 10 value bets"):
    value_bets = []
    for match in fixtures:
        player1 = fetch_player(match["first_player_key"])
        player2 = fetch_player(match["second_player_key"])
        if not player1 or not player2:
            continue
        wins1, wins2 = run_simulation(player1, player2, 1000)
        prob1, prob2 = wins1/1000, wins2/1000

        odds_data = fetch_odds(match["event_key"])
        if "Home/Away" in odds_data:
            home = odds_data["Home/Away"].get("Home", {})
            away = odds_data["Home/Away"].get("Away", {})
            if home:
                max_home = max(float(v) for v in home.values() if v)
                value_bets.append({
                    "Ottelu": f"{match['event_first_player']} vs {match['event_second_player']}",
                    "Pelaaja": match["event_first_player"],
                    "Todennäköisyys": prob1,
                    "Kerroin": max_home,
                    "Value": calculate_value(prob1, max_home)
                })
            if away:
                max_away = max(float(v) for v in away.values() if v)
                value_bets.append({
                    "Ottelu": f"{match['event_first_player']} vs {match['event_second_player']}",
                    "Pelaaja": match["event_second_player"],
                    "Todennäköisyys": prob2,
                    "Kerroin": max_away,
                    "Value": calculate_value(prob2, max_away)
                })

    df = pd.DataFrame(value_bets).sort_values("Value", ascending=False).head(10)
    st.subheader("🔥 Top 10 Value Bets")
    st.table(df[["Ottelu", "Pelaaja", "Todennäköisyys", "Kerroin", "Value"]])
