import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date
from numpy.random import beta

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# =========================
# API-funktiot
# =========================
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
        return data.get("result", {}).get(str(match_key), {})
    else:
        return {}

# =========================
# Apu-funktiot
# =========================
def calculate_probabilities(p1_key, p2_key):
    h2h = fetch_h2h(p1_key, p2_key)
    games = h2h.get("H2H", [])
    if games:
        p1_wins = sum(1 for g in games if g.get("event_winner") == "First Player")
        p2_wins = sum(1 for g in games if g.get("event_winner") == "Second Player")
        total = p1_wins + p2_wins
        if total > 0:
            return p1_wins/total, p2_wins/total
    # fallback stats
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

def get_beta_params(player):
    stats = player.get("stats")
    if stats:
        latest = sorted(stats, key=lambda x: x.get("season", ""), reverse=True)[0]
        wins = int(latest.get("matches_won") or 0)
        losses = int(latest.get("matches_lost") or 0)
        return wins+1, losses+1
    return None, None

# =========================
# UI
# =========================
st.title("🎾 Tennis-ennusteet ja value bets")

# Päivämäärä ja suodattimet
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

# =========================
# Ottelulistaus
# =========================
st.header("Ottelut ja kertoimet")
match_rows = []
for match in fixtures:
    odds_data = fetch_odds(match["event_key"])
    home_odds, away_odds = [], []
    if "Home/Away" in odds_data:
        home_vals = odds_data["Home/Away"].get("Home", {})
        away_vals = odds_data["Home/Away"].get("Away", {})
        home_odds = [float(v) for v in home_vals.values() if v]
        away_odds = [float(v) for v in away_vals.values() if v]
    max_home = max(home_odds) if home_odds else None
    max_away = max(away_odds) if away_odds else None

    p1_prob, p2_prob = calculate_probabilities(match["first_player_key"], match["second_player_key"])

    match_rows.append({
        "Aika": match.get("event_date"),
        "Ottelu": f"{match['event_first_player']} vs {match['event_second_player']}",
        "Turnaus": match["tournament_name"],
        "Perus Prob P1": f"{p1_prob:.1%}",
        "Perus Prob P2": f"{p2_prob:.1%}",
        "Kerroin 1": f"{max_home:.2f}" if max_home else "-",
        "Kerroin 2": f"{max_away:.2f}" if max_away else "-"
    })

matches_df = pd.DataFrame(match_rows)
st.table(matches_df)

# =========================
# Simulaatiot + Value Bets
# =========================
n_sims = st.sidebar.slider("Simulaatioiden määrä", 1000, 20000, 10000, step=1000)

if st.button("Simuloi päivän ottelut"):
    st.header(f"Simulaatiotulokset ({n_sims} simulaatiota / match)")

    sim_data = []
    value_rows = []

    for match in fixtures:
        player1 = fetch_player(match["first_player_key"])
        player2 = fetch_player(match["second_player_key"])
        if not player1 or not player2:
            continue

        a1, b1 = get_beta_params(player1)
        a2, b2 = get_beta_params(player2)
        if a1 is None or a2 is None:
            continue

        wins1 = 0
        wins2 = 0

        for _ in range(n_sims):
            p1 = beta(a1, b1)
            p2 = beta(a2, b2)
            total = p1 + p2
            p1 /= total
            p2 /= total
            winner = np.random.choice([1, 2], p=[p1, p2])
            if winner == 1:
                wins1 += 1
            else:
                wins2 += 1

        prob1 = wins1 / n_sims
        prob2 = wins2 / n_sims

        odds_data = fetch_odds(match["event_key"])
        max_home = max([float(v) for v in odds_data.get("Home/Away", {}).get("Home", {}).values() if v] or [0])
        max_away = max([float(v) for v in odds_data.get("Home/Away", {}).get("Away", {}).values() if v] or [0])
        max_home = max_home if max_home > 0 else None
        max_away = max_away if max_away > 0 else None

        value1 = (prob1 * max_home - 1) if max_home else None
        value2 = (prob2 * max_away - 1) if max_away else None

        row = {
            "Ottelu": f"{match['event_first_player']} vs {match['event_second_player']}",
            "Sim Prob P1": f"{prob1:.1%}",
            "Sim Prob P2": f"{prob2:.1%}",
            "Kerroin 1": f"{max_home:.2f}" if max_home else "-",
            "Kerroin 2": f"{max_away:.2f}" if max_away else "-",
            "Value P1": value1,
            "Value P2": value2
        }
        sim_data.append(row)

        if value1 is not None:
            value_rows.append((row["Ottelu"], value1, "P1"))
        if value2 is not None:
            value_rows.append((row["Ottelu"], value2, "P2"))

    sim_df = pd.DataFrame(sim_data)
    sim_df["Value P1"] = sim_df["Value P1"].apply(lambda x: f"{x:.2f}" if x is not None else "-")
    sim_df["Value P2"] = sim_df["Value P2"].apply(lambda x: f"{x:.2f}" if x is not None else "-")
    st.table(sim_df)

    # Top 10 Value Bets
    top_value = sorted(value_rows, key=lambda x: x[1], reverse=True)[:10]
    st.subheader("🔥 Top 10 Value Bets (vain datalliset ottelut)")
    for match, val, side in top_value:
        st.write(f"{match} → {side} (Value: {val:.2f})")
