import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# --- API-funktiot ---
@st.cache_data
def fetch_fixtures(date_str):
    url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    res = requests.get(url)
    data = res.json()
    if data.get("success") == 1:
        return data["result"]
    return []

@st.cache_data
def fetch_player(player_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_players&APIkey={API_KEY}&player_key={player_key}"
    res = requests.get(url)
    data = res.json()
    if data.get("success") == 1 and data.get("result"):
        return data["result"][0]
    return None

@st.cache_data
def fetch_h2h(p1_key, p2_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_H2H&APIkey={API_KEY}&first_player_key={p1_key}&second_player_key={p2_key}"
    res = requests.get(url)
    data = res.json()
    if data.get("success") == 1:
        return data["result"]
    return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}

@st.cache_data
def fetch_odds(match_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_odds&APIkey={API_KEY}&match_key={match_key}"
    res = requests.get(url)
    data = res.json()
    if data.get("success") == 1 and "result" in data:
        return data["result"].get(str(match_key), {})
    return {}

# --- Tilastofunktiot ---
def get_surface_winrate(player, surface="Hard"):
    stats = player.get("stats", [])
    if not stats:
        return 0.5
    for s in stats:
        if s.get("surface") == surface:
            w = int(s.get("matches_won") or 0)
            l = int(s.get("matches_lost") or 0)
            if w + l > 0:
                return w / (w + l)
    return 0.5

def get_hold_break_probs(player):
    stats = player.get("stats", [])
    if not stats:
        return 0.65, 0.25
    latest = sorted(stats, key=lambda x: x.get("season", ""), reverse=True)[0]
    hold = float(latest.get("service_games_won", 65)) / 100
    brk = float(latest.get("return_games_won", 25)) / 100
    return hold, brk

def simulate_match(p1, p2, surface="Hard", n_sim=3000):
    p1_hold, p1_break = get_hold_break_probs(p1)
    p2_hold, p2_break = get_hold_break_probs(p2)

    # Alustakorjaus
    p1_surface_wr = get_surface_winrate(p1, surface)
    p2_surface_wr = get_surface_winrate(p2, surface)
    adj = p1_surface_wr / (p1_surface_wr + p2_surface_wr + 1e-6)
    p1_hold = 0.5 * p1_hold + 0.5 * adj
    p2_hold = 0.5 * p2_hold + 0.5 * (1 - adj)

    p1_wins = 0
    for _ in range(n_sim):
        sets1, sets2 = 0, 0
        while sets1 < 2 and sets2 < 2:  # best of 3
            games1, games2 = 0, 0
            for _ in range(12):
                if np.random.rand() < p1_hold:
                    games1 += 1
                if np.random.rand() < p2_hold:
                    games2 += 1
            if games1 >= games2:
                sets1 += 1
            else:
                sets2 += 1
        if sets1 > sets2:
            p1_wins += 1
    return p1_wins / n_sim, 1 - (p1_wins / n_sim)

# --- UI ---
st.title("🎾 Tennis-ennusteet ja value bet -analyysi")

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

# --- Analyysi ---
results = []
for match in fixtures:
    player1 = fetch_player(match["first_player_key"])
    player2 = fetch_player(match["second_player_key"])
    if not player1 or not player2:
        continue

    odds_data = fetch_odds(match["event_key"])
    home_odds, away_odds = None, None
    if "Home/Away" in odds_data:
        home_vals = odds_data["Home/Away"].get("Home", {})
        away_vals = odds_data["Home/Away"].get("Away", {})
        if home_vals:
            home_odds = min(float(v) for v in home_vals.values() if v)
        if away_vals:
            away_odds = min(float(v) for v in away_vals.values() if v)

    model_p1, model_p2 = simulate_match(player1, player2, surface="Hard")

    implied_p1 = 1 / home_odds if home_odds else None
    implied_p2 = 1 / away_odds if away_odds else None

    value1 = model_p1 - implied_p1 if implied_p1 else None
    value2 = model_p2 - implied_p2 if implied_p2 else None

    results.append({
        "match": f"{match['event_first_player']} vs {match['event_second_player']}",
        "tournament": match["tournament_name"],
        "market_odds_p1": home_odds,
        "market_odds_p2": away_odds,
        "model_p1_prob": round(model_p1, 3),
        "model_p2_prob": round(model_p2, 3),
        "implied_p1": round(implied_p1, 3) if implied_p1 else None,
        "implied_p2": round(implied_p2, 3) if implied_p2 else None,
        "value_p1": round(value1, 3) if value1 else None,
        "value_p2": round(value2, 3) if value2 else None,
        "value_icon_p1": "💰" if value1 and value1 > 0 else "",
        "value_icon_p2": "💰" if value2 and value2 > 0 else ""
    })

df = pd.DataFrame(results)

# --- Value bets -valinta ---
show_value = st.sidebar.checkbox("Näytä parhaat value betit")

if show_value and not df.empty:
    df["best_value"] = df[["value_p1", "value_p2"]].max(axis=1)
    value_df = df[(df["value_p1"] > 0) | (df["value_p2"] > 0)].copy()
    value_df = value_df.sort_values("best_value", ascending=False)

    st.header("📊 Päivän parhaat value bets")
    st.dataframe(value_df[[
        "match", "tournament",
        "market_odds_p1", "market_odds_p2",
        "model_p1_prob", "model_p2_prob",
        "implied_p1", "implied_p2",
        "value_p1", "value_icon_p1",
        "value_p2", "value_icon_p2",
        "best_value"
    ]])
else:
    st.header("📊 Kaikki ottelut ja analyysi")
    st.dataframe(df)
