import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# --- API kutsut ---
@st.cache_data
def fetch_fixtures(date_str):
    url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    res = requests.get(url); data = res.json()
    return data["result"] if data.get("success") == 1 else []

@st.cache_data
def fetch_player(player_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_players&APIkey={API_KEY}&player_key={player_key}"
    res = requests.get(url); data = res.json()
    return data["result"][0] if data.get("success") == 1 and data.get("result") else None

@st.cache_data
def fetch_h2h(p1_key, p2_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_H2H&APIkey={API_KEY}&first_player_key={p1_key}&second_player_key={p2_key}"
    res = requests.get(url); data = res.json()
    return data.get("result") if data.get("success") == 1 else {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}

@st.cache_data
def fetch_odds(match_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_odds&APIkey={API_KEY}&match_key={match_key}"
    res = requests.get(url); data = res.json()
    return data.get("result", {}).get(str(match_key), {}) if data.get("success") == 1 else {}

# --- Todennäköisyysmalli ---
def calculate_probabilities(p1_key, p2_key):
    h2h = fetch_h2h(p1_key, p2_key)
    games = h2h.get("H2H", [])
    if games:
        p1_wins = sum(1 for g in games if g.get("event_winner") == "First Player")
        p2_wins = sum(1 for g in games if g.get("event_winner") == "Second Player")
        total = p1_wins + p2_wins
        if total > 0:
            return p1_wins / total, p2_wins / total

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

    r1, r2 = get_ratio(player1), get_ratio(player2)
    return (r1 / (r1 + r2), r2 / (r1 + r2)) if (r1 + r2) > 0 else (0.5, 0.5)

# --- UI ---
st.title("🎾 Tennis-ennusteet ja value betit")

today = date.today()
sel_date = st.sidebar.date_input("Päivämäärä", value=today)
match_type = st.sidebar.selectbox("Ottelutyyppi", ["Kaikki", "Live", "Tulevat"])
show_value = st.sidebar.checkbox("Näytä vain parhaat value bets")

fixtures = fetch_fixtures(sel_date.isoformat())
if match_type == "Live":
    fixtures = [m for m in fixtures if m.get("event_live") == "1"]
elif match_type == "Tulevat":
    fixtures = [m for m in fixtures if m.get("event_live") == "0" and not m.get("event_status")]

results = []
for match in fixtures:
    p1, p2 = match["event_first_player"], match["event_second_player"]
    prob1, prob2 = calculate_probabilities(match["first_player_key"], match["second_player_key"])
    odds_data = fetch_odds(match["event_key"])

    home_odds, away_odds = [], []
    if "Home/Away" in odds_data:
        home_odds = [float(v) for v in odds_data["Home/Away"].get("Home", {}).values() if v]
        away_odds = [float(v) for v in odds_data["Home/Away"].get("Away", {}).values() if v]

    max_home, max_away = (max(home_odds) if home_odds else None, max(away_odds) if away_odds else None)
    imp_p1, imp_p2 = (1 / max_home if max_home else None, 1 / max_away if max_away else None)

    value_p1 = (prob1 - imp_p1) if imp_p1 else 0
    value_p2 = (prob2 - imp_p2) if imp_p2 else 0

    results.append({
        "match": f"{p1} vs {p2}",
        "tournament": match["tournament_name"],
        "market_odds_p1": max_home,
        "market_odds_p2": max_away,
        "model_p1_prob": round(prob1, 3),
        "model_p2_prob": round(prob2, 3),
        "implied_p1": round(imp_p1, 3) if imp_p1 else None,
        "implied_p2": round(imp_p2, 3) if imp_p2 else None,
        "value_p1": value_p1,
        "value_p2": value_p2,
        "value_icon_p1": "💰" if value_p1 > 0 else "",
        "value_icon_p2": "💰" if value_p2 > 0 else "",
        "best_value": max(value_p1, value_p2)
    })

df = pd.DataFrame(results)

# --- TOP 3 aina näkyvissä ---
if not df.empty:
    top3 = df.sort_values("best_value", ascending=False).head(3)
    st.subheader("🔥 Päivän Top 3 Value Bettiä")
    cols = st.columns(3)
    for i, row in enumerate(top3.itertuples()):
        cols[i].metric(
            label=row.match,
            value=f"{row.best_value:.3f}",
            delta=f"Kertoimet: {row.market_odds_p1} / {row.market_odds_p2}"
        )

# --- Näytetään taulukko ---
if show_value and not df.empty:
    value_df = df[(df["value_p1"] > 0) | (df["value_p2"] > 0)].sort_values("best_value", ascending=False)
    st.header("📊 Päivän parhaat value bets")
    st.dataframe(value_df)
else:
    st.header("📊 Kaikki ottelut ja analyysi")
    st.dataframe(df)
