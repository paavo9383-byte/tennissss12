import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date
import altair as alt

# ====================
# API-avain
# ====================
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# ====================
# API-funktiot
# ====================
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
    if data.get("success") == 1:
        return data["result"].get(str(match_key), {})
    return {}

# ====================
# Malli: todennäköisyydet
# ====================
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
        if not player:
            return 0.5
        stats = player.get("stats")
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
        return r1 / (r1 + r2), r2 / (r1 + r2)
    return 0.5, 0.5

# ====================
# UI
# ====================
st.title("🎾 Tennis-ennusteet ja value bets")

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

# ====================
# Ottelut & kertoimet
# ====================
st.header("Ottelut ja kertoimet")
header_cols = st.columns([2, 2, 2, 2, 1, 1, 2, 2])
headers = ["Pelaaja 1", "Pelaaja 2", "Turnaus", "Aika", "Kerroin 1", "Kerroin 2", "Tod.näk. 1", "Tod.näk. 2"]
for i, head in enumerate(headers):
    header_cols[i].write(f"**{head}**")

match_values = []
for match in fixtures:
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

    prob1, prob2 = calculate_probabilities(match["first_player_key"], match["second_player_key"])

    cols = st.columns([2, 2, 2, 2, 1, 1, 2, 2])
    cols[0].write(match["event_first_player"])
    cols[1].write(match["event_second_player"])
    cols[2].write(match["tournament_name"])
    cols[3].write(match.get("event_date") or "-")
    cols[4].write(f"{max_home:.2f}" if max_home else "-")
    cols[5].write(f"{max_away:.2f}" if max_away else "-")
    cols[6].write(f"{prob1*100:.1f}%")
    cols[7].write(f"{prob2*100:.1f}%")

    if max_home and max_away:
        exp1 = prob1 * max_home
        exp2 = prob2 * max_away
        best_val = max(exp1, exp2)
        match_values.append({
            "match": f"{match['event_first_player']} vs {match['event_second_player']}",
            "player": match["event_first_player"] if exp1 > exp2 else match["event_second_player"],
            "value": best_val,
            "prob": prob1 if exp1 > exp2 else prob2,
            "odds": max_home if exp1 > exp2 else max_away
        })

# ====================
# Simulaatio
# ====================
if st.button("Simuloi ja piirrä"):
    st.subheader("Simulaatiotulokset (1000 ottelua)")

    sim_data = []
    for match in fixtures:
        prob1, prob2 = calculate_probabilities(match["first_player_key"], match["second_player_key"])
        wins1 = np.random.binomial(1000, prob1)
        wins2 = 1000 - wins1

        sim_data.append({
            "Ottelu": f"{match['event_first_player']} vs {match['event_second_player']}",
            "Pelaaja": match['event_first_player'],
            "Voitot": wins1
        })
        sim_data.append({
            "Ottelu": f"{match['event_first_player']} vs {match['event_second_player']}",
            "Pelaaja": match['event_second_player'],
            "Voitot": wins2
        })

    sim_df = pd.DataFrame(sim_data)
    st.dataframe(sim_df)

    chart = alt.Chart(sim_df).mark_bar().encode(
        x=alt.X("Ottelu:N", sort=None, title="Ottelu"),
        y=alt.Y("Voitot:Q", title="Voitot simulaatiossa"),
        color="Pelaaja:N",
        tooltip=["Ottelu", "Pelaaja", "Voitot"]
    ).properties(width=700, height=400)

    st.altair_chart(chart, use_container_width=True)

# ====================
# Paras value nyt
# ====================
if st.button("Etsi paras value nyt"):
    if match_values:
        best = max(match_values, key=lambda x: x["value"])
        st.success(
            f"Paras value juuri nyt: **{best['match']}** → "
            f"{best['player']} (tod.näk {best['prob']*100:.1f}%, kerroin {best['odds']:.2f}, EV {best['value']:.2f})"
        )
    else:
        st.warning("Ei löytynyt pelejä, joissa olisi kertoimia ja todennäköisyyksiä.")
