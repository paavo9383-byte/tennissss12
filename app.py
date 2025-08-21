import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# ---------------------------
# API-funktiot
# ---------------------------
@st.cache_data
def fetch_fixtures(date_str):
    url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    res = requests.get(url)
    data = res.json()
    if data["success"] == 1:
        return data["result"]
    return []

@st.cache_data
def fetch_player(player_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_players&APIkey={API_KEY}&player_key={player_key}"
    res = requests.get(url)
    data = res.json()
    if data["success"] == 1 and data["result"]:
        return data["result"][0]
    return None

@st.cache_data
def fetch_h2h(p1_key, p2_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_H2H&APIkey={API_KEY}&first_player_key={p1_key}&second_player_key={p2_key}"
    res = requests.get(url)
    data = res.json()
    if data["success"] == 1:
        return data["result"]
    return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}

@st.cache_data
def fetch_odds(match_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_odds&APIkey={API_KEY}&match_key={match_key}"
    res = requests.get(url)
    data = res.json()
    if data["success"] == 1 and "result" in data:
        return data["result"].get(str(match_key), {})
    return {}

# ---------------------------
# Analyysi
# ---------------------------
def calculate_probabilities(p1_key, p2_key):
    """Mallin perusprobabiliteetit H2H + stats"""
    h2h = fetch_h2h(p1_key, p2_key)
    games = h2h.get("H2H", [])
    if games:
        p1_wins = sum(1 for g in games if g.get("event_winner") == "First Player")
        p2_wins = sum(1 for g in games if g.get("event_winner") == "Second Player")
        total = p1_wins + p2_wins
        if total > 0:
            return p1_wins/total, p2_wins/total

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
    return 0.5, 0.5

def monte_carlo_simulation(p1_prob, p2_prob, sims=10000):
    wins1 = np.random.binomial(sims, p1_prob)
    wins2 = sims - wins1
    return wins1/sims, wins2/sims

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎾 Tennis-ennusteet ja Value Bets")

# Suodattimet
today = date.today()
sel_date = st.sidebar.date_input("Päivämäärä", value=today)
match_type = st.sidebar.selectbox("Ottelutyyppi", ["Kaikki", "Live", "Tulevat"])

fixtures = fetch_fixtures(sel_date.isoformat())

if match_type == "Live":
    fixtures = [m for m in fixtures if m.get("event_live") == "1"]
elif match_type == "Tulevat":
    fixtures = [m for m in fixtures if m.get("event_live") == "0" and not m.get("event_status")]

# Turnausvalikko
tournaments = sorted(set(m.get("tournament_name") for m in fixtures))
sel_tournament = st.sidebar.selectbox("Turnaus", ["Kaikki"] + tournaments)
if sel_tournament != "Kaikki":
    fixtures = [m for m in fixtures if m.get("tournament_name") == sel_tournament]

# ---------------------------
# Mallin analyysi
# ---------------------------
st.header("Mallin analyysi (tilastot + H2H)")

rows = []
for match in fixtures:
    p1, p2 = calculate_probabilities(match["first_player_key"], match["second_player_key"])
    odds_data = fetch_odds(match["event_key"])
    home_odds, away_odds = None, None
    if "Home/Away" in odds_data:
        home_vals = odds_data["Home/Away"].get("Home", {})
        away_vals = odds_data["Home/Away"].get("Away", {})
        if home_vals: home_odds = max(float(v) for v in home_vals.values() if v)
        if away_vals: away_odds = max(float(v) for v in away_vals.values() if v)

    implied1 = 1/home_odds if home_odds else None
    implied2 = 1/away_odds if away_odds else None

    val1 = (p1 - implied1) if (p1 and implied1) else None
    val2 = (p2 - implied2) if (p2 and implied2) else None

    rows.append({
        "Ottelu": f"{match['event_first_player']} vs {match['event_second_player']}",
        "Turnaus": match["tournament_name"],
        "Aloitusaika": match.get("event_date"),
        "Mallin P1%": round(p1*100,1),
        "Mallin P2%": round(p2*100,1),
        "Kerroin P1": home_odds,
        "Kerroin P2": away_odds,
        "Value P1": val1,
        "Value P2": val2,
        "p1_prob_model": p1,
        "p2_prob_model": p2
    })

df = pd.DataFrame(rows)
st.dataframe(df[["Ottelu","Turnaus","Aloitusaika","Mallin P1%","Mallin P2%","Kerroin P1","Kerroin P2","Value P1","Value P2"]])

# Top 3 model value bets
st.subheader("🏆 Top 3 Model Value Bets")
all_values = []
for _, row in df.iterrows():
    if row["Value P1"] is not None:
        all_values.append((row["Ottelu"], "P1", row["Value P1"]))
    if row["Value P2"] is not None:
        all_values.append((row["Ottelu"], "P2", row["Value P2"]))
top3_model = sorted(all_values, key=lambda x: x[2], reverse=True)[:3]
st.table(pd.DataFrame(top3_model, columns=["Ottelu","Pelaaja","Value"]))

# ---------------------------
# Simulaatio
# ---------------------------
st.sidebar.subheader("Simulaatio")
num_sims = st.sidebar.slider("Simulaatioiden määrä", min_value=1000, max_value=20000, step=1000, value=10000)

if st.button("Simuloi"):
    st.header(f"Monte Carlo -simulaatiot ({num_sims} ajoa)")
    sim_rows = []
    sim_values = []
    for _, row in df.iterrows():
        sim_p1, sim_p2 = monte_carlo_simulation(row["p1_prob_model"], row["p2_prob_model"], sims=num_sims)
        odds1, odds2 = row["Kerroin P1"], row["Kerroin P2"]
        val1 = (sim_p1 - (1/odds1)) if odds1 else None
        val2 = (sim_p2 - (1/odds2)) if odds2 else None

        sim_rows.append({
            "Ottelu": row["Ottelu"],
            "Sim P1%": round(sim_p1*100,1),
            "Sim P2%": round(sim_p2*100,1),
            "Kerroin P1": odds1,
            "Kerroin P2": odds2,
            "Value P1": val1,
            "Value P2": val2
        })

        if val1 is not None: sim_values.append((row["Ottelu"], "P1", val1))
        if val2 is not None: sim_values.append((row["Ottelu"], "P2", val2))

    sim_df = pd.DataFrame(sim_rows)
    st.dataframe(sim_df)

    st.subheader("🏆 Top 3 Simulation Value Bets")
    top3_sim = sorted(sim_values, key=lambda x: x[2], reverse=True)[:3]
    st.table(pd.DataFrame(top3_sim, columns=["Ottelu","Pelaaja","Value"]))
