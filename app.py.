import requests
import streamlit as st
import numpy as np

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
BASE_URL = "https://api-tennis.com/tennis"

st.title("🎾 Ammattilainen Tennis Betting Model")

# 1) Hae käynnissä olevat ottelut API:sta
@st.cache_data
def get_matches():
    url = f"{BASE_URL}/?method=get_events&APIkey={API_KEY}"
    r = requests.get(url)
    return r.json()

matches = get_matches()

match_options = {f"{m['home']} vs {m['away']}": m for m in matches.get("result", [])}
selected_match = st.selectbox("Valitse ottelu", list(match_options.keys()))

if selected_match:
    match = match_options[selected_match]
    home, away = match["home"], match["away"]

    st.subheader(f"{home} vs {away}")

    # 2) Hae pelaajatilastot
    def get_player_stats(player_id):
        url = f"{BASE_URL}/?method=get_players&player_id={player_id}&APIkey={API_KEY}"
        r = requests.get(url)
        return r.json()

    home_stats = get_player_stats(match["homeID"])
    away_stats = get_player_stats(match["awayID"])

    # Käytetään syöttöprosentteja mallin pohjana
    try:
        a_serve = float(home_stats["result"][0].get("firstServeWon", 65))
        b_serve = float(away_stats["result"][0].get("firstServeWon", 65))
    except:
        a_serve, b_serve = 65, 65

    st.write(f"{home} syöttöpisteiden voitto-%: {a_serve}")
    st.write(f"{away} syöttöpisteiden voitto-%: {b_serve}")

    # 3) Markov-chain simulaatio (10000 ottelua)
    def simulate_match(a_serve, b_serve, n_sim=10000):
        a_wins = 0
        for _ in range(n_sim):
            a_sets, b_sets = 0, 0
            while a_sets < 2 and b_sets < 2:
                a_games, b_games = 0, 0
                while a_games < 6 and b_games < 6:
                    if np.random.rand() < a_serve/(a_serve+b_serve):
                        a_games += 1
                    else:
                        b_games += 1
                if a_games > b_games:
                    a_sets += 1
                else:
                    b_sets += 1
            if a_sets > b_sets:
                a_wins += 1
        return a_wins/n_sim

    a_prob = simulate_match(a_serve, b_serve)
    b_prob = 1 - a_prob

    st.subheader("🔮 Ennusteet")
    st.write(f"{home} voittotodennäköisyys: **{a_prob:.2%}**")
    st.write(f"{away} voittotodennäköisyys: **{b_prob:.2%}**")

    # 4) Bookkerin kertoimet (manuaalisesti syötettävät)
    st.subheader("Vedonlyönti")
    a_odds = st.number_input(f"{home} kerroin", value=2.10, step=0.01)
    b_odds = st.number_input(f"{away} kerroin", value=1.75, step=0.01)

    a_value = a_prob * a_odds
    b_value = b_prob * b_odds

    st.write(f"{home}: odotusarvo = {a_value:.2f} (>{1} = value!)")
    st.write(f"{away}: odotusarvo = {b_value:.2f} (>{1} = value!)")
