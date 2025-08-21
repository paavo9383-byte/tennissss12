import requests
import streamlit as st
import numpy as np

# 🔑 Oma API-avain
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# 🔗 Oikea endpoint
BASE_URL = "https://api.api-tennis.com/tennis/"

st.title("🎾 Ammattilainen Tennis Betting Model")

# 1️⃣ Hae ottelut turvallisesti
@st.cache_data
def get_matches():
    url = f"{BASE_URL}?method=get_events&APIkey={API_KEY}"
    try:
        r = requests.get(url)
        st.write("API status code:", r.status_code)
        st.write("API response preview:", r.text[:500])  # Näyttää max 500 merkkiä

        if r.status_code != 200:
            st.error(f"API virhe: {r.status_code}. Tarkista API-avaimesi tai subscription.")
            return []

        # Yritetään parsia JSON
        try:
            data = r.json()
            return data.get("result", [])
        except Exception as e:
            st.error("API ei palauttanut JSONia. Tarkista API-avaimesi ja subscription.")
            return []
    except Exception as e:
        st.error(f"HTTP virhe: {e}")
        return []

matches = get_matches()

if not matches:
    st.warning("Ei otteludataa saatavilla.")
else:
    match_options = {f"{m.get('home','?')} vs {m.get('away','?')}": m for m in matches}
    selected_match = st.selectbox("Valitse ottelu", list(match_options.keys()))

    if selected_match:
        match = match_options[selected_match]
        home, away = match.get("home", "?"), match.get("away", "?")
        st.subheader(f"{home} vs {away}")

        # 2️⃣ Simuloidaan vain placeholder-statistiikalla, jos pelaajadataa ei ole
        a_serve = float(match.get("homeServeWon", 65)) if "homeServeWon" in match else 65
        b_serve = float(match.get("awayServeWon", 65)) if "awayServeWon" in match else 65

        st.write(f"{home} syöttöpisteiden voitto-%: {a_serve}")
        st.write(f"{away} syöttöpisteiden voitto-%: {b_serve}")

        # 3️⃣ Simulaatiomalli
        def simulate_match(a_serve, b_serve, n_sim=5000):
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

        # 4️⃣ Vedonlyöntikertoimet & odotusarvo
        st.subheader("Vedonlyönti")
        a_odds = st.number_input(f"{home} kerroin", value=2.10, step=0.01)
        b_odds = st.number_input(f"{away} kerroin", value=1.75, step=0.01)

        a_value = a_prob * a_odds
        b_value = b_prob * b_odds

        st.write(f"{home}: odotusarvo = {a_value:.2f} (>{1} = value!)")
        st.write(f"{away}: odotusarvo = {b_value:.2f} (>{1} = value!)")
