import streamlit as st
import requests
import datetime
import numpy as np
import pandas as pd

# Tennis API configuration
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
API_URL = "https://api.api-tennis.com/tennis/"

# Caching API calls to avoid repeated requests
@st.experimental_memo
def fetch_tournaments(api_key):
    params = {"method": "get_tournaments", "APIkey": api_key}
    try:
        response = requests.get(API_URL, params=params)
        data = response.json()
        if data.get("success"):
            return data.get("result", [])
        else:
            return []
    except Exception:
        return []

@st.experimental_memo
def fetch_fixtures(api_key, date_start, date_stop):
    params = {
        "method": "get_fixtures",
        "APIkey": api_key,
        "date_start": date_start,
        "date_stop": date_stop
    }
    response = requests.get(API_URL, params=params)
    data = response.json()
    if data.get("success"):
        return data.get("result", [])
    else:
        return []

@st.experimental_memo
def fetch_odds(api_key, date_start, date_stop):
    params = {
        "method": "get_odds",
        "APIkey": api_key,
        "date_start": date_start,
        "date_stop": date_stop
    }
    response = requests.get(API_URL, params=params)
    data = response.json()
    if data.get("success"):
        return data.get("result", {})
    else:
        return {}

@st.experimental_memo
def fetch_player(api_key, player_key):
    params = {"method": "get_players", "APIkey": api_key, "player_key": player_key}
    response = requests.get(API_URL, params=params)
    data = response.json()
    if data.get("success"):
        return data.get("result", [{}])[0]
    else:
        return {}

@st.experimental_memo
def fetch_h2h(api_key, key1, key2):
    params = {"method": "get_H2H", "APIkey": api_key,
              "first_player_key": key1, "second_player_key": key2}
    response = requests.get(API_URL, params=params)
    data = response.json()
    if data.get("success"):
        return data.get("result", {})
    else:
        return {}

# App title
st.title("Tennis Betting Analytics")

# Sidebar filters
st.sidebar.header("Filters")
# Date selector
selected_date = st.sidebar.date_input(
    "Select Date", value=datetime.date.today())
date_str = selected_date.strftime("%Y-%m-%d")

# Match type selector
match_type = st.sidebar.selectbox("Type", ["All", "Upcoming", "Live"])

# Tournament selector
tournament_list = fetch_tournaments(API_KEY)
# Extract unique tournament names
tournament_names = sorted({t.get("tournament_name", "") for t in tournament_list})
tournament_selection = st.sidebar.selectbox("Tournament", ["All"] + tournament_names)

# Fetch fixtures for the selected date
fixtures = fetch_fixtures(API_KEY, date_str, date_str)

# Filter by type
filtered_matches = []
for match in fixtures:
    status = match.get("event_status", "")
    # Determine live vs upcoming by status
    is_live = status not in (None, "", "Finished")
    is_upcoming = (status == "" or status is None)
    if match_type == "Live" and not is_live:
        continue
    if match_type == "Upcoming" and not is_upcoming:
        continue
    filtered_matches.append(match)

# Filter by tournament if selected
if tournament_selection != "All":
    filtered_matches = [
        m for m in filtered_matches
        if m.get("tournament_name") == tournament_selection
    ]

if not filtered_matches:
    st.warning("No matches found for selected filters.")
else:
    # Fetch odds for the date
    odds_data = fetch_odds(API_KEY, date_str, date_str)
    # Collect player keys for stats retrieval
    player_keys = set()
    for match in filtered_matches:
        player_keys.add(match.get("first_player_key"))
        player_keys.add(match.get("second_player_key"))
    # Fetch player data
    players_data = {key: fetch_player(API_KEY, key) for key in player_keys}

    # Display each match
    for match in filtered_matches:
        p1 = match.get("event_first_player", "")
        p2 = match.get("event_second_player", "")
        k1 = match.get("first_player_key")
        k2 = match.get("second_player_key")
        st.markdown(f"### {p1} vs {p2}")

        # Player logos and names
        col1, col2 = st.columns([1,1])
        with col1:
            logo1 = match.get("event_first_player_logo")
            if logo1:
                st.image(logo1, width=80)
            st.write(p1)
        with col2:
            logo2 = match.get("event_second_player_logo")
            if logo2:
                st.image(logo2, width=80)
            st.write(p2)

        # Compute win probability using H2H and recent form
        h2h = fetch_h2h(API_KEY, k1, k2)
        # Head-to-head wins
        p1_h2h_wins = 0
        p2_h2h_wins = 0
        if h2h:
            # Count H2H match outcomes
            games = h2h.get("H2H", [])
            for g in games:
                winner = g.get("event_winner", "")
                if "First Player" in str(winner):
                    p1_h2h_wins += 1
                elif "Second Player" in str(winner):
                    p2_h2h_wins += 1
        total_h2h = p1_h2h_wins + p2_h2h_wins
        h2h_prob = p1_h2h_wins / total_h2h if total_h2h > 0 else 0.5

        # Recent form (win rate from last season singles stats)
        def win_rate(player_data):
            stats = player_data.get("stats", [])
            best = None
            for s in stats:
                if s.get("type") == "singles":
                    year = int(s.get("season", "0") or 0)
                    if best is None or year > best[0]:
                        best = (year, s)
            if not best:
                return 0.5
            s = best[1]
            try:
                wins = int(s.get("matches_won") or 0)
                losses = int(s.get("matches_lost") or 0)
                total = wins + losses
                if total == 0:
                    return 0.5
                return wins / total
            except:
                return 0.5

        p1_form = win_rate(players_data.get(k1, {}))
        p2_form = win_rate(players_data.get(k2, {}))
        if p1_form + p2_form == 0:
            p1_form = p2_form = 0.5

        # Combine H2H and form (equal weights)
        p1_prob = (h2h_prob + p1_form) / 2
        p2_prob = 1 - p1_prob

        st.write(f"**Estimated Win Probability:** {p1}: {p1_prob:.2%}, {p2}: {p2_prob:.2%}")

        # Odds table (Home/Away)
        match_key = match.get("event_key")
        odds_for_match = odds_data.get(match_key, {}).get("Home/Away", {})
        if odds_for_match:
            home_odds = odds_for_match.get("Home", {})
            away_odds = odds_for_match.get("Away", {})
            df_odds = pd.DataFrame({
                p1: home_odds,
                p2: away_odds
            }).rename_axis("Bookmaker")
            df_odds = df_odds.fillna("-")
            st.write("Odds from bookmakers:")
            st.table(df_odds)
            # Best odds for value calculation
            try:
                best_odds_p1 = max(float(x) for x in home_odds.values() if x)
            except ValueError:
                best_odds_p1 = None
            try:
                best_odds_p2 = max(float(x) for x in away_odds.values() if x)
            except ValueError:
                best_odds_p2 = None
        else:
            best_odds_p1 = best_odds_p2 = None
            st.write("Odds data not available for this match.")

        # Calculate and display value (probability * best odd)
        if best_odds_p1 and best_odds_p2:
            value1 = p1_prob * best_odds_p1
            value2 = p2_prob * best_odds_p2
            st.write(f"**Value (probability * best odd):** {p1}: {value1:.3f}, {p2}: {value2:.3f}")

        # Simulation button
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            if st.button("Simulate 1000 matches", key=f"sim_{match_key}"):
                sim_wins = np.random.binomial(1000, p1_prob)
                sim_p1 = sim_wins
                sim_p2 = 1000 - sim_wins
                st.session_state[f"sim_{match_key}"] = (sim_p1, sim_p2)
        with sim_col2:
            res = st.session_state.get(f"sim_{match_key}")
            if res:
                sim_p1, sim_p2 = res
                st.write(f"{p1} won {sim_p1} times, {p2} won {sim_p2} times in 1000 simulations.")
