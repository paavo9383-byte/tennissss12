# Otteluiden listaus
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

    # Expander: näytetään simulaatiot ja pelaajadata vain kun käyttäjä avaa
    with st.expander(f"Analyysi: {match['event_first_player']} vs {match['event_second_player']}"):
        player1 = fetch_player(match["first_player_key"])
        player2 = fetch_player(match["second_player_key"])

        if player1 and player2:
            st.write(f"**{player1['player_name']}** - ranking {player1.get('player_ranking', 'N/A')}")
            st.write(f"**{player2['player_name']}** - ranking {player2.get('player_ranking', 'N/A')}")

            # Simulaatio (käyttäjän määrittämällä määrällä)
            n_sims = st.slider("Simulaatioiden määrä", 500, 5000, 1000, 500, key=match["event_key"])
            wins1, wins2 = run_simulation(player1, player2, n_sims)

            st.metric(f"{match['event_first_player']}", f"{wins1/n_sims:.1%}")
            st.metric(f"{match['event_second_player']}", f"{wins2/n_sims:.1%}")
        else:
            st.warning("Ei riittävästi pelaajadataa analyysiin.")
