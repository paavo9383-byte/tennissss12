import streamlit as st
import pandas as pd
import numpy as np
import math
from app.client import leagues, fixtures, odds, fixtures_team_last
from app.utils import implied_from_decimal_odds, expected_value, kelly_fraction, normalize_two_probs

st.set_page_config(page_title="Football App", page_icon="⚽", layout="centered")

st.title("⚽ Football App — API-Football (RapidAPI)")

with st.sidebar:
    st.subheader("Asetukset")
    st.caption("Huom: RapidAPI:n ilmaisen tason rajat voivat tulla vastaan.")
    bankroll = st.number_input("Bankroll (€)", 0.0, 1_000_000.0, 1000.0, step=50.0)
    kelly_frac = st.slider("Kelly fraction", 0.0, 1.0, 0.5, 0.05)
    min_ev = st.number_input("Min EV per 1€", 0.0, 1.0, 0.02, step=0.01, format="%.2f")

st.markdown("Valitse **välilehti**:")

tab1, tab2, tab3, tab4 = st.tabs(["Fixtures", "Odds & EV", "Value Bets", "Predictions"])

LEAGUES = {
    "Premier League (ENG)": 39,
    "La Liga (ESP)": 140,
    "Serie A (ITA)": 135,
    "Bundesliga (GER)": 78,
    "Ligue 1 (FRA)": 61,
    "Veikkausliiga (FIN)": 290
}

with tab1:
    st.subheader("Fixtures")
    col1, col2, col3 = st.columns(3)
    with col1:
        lig_name = st.selectbox("Liiga", list(LEAGUES.keys()), index=0)
    with col2:
        season = st.number_input("Kausi", 2010, 2100, 2024, step=1)
    with col3:
        date = st.date_input("Päivä (valinnainen)")
    live = st.checkbox("Vain LIVE", value=False)
    lid = LEAGUES[lig_name]
    date_str = date.strftime("%Y-%m-%d") if date else None
    if st.button("Hae ottelut"):
        data = fixtures(date=date_str, league=lid, season=int(season), live=live)
        rows = []
        for f in data.get("response", []):
            fx = f.get("fixture",{})
            teams = f.get("teams",{})
            rows.append({
                "fixture_id": fx.get("id"),
                "date": fx.get("date"),
                "home": teams.get("home",{}).get("name"),
                "away": teams.get("away",{}).get("name"),
                "status": f.get("fixture",{}).get("status",{}).get("short")
            })
        df = pd.DataFrame(rows)
        if df.empty:
            st.info("Ei tuloksia valituilla ehdoilla.")
        else:
            st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Odds & EV")
    fixture_id = st.text_input("Syötä Fixture ID")
    st.caption("Vinkki: hae ID tabista *Fixtures*.")
    prob_side = st.selectbox("Arvioitava kohde", ["Home", "Away"])
    prob_input = st.number_input("Oma todennäköisyys (%)", 0.0, 100.0, 55.0, step=0.5)/100.0
    if st.button("Hae kertoimet ja laske"):
        if not fixture_id.strip().isdigit():
            st.error("Fixture ID pitää olla numero.")
        else:
            data = odds(int(fixture_id.strip()))
            rows = []
            for r in data.get("response", []):
                for bm in r.get("bookmakers", []):
                    for bet in bm.get("bets", []):
                        if str(bet.get("name","")).lower() in ("match winner","winner","1x2","1x2 full time","fulltime result"):
                            values = bet.get("values", [])
                            home = next((v for v in values if str(v.get("value","")).lower() in ("home","1")), None)
                            away = next((v for v in values if str(v.get("value","")).lower() in ("away","2")), None)
                            if not (home and away):
                                continue
                            try:
                                o1 = float(home.get("odd"))
                                o2 = float(away.get("odd"))
                            except Exception:
                                continue
                            p1_imp = implied_from_decimal_odds(o1)
                            p2_imp = implied_from_decimal_odds(o2)
                            p1, p2 = normalize_two_probs(p1_imp, p2_imp)

                            chosen_odds = o1 if prob_side=='Home' else o2
                            chosen_prob = prob_input
                            ev = expected_value(chosen_prob, chosen_odds)
                            k = kelly_fraction(chosen_prob, chosen_odds, fraction=1.0) * bankroll

                            rows.append({
                                "bookmaker": bm.get("name"),
                                "odds_home": o1, "odds_away": o2,
                                "imp_home": round(p1,4), "imp_away": round(p2,4),
                                f"EV_{prob_side}_per_1e": round(ev,4),
                                f"Kelly_{prob_side}_eur": round(k * kelly_frac, 2),
                            })
            if not rows:
                st.warning("Kertoimia ei löytynyt tälle fixturelle.")
            else:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab3:
    st.subheader("Value Bets")
    lig_name = st.selectbox("Liiga (value-bets)", list(LEAGUES.keys()), index=0, key="vb_league")
    season = st.number_input("Kausi (value-bets)", 2010, 2100, 2024, step=1, key="vb_season")
    if st.button("Etsi value-bets"):
        lid = LEAGUES[lig_name]
        data = fixtures(league=lid, season=int(season))
        results = []
        for f in data.get("response", []):
            fx_id = f.get("fixture",{}).get("id")
            if not fx_id: 
                continue
            o = odds(fx_id)
            for r in o.get("response", []):
                for bm in r.get("bookmakers", []):
                    for bet in bm.get("bets", []):
                        if str(bet.get("name","")).lower() in ("match winner","winner","1x2","1x2 full time","fulltime result"):
                            values = bet.get("values", [])
                            home = next((v for v in values if str(v.get("value","")).lower() in ("home","1")), None)
                            away = next((v for v in values if str(v.get("value","")).lower() in ("away","2")), None)
                            if not (home and away):
                                continue
                            try:
                                o1 = float(home.get("odd"))
                                o2 = float(away.get("odd"))
                            except Exception:
                                continue
                            p1_imp = implied_from_decimal_odds(o1)
                            p2_imp = implied_from_decimal_odds(o2)
                            p1, p2 = normalize_two_probs(p1_imp, p2_imp)

                            ev1 = expected_value(p1, o1)
                            ev2 = expected_value(p2, o2)
                            if ev1 >= min_ev or ev2 >= min_ev:
                                teams = f.get("teams", {})
                                results.append({
                                    "fixture_id": fx_id,
                                    "match": f"{teams.get('home',{}).get('name','?')} vs {teams.get('away',{}).get('name','?')}",
                                    "bookmaker": bm.get("name"),
                                    "odds_home": o1,
                                    "odds_away": o2,
                                    "imp_home": round(p1,4),
                                    "imp_away": round(p2,4),
                                    "EV_home_per_1e": round(ev1,4),
                                    "EV_away_per_1e": round(ev2,4),
                                    "Kelly_home_eur": round(kelly_fraction(p1, o1, fraction=kelly_frac)*bankroll,2),
                                    "Kelly_away_eur": round(kelly_fraction(p2, o2, fraction=kelly_frac)*bankroll,2),
                                })
        df = pd.DataFrame(results)
        if df.empty:
            st.info("Ei value-betsiä annetulla min EV -rajalla — laske rajaa tai käytä omaa malliprobaa.")
        else:
            st.dataframe(df, use_container_width=True)

with tab4:
    st.subheader("Predictions (Poisson Model)")
    st.caption("Arvioi 1-X-2 ja Under/Over 2.5 Poisson-mallilla joukkueiden viime otteluiden perusteella.")
    fixture_id = st.text_input("Fixture ID (predictions)", "")
    max_goals = st.slider("Maksimi maalimäärä matriisissa", 4, 10, 6)

    if st.button("Laske Poisson-malli"):
        if not fixture_id.strip().isdigit():
            st.error("Anna kelvollinen fixture ID (numero).")
        else:
            fid = int(fixture_id.strip())
            fx = fixtures(fixture_id=fid)
            resp = fx.get("response", [])
            if not resp:
                st.warning("Ottelua ei löytynyt tällä ID:llä.")
            else:
                match = resp[0]
                home = match["teams"]["home"]["name"]
                away = match["teams"]["away"]["name"]
                league_id = match.get("league", {}).get("id")
                season = match.get("league", {}).get("season")
                h_id = match["teams"]["home"]["id"]
                a_id = match["teams"]["away"]["id"]

                st.write(f"**{home} vs {away}** — League {league_id}, Season {season}")

                # Nouda viime pelit
                h_last = fixtures_team_last(h_id, season=season, last=5, league=league_id)
                a_last = fixtures_team_last(a_id, season=season, last=5, league=league_id)

                def avg_scored_conceded(last_data, team_id):
                    scored, conceded = [], []
                    for g in last_data.get("response", []):
                        h = g["teams"]["home"]["id"]
                        a = g["teams"]["away"]["id"]
                        gh = g.get("goals",{}).get("home")
                        ga = g.get("goals",{}).get("away")
                        if gh is None or ga is None:
                            continue
                        if h == team_id:
                            scored.append(gh)
                            conceded.append(ga)
                        elif a == team_id:
                            scored.append(ga)
                            conceded.append(gh)
                    if not scored:
                        return 1.0, 1.0
                    return float(np.mean(scored)), float(np.mean(conceded))

                h_sc, h_con = avg_scored_conceded(h_last, h_id)
                a_sc, a_con = avg_scored_conceded(a_last, a_id)

                lam_home = max(0.05, h_sc * a_con)
                lam_away = max(0.05, a_sc * h_con)

                prob_matrix = np.zeros((max_goals+1, max_goals+1))
                for i in range(max_goals+1):
                    for j in range(max_goals+1):
                        p_i = math.exp(-lam_home) * lam_home**i / math.factorial(i)
                        p_j = math.exp(-lam_away) * lam_away**j / math.factorial(j)
                        prob_matrix[i, j] = p_i * p_j

                # Jos ulkopuoliset maalimaarat (yli max_goals) ovat ei-nolla, matriisi ei ole täysin 1.0; tämä on ok näytölle.
                p_home = np.tril(prob_matrix, -1).sum()
                p_draw = np.trace(prob_matrix)
                p_away = np.triu(prob_matrix, 1).sum()

                p_under25 = sum(prob_matrix[i,j] for i in range(max_goals+1) for j in range(max_goals+1) if i+j < 3)
                p_over25 = 1.0 - p_under25

                st.write(f"**1 (Home):** {p_home:.2%} | **X:** {p_draw:.2%} | **2 (Away):** {p_away:.2%}")
                st.write(f"**Under 2.5:** {p_under25:.2%} | **Over 2.5:** {p_over25:.2%}")

                show_odds = st.checkbox("Hae kertoimet ja laske EV mallilla", value=False)
                if show_odds:
                    od = odds(fid)
                    rows = []
                    for r in od.get("response", []):
                        for bm in r.get("bookmakers", []):
                            for bet in bm.get("bets", []):
                                if str(bet.get("name","")).lower() in ("match winner","winner","1x2","1x2 full time","fulltime result"):
                                    values = bet.get("values", [])
                                    home_v = next((v for v in values if str(v.get("value","")).lower() in ("home","1")), None)
                                    away_v = next((v for v in values if str(v.get("value","")).lower() in ("away","2")), None)
                                    if not (home_v and away_v):
                                        continue
                                    try:
                                        o1 = float(home_v.get("odd"))
                                        o2 = float(away_v.get("odd"))
                                    except Exception:
                                        continue
                                    ev1 = expected_value(p_home, o1)
                                    ev2 = expected_value(p_away, o2)
                                    rows.append({"bookmaker": bm.get("name"), "odds_home": o1, "odds_away": o2,
                                                 "EV_home_per_1e": round(ev1,4), "EV_away_per_1e": round(ev2,4),
                                                 "Kelly_home_eur": round(kelly_fraction(p_home, o1, fraction=kelly_frac)*bankroll,2),
                                                 "Kelly_away_eur": round(kelly_fraction(p_away, o2, fraction=kelly_frac)*bankroll,2)})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    else:
                        st.info("Ei kerroindataa tälle fixturelle.")
