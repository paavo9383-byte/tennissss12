import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date

API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"

# ---------- Asetukset / Debug ----------
st.set_page_config(page_title="Tennis-ennusteet ja kertoimet", layout="wide")
debug_mode = st.sidebar.checkbox("🔍 Näytä API-vastaukset (debug)", value=False)

# ---------- Apurit ----------
def _request_json(url: str, debug: bool = False):
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        data = res.json()
        return data if isinstance(data, (dict, list)) else {}
    except Exception as e:
        if debug:
            st.warning(f"API-pyyntö epäonnistui: {e}")
        return {}

def _show_json_expander(title: str, payload):
    # näytä debugissa vain ensimmäisellä ajolla (cache miss)
    with st.expander(title):
        st.json(payload)

def _safe_logo_url(val):
    if not isinstance(val, str):
        return None
    v = val.strip()
    if not v or v.lower() in {"null", "none", "nan"}:
        return None
    if v.startswith("http://") or v.startswith("https://"):
        return v
    return None

def _try_image(col, url, width=75):
    if not url:
        col.write("")  # tyhjä paikka
        return
    try:
        col.image(url, width=width)
    except Exception:
        # Jos kuva ei kelpaa (GIF/bytes/virheellinen) jätetään näyttämättä
        col.write("")

def _to_float_list(d):
    out = []
    if isinstance(d, dict):
        for v in d.values():
            try:
                f = float(str(v).replace(",", "."))
                if np.isfinite(f):
                    out.append(f)
            except Exception:
                continue
    return out

# ---------- API-funktiot ----------
@st.cache_data
def fetch_fixtures(date_str, debug=False):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_fixtures"
        f"&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    )
    data = _request_json(url, debug)
    if debug:
        _show_json_expander(f"📦 Fixtures API response {date_str}", data)

    if isinstance(data, dict) and data.get("success") == 1:
        result = data.get("result", [])
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return list(result.values())
    return []

@st.cache_data
def fetch_player(player_key, debug=False):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_players"
        f"&APIkey={API_KEY}&player_key={player_key}"
    )
    data = _request_json(url, debug)
    if debug:
        _show_json_expander(f"📦 Player API response {player_key}", data)

    if isinstance(data, dict) and data.get("success") == 1:
        result = data.get("result", [])
        if isinstance(result, list) and result:
            return result[0]
        if isinstance(result, dict):
            return result
    return None

@st.cache_data
def fetch_h2h(p1_key, p2_key, debug=False):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_H2H"
        f"&APIkey={API_KEY}&first_player_key={p1_key}&second_player_key={p2_key}"
    )
    data = _request_json(url, debug)
    if debug:
        _show_json_expander(f"📦 H2H API response {p1_key} vs {p2_key}", data)

    if isinstance(data, dict) and data.get("success") == 1:
        result = data.get("result", {})
        if isinstance(result, dict):
            return {
                "H2H": result.get("H2H", []),
                "firstPlayerResults": result.get("firstPlayerResults", []),
                "secondPlayerResults": result.get("secondPlayerResults", []),
            }
        if isinstance(result, list) and result:
            # joillain vasteilla result voi olla lista yhdellä dict-alkiolla
            r0 = result[0] if isinstance(result[0], dict) else {}
            return {
                "H2H": r0.get("H2H", []),
                "firstPlayerResults": r0.get("firstPlayerResults", []),
                "secondPlayerResults": r0.get("secondPlayerResults", []),
            }
    return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}

@st.cache_data
def fetch_odds(match_key, debug=False):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_odds"
        f"&APIkey={API_KEY}&match_key={match_key}"
    )
    data = _request_json(url, debug)
    if debug:
        _show_json_expander(f"📦 Odds API response match {match_key}", data)

    if isinstance(data, dict) and data.get("success") == 1:
        result = data.get("result", {})
        if isinstance(result, dict):
            # useita palvelun muotoja nähty: joko result[match_key] tai suoraan marketit
            return result.get(str(match_key), result)
        if isinstance(result, list) and result:
            return result[0]
    return {}

# ---------- Todennäköisyyslaskenta ----------
def calculate_probabilities(p1_key, p2_key):
    h2h = fetch_h2h(p1_key, p2_key, debug_mode)
    games = h2h.get("H2H", []) or []
    if games:
        p1_wins = sum(1 for g in games if g.get("event_winner") == "First Player")
        p2_wins = sum(1 for g in games if g.get("event_winner") == "Second Player")
        total = p1_wins + p2_wins
        if total > 0:
            return p1_wins / total, p2_wins / total

    # fallback: kausitilastot
    player1 = fetch_player(p1_key, debug_mode)
    player2 = fetch_player(p2_key, debug_mode)

    def get_ratio(player):
        stats = player.get("stats") if isinstance(player, dict) else None
        if isinstance(stats, list) and stats:
            stats_sorted = sorted(stats, key=lambda x: str(x.get("season", "")), reverse=True)
            latest = stats_sorted[0] if stats_sorted else {}
            wins = int((latest.get("matches_won") or 0))
            losses = int((latest.get("matches_lost") or 0))
            if wins + losses > 0:
                return wins / (wins + losses)
        return 0.5

    r1 = get_ratio(player1)
    r2 = get_ratio(player2)
    s = r1 + r2
    return (r1 / s, r2 / s) if s > 0 else (0.5, 0.5)

# ---------- UI ----------
st.title("🎾 Tennis-ennusteet ja kertoimet")

# Suodattimet
today = date.today()
sel_date = st.sidebar.date_input("Päivämäärä", value=today)
match_type = st.sidebar.selectbox("Ottelutyyppi", ["Kaikki", "Live", "Tulevat"])

fixtures = fetch_fixtures(sel_date.isoformat(), debug_mode) or []
if match_type == "Live":
    fixtures = [m for m in fixtures if m.get("event_live") == "1"]
elif match_type == "Tulevat":
    fixtures = [m for m in fixtures if m.get("event_live") == "0" and not m.get("event_status")]

tournaments = sorted({m.get("tournament_name", "-") for m in fixtures})
sel_tournaments = st.sidebar.multiselect("Turnaus", options=tournaments, default=tournaments)
if sel_tournaments:
    fixtures = [m for m in fixtures if m.get("tournament_name", "-") in sel_tournaments]

# Ottelut ja kertoimet
st.header("Ottelut ja kertoimet")
header_cols = st.columns([1, 2, 1, 2, 2, 1, 1])
headers = ["", "Pelaaja 1", "", "Pelaaja 2", "Turnaus", "Kerroin 1", "Kerroin 2"]
for i, head in enumerate(headers):
    header_cols[i].write(f"**{head}**")

for match in fixtures:
    p1_key = match.get("first_player_key")
    p2_key = match.get("second_player_key")
    if not p1_key or not p2_key:
        continue  # ohita vajaamuotoinen rivi

    player1 = fetch_player(p1_key, debug_mode)
    player2 = fetch_player(p2_key, debug_mode)

    img1 = _safe_logo_url((player1 or {}).get("player_logo"))
    img2 = _safe_logo_url((player2 or {}).get("player_logo"))

    # Todennäköisyydet ja kertoimet
    prob1, prob2 = calculate_probabilities(p1_key, p2_key)
    odds_data = fetch_odds(match.get("event_key"), debug_mode) or {}

    # Yritä löytää paras market kahden voittajan kertoimille
    market_candidates = ["Home/Away", "Match Winner", "To Win Match", "1x2", "Winner", "Moneyline"]
    chosen_market = next((m for m in market_candidates if isinstance(odds_data, dict) and m in odds_data), None)
    section = odds_data.get(chosen_market, {}) if chosen_market else {}

    home_vals = {}
    away_vals = {}
    if isinstance(section, dict):
        for k1, k2 in [("Home", "Away"), ("1", "2"), ("Player 1", "Player 2")]:
            if k1 in section and k2 in section:
                home_vals = section.get(k1) or {}
                away_vals = section.get(k2) or {}
                break
        # fallback: ota ensimmäiset kaksi alidictiä
        if not home_vals and not away_vals:
            vals = [v for v in section.values() if isinstance(v, dict)]
            if len(vals) >= 2:
                home_vals, away_vals = vals[0], vals[1]

    home_odds = _to_float_list(home_vals)
    away_odds = _to_float_list(away_vals)
    max_home = max(home_odds) if home_odds else None
    max_away = max(away_odds) if away_odds else None

    cols = st.columns([1, 2, 1, 2, 2, 1, 1])
    _try_image(cols[0], img1, width=75)
    cols[1].write(match.get("event_first_player", "-"))
    _try_image(cols[2], img2, width=75)
    cols[3].write(match.get("event_second_player", "-"))
    cols[4].write(match.get("tournament_name", "-"))
    cols[5].write(f"{max_home:.2f}" if isinstance(max_home, (int, float)) else "-")
    cols[6].write(f"{max_away:.2f}" if isinstance(max_away, (int, float)) else "-")

# Simulointipainike
if st.button("Simuloi 1000 ottelua"):
    st.header("Simulaatiotulokset (1000 ottelua)")
    sim_rows = []
    for match in fixtures:
        p1_key = match.get("first_player_key")
        p2_key = match.get("second_player_key")
        if not p1_key or not p2_key:
            continue
        prob1, prob2 = calculate_probabilities(p1_key, p2_key)
        wins1 = int(np.random.binomial(1000, prob1))
        wins2 = 1000 - wins1
        sim_rows.append({
            "Ottelu": f"{match.get('event_first_player', '-') } vs { match.get('event_second_player', '-')}",
            "Pelaaja 1 voitot": wins1,
            "Pelaaja 2 voitot": wins2,
            "P(1)": round(prob1, 3),
            "P(2)": round(prob2, 3),
        })
    st.table(pd.DataFrame(sim_rows))
