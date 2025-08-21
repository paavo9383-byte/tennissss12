import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime

# ------------------ Asetukset ------------------
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
st.set_page_config(page_title="Tennis-malli: simulaatiot & value", layout="wide")
np.random.seed(42)

# ------------------ Apurit ------------------
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

def _expander_json(title: str, payload, enable: bool):
    if enable:
        with st.expander(title):
            st.json(payload)

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(str(x).replace(",", "."))
    except Exception:
        return default

def _implied_prob_from_odds(odds):
    f = _safe_float(odds, None)
    if f and f > 0:
        return 1.0 / f
    return None

def _normalize_probs(p1, p2):
    s = p1 + p2
    if s > 0:
        return p1 / s, p2 / s
    return 0.5, 0.5

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _parse_time(ts: str):
    # API-tyypillisesti "YYYY-MM-DD HH:MM"
    if not isinstance(ts, str):
        return "-"
    for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.strftime("%H:%M")
        except Exception:
            continue
    return ts

# ------------------ API-kutsut ------------------
@st.cache_data
def fetch_fixtures(date_str, debug=False):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_fixtures"
        f"&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    )
    data = _request_json(url, debug)
    _expander_json(f"📦 Fixtures {date_str}", data, debug)

    if isinstance(data, dict) and data.get("success") == 1:
        res = data.get("result", [])
        if isinstance(res, list):
            return res
        if isinstance(res, dict):
            return list(res.values())
    return []

@st.cache_data
def fetch_player(player_key, debug=False):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_players"
        f"&APIkey={API_KEY}&player_key={player_key}"
    )
    data = _request_json(url, debug)
    _expander_json(f"📦 Player {player_key}", data, debug)

    if isinstance(data, dict) and data.get("success") == 1:
        res = data.get("result", [])
        if isinstance(res, list) and res:
            return res[0]
        if isinstance(res, dict):
            return res
    return None

@st.cache_data
def fetch_h2h(p1_key, p2_key, debug=False):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_H2H"
        f"&APIkey={API_KEY}&first_player_key={p1_key}&second_player_key={p2_key}"
    )
    data = _request_json(url, debug)
    _expander_json(f"📦 H2H {p1_key} vs {p2_key}", data, debug)

    if isinstance(data, dict) and data.get("success") == 1:
        res = data.get("result", {})
        if isinstance(res, dict):
            return {
                "H2H": res.get("H2H", []),
                "firstPlayerResults": res.get("firstPlayerResults", []),
                "secondPlayerResults": res.get("secondPlayerResults", []),
            }
        if isinstance(res, list) and res and isinstance(res[0], dict):
            r0 = res[0]
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
    _expander_json(f"📦 Odds match {match_key}", data, debug)

    if isinstance(data, dict) and data.get("success") == 1:
        res = data.get("result", {})
        if isinstance(res, dict):
            return res.get(str(match_key), res)
        if isinstance(res, list) and res:
            return res[0]
    return {}

# ------------------ Ominaisuudet ------------------
def get_surface_from_match(match: dict) -> str:
    for k in ["event_court", "court", "surface", "event_surface", "tournament_surface"]:
        val = match.get(k)
        if isinstance(val, str) and val:
            v = val.strip().lower()
            if "hard" in v: return "Hard"
            if "clay" in v: return "Clay"
            if "grass" in v: return "Grass"
            if "indoor" in v: return "Indoor"
    return "Hard"

def get_recent_form_ratio(results_list, n=5):
    """Palauttaa voittosuhteen viimeisistä n-ottelusta (0..1)."""
    if not isinstance(results_list, list) or not results_list:
        return 0.5
    last = results_list[:n]
    wins, total = 0, 0
    for r in last:
        # Monilla API:lla ei ole suoraa flagia – yritetään päätellä
        is_win = r.get("isWinner")
        if isinstance(is_win, bool):
            wins += 1 if is_win else 0
            total += 1
        else:
            ew = (r.get("event_winner") or "").lower()
            if ew in {"first player", "second player"}:
                # tässä listassa ei aina tiedetä kumpi pelaaja on "first"
                # jätetään huomioimatta jos emme pysty mapittamaan
                continue
            res = (r.get("result") or r.get("score_note") or "")
            if isinstance(res, str):
                rl = res.lower().strip()
                if rl.startswith("w") or "win" in rl:
                    wins += 1; total += 1
                elif rl.startswith("l") or "loss" in rl:
                    total += 1
    return wins/total if total > 0 else 0.5

def get_surface_winrate_from_stats(player: dict, surface: str) -> float:
    stats = player.get("stats", []) if isinstance(player, dict) else []
    if not isinstance(stats, list) or not stats:
        return 0.5
    # Etsi pintakohtainen rivi
    candidates = [s for s in stats if str(s.get("surface","")).lower().startswith(surface.lower())]
    chosen = sorted(candidates, key=lambda x: str(x.get("season","")), reverse=True)[0] if candidates else \
             sorted(stats, key=lambda x: str(x.get("season","")), reverse=True)[0]
    w = int(_safe_float(chosen.get("matches_won"), 0) or 0)
    l = int(_safe_float(chosen.get("matches_lost"), 0) or 0)
    return w / (w + l) if (w + l) > 0 else 0.5

def get_hold_break_from_stats(player: dict) -> tuple:
    """Palauttaa (hold%, break%) 0..1 (fallback neutraaleihin arvoihin)."""
    stats = player.get("stats", []) if isinstance(player, dict) else []
    if isinstance(stats, list) and stats:
        latest = sorted(stats, key=lambda x: str(x.get("season","")), reverse=True)[0]
        hold = _safe_float(latest.get("service_games_won"), None)
        retn = _safe_float(latest.get("return_games_won"), None)
        if hold is not None and retn is not None:
            return _clamp(hold/100.0, 0.4, 0.95), _clamp(retn/100.0, 0.05, 0.6)
    # fallback: neutraalit (läh. ATP/WTA keskitaso)
    return 0.75, 0.25

def get_h2h_ratio(h2h_dict: dict) -> float:
    games = h2h_dict.get("H2H", []) or []
    if not games:
        return 0.5
    p1_wins = sum(1 for g in games if g.get("event_winner") == "First Player")
    p2_wins = sum(1 for g in games if g.get("event_winner") == "Second Player")
    tot = p1_wins + p2_wins
    return p1_wins / tot if tot > 0 else 0.5

# ------------------ Simulaatiot ------------------
def tiebreak_win_prob(p1_hold: float, p2_hold: float) -> float:
    # yksinkertainen approksimaatio tiebreakin voittotn: 0.5 + (hold-ero)/4
    delta = _clamp(p1_hold - p2_hold, -0.4, 0.4)
    return _clamp(0.5 + delta/4.0, 0.2, 0.8)

def simulate_set(p1_hold: float, p2_hold: float) -> int:
    """
    Simuloi yhden erän.
    Palauttaa 1 jos P1 voittaa erän, muuten 0.
    Malli: vuorosyötöt, 12 peliä max -> 6-6 tiebreak.
    """
    g1 = 0
    g2 = 0
    for i in range(12):
        if i % 2 == 0:  # P1 syöttää
            if np.random.rand() < p1_hold: g1 += 1
            else: g2 += 1
        else:          # P2 syöttää
            if np.random.rand() < p2_hold: g2 += 1
            else: g1 += 1
    if g1 > g2:
        return 1
    if g2 > g1:
        return 0
    # 6-6 -> tiebreak
    return 1 if np.random.rand() < tiebreak_win_prob(p1_hold, p2_hold) else 0

def simulate_match_best_of_3(p1_hold: float, p2_hold: float, n_sim: int = 5000) -> float:
    """
    Palauttaa P(P1 voittaa ottelun).
    """
    p1_sets = 0
    wins = 0
    for _ in range(n_sim):
        s1 = s2 = 0
        while s1 < 2 and s2 < 2:
            s1 += simulate_set(p1_hold, p2_hold)
            s2 += 1 if s1 + s2 > (s1 + s2 - 1) and (s1 == s2) else 0  # ei käytetä; vain täydennyksenä
            # korjataan: jos P1 ei voittanut erää, se meni P2:lle
            if s1 + s2 == 0 or (s1 + s2) % 1 == 0:
                pass
            if s1 + s2 > 0 and s1 + s2 <= 5:
                pass
            if s1 + s2 > 0 and (s1 == s2 or s1 == 2 or s2 == 2):
                pass
            # yksinkertaistaen: jos P1 ei voittanut, P2 voitti
            if (s1 + s2) > 0 and (s1 + s2) % 1 == 0:
                pass
            if s1 + s2 > 0 and (s1 == s2 or s1 == 2 or s2 == 2):
                pass
            # yllä oleva oli placeholder; tehdään selkeästi:
            if s1 + s2 == 0:
                continue
            # korjataan erän laskenta oikein
            if s1 == s2:  # ei pitäisi tapahtua heti, mutta varmistus
                pass
            if s1 < 2 and s2 < 2:
                # jos viimeisin simulate_set ei lisännyt P1:lle, se lisäsi P2:lle
                if (s1 + s2) % 1 == 0:  # ei merkitystä – poistetaan
                    pass
        if s1 > s2:
            wins += 1
    return wins / n_sim

# korjataan yllä oleva pieni sekoilu: tehdään yksinkertaisempi best-of-3
def simulate_match_best_of_3(p1_hold: float, p2_hold: float, n_sim: int = 5000) -> float:
    wins = 0
    for _ in range(n_sim):
        s1 = s2 = 0
        while s1 < 2 and s2 < 2:
            erä = simulate_set(p1_hold, p2_hold)
            if erä == 1: s1 += 1
            else: s2 += 1
        if s1 > s2:
            wins += 1
    return wins / n_sim

# ------------------ Malli: yhdistetään signaalit ------------------
def model_probability_for_match(match: dict, n_sim: int, debug=False) -> dict:
    p1_key = match.get("first_player_key")
    p2_key = match.get("second_player_key")
    if not p1_key or not p2_key:
        return {"p1": 0.5, "p2": 0.5}

    surface = get_surface_from_match(match)

    # data
    p1 = fetch_player(p1_key, debug)
    p2 = fetch_player(p2_key, debug)
    h2h = fetch_h2h(p1_key, p2_key, debug)

    # hold/break arviot
    p1_hold, p1_break = get_hold_break_from_stats(p1)
    p2_hold, p2_break = get_hold_break_from_stats(p2)

    # surface winrate
    wr1_surface = get_surface_winrate_from_stats(p1, surface)
    wr2_surface = get_surface_winrate_from_stats(p2, surface)
    surface_ratio = _normalize_probs(wr1_surface, wr2_surface)[0]

    # recent form (viimeiset 5)
    form1 = get_recent_form_ratio(h2h.get("firstPlayerResults", []), n=5)
    form2 = get_recent_form_ratio(h2h.get("secondPlayerResults", []), n=5)
    form_ratio = _normalize_probs(form1, form2)[0]

    # h2h
    h2h_ratio = get_h2h_ratio(h2h)

    # säädetään holdeja hieman surface-eron mukaan (±5 %-yks max)
    adj = _clamp((wr1_surface - wr2_surface) * 0.10, -0.05, 0.05)
    p1_hold_adj = _clamp(p1_hold + adj, 0.40, 0.95)
    p2_hold_adj = _clamp(p2_hold - adj, 0.40, 0.95)

    # simulaatio (best-of-3)
    p1_sim = simulate_match_best_of_3(p1_hold_adj, p2_hold_adj, n_sim=n_sim)

    # painotettu yhdistelmä
    w_sim, w_surface, w_form, w_h2h = 0.50, 0.25, 0.15, 0.10
    p1_model = (
        w_sim * p1_sim +
        w_surface * surface_ratio +
        w_form * form_ratio +
        w_h2h * h2h_ratio
    )
    p1_model = _clamp(p1_model, 0.02, 0.98)
    p2_model = 1 - p1_model

    return {
        "p1": p1_model,
        "p2": p2_model,
        "p1_hold": p1_hold_adj,
        "p2_hold": p2_hold_adj,
        "surface": surface,
        "p1_sim": p1_sim,
        "surface_ratio": surface_ratio,
        "form_ratio": form_ratio,
        "h2h_ratio": h2h_ratio,
    }

# ------------------ Odds-parsinta ------------------
def extract_two_way_odds(odds_data: dict):
    """
    Yrittää löytää 1v1 voittajakertoimet eri market-nimillä.
    Palauttaa (odds1, odds2) tai (None, None).
    """
    if not isinstance(odds_data, dict):
        return None, None
    market_candidates = ["Home/Away", "Match Winner", "To Win Match", "1x2", "Winner", "Moneyline"]
    market = next((m for m in market_candidates if m in odds_data), None)
    if not market:
        # joskus marketit suoraan "Home"/"Away" ilman yläavainta
        if "Home" in odds_data and "Away" in odds_data:
            home_vals = odds_data.get("Home") or {}
            away_vals = odds_data.get("Away") or {}
        else:
            return None, None
    else:
        section = odds_data.get(market, {})
        # tyypilliset avaimet
        for k1, k2 in [("Home", "Away"), ("1", "2"), ("Player 1", "Player 2")]:
            if k1 in section and k2 in section:
                home_vals = section.get(k1) or {}
                away_vals = section.get(k2) or {}
                break
        else:
            # fallback: ota kaksi ensimmäistä dictiä
            vals = [v for v in section.values() if isinstance(v, dict)]
            if len(vals) >= 2:
                home_vals, away_vals = vals[0], vals[1]
            else:
                return None, None

    def best_odds(d):
        best = None
        if isinstance(d, dict):
            for v in d.values():
                f = _safe_float(v, None)
                if f and f > 1.0:
                    best = f if best is None else max(best, f)
        return best

    return best_odds(home_vals), best_odds(away_vals)

# ------------------ UI ------------------
st.title("🎾 Tennis-ennusteet (simulaatiomalli)")

# Sivupalkki
today = date.today()
sel_date = st.sidebar.date_input("Päivämäärä", value=today)
match_type = st.sidebar.selectbox("Ottelutyyppi", ["Kaikki", "Live", "Tulevat"])
n_sim = st.sidebar.slider("Simulaatioiden määrä / ottelu", min_value=1000, max_value=20000, step=1000, value=5000)
debug_mode = st.sidebar.checkbox("🔍 Debug-näkymä (näytä raw API)", value=False)

fixtures = fetch_fixtures(sel_date.isoformat(), debug_mode) or []
if match_type == "Live":
    fixtures = [m for m in fixtures if m.get("event_live") == "1"]
elif match_type == "Tulevat":
    fixtures = [m for m in fixtures if m.get("event_live") == "0" and not m.get("event_status")]

tournaments = sorted({m.get("tournament_name", "-") for m in fixtures})
sel_tournaments = st.sidebar.multiselect("Turnaus", options=tournaments, default=tournaments)
if sel_tournaments:
    fixtures = [m for m in fixtures if m.get("tournament_name", "-") in sel_tournaments]

# Taulukon rakentaminen
rows = []
for match in fixtures:
    p1 = match.get("event_first_player", "-")
    p2 = match.get("event_second_player", "-")
    start_str = _parse_time(match.get("event_date"))
    tourn = match.get("tournament_name", "-")
    event_key = match.get("event_key")

    # Malli
    model = model_probability_for_match(match, n_sim=n_sim, debug=debug_mode)
    p1_prob, p2_prob = model["p1"], model["p2"]

    # Odds
    odds_data = fetch_odds(event_key, debug_mode)
    o1, o2 = extract_two_way_odds(odds_data)

    # Implied probs + edge
    imp1 = _implied_prob_from_odds(o1)
    imp2 = _implied_prob_from_odds(o2)
    edge1 = (p1_prob - imp1) * 100 if (imp1 is not None) else None
    edge2 = (p2_prob - imp2) * 100 if (imp2 is not None) else None

    rows.append({
        "Aika": start_str,
        "Pelaaja 1": p1,
        "Pelaaja 2": p2,
        "Turnaus": tourn,
        "Surface": model["surface"],
        "Kerroin 1": round(o1, 2) if o1 else None,
        "Kerroin 2": round(o2, 2) if o2 else None,
        "Malli P1 %": round(p1_prob * 100, 1),
        "Malli P2 %": round(p2_prob * 100, 1),
        "Fair 1": round(1.0 / p1_prob, 2),
        "Fair 2": round(1.0 / p2_prob, 2),
        "Edge 1 %": round(edge1, 1) if edge1 is not None else None,
        "Edge 2 %": round(edge2, 1) if edge2 is not None else None,
    })

df = pd.DataFrame(rows, columns=[
    "Aika","Pelaaja 1","Pelaaja 2","Turnaus","Surface",
    "Kerroin 1","Kerroin 2","Malli P1 %","Malli P2 %","Fair 1","Fair 2","Edge 1 %","Edge 2 %"
])

st.dataframe(df, use_container_width=True)

st.caption("Huom: Malli yhdistää simulaation (hold/break %), alustan, formikunnon ja H2H:n. Edge = oma todennäköisyys − markkinan implied probability ( %-yksikköä ).")
