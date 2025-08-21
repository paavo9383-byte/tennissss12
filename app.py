# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import altair as alt
url = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/charting-m-matches.csv"
tennis_stats = pd.read_csv(url)
def get_player_stats(player_name):
    # yritetään matchata pelaajan nimi dataan
    row = tennis_stats[tennis_stats["player"].str.contains(player_name, case=False, na=False)]
    if not row.empty:
        return row.iloc[0].to_dict()
    return None
    stats1 = get_player_stats(match["event_first_player"])
stats2 = get_player_stats(match["event_second_player"])

if stats1 and stats2:
    p1_hold = stats1.get("hold_pct", 0.7)
    p2_hold = stats2.get("hold_pct", 0.7)

    # arvioidaan todennäköisyys pelaajan 1 voitolle
    prob1 = p1_hold / (p1_hold + (1 - p2_hold))
    prob2 = 1 - prob1
else:
    prob1, prob2 = 0.5, 0.5
# =========================================
# Asetukset
# =========================================
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
st.set_page_config(page_title="Tennis-malli PRO: simulaatiot, value & Kelly", layout="wide")
np.random.seed(42)

# =========================================
# Apu / util
# =========================================
def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(str(x).replace(",", "."))
    except Exception:
        return default

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _normalize(p1, p2, floor=0.02):
    s = (p1 or 0) + (p2 or 0)
    if s <= 0:
        return 0.5, 0.5
    p1, p2 = p1 / s, p2 / s
    p1 = _clamp(p1, floor, 1 - floor)
    return p1, 1 - p1

def _parse_time_any(match: dict):
    """
    Koittaa useita kenttiä: event_time, event_date (datetime), event_utc (ISO), event_timestamp, jne.
    Palauttaa "HH:MM" tai "-" jos ei löydy.
    """
    candidates = [
        match.get("event_time"),
        match.get("event_date"),
        match.get("event_utc"),
        match.get("event_start"),
        match.get("time"),
    ]
    for ts in candidates:
        if not ts or not isinstance(ts, str):
            continue
        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                    "%d.%m.%Y %H:%M", "%H:%M"]:
            try:
                dt = datetime.strptime(ts, fmt)
                return dt.strftime("%H:%M")
            except Exception:
                continue
        # jos on pelkkä HH:MM
        if len(ts) == 5 and ":" in ts:
            return ts
    # joskus on unix timestamp
    for ts in ["event_timestamp", "timestamp", "start_timestamp"]:
        v = match.get(ts)
        try:
            if v:
                dt = datetime.fromtimestamp(int(v))
                return dt.strftime("%H:%M")
        except Exception:
            pass
    return "-"

def _implied_from_odds(o1, o2):
    if not (o1 and o2) or (o1 <= 1.0 or o2 <= 1.0):
        return None, None
    p1 = 1.0 / o1
    p2 = 1.0 / o2
    s = p1 + p2
    if s <= 0:
        return None, None
    return p1 / s, p2 / s

def _pct(x):  # string -> 0..1
    f = _safe_float(x, None)
    if f is None:
        return None
    # jos >1, tulkitaan prosenteiksi
    return f/100.0 if f > 1.5 else f

# =========================================
# API-kutsut (cache_data)
# =========================================
@st.cache_data(show_spinner=False)
def fetch_fixtures(date_str):
    url = (f"https://api.api-tennis.com/tennis/?method=get_fixtures"
           f"&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}")
    try:
        r = requests.get(url, timeout=20)
        j = r.json()
        if j.get("success") == 1:
            res = j.get("result", [])
            if isinstance(res, list):
                return res
            if isinstance(res, dict):
                return list(res.values())
    except Exception:
        pass
    return []

@st.cache_data(show_spinner=False)
def fetch_player(player_key):
    url = (f"https://api.api-tennis.com/tennis/?method=get_players"
           f"&APIkey={API_KEY}&player_key={player_key}")
    try:
        r = requests.get(url, timeout=20)
        j = r.json()
        if j.get("success") == 1:
            res = j.get("result", [])
            if isinstance(res, list) and res:
                return res[0]
            if isinstance(res, dict):
                return res
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def fetch_h2h(p1_key, p2_key):
    url = (f"https://api.api-tennis.com/tennis/?method=get_H2H"
           f"&APIkey={API_KEY}&first_player_key={p1_key}&second_player_key={p2_key}")
    try:
        r = requests.get(url, timeout=20)
        j = r.json()
        if j.get("success") == 1:
            res = j.get("result", {})
            if isinstance(res, dict):
                return {
                    "H2H": res.get("H2H", []) or [],
                    "firstPlayerResults": res.get("firstPlayerResults", []) or [],
                    "secondPlayerResults": res.get("secondPlayerResults", []) or [],
                }
            if isinstance(res, list) and res and isinstance(res[0], dict):
                r0 = res[0]
                return {
                    "H2H": r0.get("H2H", []) or [],
                    "firstPlayerResults": r0.get("firstPlayerResults", []) or [],
                    "secondPlayerResults": r0.get("secondPlayerResults", []) or [],
                }
    except Exception:
        pass
    return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}

@st.cache_data(show_spinner=False)
def fetch_odds(match_key):
    url = (f"https://api.api-tennis.com/tennis/?method=get_odds"
           f"&APIkey={API_KEY}&match_key={match_key}")
    try:
        r = requests.get(url, timeout=20)
        j = r.json()
        if j.get("success") == 1:
            res = j.get("result", {})
            if isinstance(res, dict):
                return res.get(str(match_key), res)
            if isinstance(res, list) and res:
                return res[0]
    except Exception:
        pass
    return {}

# =========================================
# Tilastojen nosto & tulkinta
# =========================================
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

def _latest_stats(player: dict):
    stats = player.get("stats", []) if player else []
    if not isinstance(stats, list) or not stats:
        return {}
    return sorted(stats, key=lambda x: str(x.get("season","")), reverse=True)[0]

def get_recent_form_ratio(results_list, n=5):
    if not isinstance(results_list, list) or not results_list:
        return 0.5
    last = results_list[:n]
    wins, total = 0, 0
    for r in last:
        is_win = r.get("isWinner")
        if isinstance(is_win, bool):
            wins += 1 if is_win else 0
            total += 1
        else:
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
    candidates = [s for s in stats if str(s.get("surface","")).lower().startswith(surface.lower())]
    chosen = sorted(candidates, key=lambda x: str(x.get("season","")), reverse=True)[0] if candidates else \
             sorted(stats, key=lambda x: str(x.get("season","")), reverse=True)[0]
    w = int(_safe_float(chosen.get("matches_won"), 0) or 0)
    l = int(_safe_float(chosen.get("matches_lost"), 0) or 0)
    return w / (w + l) if (w + l) > 0 else 0.5

def extract_two_way_odds(odds_data: dict):
    if not isinstance(odds_data, dict):
        return None, None
    market_candidates = ["Home/Away", "Match Winner", "To Win Match", "1x2", "Winner", "Moneyline"]
    market = next((m for m in market_candidates if m in odds_data), None)

    if not market:
        if "Home" in odds_data and "Away" in odds_data:
            home_vals = odds_data.get("Home") or {}
            away_vals = odds_data.get("Away") or {}
        else:
            return None, None
    else:
        section = odds_data.get(market, {})
        for k1, k2 in [("Home", "Away"), ("1", "2"), ("Player 1", "Player 2")]:
            if k1 in section and k2 in section:
                home_vals = section.get(k1) or {}
                away_vals = section.get(k2) or {}
                break
        else:
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

# =========================================
# Edistynyt “piste→game→set→matsi” -malli
# =========================================
def _serve_point_win_prob_from_stats(stats: dict):
    """Arvioi syöttöpisteen voittotodennäköisyys p ~ P(win point on serve)."""
    fs_in = _pct(stats.get("first_serve_in") or stats.get("firstServeIn") or stats.get("first_serve_"))
    fs_won = _pct(stats.get("first_serve_points_won") or stats.get("firstServePointsWon"))
    ss_won = _pct(stats.get("second_serve_points_won") or stats.get("secondServePointsWon"))
    # fallbackit
    if fs_in is None: fs_in = 0.60
    if fs_won is None: fs_won = 0.70
    if ss_won is None: ss_won = 0.50
    p = fs_in * fs_won + (1 - fs_in) * ss_won  # sekoite
    return _clamp(p, 0.45, 0.80)

def _return_strength_from_stats(stats: dict):
    """Palautuspisteiden vahvuus (enemmän = parempi paluuja) 0..1."""
    rgw = _pct(stats.get("return_games_won") or stats.get("returnGamesWon"))  # games
    bpc = _pct(stats.get("break_points_converted") or stats.get("breakPointsConverted") or stats.get("bp_converted"))
    if rgw is None: rgw = 0.25
    if bpc is None: bpc = 0.40
    # Skaalaa 0..1
    return _clamp(0.5 * rgw / 0.35 + 0.5 * bpc / 0.45, 0.0, 1.0)

def _clutch_from_stats(stats: dict):
    """Paineensietokyky / clutch: tiebreak- ja bp-saved signaali."""
    t_won = _safe_float(stats.get("tiebreaks_won"), None)
    t_play = _safe_float(stats.get("tiebreaks_played"), None)
    tb_ratio = (t_won / t_play) if (t_won is not None and t_play and t_play > 0) else 0.5
    bps = _pct(stats.get("break_points_saved") or stats.get("breakPointsSaved") or stats.get("bp_saved"))
    if bps is None: bps = 0.60
    c = 0.6 * tb_ratio + 0.4 * bps
    return _clamp(c, 0.35, 0.80)

def _fatigue_penalty(recent_results: list):
    """Rasitus: viimeisen 7 päivän matsimäärä → pieni miinus winprobille."""
    if not isinstance(recent_results, list):
        return 0.0
    now = datetime.utcnow()
    cnt = 0
    for r in recent_results:
        ts = r.get("event_date") or r.get("date") or r.get("event_time")
        dt = None
        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
            try:
                dt = datetime.strptime(ts, fmt); break
            except Exception:
                continue
        if dt and (now - dt).days <= 7:
            cnt += 1
    # 0m = 0, 1-2m = -0.01, 3-4m = -0.02, 5+ = -0.03
    if cnt <= 0: return 0.0
    if cnt <= 2: return 0.01
    if cnt <= 4: return 0.02
    return 0.03

def _adjust_pserve_with_opponent_return(p_serve, opp_return_strength):
    # Opp return 0..1: suurempi → heikennä p_serveä hieman
    return _clamp(p_serve - 0.05 * (opp_return_strength - 0.5), 0.40, 0.85)

def _game_win_prob_from_point(p):
    """
    Voita game omalla syötöllä pistevoittotn p. Approksimaatio (deuce-malli on raskas).
    Hyvä käytännön likiarvo tennis-analyyseissä:
    """
    # Käytännön smooth-approksimaatio (sovite yleisiin arvoihin):
    # p=0.55 → ~0.70, p=0.60 → ~0.80, p=0.65 → ~0.88
    return _clamp(1 / (1 + ((1 - p) / p)**4.0), 0.05, 0.99)

def _tiebreak_win_prob_from_clutch(p1_game, p2_game, c1, c2):
    # Lähtötaso game-eroista
    base = _clamp(0.5 + (p1_game - p2_game) * 0.25, 0.20, 0.80)
    # Clutch-säätö +/- 5 %-yks
    adj = _clamp((c1 - c2) * 0.10, -0.05, 0.05)
    return _clamp(base + adj, 0.20, 0.80)

def _simulate_set_once(p1_hold, p2_hold, tbp):
    g1 = g2 = 0
    # 12 gamea max → TB
    for i in range(12):
        if i % 2 == 0:  # P1 syöttää
            g1 += 1 if np.random.rand() < p1_hold else 0
        else:
            g2 += 1 if np.random.rand() < p2_hold else 0
        # jos jompikumpi 6 ja eroa >=2 → loppu
        if g1 >= 6 or g2 >= 6:
            if abs(g1 - g2) >= 2:
                return 1 if g1 > g2 else 0
    # tiebreak
    return 1 if np.random.rand() < tbp else 0

def simulate_match_best_of_3(p1_hold, p2_hold, tbp, n_sim=3000):
    wins = 0
    for _ in range(n_sim):
        s1 = s2 = 0
        while s1 < 2 and s2 < 2:
            s = _simulate_set_once(p1_hold, p2_hold, tbp)
            if s == 1: s1 += 1
            else: s2 += 1
        if s1 > s2:
            wins += 1
    return wins / n_sim

def advanced_model_probability_for_match(match: dict, n_sim_match: int = 4000) -> dict:
    p1_key, p2_key = match.get("first_player_key"), match.get("second_player_key")
    if not p1_key or not p2_key:
        return {"p1": 0.5, "p2": 0.5, "surface": "Hard"}

    surface = get_surface_from_match(match)
    p1 = fetch_player(p1_key)
    p2 = fetch_player(p2_key)
    h2h = fetch_h2h(p1_key, p2_key)

    s1 = _latest_stats(p1)
    s2 = _latest_stats(p2)

    # pisteen voittotn omalla syötöllä
    p1_point_srv = _serve_point_win_prob_from_stats(s1)
    p2_point_srv = _serve_point_win_prob_from_stats(s2)

    # paluuvoima
    p1_ret_strength = _return_strength_from_stats(s1)
    p2_ret_strength = _return_strength_from_stats(s2)

    # säädä serve point opp-paluuvoimalla
    p1_point_srv = _adjust_pserve_with_opponent_return(p1_point_srv, p2_ret_strength)
    p2_point_srv = _adjust_pserve_with_opponent_return(p2_point_srv, p1_ret_strength)

    # game-hold approksimaatio
    p1_hold = _game_win_prob_from_point(p1_point_srv)
    p2_hold = _game_win_prob_from_point(p2_point_srv)

    # clutch / TB
    c1 = _clutch_from_stats(s1)
    c2 = _clutch_from_stats(s2)
    tbp = _tiebreak_win_prob_from_clutch(p1_hold, p2_hold, c1, c2)

    # rasitus
    form1 = get_recent_form_ratio(h2h.get("firstPlayerResults", []), n=6)
    form2 = get_recent_form_ratio(h2h.get("secondPlayerResults", []), n=6)
    fatigue1 = _fatigue_penalty(h2h.get("firstPlayerResults", []))
    fatigue2 = _fatigue_penalty(h2h.get("secondPlayerResults", []))

    # alusta
    wr1_surface = get_surface_winrate_from_stats(p1, surface)
    wr2_surface = get_surface_winrate_from_stats(p2, surface)
    surface_ratio = _normalize(wr1_surface, wr2_surface)[0]

    # H2H
    games = h2h.get("H2H", []) or []
    if games:
        p1_wins = sum(1 for g in games if g.get("event_winner") == "First Player")
        p2_wins = sum(1 for g in games if g.get("event_winner") == "Second Player")
        tot = p1_wins + p2_wins
        h2h_ratio = p1_wins / tot if tot > 0 else 0.5
    else:
        h2h_ratio = 0.5

    # simulaatio
    p1_sim = simulate_match_best_of_3(p1_hold, p2_hold, tbp, n_sim=n_sim_match)

    # kombinaatio (painot)
    w_sim, w_surface, w_form, w_h2h = 0.55, 0.20, 0.15, 0.10
    p1_model = (w_sim * p1_sim +
                w_surface * surface_ratio +
                w_form * _normalize(form1, form2)[0] +
                w_h2h * h2h_ratio)

    # rasitus alas
    p1_model -= (fatigue1 - fatigue2) * 0.5  # jos p1 väsyneempi → pienennä
    p1_model = _clamp(p1_model, 0.02, 0.98)
    return {
        "p1": p1_model,
        "p2": 1 - p1_model,
        "p1_hold": p1_hold,
        "p2_hold": p2_hold,
        "tbp": tbp,
        "surface": surface,
        "p1_sim": p1_sim,
        "surface_ratio": surface_ratio,
        "form_ratio": _normalize(form1, form2)[0],
        "h2h_ratio": h2h_ratio,
        "fatigue1": fatigue1,
        "fatigue2": fatigue2,
        "p1_point_srv": p1_point_srv,
        "p2_point_srv": p2_point_srv,
        "clutch1": c1,
        "clutch2": c2,
    }

# =========================================
# Kelly & EV
# =========================================
def kelly(prob, odds, bankroll, fraction=0.5):
    if not odds or odds <= 1.0 or prob <= 0 or prob >= 1:
        return 0.0, -1.0
    b = odds - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    f = max(0.0, f)
    stake = f * bankroll * fraction
    edge_abs = prob - (1.0 / odds)
    return stake, edge_abs

def expected_value(prob, odds):
    if not odds or odds <= 1.0:
        return 0.0
    # EV per 1€ panos
    return prob * (odds - 1) - (1 - prob)

# =========================================
# UI
# =========================================
st.title("🎾 Tennis-malli PRO (edistynyt): simulaatiot, value & Kelly")

today = date.today()
sel_date = st.sidebar.date_input("Päivämäärä", value=today)
match_type = st.sidebar.selectbox("Ottelutyyppi", ["Kaikki", "Live", "Tulevat"])
n_sim = st.sidebar.slider("Simulaatioiden määrä / ottelu", min_value=1000, max_value=20000, step=1000, value=4000)
bankroll = st.sidebar.number_input("Bankroll (€)", value=1000, step=100)
market_shrink = st.sidebar.slider("Kalibrointi markkinaan", 0.0, 0.5, 0.20, 0.05,
                                  help="0 = ei shrinkkausta, 0.2 = maltillinen, 0.5 = vahva")

# Hae & suodata
fixtures = fetch_fixtures(sel_date.isoformat()) or []
if match_type == "Live":
    fixtures = [m for m in fixtures if m.get("event_live") == "1"]
elif match_type == "Tulevat":
    fixtures = [m for m in fixtures if m.get("event_live") == "0" and not m.get("event_status")]

tournaments = sorted({m.get("tournament_name", "-") for m in fixtures})
sel_tournaments = st.sidebar.multiselect("Turnaus", options=tournaments, default=tournaments)
if sel_tournaments:
    fixtures = [m for m in fixtures if m.get("tournament_name", "-") in sel_tournaments]

# Päätabla
rows = []
for match in fixtures:
    p1 = match.get("event_first_player", "-")
    p2 = match.get("event_second_player", "-")
    start_str = _parse_time_any(match)
    tourn = match.get("tournament_name", "-")
    event_key = match.get("event_key")

    model = advanced_model_probability_for_match(match, n_sim_match=n_sim)
    p1_prob, p2_prob = model["p1"], model["p2"]

    odds_data = fetch_odds(event_key)
    o1, o2 = extract_two_way_odds(odds_data)

    # markkinakalibrointi
    imp1, imp2 = _implied_from_odds(o1, o2)
    if imp1 is not None:
        p1_prob = (1 - market_shrink) * p1_prob + market_shrink * imp1
        p2_prob = 1 - p1_prob

    edge1 = (p1_prob - (imp1 or 0)) * 100 if imp1 is not None else None
    edge2 = (p2_prob - (imp2 or 0)) * 100 if imp2 is not None else None

    k1, edge_abs1 = kelly(p1_prob, o1, bankroll, fraction=0.5) if o1 else (0.0, -1.0)
    k2, edge_abs2 = kelly(p2_prob, o2, bankroll, fraction=0.5) if o2 else (0.0, -1.0)

    ev1 = expected_value(p1_prob, o1) if o1 else None
    ev2 = expected_value(p2_prob, o2) if o2 else None

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
        "Fair 1": round(1.0 / p1_prob, 2) if p1_prob else None,
        "Fair 2": round(1.0 / p2_prob, 2) if p2_prob else None,
        "Edge 1 %": round(edge1, 1) if edge1 is not None else None,
        "Edge 2 %": round(edge2, 1) if edge2 is not None else None,
        "EV 1 €/€": round(ev1, 3) if ev1 is not None else None,
        "EV 2 €/€": round(ev2, 3) if ev2 is not None else None,
        "Kelly 1 (€)": round(k1, 2),
        "Kelly 2 (€)": round(k2, 2),
        # diagnostiset
        "P1 hold": round(model["p1_hold"], 3),
        "P2 hold": round(model["p2_hold"], 3),
        "TB P1": round(model["tbp"], 3),
        "Form P1": round(model["form_ratio"], 3),
        "H2H P1": round(model["h2h_ratio"], 3),
    })

df_cols = ["Aika","Pelaaja 1","Pelaaja 2","Turnaus","Surface",
           "Kerroin 1","Kerroin 2","Malli P1 %","Malli P2 %","Fair 1","Fair 2",
           "Edge 1 %","Edge 2 %","EV 1 €/€","EV 2 €/€","Kelly 1 (€)","Kelly 2 (€)",
           "P1 hold","P2 hold","TB P1","Form P1","H2H P1"]
df = pd.DataFrame(rows, columns=df_cols)

st.subheader("Ottelut, kertoimet, malli & panossuositukset")
st.dataframe(df, use_container_width=True)
st.caption("Edge = oma todennäköisyys − markkinan implied. EV = odotusarvo €/€. Kelly = puolikas Kelly.")

# Top 10 value
with st.expander("🔥 Päivän Top 10 Value Bets (EV & Kelly)"):
    value_rows = []
    for match in fixtures:
        event_key = match.get("event_key")
        odds_data = fetch_odds(event_key)
        o1, o2 = extract_two_way_odds(odds_data)
        if not (o1 and o2):
            continue

        model = advanced_model_probability_for_match(match, n_sim_match=max(2000, n_sim//2))
        p1_prob, p2_prob = model["p1"], model["p2"]

        imp1, imp2 = _implied_from_odds(o1, o2)
        if imp1 is not None:
            p1_prob = (1 - market_shrink) * p1_prob + market_shrink * imp1
            p2_prob = 1 - p1_prob

        ev1 = expected_value(p1_prob, o1)
        ev2 = expected_value(p2_prob, o2)
        best_side = "1" if ev1 > ev2 else "2"
        best_ev = max(ev1, ev2)
        if best_ev > 0:
            odds_sel = o1 if best_side == "1" else o2
            k_stake, _ = kelly(p1_prob if best_side=="1" else p2_prob, odds_sel, bankroll, fraction=0.5)
            value_rows.append({
                "Aika": _parse_time_any(match),
                "Ottelu": f"{match.get('event_first_player','-')} vs {match.get('event_second_player','-')}",
                "Turnaus": match.get("tournament_name", "-"),
                "Puoli": best_side,
                "Kerroin": round(odds_sel, 2),
                "Malli %": round((p1_prob if best_side=="1" else p2_prob)*100, 1),
                "EV €/€": round(best_ev, 3),
                "Kelly (€)": round(k_stake, 2),
            })
    if value_rows:
        top10 = pd.DataFrame(sorted(value_rows, key=lambda x: x["EV €/€"], reverse=True)[:10])
        st.table(top10)
    else:
        st.info("Ei value-kohteita valituilla suodattimilla juuri nyt.")

# Paras value juuri nyt
st.markdown("---")
st.subheader("🚨 Paras value juuri nyt (1 klikkaus)")
go_best = st.button("Etsi paras value nyt", key="go_best_value_unique")
if go_best:
    candidates = []
    for match in fixtures:
        event_key = match.get("event_key")
        odds_data = fetch_odds(event_key)
        o1, o2 = extract_two_way_odds(odds_data)
        if not (o1 and o2):
            continue
        model = advanced_model_probability_for_match(match, n_sim_match=max(2000, n_sim//2))
        p1_prob, p2_prob = model["p1"], model["p2"]

        imp1, imp2 = _implied_from_odds(o1, o2)
        if imp1 is not None:
            p1_prob = (1 - market_shrink) * p1_prob + market_shrink * imp1
            p2_prob = 1 - p1_prob

        ev1 = expected_value(p1_prob, o1)
        ev2 = expected_value(p2_prob, o2)
        if ev1 is not None and ev2 is not None:
            if ev1 > 0 or ev2 > 0:
                if ev1 >= ev2:
                    k_stake, _ = kelly(p1_prob, o1, bankroll, fraction=0.5)
                    candidates.append((
                        ev1,
                        {
                            "Ottelu": f"{match.get('event_first_player','-')} vs {match.get('event_second_player','-')}",
                            "Puoli": "1",
                            "Kerroin": round(o1, 2),
                            "Malli %": round(p1_prob*100, 1),
                            "EV €/€": round(ev1, 3),
                            "Suositus (€)": round(k_stake, 2),
                        }
                    ))
                else:
                    k_stake, _ = kelly(p2_prob, o2, bankroll, fraction=0.5)
                    candidates.append((
                        ev2,
                        {
                            "Ottelu": f"{match.get('event_first_player','-')} vs {match.get('event_second_player','-')}",
                            "Puoli": "2",
                            "Kerroin": round(o2, 2),
                            "Malli %": round(p2_prob*100, 1),
                            "EV €/€": round(ev2, 3),
                            "Suositus (€)": round(k_stake, 2),
                        }
                    ))
    if candidates:
        best = max(candidates, key=lambda x: x[0])[1]
        st.success(f"Paras value juuri nyt: {best['Ottelu']} | Puoli {best['Puoli']} | "
                   f"Kerroin {best['Kerroin']} | Malli {best['Malli %']}% | EV {best['EV €/€']} €/€ | "
                   f"Kelly-suositus {best['Suositus (€)']} €")
    else:
        st.warning("Ei positiivisen EV:n kohteita juuri nyt suodattimilla.")

# Monte Carlo visualisointi (Altair)
st.markdown("---")
st.subheader("📈 Monte Carlo ‑jakauma (malli vs. markkina)")

match_labels = [f"{m.get('event_first_player','-')} vs {m.get('event_second_player','-')} — {m.get('tournament_name','-')} ({_parse_time_any(m)})" for m in fixtures]
selected_idx = st.selectbox("Valitse ottelu", options=list(range(len(fixtures))) if fixtures else [0],
                            format_func=lambda i: match_labels[i] if fixtures else "-", index=0 if fixtures else 0)

col_mc1, col_mc2, col_mc3 = st.columns([2,1,1])
with col_mc2:
    n_iter = st.number_input("Simulointeja", min_value=2000, max_value=50000, step=1000, value=10000)
with col_mc3:
    run_mc = st.button("Simuloi jakauma", key="run_mc_unique")

def run_mc_for_match(match: dict, n_iter: int = 10000, shrink_to_market: float = 0.20):
    p1_key = match.get("first_player_key")
    p2_key = match.get("second_player_key")
    if not p1_key or not p2_key:
        return [], 0.5, (None, None)

    odds_data = fetch_odds(match.get("event_key"))
    o1, o2 = extract_two_way_odds(odds_data)
    imp1, imp2 = _implied_from_odds(o1, o2)

    model = advanced_model_probability_for_match(match, n_sim_match=max(2000, n_iter//5))
    p1_point = model["p1"]
    p1_hold, p2_hold, tbp = model["p1_hold"], model["p2_hold"], model["tbp"]

    samples = []
    for _ in range(int(n_iter)):
        # sattumanvarainen pieni vaihtelu holdeihin ja tb:hen
        h1 = _clamp(np.random.normal(p1_hold, 0.02), 0.40, 0.95)
        h2 = _clamp(np.random.normal(p2_hold, 0.02), 0.40, 0.95)
        tb = _clamp(np.random.normal(tbp, 0.03), 0.20, 0.80)

        # set-win approx bo3
        set_p = _clamp(0.5 + (h1 - h2)/4.0, 0.15, 0.85)
        match_p = _clamp(set_p*set_p*(3 - 2*set_p), 0.05, 0.95)
        if imp1 is not None:
            match_p = (1 - shrink_to_market) * match_p + shrink_to_market * imp1
        samples.append(match_p)
    return samples, p1_point, (imp1, imp2)

if run_mc and fixtures:
    m = fixtures[selected_idx]
    samples, point_est, implied = run_mc_for_match(m, n_iter=int(n_iter), shrink_to_market=market_shrink)
    if samples:
        samp_df = pd.DataFrame({"p": samples})
        p_low, p_med, p_high = np.percentile(samples, [5, 50, 95])

        hist = alt.Chart(samp_df).mark_bar().encode(
            x=alt.X("p:Q", bin=alt.Bin(maxbins=40), title="P1 voittotodennäköisyys"),
            y=alt.Y("count():Q", title="Frekvenssi"),
            tooltip=[alt.Tooltip("count()", title="Frekvenssi")]
        ).properties(width=700, height=300)

        rules = []
        rules.append(alt.Chart(pd.DataFrame({"x":[point_est]})).mark_rule(strokeWidth=2, strokeDash=[6,4]).encode(x="x:Q"))
        if implied[0] is not None:
            rules.append(alt.Chart(pd.DataFrame({"x":[implied[0]]})).mark_rule(strokeWidth=2).encode(x="x:Q"))

        st.altair_chart(hist + sum(rules[1:], start=rules[0]), use_container_width=True)
        st.markdown(
            f"**Mallin pointti**: {point_est*100:.1f}% &nbsp;|&nbsp; "
            f"**MC mediaani**: {p_med*100:.1f}% (5–95%: {p_low*100:.1f}–{p_high*100:.1f}%)"
            + (f" &nbsp;|&nbsp; **Implied**: {implied[0]*100:.1f}%" if implied[0] is not None else "")
        )
    else:
        st.info("Ei riittäviä tietoja simulaatioon tälle ottelulle.")
