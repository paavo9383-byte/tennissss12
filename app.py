import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

# -------------------------------------------------
# Asetukset
# -------------------------------------------------
API_KEY = "52a26b83b9dcb13ebcf0790bbac97ea14eeaaf75f88ec4661f98b9ab9009bf76"
st.set_page_config(page_title="Tennis-malli: simulaatiot, value & Kelly", layout="wide")
np.random.seed(42)

# -------------------------------------------------
# Apuja
# -------------------------------------------------
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
    s = p1 + p2
    if s <= 0:
        return 0.5, 0.5
    p1, p2 = p1 / s, p2 / s
    p1 = _clamp(p1, floor, 1 - floor)
    p2 = 1 - p1
    return p1, p2

def _parse_time(ts: str):
    if not isinstance(ts, str):
        return "-"
    for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.strftime("%H:%M")
        except Exception:
            continue
    return "-"

def _implied_from_odds(o1, o2):
    if not (o1 and o2):
        return None, None
    p1 = 1.0 / o1
    p2 = 1.0 / o2
    s = p1 + p2
    if s <= 0:
        return None, None
    return p1 / s, p2 / s

# -------------------------------------------------
# API-kutsut (vanha Streamlit-yhteensopiva cache)
# -------------------------------------------------
@st.cache(allow_output_mutation=True, show_spinner=False)
def fetch_fixtures(date_str):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_fixtures"
        f"&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    )
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        data = res.json()
        if data.get("success") == 1:
            resu = data.get("result", [])
            if isinstance(resu, list):
                return resu
            if isinstance(resu, dict):
                return list(resu.values())
    except Exception:
        pass
    return []

@st.cache(allow_output_mutation=True, show_spinner=False)
def fetch_player(player_key):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_players"
        f"&APIkey={API_KEY}&player_key={player_key}"
    )
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        data = res.json()
        if data.get("success") == 1:
            resu = data.get("result", [])
            if isinstance(resu, list) and resu:
                return resu[0]
            if isinstance(resu, dict):
                return resu
    except Exception:
        pass
    return None

@st.cache(allow_output_mutation=True, show_spinner=False)
def fetch_h2h(p1_key, p2_key):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_H2H"
        f"&APIkey={API_KEY}&first_player_key={p1_key}&second_player_key={p2_key}"
    )
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        data = res.json()
        if data.get("success") == 1:
            resu = data.get("result", {})
            if isinstance(resu, dict):
                return {
                    "H2H": resu.get("H2H", []),
                    "firstPlayerResults": resu.get("firstPlayerResults", []),
                    "secondPlayerResults": resu.get("secondPlayerResults", []),
                }
            if isinstance(resu, list) and resu and isinstance(resu[0], dict):
                r0 = resu[0]
                return {
                    "H2H": r0.get("H2H", []),
                    "firstPlayerResults": r0.get("firstPlayerResults", []),
                    "secondPlayerResults": r0.get("secondPlayerResults", []),
                }
    except Exception:
        pass
    return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}

@st.cache(allow_output_mutation=True, show_spinner=False)
def fetch_odds(match_key):
    url = (
        f"https://api.api-tennis.com/tennis/?method=get_odds"
        f"&APIkey={API_KEY}&match_key={match_key}"
    )
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        data = res.json()
        if data.get("success") == 1:
            resu = data.get("result", {})
            if isinstance(resu, dict):
                return resu.get(str(match_key), resu)
            if isinstance(resu, list) and resu:
                return resu[0]
    except Exception:
        pass
    return {}

# -------------------------------------------------
# Ominaisuudet / tilastot
# -------------------------------------------------
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

def get_hold_break_from_stats(player: dict):
    stats = player.get("stats", []) if isinstance(player, dict) else []
    if isinstance(stats, list) and stats:
        latest = sorted(stats, key=lambda x: str(x.get("season","")), reverse=True)[0]
        hold = _safe_float(latest.get("service_games_won"), None)
        retn = _safe_float(latest.get("return_games_won"), None)
        if hold is not None and retn is not None:
            return _clamp(hold/100.0, 0.40, 0.95), _clamp(retn/100.0, 0.05, 0.60)
    return 0.75, 0.25  # fallback

def get_h2h_ratio(h2h_dict: dict) -> float:
    games = h2h_dict.get("H2H", []) or []
    if not games:
        return 0.5
    p1_wins = sum(1 for g in games if g.get("event_winner") == "First Player")
    p2_wins = sum(1 for g in games if g.get("event_winner") == "Second Player")
    tot = p1_wins + p2_wins
    return p1_wins / tot if tot > 0 else 0.5

# -------------------------------------------------
# Simulaatiot (setti / matsi)
# -------------------------------------------------
def tiebreak_win_prob(p1_hold: float, p2_hold: float) -> float:
    delta = _clamp(p1_hold - p2_hold, -0.4, 0.4)
    return _clamp(0.5 + delta/4.0, 0.2, 0.8)

def simulate_set(p1_hold: float, p2_hold: float) -> int:
    g1 = g2 = 0
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
    return 1 if np.random.rand() < tiebreak_win_prob(p1_hold, p2_hold) else 0

def simulate_match_best_of_3(p1_hold: float, p2_hold: float, n_sim: int = 3000) -> float:
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

# -------------------------------------------------
# Malli: yhdistetään signaalit (sim + surface + form + H2H + markkinakalibrointi)
# -------------------------------------------------
def model_probability_for_match(match: dict, n_sim: int) -> dict:
    p1_key = match.get("first_player_key")
    p2_key = match.get("second_player_key")
    if not p1_key or not p2_key:
        return {"p1": 0.5, "p2": 0.5, "surface": "Hard"}

    surface = get_surface_from_match(match)
    p1 = fetch_player(p1_key)
    p2 = fetch_player(p2_key)
    h2h = fetch_h2h(p1_key, p2_key)

    p1_hold, p1_break = get_hold_break_from_stats(p1)
    p2_hold, p2_break = get_hold_break_from_stats(p2)

    wr1_surface = get_surface_winrate_from_stats(p1, surface)
    wr2_surface = get_surface_winrate_from_stats(p2, surface)
    surface_ratio = _normalize(wr1_surface, wr2_surface)[0]

    form1 = get_recent_form_ratio(h2h.get("firstPlayerResults", []), n=5)
    form2 = get_recent_form_ratio(h2h.get("secondPlayerResults", []), n=5)
    form_ratio = _normalize(form1, form2)[0]

    h2h_ratio = get_h2h_ratio(h2h)

    # pinta → pieni säätö holdeihin (±5 %-yks)
    adj = _clamp((wr1_surface - wr2_surface) * 0.10, -0.05, 0.05)
    p1_hold_adj = _clamp(p1_hold + adj, 0.40, 0.95)
    p2_hold_adj = _clamp(p2_hold - adj, 0.40, 0.95)

    p1_sim = simulate_match_best_of_3(p1_hold_adj, p2_hold_adj, n_sim=n_sim)

    # painotus
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

# -------------------------------------------------
# Kertoimien parsinta
# -------------------------------------------------
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

# -------------------------------------------------
# Kelly-kriteeri
# -------------------------------------------------
def kelly(prob, odds, bankroll, fraction=0.5):
    """Puolikas Kelly oletuksena."""
    if not odds or odds <= 1.0:
        return 0.0, -1.0
    b = odds - 1.0
    q = 1.0 - prob
    f = (b*prob - q) / b
    f = max(0.0, f)
    stake = f * bankroll * fraction
    edge = prob - (1.0 / odds)  # absoluuttinen edge (ei oikaistu marginaalista)
    return stake, edge

# -------------------------------------------------
# UI: sivupalkki
# -------------------------------------------------
st.title("🎾 Tennis-ennusteet (simulaatiomalli + value + Kelly)")

today = date.today()
sel_date = st.sidebar.date_input("Päivämäärä", value=today)
match_type = st.sidebar.selectbox("Ottelutyyppi", ["Kaikki", "Live", "Tulevat"])
n_sim = st.sidebar.slider("Simulaatioiden määrä / ottelu", min_value=1000, max_value=20000, step=1000, value=5000)
bankroll = st.sidebar.number_input("Bankroll (€)", value=1000, step=100)

# -------------------------------------------------
# Hae ottelut ja suodata
# -------------------------------------------------
fixtures = fetch_fixtures(sel_date.isoformat()) or []
if match_type == "Live":
    fixtures = [m for m in fixtures if m.get("event_live") == "1"]
elif match_type == "Tulevat":
    fixtures = [m for m in fixtures if m.get("event_live") == "0" and not m.get("event_status")]

tournaments = sorted({m.get("tournament_name", "-") for m in fixtures})
sel_tournaments = st.sidebar.multiselect("Turnaus", options=tournaments, default=tournaments)
if sel_tournaments:
    fixtures = [m for m in fixtures if m.get("tournament_name", "-") in sel_tournaments]

# -------------------------------------------------
# Päätabla (ottelut, kertoimet, malli, edget, Kelly)
# -------------------------------------------------
rows = []
for match in fixtures:
    p1 = match.get("event_first_player", "-")
    p2 = match.get("event_second_player", "-")
    start_str = _parse_time(match.get("event_date"))
    tourn = match.get("tournament_name", "-")
    event_key = match.get("event_key")

    model = model_probability_for_match(match, n_sim=n_sim)
    p1_prob, p2_prob = model["p1"], model["p2"]

    odds_data = fetch_odds(event_key)
    o1, o2 = extract_two_way_odds(odds_data)

    # markkinakalibrointi (maltillinen), vähentää 100–0 outliereita
    if o1 and o2:
        imp1, imp2 = _implied_from_odds(o1, o2)
        if imp1 and imp2:
            p1_prob = 0.8 * p1_prob + 0.2 * imp1
            p2_prob = 1 - p1_prob

    imp1, imp2 = _implied_from_odds(o1, o2)
    edge1 = (p1_prob - imp1) * 100 if (imp1 is not None) else None
    edge2 = (p2_prob - imp2) * 100 if (imp2 is not None) else None

    k1, raw_edge1 = kelly(p1_prob, o1, bankroll, fraction=0.5) if o1 else (0.0, -1.0)
    k2, raw_edge2 = kelly(p2_prob, o2, bankroll, fraction=0.5) if o2 else (0.0, -1.0)

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
        "Kelly 1 (€)": round(k1, 2),
        "Kelly 2 (€)": round(k2, 2),
    })

df = pd.DataFrame(rows, columns=[
    "Aika","Pelaaja 1","Pelaaja 2","Turnaus","Surface",
    "Kerroin 1","Kerroin 2","Malli P1 %","Malli P2 %",
    "Fair 1","Fair 2","Edge 1 %","Edge 2 %","Kelly 1 (€)","Kelly 2 (€)"
])

st.dataframe(df, use_container_width=True)
st.caption("Malli yhdistää: simulaation (hold/break), alustan, formikunnon ja H2H:n. Markkinakalibrointi vähentää yli-itsevarmuutta. Edge = oma todennäköisyys − implied (%‑yks.). Kelly = puolikas Kelly.")

# -------------------------------------------------
# Value bets (Top 10)
# -------------------------------------------------
with st.expander("🔥 Päivän Top 10 Value Bets"):
    value_rows = []
    for match in fixtures:
        p1 = match.get("event_first_player", "-")
        p2 = match.get("event_second_player", "-")
        event_key = match.get("event_key")
        tourn = match.get("tournament_name", "-")

        odds_data = fetch_odds(event_key)
        o1, o2 = extract_two_way_odds(odds_data)
        if not (o1 and o2):
            continue

        model = model_probability_for_match(match, n_sim=max(2000, n_sim//2))
        p1_prob, p2_prob = model["p1"], model["p2"]

        imp1, imp2 = _implied_from_odds(o1, o2)
        if imp1 is None:
            continue

        # pieni markkinashrinkkaus
        p1_prob = 0.85 * p1_prob + 0.15 * imp1
        p2_prob = 1 - p1_prob

        v1 = p1_prob - imp1
        v2 = p2_prob - imp2
        best_side = "1" if v1 > v2 else "2"
        best_edge = max(v1, v2)

        if best_edge > 0:
            odds_sel = o1 if best_side == "1" else o2
            k_stake, _ = kelly(p1_prob if best_side == "1" else p2_prob, odds_sel, bankroll, fraction=0.5)
            value_rows.append({
                "Aika": _parse_time(match.get("event_date")),
                "Ottelu": f"{p1} vs {p2}",
                "Turnaus": tourn,
                "Puoli": best_side,
                "Edge %": round(best_edge*100, 2),
                "Kerroin": round(odds_sel, 2),
                "Malli %": round((p1_prob if best_side=="1" else p2_prob)*100, 1),
                "Kelly (€)": round(k_stake, 2),
            })

    if value_rows:
        top10 = pd.DataFrame(sorted(value_rows, key=lambda x: x["Edge %"], reverse=True)[:10])
        st.table(top10)
    else:
        st.info("Ei value-kohteita valituille suodattimille.")

# -------------------------------------------------
# Monte Carlo -visualisaatio
# -------------------------------------------------
st.markdown("---")
st.subheader("📈 Monte Carlo ‑visualisaatio (malli vs. markkina)")

def _get_hold_return_for_player(player_key):
    p = fetch_player(player_key)
    if not p:
        return 0.65, 0.35
    stats = p.get("stats") or []
    if not stats:
        return 0.65, 0.35
    latest = sorted(stats, key=lambda x: str(x.get("season","")), reverse=True)[0]
    hold = _safe_float(latest.get("service_games_won"), 65) / 100.0
    retn = _safe_float(latest.get("return_games_won"), 35) / 100.0
    hold = _clamp(hold, 0.40, 0.95)
    retn = _clamp(retn, 0.05, 0.55)
    return hold, retn

def _tiebreak_win_prob_mc(p1_hold, p2_hold):
    delta = _clamp(p1_hold - p2_hold, -0.4, 0.4)
    return _clamp(0.5 + delta/4.0, 0.2, 0.8)

def _simulate_set_once(p1_hold, p2_hold):
    g1 = g2 = 0
    for i in range(12):
        if i % 2 == 0:
            g1 += 1 if np.random.rand() < p1_hold else 0
        else:
            g2 += 1 if np.random.rand() < p2_hold else 0
    if g1 > g2: return 1
    if g2 > g1: return 0
    return 1 if np.random.rand() < _tiebreak_win_prob_mc(p1_hold, p2_hold) else 0

def _simulate_match_best_of_3_once(p1_hold, p2_hold):
    s1 = s2 = 0
    while s1 < 2 and s2 < 2:
        s = _simulate_set_once(p1_hold, p2_hold)
        if s == 1: s1 += 1
        else: s2 += 1
    return 1 if s1 > s2 else 0

def _sample_beta_around(p, strength=200):
    alpha = max(1e-3, p * strength)
    beta = max(1e-3, (1 - p) * strength)
    return np.random.beta(alpha, beta)

def run_mc_for_match(match: dict, n_iter: int = 8000, shrink_to_market: float = 0.25):
    p1_key = match.get("first_player_key")
    p2_key = match.get("second_player_key")
    if not p1_key or not p2_key:
        return [], 0.5, (None, None)

    odds_data = fetch_odds(match.get("event_key"))
    # käytä samaa odds-parsintaa
    o1, o2 = extract_two_way_odds(odds_data)
    imp1 = (1/o1)/(1/o1 + 1/o2) if (o1 and o2) else None
    imp2 = 1 - imp1 if imp1 is not None else None

    # point estimate mallista
    model = model_probability_for_match(match, n_sim=max(2000, n_iter//4))
    p1_point = model["p1"]

    # hold/return peruspisteet
    h1, r1 = _get_hold_return_for_player(p1_key)
    h2, r2 = _get_hold_return_for_player(p2_key)

    # MC – nopea malli jakaumaan
    samples = []
    for _ in range(n_iter):
        h1_s = _sample_beta_around(h1, strength=300)
        h2_s = _sample_beta_around(h2, strength=300)
        # set-win approksimaatio
        set_p = _clamp(0.5 + (h1_s - h2_s)/4.0, 0.15, 0.85)
        match_p = _clamp(set_p*set_p*(3 - 2*set_p), 0.05, 0.95)  # Bo3 approx
        if imp1 is not None:
            match_p = (1 - shrink_to_market) * match_p + shrink_to_market * imp1
        samples.append(match_p)

    return samples, p1_point, (imp1, imp2)

# UI ohjaimet
match_labels = [f"{m.get('event_first_player','-')} vs {m.get('event_second_player','-')} — {m.get('tournament_name','-')} ({_parse_time(m.get('event_date'))})" for m in fixtures]
selected_idx = st.selectbox("Valitse ottelu", options=list(range(len(fixtures))) if fixtures else [0],
                            format_func=lambda i: match_labels[i] if fixtures else "-", index=0 if fixtures else 0)

mc_cols = st.columns([2,1,1,1])
with mc_cols[1]:
    n_iter = st.number_input("Simulointeja", min_value=1000, max_value=40000, step=1000, value=8000)
with mc_cols[2]:
    shrink = st.slider("Kalibrointi markkinaan", min_value=0.0, max_value=0.6, value=0.25, step=0.05,
                       help="0 = ei shrinkkausta, 0.25 = maltillinen, 0.5 = vahva")
with mc_cols[3]:
    run_btn = st.button("Simuloi & piirrä")

if run_btn and fixtures:
    m = fixtures[selected_idx]
    samples, point_est, implied = run_mc_for_match(m, n_iter=int(n_iter), shrink_to_market=shrink)

    if samples:
        fig = plt.figure(figsize=(8,4))
        plt.hist(samples, bins=40, alpha=0.85)
        plt.xlabel("P1 voittotodennäköisyys")
        plt.ylabel("Frekvenssi")
        plt.axvline(point_est, linestyle="--")  # mallin point
        if implied[0] is not None:
            plt.axvline(implied[0], linestyle=":")  # markkina
        plt.title(f"Jakauma: {m.get('event_first_player','-')} vs {m.get('event_second_player','-')}")
        st.pyplot(fig, use_container_width=True)

        p_low, p_med, p_high = np.percentile(samples, [5, 50, 95])
        st.markdown(
            f"**Mallin pointti**: {point_est*100:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**MC mediaani**: {p_med*100:.1f}% (5–95%: {p_low*100:.1f}–{p_high*100:.1f}%) "
            + (f"&nbsp;&nbsp;|&nbsp;&nbsp; **Implied**: {implied[0]*100:.1f}%" if implied[0] is not None else "")
        )
    else:
        st.info("Ei riittäviä tietoja simulaatioon tälle ottelulle.")
        # ====================
# Value bet evaluointi
# ====================
def evaluate_value(match, n_sim=2000):
    odds_data = fetch_odds(match["event_key"])
    if "Home/Away" not in odds_data:
        return None

    home_vals = odds_data["Home/Away"].get("Home", {})
    away_vals = odds_data["Home/Away"].get("Away", {})

    try:
        home_odds = max([float(v) for v in home_vals.values() if v])
        away_odds = max([float(v) for v in away_vals.values() if v])
    except ValueError:
        return None

    prob1, prob2 = calculate_probabilities(match["first_player_key"], match["second_player_key"])

    ev_home = prob1 * home_odds
    ev_away = prob2 * away_odds

    return [
        {
            "Ottelu": f"{match['event_first_player']} vs {match['event_second_player']}",
            "Pelaaja": match["event_first_player"],
            "Todennäköisyys": f"{prob1:.2%}",
            "Keroin": home_odds,
            "EV": ev_home,
            "Turnaus": match.get("tournament_name", "N/A"),
            "Aika": match.get("event_date", "N/A")
        },
        {
            "Ottelu": f"{match['event_first_player']} vs {match['event_second_player']}",
            "Pelaaja": match["event_second_player"],
            "Todennäköisyys": f"{prob2:.2%}",
            "Keroin": away_odds,
            "EV": ev_away,
            "Turnaus": match.get("tournament_name", "N/A"),
            "Aika": match.get("event_date", "N/A")
        }
    ]

# ====================
# Top 10 value bets
# ====================
if st.button("Näytä Top 10 Value Bets"):
    value_bets = []
    for match in fixtures:
        res = evaluate_value(match)
        if res:
            value_bets.extend(res)

    if value_bets:
        df_value = pd.DataFrame(value_bets).sort_values("EV", ascending=False).head(10)
        st.subheader("📊 Top 10 Value Bets")
        st.dataframe(df_value)
    else:
        st.info("Ei tarpeeksi dataa value betseihin.")

# ====================
# Paras value juuri nyt
# ====================
st.subheader("🎯 Paras value juuri nyt")
go_best = st.button("🔍 Hae paras value nyt")

if go_best:
    candidates = []
    for match in fixtures:
        res = evaluate_value(match, n_sim=2000)
        if res:
            candidates.extend(res)

    if not candidates:
        st.warning("Ei löytynyt value betsiä juuri nyt.")
    else:
        best = sorted(candidates, key=lambda x: x["EV"], reverse=True)[0]
        st.success("Paras value löytyi!")
        st.json(best)
