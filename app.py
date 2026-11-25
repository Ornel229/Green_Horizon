"""ClimaScope - global climate risk map for smallholder farmers.

Run locally:   streamlit run app.py
Dependencies:  see requirements.txt
Data:          Open-Meteo (forecast + historical archive), no API key needed.
"""

import datetime as dt

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

# ----------------------------------------------------------------- settings
RECENT_DAYS = 30          # window analysed ("the last 30 days")
NORMAL_YEARS = 10         # number of past years used for the local normal
DRY_MM = 1.0              # a day under 1 mm counts as dry
HEAT_C = 35.0             # a day with Tmax >= 35 C counts as a heat day
HEAVY_MM = 20.0           # a day with >= 20 mm counts as a heavy-rain day

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

st.set_page_config(page_title="ClimaScope", page_icon="🌍", layout="wide")


# --------------------------------------------------------------- data access
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent(lat: float, lon: float) -> pd.DataFrame:
    """Last 30 days + next 7 days of daily weather."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "past_days": RECENT_DAYS,
        "forecast_days": 7,
        "timezone": "auto",
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(d["time"]),
            "rain": d["precipitation_sum"],
            "tmax": d["temperature_2m_max"],
            "tmin": d["temperature_2m_min"],
        }
    )
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_normal_rain(lat: float, lon: float, start: dt.date, end: dt.date) -> float:
    """Average rainfall (mm) over the same calendar window in past years."""
    this_year = dt.date.today().year
    first_year = this_year - NORMAL_YEARS
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{first_year}-01-01",
        "end_date": f"{this_year - 1}-12-31",
        "daily": "precipitation_sum",
        "timezone": "auto",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    hist = pd.DataFrame({"date": pd.to_datetime(d["time"]), "rain": d["precipitation_sum"]})
    hist["rain"] = hist["rain"].fillna(0)

    totals = []
    for year in range(first_year, this_year):
        try:
            s = pd.Timestamp(start.replace(year=year))
            e = pd.Timestamp(end.replace(year=year))
        except ValueError:  # 29 February
            continue
        if s > e:  # window crossing the new year: shift start one year back
            s = s.replace(year=year - 1)
        window = hist[(hist["date"] >= s) & (hist["date"] <= e)]
        if len(window) > 0:
            totals.append(window["rain"].sum())
    return float(sum(totals) / len(totals)) if totals else float("nan")


# --------------------------------------------------------------- indicators
def longest_dry_spell(rain: pd.Series) -> int:
    best = cur = 0
    for v in rain:
        if v < DRY_MM:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_indicators(df: pd.DataFrame, normal_mm: float) -> dict:
    today = pd.Timestamp(dt.date.today())
    past = df[df["date"] <= today].copy()
    future = df[df["date"] > today]
    past["rain"] = past["rain"].fillna(0)

    rain_total = float(past["rain"].sum())
    anomaly = (rain_total - normal_mm) / normal_mm * 100 if normal_mm and normal_mm > 0 else float("nan")

    return {
        "rain_total": rain_total,
        "normal_mm": normal_mm,
        "anomaly_pct": anomaly,
        "dry_spell": longest_dry_spell(past["rain"]),
        "heat_days": int((past["tmax"] >= HEAT_C).sum()),
        "heavy_days": int((past["rain"] >= HEAVY_MM).sum()),
        "tmax_mean": float(past["tmax"].mean()),
        "forecast_rain": float(future["rain"].fillna(0).sum()),
        "forecast_tmax": float(future["tmax"].max()) if len(future) else float("nan"),
        "past": past,
        "future": future,
    }


def risk_level(ind: dict) -> tuple[str, list[str]]:
    """Simple additive score -> low / medium / high, with the reasons."""
    score, reasons = 0, []

    a = ind["anomaly_pct"]
    if a == a:  # not NaN
        if a <= -40:
            score += 2
            reasons.append("Rainfall is far below the local normal (drought risk).")
        elif a <= -20:
            score += 1
            reasons.append("Rainfall is below the local normal.")
        elif a >= 60:
            score += 1
            reasons.append("Rainfall is far above the local normal (flood risk).")

    if ind["dry_spell"] >= 14:
        score += 2
        reasons.append(f"Long dry spell: {ind['dry_spell']} days in a row without rain.")
    elif ind["dry_spell"] >= 7:
        score += 1
        reasons.append(f"Dry spell of {ind['dry_spell']} days.")

    if ind["heat_days"] >= 10:
        score += 2
        reasons.append(f"{ind['heat_days']} heat-stress days (Tmax >= {HEAT_C:.0f} °C).")
    elif ind["heat_days"] >= 3:
        score += 1
        reasons.append(f"{ind['heat_days']} heat-stress days.")

    if ind["heavy_days"] >= 3:
        score += 2
        reasons.append(f"{ind['heavy_days']} heavy-rain days (>= {HEAVY_MM:.0f} mm).")
    elif ind["heavy_days"] >= 1:
        score += 1
        reasons.append(f"{ind['heavy_days']} heavy-rain day(s).")

    level = "High" if score >= 5 else "Medium" if score >= 2 else "Low"
    if not reasons:
        reasons.append("No significant anomaly detected over the last 30 days.")
    return level, reasons


def crop_advice(ind: dict, level: str) -> list[str]:
    """Indicative advice only: needs validation by agronomy experts."""
    t, normal = ind["tmax_mean"], ind["normal_mm"]
    tips = []
    if t >= 30 and normal == normal and normal < 100:
        tips.append("Hot, dry-leaning zone: drought-tolerant crops such as sorghum, millet, cowpea.")
    elif t >= 27:
        tips.append("Warm zone: maize, cassava, yam, groundnut, rice where water is available.")
    elif t >= 18:
        tips.append("Mild zone: maize, beans, potato, vegetables.")
    else:
        tips.append("Cool zone: wheat, barley, potato, peas.")

    if level == "High":
        tips.append("Delay sowing if possible and prefer short-cycle or tolerant varieties.")
    elif level == "Medium":
        tips.append("Sow with caution: watch the forecast and keep some seed in reserve.")
    else:
        tips.append("Conditions look normal: follow the usual local planting calendar.")

    if ind["dry_spell"] >= 7:
        tips.append("Mulch and prioritise irrigation for young plants.")
    if ind["heavy_days"] >= 1:
        tips.append("Check drainage to limit erosion and waterlogging.")
    return tips


# ----------------------------------------------------------------------- UI
st.title("🌍 ClimaScope")
st.caption("A global climate risk map for smallholder farmers · open data (Open-Meteo)")

if "point" not in st.session_state:
    st.session_state["point"] = None

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("1. Click a location on the map")
    center = st.session_state["point"] or (8.0, 2.3)
    m = folium.Map(location=center, zoom_start=5 if st.session_state["point"] else 3)
    if st.session_state["point"]:
        folium.Marker(st.session_state["point"], tooltip="Selected location").add_to(m)
    out = st_folium(m, height=520, use_container_width=True, returned_objects=["last_clicked"])
    if out and out.get("last_clicked"):
        new = (round(out["last_clicked"]["lat"], 3), round(out["last_clicked"]["lng"], 3))
        if new != st.session_state["point"]:
            st.session_state["point"] = new
            st.rerun()

with right:
    st.subheader("2. Climate risk for farming")
    pt = st.session_state["point"]
    if not pt:
        st.info("Click anywhere on the map to see the climate risk there.")
    else:
        lat, lon = pt
        st.write(f"**Location:** {lat:.3f}, {lon:.3f}")
        try:
            with st.spinner("Loading open weather data..."):
                df = fetch_recent(lat, lon)
                today = dt.date.today()
                normal = fetch_normal_rain(lat, lon, today - dt.timedelta(days=RECENT_DAYS), today)
            ind = compute_indicators(df, normal)
            level, reasons = risk_level(ind)

            color = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}[level]
            st.markdown(f"### {color} Risk level: {level}")
            for r_ in reasons:
                st.write(f"- {r_}")

            c1, c2 = st.columns(2)
            c1.metric("Rain, last 30 days", f"{ind['rain_total']:.0f} mm",
                      None if ind["anomaly_pct"] != ind["anomaly_pct"] else f"{ind['anomaly_pct']:+.0f}% vs normal")
            c2.metric("Local normal", f"{ind['normal_mm']:.0f} mm" if ind["normal_mm"] == ind["normal_mm"] else "n/a")
            c3, c4, c5 = st.columns(3)
            c3.metric("Longest dry spell", f"{ind['dry_spell']} d")
            c4.metric("Heat days", ind["heat_days"])
            c5.metric("Heavy-rain days", ind["heavy_days"])

            st.markdown("#### Indicative crop and timing advice")
            for tip in crop_advice(ind, level):
                st.write(f"- {tip}")
            st.caption("Indicative only: to be validated with local agronomy experts.")
        except requests.RequestException as err:
            st.error(f"Could not load weather data. Please try again. ({err})")

# charts under the two columns
if st.session_state["point"] and "ind" in dir():
    st.subheader("3. Charts")
    g1, g2 = st.columns(2)
    full = pd.concat([ind["past"], ind["future"]]).set_index("date")
    with g1:
        st.write("Daily rainfall (mm): last 30 days and next 7 days")
        st.bar_chart(full["rain"].fillna(0))
    with g2:
        st.write("Daily temperature (°C)")
        st.line_chart(full[["tmax", "tmin"]])
    st.write(
        f"Next 7 days: about {ind['forecast_rain']:.0f} mm of rain expected, "
        f"maximum temperature {ind['forecast_tmax']:.0f} °C."
    )
