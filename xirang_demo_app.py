#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from os.path import join
from pathlib import Path
from difflib import SequenceMatcher, get_close_matches

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt
import plotly.express as px

from Modules.CRM_module import CRMP

@dataclass(frozen=True)
class WellSite:
    name: str
    lat: float
    lon: float
    region: str


WELLS = [
    WellSite("UK-W01-London", 51.5074, -0.1278, "London"),
    WellSite("UK-W02-Manchester", 53.4808, -2.2426, "North West"),
    WellSite("UK-W03-Birmingham", 52.4862, -1.8904, "West Midlands"),
    WellSite("UK-W04-Glasgow", 55.8642, -4.2518, "Scotland"),
    WellSite("UK-W05-Bristol", 51.4545, -2.5879, "South West"),
    WellSite("UK-W06-Leeds", 53.8008, -1.5491, "Yorkshire"),
    WellSite("UK-W07-Liverpool", 53.4084, -2.9916, "North West"),
    WellSite("UK-W08-Sheffield", 53.3811, -1.4701, "Yorkshire"),
    WellSite("UK-W09-Newcastle", 54.9783, -1.6178, "North East"),
    WellSite("UK-W10-Nottingham", 52.9548, -1.1581, "East Midlands"),
    WellSite("UK-W11-Leicester", 52.6369, -1.1398, "East Midlands"),
    WellSite("UK-W12-Southampton", 50.9097, -1.4044, "South East"),
    WellSite("UK-W13-Portsmouth", 50.8198, -1.0880, "South East"),
    WellSite("UK-W14-Cambridge", 52.2053, 0.1218, "East of England"),
    WellSite("UK-W15-Oxford", 51.7520, -1.2577, "South East"),
    WellSite("UK-W16-Cardiff", 51.4816, -3.1791, "Wales"),
    WellSite("UK-W17-Swansea", 51.6214, -3.9436, "Wales"),
    WellSite("UK-W18-Edinburgh", 55.9533, -3.1883, "Scotland"),
    WellSite("UK-W19-Aberdeen", 57.1497, -2.0943, "Scotland"),
    WellSite("UK-W20-Inverness", 57.4778, -4.2247, "Scotland"),
    WellSite("UK-W21-Belfast", 54.5973, -5.9301, "Northern Ireland"),
    WellSite("UK-W22-Derry", 54.9966, -7.3086, "Northern Ireland"),
    WellSite("UK-W23-Plymouth", 50.3755, -4.1427, "South West"),
    WellSite("UK-W24-Norwich", 52.6309, 1.2974, "East of England"),
    WellSite("UK-W25-Hull", 53.7676, -0.3274, "Yorkshire"),
]

CRM_WELLS = [
    WellSite("P1", 51.5074, -0.1278, "London"),
    WellSite("P2", 53.4808, -2.2426, "North West"),
    WellSite("P3", 52.4862, -1.8904, "West Midlands"),
    WellSite("P4", 55.8642, -4.2518, "Scotland"),
]

# Coarse country boundary overlays for quick geographic orientation.
UK_BOUNDARIES = {
    "England": [
        [-5.8, 50.0], [1.8, 50.0], [1.8, 55.9], [-2.1, 55.9], [-3.2, 54.7], [-5.8, 50.0]
    ],
    "Scotland": [
        [-7.8, 54.6], [-0.5, 54.6], [-0.5, 58.8], [-7.8, 58.8], [-7.8, 54.6]
    ],
    "Wales": [
        [-5.9, 51.3], [-2.5, 51.3], [-2.5, 53.6], [-5.9, 53.6], [-5.9, 51.3]
    ],
    "Northern Ireland": [
        [-8.3, 54.0], [-5.2, 54.0], [-5.2, 55.4], [-8.3, 55.4], [-8.3, 54.0]
    ],
}

CLUSTER_COLORS_HEX = {
    "C1": "#1d3557",  # deep blue
    "C2": "#2a9d8f",  # teal
    "C3": "#e9c46a",  # amber
    "C4": "#e76f51",  # terracotta
}
CLUSTER_NOTES = {
    "C1": "Latitude >= 56.0",
    "C2": "53.5 <= Latitude < 56.0",
    "C3": "Latitude < 53.5 and Longitude < -2.0",
    "C4": "Latitude < 53.5 and Longitude >= -2.0",
}
PROJECT_ROOT = Path(__file__).resolve().parent
GCAM_SAMPLE_PATH = PROJECT_ROOT / "data" / "gcam" / "gcam_global_sample.csv"
GCAM_CENTROID_PATH = PROJECT_ROOT / "data" / "gcam" / "region_centroids.csv"
GCAM_REGION_COLOR_PATH = PROJECT_ROOT / "data" / "gcam" / "region_colors.csv"
GCAM_ISO3_PATH = PROJECT_ROOT / "data" / "gcam" / "region_iso3.csv"
GEOTHERMAL_POTENTIAL_PATH = PROJECT_ROOT / "data" / "geothermal" / "global_geothermal_potential.csv"

I18N = {
    "en": {
        "subtitle": "eXplainable Intelligent Resilience Agent Network for Geothermal systems",
        "tab_monitor": "UK Well Monitoring",
        "tab_gcam": "Global GCAM Scenarios",
        "data_source": "Data Source",
        "choose_source": "Choose source",
        "synthetic_well_count": "Synthetic Well Count",
        "dashboard_filters": "Dashboard Filters",
        "region_grouping": "Region Grouping",
        "regions": "Regions",
        "well_search": "Well Search",
        "wells": "Wells",
        "date_range": "Date Range",
        "show_boundaries": "Show UK Country Boundaries",
        "show_interactive": "Interactive Map Layer",
        "lang": "Language",
        "grouping_c": "C1-C4",
        "grouping_admin": "Administrative",
        "need_one_well": "Please select at least one well.",
        "need_valid_date": "Please select a valid start and end date.",
        "no_data_range": "No data in the selected range.",
        "metrics_dict": "Metrics Dictionary",
    },
    "zh": {
        "subtitle": "eXplainable Intelligent Resilience Agent Network for Geothermal systems",
        "tab_monitor": "英国井网监测",
        "tab_gcam": "全球GCAM情景",
        "data_source": "数据源",
        "choose_source": "选择数据源",
        "synthetic_well_count": "模拟井数量",
        "dashboard_filters": "筛选条件",
        "region_grouping": "区域分组",
        "regions": "区域",
        "well_search": "井名搜索",
        "wells": "井口",
        "date_range": "日期范围",
        "show_boundaries": "显示英国边界",
        "show_interactive": "交互地图层",
        "lang": "语言",
        "grouping_c": "C1-C4",
        "grouping_admin": "行政区",
        "need_one_well": "请至少选择一口井。",
        "need_valid_date": "请选择有效的起止日期。",
        "no_data_range": "该时间范围无数据。",
        "metrics_dict": "指标字典",
    },
}


def tr(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return I18N.get(lang, I18N["en"]).get(key, key)


def metrics_dictionary_df() -> pd.DataFrame:
    rows = [
        {
            "Metric": "flow_m3h",
            "Unit": "m3/h",
            "Definition": "Injection/flow proxy used in UK monitoring series.",
            "Source": "Synthetic UK generator or CRM-derived aggregate injection",
        },
        {
            "Metric": "pressure_bar",
            "Unit": "bar",
            "Definition": "Pressure signal used for alert and agent checks.",
            "Source": "Synthetic proxy or scaled production-derived proxy (CRM mode)",
        },
        {
            "Metric": "actual",
            "Unit": "signal-dependent",
            "Definition": "Observed series (temperature-like in synthetic mode; production in CRM mode).",
            "Source": "Synthetic UK data or Streak Production.xlsx",
        },
        {
            "Metric": "pred / p10 / p90",
            "Unit": "same as actual",
            "Definition": "Model prediction and uncertainty band.",
            "Source": "CRM fit/prediction or synthetic trend model",
        },
        {
            "Metric": "value (GCAM)",
            "Unit": "scenario dependent (e.g., EJ, GtCO2)",
            "Definition": "GCAM output value for selected variable/region/year.",
            "Source": "Uploaded GCAM CSV or built-in sample",
        },
        {
            "Metric": "potential_mwe",
            "Unit": "MWe",
            "Definition": "Estimated geothermal power potential at country level.",
            "Source": "Uploaded geothermal potential CSV or built-in sample",
        },
    ]
    return pd.DataFrame(rows)


def generate_synthetic_wells(target_count: int) -> list[WellSite]:
    base = WELLS.copy()
    if target_count <= len(base):
        return base[:target_count]

    rng = np.random.default_rng(20260222)
    expanded = base.copy()
    idx = len(base) + 1
    while len(expanded) < target_count:
        anchor = base[(idx - 1) % len(base)]
        lat = float(anchor.lat + rng.normal(0.18, 0.14))
        lon = float(anchor.lon + rng.normal(0.12, 0.16))
        lat = min(max(lat, 49.8), 58.9)
        lon = min(max(lon, -8.1), 2.0)
        expanded.append(WellSite(f"UK-W{idx:03d}-{anchor.region}", lat, lon, anchor.region))
        idx += 1
    return expanded


def cluster_from_lat_lon(lat: float, lon: float) -> str:
    """Deterministic C1-C4 zoning using only latitude/longitude thresholds."""
    if lat >= 56.0:
        return "C1"
    if lat >= 53.5:
        return "C2"
    if lon < -2.0:
        return "C3"
    return "C4"


def simulate_well_history(site: WellSite, n_days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(site.name)) % (2**32))
    time_index = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    t = np.arange(n_days)

    # Synthetic but stable signals for demo dashboards.
    flow = 60 + 8 * np.sin(t / 11.0) + rng.normal(0, 1.8, n_days)
    pressure = 92 + 6 * np.sin(t / 17.0 + 0.8) + rng.normal(0, 1.2, n_days)
    temperature = 68 + 3.5 * np.sin(t / 22.0 + 1.1) + rng.normal(0, 0.9, n_days)

    # A simple "model forecast" and uncertainty band.
    trend = pd.Series(temperature).rolling(10, min_periods=1).mean().to_numpy()
    pred = trend + rng.normal(0, 0.35, n_days)
    p10 = pred - 1.0
    p90 = pred + 1.0

    df = pd.DataFrame(
        {
            "date": time_index,
            "well": site.name,
            "region": site.region,
            "lat": site.lat,
            "lon": site.lon,
            "flow_m3h": flow,
            "pressure_bar": pressure,
            "actual": temperature,
            "pred": pred,
            "p10": p10,
            "p90": p90,
            "source": "Synthetic UK",
            "cluster": cluster_from_lat_lon(site.lat, site.lon),
        }
    )
    return df


@st.cache_data(show_spinner=False)
def build_synthetic_dataset(target_count: int = 100) -> pd.DataFrame:
    sites = generate_synthetic_wells(target_count)
    return pd.concat([simulate_well_history(site) for site in sites], ignore_index=True)


@st.cache_data(show_spinner=True)
def build_crm_dataset() -> pd.DataFrame:
    filepath = "Datasets/Streak"
    qi = pd.read_excel(join(filepath, "Injection.xlsx"))
    qp = pd.read_excel(join(filepath, "Production.xlsx"))

    qi["Time [days]"] = (qi["Date"] - qi["Date"].iloc[0]) / pd.to_timedelta(1, unit="D")
    t_arr = qi["Time [days]"].values

    inj_list = [c for c in qi.columns if c.startswith("I")]
    prd_list = [c for c in qp.columns if c.startswith("P")]

    qi_arr = qi[inj_list].values
    q_obs = qp[prd_list].values

    n_inj = len(inj_list)
    n_prd = len(prd_list)
    n_train = int(0.7 * len(t_arr))

    tau = np.ones(n_prd)
    gain_mat = np.ones((n_inj, n_prd))
    gain_mat = gain_mat / np.sum(gain_mat, axis=1, keepdims=True)
    qp0 = np.zeros((1, n_prd))
    init = [tau, gain_mat, qp0]

    model = CRMP(init, include_press=False)
    model.fit_model([t_arr[:n_train], qi_arr[:n_train, :]], q_obs[:n_train, :], init)
    pred_all = model.prod_pred([t_arr, qi_arr], train=True)

    train_residual = q_obs[:n_train, :] - pred_all[:n_train, :]
    sigma = np.std(train_residual, axis=0)

    total_inj = np.sum(qi_arr, axis=1)
    records = []
    for i, well in enumerate(prd_list):
        site = CRM_WELLS[i] if i < len(CRM_WELLS) else WellSite(well, 54.0, -2.0, "UK")
        actual = q_obs[:, i]
        pred = pred_all[:, i]
        p10 = pred - 1.28 * sigma[i]
        p90 = pred + 1.28 * sigma[i]
        pressure = 90 + 12 * (actual - np.min(actual)) / (np.ptp(actual) + 1e-6)
        for k in range(len(t_arr)):
            records.append(
                {
                    "date": pd.to_datetime(qi["Date"].iloc[k]),
                    "well": well,
                    "region": site.region,
                    "lat": site.lat,
                    "lon": site.lon,
                    "flow_m3h": total_inj[k],
                    "pressure_bar": pressure[k],
                    "actual": actual[k],
                    "pred": pred[k],
                    "p10": p10[k],
                    "p90": p90[k],
                    "source": "CRM Streak",
                    "cluster": cluster_from_lat_lon(site.lat, site.lon),
                }
            )

    return pd.DataFrame.from_records(records)


def latest_status_table(df: pd.DataFrame, is_crm: bool) -> pd.DataFrame:
    latest = df.sort_values("date").groupby("well").tail(1).copy()
    actual_high = 2500 if is_crm else 74
    latest["alert"] = np.where(
        (latest["pressure_bar"] > 101) | (latest["actual"] > actual_high),
        "HIGH",
        "OK",
    )
    return latest[["well", "region", "cluster", "lat", "lon", "flow_m3h", "pressure_bar", "actual", "alert"]]


def agent_status_block(selected_df: pd.DataFrame) -> pd.DataFrame:
    latest = selected_df.sort_values("date").tail(1).iloc[0]
    drift = abs(float(latest["actual"]) - float(latest["pred"]))
    pressure = float(latest["pressure_bar"])

    status = [
        ("Data Agent", "PASS", "Timestamp aligned, no missing values."),
        ("Model Agent", "PASS", "Daily retrain completed with stable loss."),
        (
            "Physics Agent",
            "WARN" if pressure > 101 else "PASS",
            "Pressure bound check triggered." if pressure > 101 else "Constraints satisfied.",
        ),
        (
            "Forecast Agent",
            "WARN" if drift > 1.3 else "PASS",
            f"Current drift {drift:.2f} C.",
        ),
        ("Decision Agent", "PASS", "Recommendation generated for next control window."),
    ]
    return pd.DataFrame(status, columns=["agent", "status", "note"])


def recommendation_text(selected_df: pd.DataFrame) -> str:
    latest = selected_df.sort_values("date").tail(1).iloc[0]
    pressure = float(latest["pressure_bar"])
    actual = float(latest["actual"])

    if pressure > 101 and actual > 73:
        return "Reduce injection setpoint by 5% and schedule a short cooldown interval."
    if pressure > 101:
        return "Hold current thermal load and reduce pump rate by 3% for one cycle."
    if actual < 65:
        return "Increase thermal extraction target by 4% for the next 24 hours."
    return "Maintain current operating strategy and continue monitoring."


def agent_intelligence(selected_df: pd.DataFrame) -> dict:
    s = selected_df.sort_values("date").tail(21).copy()
    latest = s.iloc[-1]
    prev = s.iloc[:-1] if len(s) > 1 else s

    drift = abs(float(latest["actual"]) - float(latest["pred"]))
    pressure = float(latest["pressure_bar"])
    flow = float(latest["flow_m3h"])
    actual = float(latest["actual"])
    trend = float(s["actual"].tail(7).mean() - s["actual"].head(7).mean()) if len(s) >= 14 else 0.0
    variability = float(np.std(s["actual"] - s["pred"]))
    pressure_jump = float(abs(pressure - float(prev["pressure_bar"].tail(1).iloc[0]))) if len(prev) else 0.0

    score = min(
        100.0,
        25.0 * min(drift, 2.0)
        + 0.8 * max(pressure - 98.0, 0.0)
        + 4.5 * abs(trend)
        + 6.0 * min(variability, 2.0)
        + 1.2 * min(pressure_jump, 10.0),
    )
    if score >= 70:
        risk_level = "HIGH"
    elif score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    causes = []
    if drift > 1.2:
        causes.append("Forecast drift above tolerance")
    if pressure > 101:
        causes.append("Pressure above operating envelope")
    if abs(trend) > 1.0:
        causes.append("Fast signal trend shift")
    if variability > 0.9:
        causes.append("Residual volatility elevated")
    if not causes:
        causes.append("No critical anomaly detected")

    confidence = max(0.55, min(0.98, 0.94 - 0.0035 * score))
    plan = [
        {"step": 1, "owner": "Data Agent", "action": "Validate last 24h completeness and outliers", "eta_min": 2},
        {"step": 2, "owner": "Model Agent", "action": "Run fast rolling retrain on latest window", "eta_min": 4},
        {"step": 3, "owner": "Physics Agent", "action": "Check pressure and constraint violations", "eta_min": 2},
        {"step": 4, "owner": "Forecast Agent", "action": "Publish refreshed P10/P50/P90 band", "eta_min": 2},
        {"step": 5, "owner": "Decision Agent", "action": "Issue control suggestion with rationale", "eta_min": 1},
    ]
    if risk_level == "LOW":
        plan[1]["action"] = "Keep current model; run lightweight drift check"
        plan[3]["action"] = "Maintain forecast and monitor uncertainty"
    if risk_level == "HIGH":
        plan[4]["action"] = "Escalate: reduce setpoint and trigger operator approval"

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "confidence": confidence,
        "drift": drift,
        "pressure": pressure,
        "flow": flow,
        "actual": actual,
        "trend": trend,
        "variability": variability,
        "causes": causes,
        "plan": pd.DataFrame(plan),
    }


def boundary_lines_df() -> pd.DataFrame:
    rows = []
    for name, poly in UK_BOUNDARIES.items():
        for lon, lat in poly:
            rows.append({"country": name, "lon": lon, "lat": lat})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_gcam_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"year", "region", "variable", "value", "unit", "scenario"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["region"] = df["region"].astype(str).str.strip()
    df["variable"] = df["variable"].astype(str).str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    df = df.dropna(subset=["value"])
    return df


@st.cache_data(show_spinner=False)
def load_region_centroids(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"region", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing centroid columns: {sorted(missing)}")
    df = df.copy()
    df["region"] = df["region"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"])


@st.cache_data(show_spinner=False)
def load_region_colors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"region", "color_hex"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing color columns: {sorted(missing)}")
    df = df.copy()
    df["region"] = df["region"].astype(str).str.strip()
    df["color_hex"] = df["color_hex"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_region_iso3(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"region", "iso3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing iso3 columns: {sorted(missing)}")
    df = df.copy()
    df["region"] = df["region"].astype(str).str.strip()
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_geothermal_potential(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"region", "iso3", "potential_mwe"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing geothermal potential columns: {sorted(missing)}")
    out = df.copy()
    out["region"] = out["region"].astype(str).str.strip()
    out["iso3"] = out["iso3"].astype(str).str.strip().str.upper()
    out["potential_mwe"] = pd.to_numeric(out["potential_mwe"], errors="coerce")
    if "category" not in out.columns:
        out["category"] = "Unknown"
    out["category"] = out["category"].astype(str).str.strip()
    return out.dropna(subset=["iso3", "potential_mwe"])


@st.cache_data(show_spinner=False)
def iso3_catalog() -> pd.DataFrame:
    gm = px.data.gapminder()[["country", "iso_alpha"]].drop_duplicates().copy()
    gm = gm.rename(columns={"country": "name", "iso_alpha": "iso3"})
    gm["name_norm"] = gm["name"].str.lower().str.strip()
    return gm


def normalize_region_name(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("&", "and").split())


def suggest_iso3(region: str, catalog: pd.DataFrame) -> tuple[str | None, float, str]:
    aliases = {
        "united states": "USA",
        "usa": "USA",
        "uk": "GBR",
        "united kingdom": "GBR",
        "russia": "RUS",
        "south korea": "KOR",
        "north korea": "PRK",
        "iran": "IRN",
        "viet nam": "VNM",
        "czech republic": "CZE",
        "lao pdr": "LAO",
    }
    r = normalize_region_name(region)
    if r in aliases:
        return aliases[r], 0.99, "alias"
    names = catalog["name_norm"].tolist()
    close = get_close_matches(r, names, n=1, cutoff=0.72)
    if not close:
        return None, 0.0, "none"
    best = close[0]
    score = SequenceMatcher(None, r, best).ratio()
    iso = catalog.loc[catalog["name_norm"] == best, "iso3"].iloc[0]
    return str(iso), float(score), "fuzzy"


def hex_to_rgba(color_hex: str, alpha: int = 220) -> list[int]:
    c = color_hex.strip().lstrip("#")
    if len(c) != 6:
        return [255, 140, 0, alpha]
    try:
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        return [r, g, b, alpha]
    except ValueError:
        return [255, 140, 0, alpha]


def value_to_rgba(value: float, vmin: float, vmax: float, alpha: int = 230) -> list[int]:
    """Continuous blue->cyan->yellow->red ramp for numeric value display."""
    if vmax <= vmin:
        return [239, 68, 68, alpha]
    t = (value - vmin) / (vmax - vmin)
    t = float(max(0.0, min(1.0, t)))
    if t < 0.33:
        # blue -> cyan
        u = t / 0.33
        r = int(30 + 20 * u)
        g = int(90 + 150 * u)
        b = int(220 + 20 * u)
    elif t < 0.66:
        # cyan -> yellow
        u = (t - 0.33) / 0.33
        r = int(50 + 205 * u)
        g = int(240 - 20 * u)
        b = int(240 - 200 * u)
    else:
        # yellow -> red
        u = (t - 0.66) / 0.34
        r = int(255)
        g = int(220 - 150 * u)
        b = int(40 - 20 * u)
    return [r, g, b, alpha]


def render_monitoring_tab(
    filtered: pd.DataFrame,
    selected_wells: list[str],
    is_crm: bool,
    show_boundaries: bool,
    show_interactive_map: bool,
) -> None:
    time_start = filtered["date"].min().strftime("%Y-%m-%d")
    time_end = filtered["date"].max().strftime("%Y-%m-%d")
    source_text = filtered["source"].iloc[0] if "source" in filtered.columns else "Unknown"
    meta_df = pd.DataFrame(
        [
            {
                "Data Source": source_text,
                "Time Range": f"{time_start} to {time_end}",
                "Signal Unit": "proxy / modeled",
                "Scenario": "Operational Monitoring",
            }
        ]
    )
    st.caption("Figure Metadata")
    st.dataframe(meta_df, use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    latest_all = latest_status_table(filtered, is_crm=is_crm)
    c1.metric("Selected Wells", len(selected_wells))
    c2.metric("Mean Signal", f"{latest_all['actual'].mean():.2f}")
    c3.metric("Mean Pressure (bar)", f"{latest_all['pressure_bar'].mean():.2f}")
    c4.metric("High Alerts", int((latest_all["alert"] == "HIGH").sum()))

    st.subheader("UK Well Map (Latest Snapshot)")
    map_df = latest_all.copy()
    map_df["color_hex"] = map_df["cluster"].map(CLUSTER_COLORS_HEX).fillna("#6b7280")
    map_df["color"] = map_df["color_hex"].apply(lambda h: hex_to_rgba(h, 220))
    map_df["line_color"] = map_df["alert"].map({"HIGH": [165, 0, 38, 255], "OK": [20, 20, 20, 190]})
    map_df["line_width"] = map_df["alert"].map({"HIGH": 2.8, "OK": 1.2})
    uk_view = pdk.ViewState(latitude=54.5, longitude=-2.5, zoom=4.6, pitch=0)
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=20000,
        stroked=True,
        get_line_color="line_color",
        get_line_width="line_width",
        line_width_min_pixels=1,
        pickable=True,
    )
    layers = []
    if show_boundaries:
        boundary_data = pd.DataFrame(
            {"name": list(UK_BOUNDARIES.keys()), "polygon": list(UK_BOUNDARIES.values())}
        )
        boundary_layer = pdk.Layer(
            "PolygonLayer",
            data=boundary_data,
            get_polygon="polygon",
            get_line_color=[55, 65, 81],
            get_fill_color=[0, 0, 0, 0],
            line_width_min_pixels=2,
            stroked=True,
            filled=False,
            pickable=False,
        )
        layers.append(boundary_layer)
    layers.append(point_layer)
    tooltip = {
        "html": "<b>{well}</b><br/>Region: {region}<br/>Cluster: {cluster}<br/>Signal: {actual}<br/>Pressure: {pressure_bar}<br/>Alert: {alert}",
        "style": {"backgroundColor": "#111827", "color": "white"},
    }
    if show_interactive_map:
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=uk_view,
                layers=layers,
                tooltip=tooltip,
            ),
            use_container_width=True,
        )

    st.markdown("**Fallback UK Coordinate Plot (always visible)**")
    point_select = alt.selection_point(fields=["well"], name="well_pick", on="click", clear="dblclick")
    x_scale = alt.Scale(domain=[-8.5, 2.2])
    y_scale = alt.Scale(domain=[49.7, 59.0])
    base_points = (
        alt.Chart(map_df)
        .mark_circle(size=140)
        .encode(
            x=alt.X("lon:Q", title="Longitude", scale=x_scale),
            y=alt.Y("lat:Q", title="Latitude", scale=y_scale),
            color=alt.Color("alert:N", scale=alt.Scale(domain=["OK", "HIGH"], range=["#22c55e", "#dc3545"])),
            shape=alt.Shape("cluster:N", title="Cluster"),
            opacity=alt.condition(point_select, alt.value(1.0), alt.value(0.75)),
            tooltip=["well:N", "region:N", "cluster:N", "alert:N", "actual:Q", "pressure_bar:Q"],
        )
        .add_params(point_select)
    )
    if show_boundaries:
        boundary_chart = (
            alt.Chart(boundary_lines_df())
            .mark_line(color="#4b5563", strokeWidth=2)
            .encode(
                x=alt.X("lon:Q", title=None, scale=x_scale),
                y=alt.Y("lat:Q", title=None, scale=y_scale),
                detail="country:N",
            )
        )
        # Streamlit currently disallows on_select for multi-view specs.
        st.altair_chart(boundary_chart.properties(height=170), use_container_width=True)
    selection = st.altair_chart(
        base_points.properties(height=360),
        use_container_width=True,
        on_select="rerun",
    )
    st.dataframe(latest_all.reset_index(drop=True), use_container_width=True)
    cluster_legend = pd.DataFrame(
        [{"Cluster": k, "Zone Logic": CLUSTER_NOTES.get(k, "")} for k in CLUSTER_COLORS_HEX.keys()]
    )
    st.dataframe(cluster_legend, use_container_width=True, hide_index=True)

    st.subheader("Single-Well Detail")
    selected_from_map = None
    if isinstance(selection, dict):
        pick = selection.get("selection", {}).get("well_pick", [])
        if pick and isinstance(pick, list):
            maybe = pick[0].get("well")
            if maybe in selected_wells:
                selected_from_map = maybe

    default_focus_idx = 0
    if selected_from_map is not None:
        default_focus_idx = selected_wells.index(selected_from_map)

    focus_well = st.selectbox("Focus Well", selected_wells, index=default_focus_idx)
    focus = filtered[filtered["well"] == focus_well].sort_values("date")
    intel = agent_intelligence(focus)

    chart_df = focus[["date", "actual", "pred", "p10", "p90"]].set_index("date")
    st.line_chart(chart_df, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.line_chart(focus.set_index("date")[["flow_m3h"]], use_container_width=True)
    m2.line_chart(focus.set_index("date")[["pressure_bar"]], use_container_width=True)
    m3.line_chart(focus.set_index("date")[["actual"]], use_container_width=True)

    st.subheader("Agent Panel")
    st.dataframe(agent_status_block(focus), use_container_width=True, hide_index=True)

    ia, ib, ic, idm = st.columns(4)
    ia.metric("Risk Score", f"{intel['risk_score']:.1f}/100")
    ib.metric("Risk Level", intel["risk_level"])
    ic.metric("Agent Confidence", f"{intel['confidence']:.2f}")
    idm.metric("Drift", f"{intel['drift']:.2f}")

    st.markdown("**Agent Diagnosis**")
    for cause in intel["causes"]:
        st.write(f"- {cause}")

    st.markdown("**Agent Execution Plan**")
    st.dataframe(intel["plan"], use_container_width=True, hide_index=True)

    st.subheader("Decision Suggestion")
    st.info(recommendation_text(focus))


def render_gcam_tab() -> None:
    st.subheader("GCAM Global Explorer")
    st.caption("Interactive global scenario analytics for GCAM-style outputs")

    use_uploaded = st.checkbox("Upload custom GCAM CSV", value=False)
    use_region_palette = st.checkbox("Use GCAM Region Colors", value=False)
    upload_palette = st.checkbox("Upload custom region color mapping (CSV)", value=False)
    upload_iso3 = st.checkbox("Upload custom region->ISO3 mapping (CSV)", value=False)
    gcam_df = None
    if use_uploaded:
        uploaded = st.file_uploader("Upload GCAM CSV", type=["csv"])
        if uploaded is not None:
            try:
                gcam_df = load_gcam_data(uploaded)
            except Exception as exc:
                st.error(f"Failed to read uploaded GCAM file: {exc}")
                return
    else:
        if not GCAM_SAMPLE_PATH.exists():
            st.warning(f"Sample file not found: {GCAM_SAMPLE_PATH}")
            return
        gcam_df = load_gcam_data(str(GCAM_SAMPLE_PATH))

    region_color_df = None
    if upload_palette:
        color_file = st.file_uploader("Upload region color CSV (columns: region,color_hex)", type=["csv"], key="region_colors")
        if color_file is not None:
            try:
                region_color_df = load_region_colors(color_file)
            except Exception as exc:
                st.error(f"Failed to read region color file: {exc}")
                return
    elif GCAM_REGION_COLOR_PATH.exists():
        region_color_df = load_region_colors(str(GCAM_REGION_COLOR_PATH))

    iso3_df = None
    if upload_iso3:
        iso3_file = st.file_uploader("Upload region ISO3 CSV (columns: region,iso3)", type=["csv"], key="region_iso3")
        if iso3_file is not None:
            try:
                iso3_df = load_region_iso3(iso3_file)
            except Exception as exc:
                st.error(f"Failed to read region ISO3 file: {exc}")
                return
    elif GCAM_ISO3_PATH.exists():
        iso3_df = load_region_iso3(str(GCAM_ISO3_PATH))

    scenarios = sorted(gcam_df["scenario"].unique())
    variables = sorted(gcam_df["variable"].unique())
    regions_all = sorted(gcam_df["region"].unique())
    years = sorted(gcam_df["year"].unique())

    g1, g2, g3, g4 = st.columns(4)
    sel_scenarios = g1.multiselect("Scenarios", scenarios, default=scenarios[:2] if len(scenarios) > 1 else scenarios)
    sel_variable = g2.selectbox("Variable", variables, index=0)
    sel_regions = g3.multiselect("Regions", regions_all, default=regions_all[: min(12, len(regions_all))])
    sel_year = g4.selectbox("Ranking Year", years, index=len(years) - 1)
    if not sel_scenarios:
        st.warning("Please select at least one scenario.")
        return
    if not sel_regions:
        st.warning("Please select at least one region.")
        return

    g_filtered = gcam_df[
        (gcam_df["scenario"].isin(sel_scenarios))
        & (gcam_df["variable"] == sel_variable)
        & (gcam_df["region"].isin(sel_regions))
    ].copy()
    if g_filtered.empty:
        st.warning("No GCAM data matches current filters.")
        return

    unit = g_filtered["unit"].iloc[0]
    latest_year = int(g_filtered["year"].max())
    total_latest = g_filtered[g_filtered["year"] == latest_year]["value"].sum()
    r1, r2, r3 = st.columns(3)
    r1.metric("Variable", sel_variable)
    r2.metric(f"Global Total ({latest_year})", f"{total_latest:,.2f} {unit}")
    r3.metric("Scenarios Selected", len(sel_scenarios))

    g_meta = pd.DataFrame(
        [
            {
                "Data Source": "Uploaded GCAM CSV" if use_uploaded else "Built-in GCAM sample",
                "Time Range": f"{int(g_filtered['year'].min())} to {int(g_filtered['year'].max())}",
                "Signal Unit": unit,
                "Scenario": ", ".join(sel_scenarios),
            }
        ]
    )
    st.caption("Figure Metadata")
    st.dataframe(g_meta, use_container_width=True, hide_index=True)

    st.markdown("**Trend by Year**")
    trend = (
        alt.Chart(g_filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("sum(value):Q", title=f"Value ({unit})"),
            color=alt.Color("scenario:N", title="Scenario"),
            tooltip=["scenario:N", "year:O", alt.Tooltip("sum(value):Q", title=f"Value ({unit})")],
        )
        .properties(height=320)
    )
    st.altair_chart(trend, use_container_width=True)

    st.markdown("**Region-Year Heatmap (first selected scenario)**")
    heat_scenario = sel_scenarios[0]
    heat_df = g_filtered[g_filtered["scenario"] == heat_scenario]
    heat = (
        alt.Chart(heat_df)
        .mark_rect()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("region:N", sort="-x", title="Region"),
            color=alt.Color("value:Q", title=unit),
            tooltip=["region:N", "year:O", alt.Tooltip("value:Q", title=f"Value ({unit})")],
        )
        .properties(height=360)
    )
    st.altair_chart(heat, use_container_width=True)

    st.markdown("**Regional Ranking**")
    rank_df = g_filtered[g_filtered["year"] == sel_year].groupby("region", as_index=False)["value"].sum().sort_values("value", ascending=False)
    if use_region_palette and region_color_df is not None:
        rank_df = rank_df.merge(region_color_df, on="region", how="left")
    else:
        rank_df["color_hex"] = "#f59e0b"
    bar = (
        alt.Chart(rank_df.head(20))
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title=f"Value ({unit})"),
            y=alt.Y("region:N", sort="-x", title="Region"),
            color=alt.Color("color_hex:N", scale=None, legend=None),
            tooltip=["region:N", alt.Tooltip("value:Q", title=f"Value ({unit})")],
        )
        .properties(height=420)
    )
    st.altair_chart(bar, use_container_width=True)

    st.markdown("**Country Choropleth (GCAM Values)**")
    choropleth_df = rank_df.copy()
    if "iso3" in g_filtered.columns:
        tmp = g_filtered[g_filtered["year"] == sel_year].groupby(["region", "iso3"], as_index=False)["value"].sum()
        choropleth_df = tmp.copy()
    elif iso3_df is not None:
        choropleth_df = choropleth_df.merge(iso3_df, on="region", how="left")
    else:
        choropleth_df["iso3"] = np.nan

    st.markdown("**Mapping Agent**")
    catalog = iso3_catalog()
    missing_regions = choropleth_df[choropleth_df["iso3"].isna()]["region"].drop_duplicates().tolist()
    sugg_rows = []
    for r in missing_regions:
        iso, conf, mode = suggest_iso3(r, catalog)
        sugg_rows.append(
            {
                "region": r,
                "suggested_iso3": iso if iso is not None else "",
                "confidence": round(conf, 3),
                "method": mode,
                "auto_apply": bool(iso is not None and conf >= 0.86),
            }
        )
    sugg_df = pd.DataFrame(sugg_rows)
    auto_apply = st.checkbox("Auto-apply high-confidence ISO3 suggestions", value=True)
    if not sugg_df.empty:
        st.dataframe(sugg_df, use_container_width=True, hide_index=True)
        csv_bytes = sugg_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Mapping Suggestions CSV",
            data=csv_bytes,
            file_name="gcam_iso3_mapping_suggestions.csv",
            mime="text/csv",
        )
        if auto_apply:
            apply_map = {row["region"]: row["suggested_iso3"] for _, row in sugg_df.iterrows() if row["auto_apply"] and row["suggested_iso3"]}
            if apply_map:
                choropleth_df["iso3"] = choropleth_df.apply(
                    lambda row: apply_map.get(row["region"], row["iso3"]) if pd.isna(row["iso3"]) else row["iso3"],
                    axis=1,
                )
    else:
        st.success("All selected regions already mapped to ISO3.")

    valid_map = choropleth_df.dropna(subset=["iso3"]).copy()
    if not valid_map.empty:
        fig = px.choropleth(
            valid_map,
            locations="iso3",
            color="value",
            hover_name="region",
            hover_data={"value": ":.3f"},
            color_continuous_scale="Viridis",
            projection="natural earth",
            title=f"{sel_variable} ({sel_year}, {heat_scenario if sel_scenarios else ''})",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=45, b=0), coloraxis_colorbar_title=unit)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid ISO3 mapping found. Upload mapping CSV or include `iso3` in your GCAM data.")

    unmapped = choropleth_df[choropleth_df["iso3"].isna()][["region"]].drop_duplicates()
    if not unmapped.empty:
        st.warning(f"Unmapped regions (not shown on country fill map): {', '.join(unmapped['region'].tolist()[:12])}")

    if GCAM_CENTROID_PATH.exists():
        st.markdown("**Global Region Map (Centroid Bubble)**")
        centroid = load_region_centroids(str(GCAM_CENTROID_PATH))
        map_year_df = rank_df.merge(centroid, on="region", how="left").dropna(subset=["lat", "lon"])
        if not map_year_df.empty:
            vmax = float(map_year_df["value"].max() + 1e-9)
            vmin = float(map_year_df["value"].min())
            # Strong visual emphasis: larger bubbles + optional region palette.
            map_year_df["norm"] = map_year_df["value"] / (vmax + 1e-9)
            map_year_df["radius"] = 60000 + 240000 * np.power(map_year_df["norm"], 0.7)
            if use_region_palette and "color_hex" in map_year_df.columns:
                map_year_df["color"] = map_year_df["color_hex"].fillna("#f59e0b").apply(lambda x: hex_to_rgba(x, 230))
            else:
                map_year_df["color"] = map_year_df["value"].apply(lambda x: value_to_rgba(float(x), vmin, vmax, 230))
            map_year_df["rank"] = map_year_df["value"].rank(method="dense", ascending=False).astype(int)
            map_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_year_df,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="color",
                stroked=True,
                get_line_color=[20, 20, 20, 200],
                line_width_min_pixels=1.2,
                pickable=True,
            )
            st.pydeck_chart(
                pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.1),
                    layers=[map_layer],
                    tooltip={
                        "html": "<b>{region}</b><br/>Rank: {rank}<br/>Value: {value}",
                        "style": {"backgroundColor": "#111827", "color": "white"},
                    },
                ),
                use_container_width=True,
            )
            if use_region_palette and "color_hex" in map_year_df.columns:
                legend_df = map_year_df[["region", "color_hex"]].drop_duplicates().head(20)
                st.dataframe(legend_df, use_container_width=True, hide_index=True)
            else:
                st.markdown("**Numeric Color Scale**")
                legend_vals = np.linspace(vmin, vmax, 120)
                legend_df = pd.DataFrame({"value": legend_vals, "strip": ["scale"] * len(legend_vals)})
                legend_chart = (
                    alt.Chart(legend_df)
                    .mark_rect()
                    .encode(
                        x=alt.X("value:Q", title=f"Value ({unit})"),
                        y=alt.Y("strip:N", title=None, axis=None),
                        color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis"), legend=None),
                    )
                    .properties(height=36)
                )
                st.altair_chart(legend_chart, use_container_width=True)
                s1, s2, s3 = st.columns(3)
                s1.metric("Min", f"{vmin:,.2f} {unit}")
                s2.metric("Median", f"{float(np.median(map_year_df['value'])):,.2f} {unit}")
                s3.metric("Max", f"{vmax:,.2f} {unit}")
        else:
            st.info("No centroid match found for currently selected regions.")

    st.markdown("**Global Geothermal Potential**")
    use_uploaded_geo_potential = st.checkbox("Upload custom geothermal potential CSV", value=False)
    potential_df = None
    if use_uploaded_geo_potential:
        uploaded_pot = st.file_uploader(
            "Upload geothermal potential CSV (columns: region,iso3,potential_mwe[,category])",
            type=["csv"],
            key="geo_potential",
        )
        if uploaded_pot is not None:
            try:
                potential_df = load_geothermal_potential(uploaded_pot)
            except Exception as exc:
                st.error(f"Failed to read geothermal potential file: {exc}")
                return
    else:
        if GEOTHERMAL_POTENTIAL_PATH.exists():
            potential_df = load_geothermal_potential(str(GEOTHERMAL_POTENTIAL_PATH))

    if potential_df is None or potential_df.empty:
        st.info("No geothermal potential data loaded.")
    else:
        cat_list = sorted(potential_df["category"].dropna().unique().tolist())
        selected_cat = st.multiselect("Potential Category", cat_list, default=cat_list)
        pot_view = potential_df[potential_df["category"].isin(selected_cat)] if selected_cat else potential_df.copy()

        if pot_view.empty:
            st.info("No geothermal potential records match selected category.")
        else:
            fig_pot = px.choropleth(
                pot_view,
                locations="iso3",
                color="potential_mwe",
                hover_name="region",
                hover_data={"potential_mwe": ":,.0f", "iso3": False, "category": True},
                color_continuous_scale="YlOrRd",
                projection="natural earth",
                title="Estimated Geothermal Potential (MWe)",
            )
            fig_pot.update_layout(margin=dict(l=0, r=0, t=45, b=0), coloraxis_colorbar_title="MWe")
            st.plotly_chart(fig_pot, use_container_width=True)

            top_pot = pot_view.sort_values("potential_mwe", ascending=False).head(15)
            bar_pot = (
                alt.Chart(top_pot)
                .mark_bar()
                .encode(
                    x=alt.X("potential_mwe:Q", title="Potential (MWe)"),
                    y=alt.Y("region:N", sort="-x", title="Region"),
                    color=alt.Color("potential_mwe:Q", scale=alt.Scale(scheme="yelloworangered"), legend=None),
                    tooltip=["region:N", "iso3:N", alt.Tooltip("potential_mwe:Q", title="MWe"), "category:N"],
                )
                .properties(height=360)
            )
            st.altair_chart(bar_pot, use_container_width=True)

            p1, p2, p3 = st.columns(3)
            p1.metric("Countries", int(pot_view["iso3"].nunique()))
            p2.metric("Total Potential (MWe)", f"{float(pot_view['potential_mwe'].sum()):,.0f}")
            p3.metric("Max Country (MWe)", f"{float(pot_view['potential_mwe'].max()):,.0f}")


def main() -> None:
    st.set_page_config(page_title="XIRANG Demo", layout="wide")
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

    st.sidebar.header(tr("lang"))
    st.session_state["lang"] = st.sidebar.radio(
        tr("lang"),
        options=["en", "zh"],
        format_func=lambda x: "English" if x == "en" else "中文",
        index=0 if st.session_state.get("lang", "en") == "en" else 1,
        label_visibility="collapsed",
    )
    with st.sidebar.expander(tr("metrics_dict"), expanded=False):
        st.dataframe(metrics_dictionary_df(), use_container_width=True, hide_index=True)

    banner_html = """
    <style>
    .xirang-title {
        font-size: 3.6rem;
        font-weight: 800;
        line-height: 1.08;
        margin: 0.15rem 0 0.35rem 0;
        color: #1f2937;
        letter-spacing: 0.5px;
    }
    .xirang-subtitle {
        font-size: 1.42rem;
        font-weight: 600;
        color: #334155;
        margin-top: 0.15rem;
        margin-bottom: 1.1rem;
        line-height: 1.45;
        max-width: 980px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.28rem;
        font-weight: 700;
        min-height: 52px;
        padding-top: 0.28rem;
        padding-bottom: 0.28rem;
    }
    </style>
    <div class="xirang-title">XIRANG (息壤)</div>
    <div class="xirang-subtitle">%s</div>
    """ % tr("subtitle")
    st.markdown(banner_html, unsafe_allow_html=True)

    st.sidebar.header(tr("data_source"))
    source = st.sidebar.radio(tr("choose_source"), options=["Synthetic UK", "CRM Streak"], index=0)
    is_crm = source == "CRM Streak"

    synthetic_well_count = 100
    if not is_crm:
        synthetic_well_count = st.sidebar.slider(tr("synthetic_well_count"), min_value=50, max_value=400, value=200, step=25)

    df = build_crm_dataset() if is_crm else build_synthetic_dataset(synthetic_well_count)
    wells = sorted(df["well"].unique())
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    st.sidebar.header(tr("dashboard_filters"))
    region_mode = st.sidebar.radio(tr("region_grouping"), options=[tr("grouping_c"), tr("grouping_admin")], index=0)
    is_c_group = region_mode == tr("grouping_c")
    groups = sorted(df["cluster"].unique()) if is_c_group else sorted(df["region"].unique())
    selected_groups = st.sidebar.multiselect(tr("regions"), options=groups, default=groups)
    search = st.sidebar.text_input(tr("well_search"), value="").strip().lower()
    if is_c_group:
        candidate_wells = [w for w in wells if df[df["well"] == w]["cluster"].iloc[0] in selected_groups]
    else:
        candidate_wells = [w for w in wells if df[df["well"] == w]["region"].iloc[0] in selected_groups]
    if search:
        candidate_wells = [w for w in candidate_wells if search in w.lower()]
    default_wells = candidate_wells
    selected_wells = st.sidebar.multiselect(tr("wells"), options=candidate_wells, default=default_wells)
    date_range = st.sidebar.date_input(tr("date_range"), value=(min_date, max_date), min_value=min_date, max_value=max_date)
    show_boundaries = st.sidebar.checkbox(tr("show_boundaries"), value=True)
    show_interactive_map = st.sidebar.checkbox(tr("show_interactive"), value=True)

    tab_monitor, tab_gcam = st.tabs([tr("tab_monitor"), tr("tab_gcam")])

    with tab_monitor:
        if not selected_wells:
            st.warning(tr("need_one_well"))
        elif not isinstance(date_range, tuple) or len(date_range) != 2:
            st.warning(tr("need_valid_date"))
        else:
            start_date, end_date = date_range
            mask = (
                df["well"].isin(selected_wells)
                & (df["date"].dt.date >= start_date)
                & (df["date"].dt.date <= end_date)
            )
            filtered = df.loc[mask].copy()
            if filtered.empty:
                st.warning(tr("no_data_range"))
            else:
                render_monitoring_tab(
                    filtered=filtered,
                    selected_wells=selected_wells,
                    is_crm=is_crm,
                    show_boundaries=show_boundaries,
                    show_interactive_map=show_interactive_map,
                )

    with tab_gcam:
        render_gcam_tab()


if __name__ == "__main__":
    main()
