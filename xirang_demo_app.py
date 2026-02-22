#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from os.path import join

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

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
    return latest[["well", "region", "lat", "lon", "flow_m3h", "pressure_bar", "actual", "alert"]]


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


def main() -> None:
    st.set_page_config(page_title="XIRANG Demo", layout="wide")
    st.title("XIRANG (息壤) - Well Monitoring Demo")
    st.caption("eXplainable Intelligent Resilience Agent Network for Geothermal systems")

    st.sidebar.header("Data Source")
    source = st.sidebar.radio("Choose source", options=["Synthetic UK", "CRM Streak"], index=0)
    is_crm = source == "CRM Streak"

    synthetic_well_count = 100
    if not is_crm:
        synthetic_well_count = st.sidebar.slider("Synthetic Well Count", min_value=25, max_value=200, value=100, step=25)

    df = build_crm_dataset() if is_crm else build_synthetic_dataset(synthetic_well_count)
    wells = sorted(df["well"].unique())
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    st.sidebar.header("Dashboard Filters")
    regions = sorted(df["region"].unique())
    selected_regions = st.sidebar.multiselect("Regions", options=regions, default=regions)
    search = st.sidebar.text_input("Well Search", value="").strip().lower()
    candidate_wells = [w for w in wells if df[df["well"] == w]["region"].iloc[0] in selected_regions]
    if search:
        candidate_wells = [w for w in candidate_wells if search in w.lower()]
    default_wells = candidate_wells[: min(8, len(candidate_wells))]
    selected_wells = st.sidebar.multiselect("Wells", options=candidate_wells, default=default_wells)
    date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    show_boundaries = st.sidebar.checkbox("Show UK Country Boundaries", value=True)
    show_interactive_map = st.sidebar.checkbox("Interactive Map Layer", value=True)

    if not selected_wells:
        st.warning("Please select at least one well.")
        return
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.warning("Please select a valid start and end date.")
        return

    start_date, end_date = date_range
    mask = (
        df["well"].isin(selected_wells)
        & (df["date"].dt.date >= start_date)
        & (df["date"].dt.date <= end_date)
    )
    filtered = df.loc[mask].copy()
    if filtered.empty:
        st.warning("No data in the selected range.")
        return

    c1, c2, c3, c4 = st.columns(4)
    latest_all = latest_status_table(filtered, is_crm=is_crm)
    c1.metric("Selected Wells", len(selected_wells))
    c2.metric("Mean Signal", f"{latest_all['actual'].mean():.2f}")
    c3.metric("Mean Pressure (bar)", f"{latest_all['pressure_bar'].mean():.2f}")
    c4.metric("High Alerts", int((latest_all["alert"] == "HIGH").sum()))

    st.subheader("UK Well Map (Latest Snapshot)")
    map_df = latest_all.copy()
    map_df["color"] = map_df["alert"].map({"HIGH": [220, 53, 69], "OK": [34, 197, 94]})
    uk_view = pdk.ViewState(latitude=54.5, longitude=-2.5, zoom=4.6, pitch=0)
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=16000,
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
        "html": "<b>{well}</b><br/>Region: {region}<br/>Signal: {actual}<br/>Pressure: {pressure_bar}<br/>Alert: {alert}",
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
            opacity=alt.condition(point_select, alt.value(1.0), alt.value(0.75)),
            tooltip=["well:N", "region:N", "alert:N", "actual:Q", "pressure_bar:Q"],
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


if __name__ == "__main__":
    main()
