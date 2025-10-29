from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path

from .loaders import load_lap_times, load_lap_starts, load_lap_ends, load_weather, build_lap_table


st.set_page_config(page_title="GR Cup Strategy", layout="wide")
st.title("GR Cup Real-Time Strategy & Analytics")

base_dir = Path(__file__).resolve().parents[2]

race = st.sidebar.selectbox("Race", ["Race 1", "Race 2"], index=0)
race_dir = base_dir / race

lap_time_file = next((race_dir.glob("vir_lap_time_*.csv")), None)
lap_start_file = next((race_dir.glob("vir_lap_start_*.csv")), None)
lap_end_file = next((race_dir.glob("vir_lap_end_*.csv")), None)
weather_file = next((race_dir.glob("26_Weather_*.CSV")), None)

if not all([lap_time_file, lap_start_file, lap_end_file]):
    st.error("Lap timing files not found. Ensure CSVs exist in Race folders.")
    st.stop()

lt = load_lap_times(lap_time_file)
ls = load_lap_starts(lap_start_file)
le = load_lap_ends(lap_end_file)
laps = build_lap_table(lt, ls, le)

st.subheader("Lap Summary")
st.dataframe(laps.head(200))

st.subheader("Per-Vehicle Pace")
vehicle_ids = sorted([v for v in laps["vehicle_id"].dropna().unique() if isinstance(v, str)])
selected = st.multiselect("Vehicles", vehicle_ids, default=vehicle_ids[:3])

chart_df = (
    laps.dropna(subset=["lap", "lap_time_ms", "vehicle_id"]) 
        .query("vehicle_id in @selected")
        .copy()
)
chart_df["lap_time_s"] = chart_df["lap_time_ms"] / 1000.0
chart_df = chart_df.sort_values(["vehicle_id", "lap"]) 

st.line_chart(chart_df.pivot_table(index="lap", columns="vehicle_id", values="lap_time_s"))

if weather_file is not None:
    st.subheader("Weather Snapshot")
    wx = load_weather(weather_file)
    st.dataframe(wx.head(200))


