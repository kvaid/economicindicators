import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

BASE_DIR = Path(__file__).resolve().parent

SERIES = {
    "1Y": "BC_1YEAR",
    "2Y": "BC_2YEAR",
    "5Y": "BC_5YEAR",
    "10Y": "BC_10YEAR",
    "30Y": "BC_30YEAR",
}
MATURITY_PRESETS = list(SERIES.keys())

PRESETS = ["1W", "YTD", "1M", "3M", "6M", "1Y", "5Y", "10Y"]
YIELD_COLORS = {
    "1Y": "#B7D7F5",
    "2Y": "#8EBFEA",
    "5Y": "#5B9EDB",
    "10Y": "#2E73B8",
    "30Y": "#124A80",
}
REFRESH_SCRIPTS = [
    "download_fedrate.py",
    "download_inflation.py",
    "download_treasury_data.py",
    "download_unemployment.py",
]

refresh_lock = threading.Lock()
refresh_state = {
    "running": False,
    "done": False,
    "ok": True,
    "message": "",
    "progress": 0,
}


def load_and_process_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    csv_path = BASE_DIR / path
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, parse_dates=[date_col])
        df = df.rename(columns={date_col: "DATE"}).sort_values("DATE")
        return df
    except Exception:
        return pd.DataFrame()


def align_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["DATE"] = df["DATE"] + pd.offsets.MonthEnd(0)
    return df.sort_values("DATE").reset_index(drop=True)


def align_to_treasury_daily_calendar(
    indicator_df: pd.DataFrame,
    value_cols: list[str],
    treasury_dates: pd.Series,
) -> pd.DataFrame:
    if indicator_df.empty:
        return indicator_df

    base = indicator_df[["DATE"] + value_cols].copy()
    base = base.sort_values("DATE").drop_duplicates(subset=["DATE"]).set_index("DATE")

    daily_index = pd.DatetimeIndex(pd.Series(treasury_dates).dropna().sort_values().unique())
    aligned = base.reindex(daily_index).ffill()
    aligned.index.name = "DATE"
    return aligned.reset_index()


def filter_by_date(df: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["DATE"].dt.date >= start) & (df["DATE"].dt.date <= end)
    return df.loc[mask].copy()


def _set_refresh_state(**kwargs) -> None:
    with refresh_lock:
        refresh_state.update(kwargs)


def _refresh_worker() -> None:
    total = len(REFRESH_SCRIPTS)
    ok = True
    _set_refresh_state(running=True, done=False, ok=True, message="Starting refresh...", progress=0)

    for i, script in enumerate(REFRESH_SCRIPTS, start=1):
        _set_refresh_state(message=f"Running {script}...", progress=int(((i - 1) / total) * 100))
        try:
            subprocess.run(
                [sys.executable, str(BASE_DIR / script)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            ok = False
            err_lines = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
            summary = err_lines[-1] if err_lines else str(exc)
            _set_refresh_state(
                message=f"Error in {script}: {summary}",
                progress=int((i / total) * 100),
                ok=False,
            )
        else:
            _set_refresh_state(
                message=f"Completed {script}",
                progress=int((i / total) * 100),
            )

    final_message = "Refresh complete." if ok else "Refresh complete with errors."
    _set_refresh_state(running=False, done=True, ok=ok, message=final_message, progress=100)


def start_refresh_worker() -> bool:
    with refresh_lock:
        if refresh_state["running"]:
            return False
        refresh_state.update({"running": True, "done": False, "ok": True, "message": "Starting refresh...", "progress": 0})
    worker = threading.Thread(target=_refresh_worker, daemon=True)
    worker.start()
    return True


def get_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    ust_df = load_and_process_csv("ust.csv")
    if ust_df.empty:
        now = pd.Timestamp.today().normalize()
        return now, now
    return ust_df["DATE"].min(), ust_df["DATE"].max()


def compute_preset_range(preset: str, min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple[str, str]:
    now = pd.Timestamp(max_date)
    if preset == "1W":
        start = now - pd.DateOffset(weeks=1)
    elif preset == "1M":
        start = now - pd.DateOffset(months=1)
    elif preset == "YTD":
        start = now.replace(month=1, day=1)
    elif preset == "3M":
        start = now - pd.DateOffset(months=3)
    elif preset == "6M":
        start = now - pd.DateOffset(months=6)
    elif preset == "1Y":
        start = now - pd.DateOffset(years=1)
    elif preset == "5Y":
        start = now - pd.DateOffset(years=5)
    elif preset == "10Y":
        start = now - pd.DateOffset(years=10)
    else:
        start = now

    start = max(pd.Timestamp(min_date), start)
    return start.date().isoformat(), pd.Timestamp(max_date).date().isoformat()


def date_to_timeline_idx(date_value: object, min_date: pd.Timestamp, max_date: pd.Timestamp) -> int:
    dt = pd.Timestamp(date_value)
    dt = max(min_date, min(max_date, dt))
    return int((dt - min_date).days)


def timeline_idx_to_date(idx: int, min_date: pd.Timestamp, max_date: pd.Timestamp) -> str:
    total_days = max((max_date - min_date).days, 1)
    idx = int(max(0, min(total_days, idx)))
    return (min_date + pd.Timedelta(days=idx)).date().isoformat()


def build_timeline_marks(min_date: pd.Timestamp, max_date: pd.Timestamp) -> dict[int, str]:
    total_days = max((max_date - min_date).days, 1)
    marks = {0: min_date.strftime("%Y"), total_days: max_date.strftime("%Y")}
    step = 10
    for year in range(min_date.year + 1, max_date.year):
        if year % step != 0:
            continue
        year_start = pd.Timestamp(year=year, month=1, day=1)
        marks[int((year_start - min_date).days)] = str(year)
    return dict(sorted(marks.items()))


def build_figure(
    plot_ust: pd.DataFrame,
    plot_fed: pd.DataFrame,
    plot_infl: pd.DataFrame,
    plot_unrate: pd.DataFrame,
    selected_maturities: list[str],
    show_yields: bool,
    show_spread: bool,
    show_fed_rate: bool,
    show_inflation: bool,
    show_unemployment: bool,
    show_u6_unemployment: bool,
    show_unemp_ind: bool,
) -> go.Figure:
    all_vals: list[float] = []

    if show_yields and selected_maturities:
        for maturity in selected_maturities:
            all_vals.extend(plot_ust[SERIES[maturity]].dropna().tolist())
    if show_spread and not plot_ust.empty:
        all_vals.extend(plot_ust["SPREAD_10Y_2Y"].dropna().tolist())
    if show_inflation and not plot_infl.empty:
        all_vals.extend(plot_infl["PCE_YoY"].dropna().tolist())
    if show_unemployment and not plot_unrate.empty:
        all_vals.extend(plot_unrate["UNRATE"].dropna().tolist())
    if show_u6_unemployment and not plot_unrate.empty and "U6RATE" in plot_unrate.columns:
        all_vals.extend(plot_unrate["U6RATE"].dropna().tolist())
    if show_unemp_ind and not plot_unrate.empty:
        all_vals.extend(plot_unrate["UNEMP_INDICATOR"].dropna().tolist())
    if show_fed_rate and not plot_fed.empty:
        all_vals.extend(plot_fed["FED_RATE"].dropna().tolist())

    if all_vals:
        v_min, v_max = min(all_vals), max(all_vals)
        # Force zero to remain visible in all chart states.
        v_min = min(v_min, 0.0)
        v_max = max(v_max, 0.0)
        pad = 0.05 * (v_max - v_min) if v_max != v_min else 1.0
        y_range = [v_min - pad, v_max + pad]
    else:
        y_range = [0, 10]

    fig = go.Figure()

    if show_yields:
        for maturity in selected_maturities:
            fig.add_trace(
                go.Scatter(
                    x=plot_ust["DATE"],
                    y=plot_ust[SERIES[maturity]],
                    name=f"Yield: {maturity}",
                    mode="lines",
                    line={"color": YIELD_COLORS.get(maturity, "#2E73B8"), "width": 2.4},
                    hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
                )
            )

    if show_fed_rate and not plot_fed.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_fed["DATE"],
                y=plot_fed["FED_RATE"],
                name="Fed Rate",
                mode="lines",
                line={"color": "#1B8F3A", "width": 2.4, "dash": "dot"},
                line_shape="hv",
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
        changes = plot_fed[plot_fed["rate_changed"]]
        if not changes.empty:
            fig.add_trace(
                go.Scatter(
                    x=changes["DATE"],
                    y=changes["FED_RATE"],
                    mode="markers",
                    marker={"color": "green", "size": 6},
                    showlegend=False,
                )
            )

    if show_spread and not plot_ust.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_ust["DATE"],
                y=plot_ust["SPREAD_10Y_2Y"],
                name="10Y-2Y Spread",
                line={"color": "#0B3A63", "width": 2, "dash": "dot"},
                yaxis="y2",
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )

    if show_inflation and not plot_infl.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_infl["DATE"],
                y=plot_infl["PCE_YoY"],
                name="Core PCE Inflation",
                line={"color": "#FF1F1F", "width": 2.4, "dash": "dot"},
                yaxis="y2",
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )

    if show_unemployment and not plot_unrate.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_unrate["DATE"],
                y=plot_unrate["UNRATE"],
                name="Unemployment Rate",
                line={"color": "#F28C28", "width": 2, "dash": "dot"},
                connectgaps=True,
                yaxis="y2",
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )

    if show_u6_unemployment and not plot_unrate.empty and "U6RATE" in plot_unrate.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_unrate["DATE"],
                y=plot_unrate["U6RATE"],
                name="U-6 Unemployment Rate",
                line={"color": "#8B5A2B", "width": 2, "dash": "dot"},
                connectgaps=True,
                yaxis="y2",
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )

    if show_unemp_ind and not plot_unrate.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_unrate["DATE"],
                y=plot_unrate["UNEMP_INDICATOR"],
                name="Unemp. Indicator (U3-NROU)",
                line={"color": "#8B0000", "width": 2, "dash": "dot"},
                connectgaps=True,
                yaxis="y2",
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        font={"family": "Plus Jakarta Sans, Segoe UI, Arial, sans-serif", "size": 13, "color": "#1f2a37"},
        xaxis_title="Date",
        xaxis={
            "showgrid": True,
            "gridcolor": "rgba(15, 23, 42, 0.08)",
            "linecolor": "rgba(15, 23, 42, 0.25)",
        },
        yaxis={
            "title": "Yield (%)",
            "side": "left",
            "range": y_range,
            "tickformat": ".1f",
            "showgrid": True,
            "gridcolor": "rgba(15, 23, 42, 0.08)",
            "zeroline": True,
            "zerolinecolor": "rgba(15, 23, 42, 0.28)",
            "linecolor": "rgba(15, 23, 42, 0.25)",
        },
        yaxis2={
            "title": "%",
            "side": "right",
            "overlaying": "y",
            "showgrid": False,
            "range": y_range,
            "tickformat": ".1f",
            "linecolor": "rgba(15, 23, 42, 0.25)",
        },
        hovermode="x unified",
        hoverlabel={"bgcolor": "white", "font_size": 12, "font_family": "Plus Jakarta Sans, Segoe UI, Arial"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "rgba(15, 23, 42, 0.15)",
            "borderwidth": 1,
        },
        height=650,
        margin={"t": 72, "b": 40, "l": 52, "r": 52},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(15,23,42,0.45)", opacity=0.7, yref="y2")

    return fig


def latest_non_null_value(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def indicator_card(label: str, value: float | None) -> html.Div:
    value_txt = "N/A" if value is None else f"{value:.2f}%"
    return html.Div(
        [
            html.Div(label, className="indicator-label"),
            html.Div(value_txt, className="indicator-value"),
        ],
        className="indicator-card",
    )


min_dt, max_dt = get_date_bounds()
timeline_min_dt = max(min_dt, pd.to_datetime("2000-01-01"))
default_start = max(timeline_min_dt, pd.to_datetime("2020-01-01"))
timeline_total_days = max((max_dt - timeline_min_dt).days, 1)
default_slider_range = [
    date_to_timeline_idx(default_start, timeline_min_dt, max_dt),
    timeline_total_days,
]
timeline_marks = build_timeline_marks(timeline_min_dt, max_dt)
dataset_range_text = f"Downloaded range: {min_dt.strftime('%b-%Y')} to {max_dt.strftime('%b-%Y')}"

app = Dash(__name__)
app.title = "Key Economic Indicators"

app.layout = html.Div(
    [
        dcc.Store(id="refresh-token", data=0),
        dcc.Store(id="active-preset", data=None),
        dcc.Store(id="selected-maturities", data=["10Y"]),
        dcc.Interval(id="refresh-progress-interval", interval=600, n_intervals=0, disabled=True),
        html.Div(
            [
                html.H2("Controls", className="sidebar-title"),
                dcc.Checklist(
                    id="show-yields",
                    options=[{"label": "Treasury Yields", "value": "on"}],
                    value=["on"],
                    className="control-group",
                ),
                html.Div(
                    [html.Button(m, id=f"maturity-{m}", n_clicks=0, className="maturity-btn") for m in MATURITY_PRESETS],
                    className="maturity-grid",
                ),
                dcc.Checklist(
                    id="show-spread",
                    options=[{"label": "10Y-2Y Spread", "value": "on"}],
                    value=[],
                    className="control-group",
                ),
                dcc.Checklist(
                    id="show-fed-rate",
                    options=[{"label": "Federal Reserve Rate", "value": "on"}],
                    value=["on"],
                    className="control-group",
                ),
                dcc.Checklist(
                    id="show-inflation",
                    options=[{"label": "Core PCE Inflation", "value": "on"}],
                    value=["on"],
                    className="control-group",
                ),
                dcc.Checklist(
                    id="show-unemployment",
                    options=[{"label": "U-3 Unemployment Rate", "value": "on"}],
                    value=[],
                    className="control-group",
                ),
                dcc.Checklist(
                    id="show-u6-unemployment",
                    options=[{"label": "U-6 Unemployment Rate", "value": "on"}],
                    value=[],
                    className="control-group",
                ),
                dcc.Checklist(
                    id="show-unemp-ind",
                    options=[{"label": "Unemployment Indicator (U3-NROU)", "value": "on"}],
                    value=[],
                    className="control-group",
                ),
                html.Hr(className="divider"),
                html.Div("Manual Date Entry", className="control-label"),
                html.Div(
                    [
                        dcc.DatePickerSingle(
                            id="start-date",
                            min_date_allowed=timeline_min_dt.date(),
                            max_date_allowed=max_dt.date(),
                            date=default_start.date(),
                            display_format="YYYY-MM-DD",
                            className="date-single",
                        ),
                        dcc.DatePickerSingle(
                            id="end-date",
                            min_date_allowed=timeline_min_dt.date(),
                            max_date_allowed=max_dt.date(),
                            date=max_dt.date(),
                            display_format="YYYY-MM-DD",
                            className="date-single",
                        ),
                    ],
                    className="date-range",
                ),
                dcc.RangeSlider(
                    id="timeline-slider",
                    min=0,
                    max=timeline_total_days,
                    step=1,
                    value=default_slider_range,
                    marks=timeline_marks,
                    allowCross=False,
                    className="timeline-slider",
                ),
                html.Div(
                    [html.Button(p, id=f"preset-{p}", n_clicks=0, className="preset-btn") for p in PRESETS],
                    className="preset-grid",
                ),
                html.Div(id="freq-label", className="freq-label"),
                html.Hr(className="divider"),
                html.Div(dataset_range_text, id="dataset-range-label", className="freq-label"),
                html.Button("Refresh Data", id="refresh-btn", n_clicks=0, className="primary-btn"),
                html.Div(
                    [
                        html.Div(id="refresh-progress-msg", className="refresh-status"),
                        html.Progress(id="refresh-progress-bar", value="0", max=100, className="refresh-progress-bar"),
                    ],
                    id="refresh-progress-wrap",
                    style={"display": "none"},
                ),
                html.Hr(className="divider"),
                html.Div("(c) Kushagra Vaid 2026", className="copyright"),
            ],
            className="sidebar",
        ),
        html.Div(
            [
                html.H1("Key Economic Indicators", className="page-title"),
                html.Div(id="warning-message", className="warning-message"),
                html.Div(id="latest-indicators", className="latest-indicators"),
                dcc.Graph(id="indicator-graph"),
            ],
            className="main-content",
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("start-date", "date"),
    Output("end-date", "date"),
    Output("timeline-slider", "value"),
    Output("active-preset", "data"),
    [Input(f"preset-{p}", "n_clicks") for p in PRESETS],
    Input("start-date", "date"),
    Input("end-date", "date"),
    Input("timeline-slider", "value"),
    prevent_initial_call=True,
)
def apply_preset(*args):
    trigger = callback_context.triggered_id
    if not trigger:
        return no_update, no_update, no_update, no_update

    start_date_str = args[-3]
    end_date_str = args[-2]
    slider_range = args[-1]

    if str(trigger).startswith("preset-"):
        preset = str(trigger).replace("preset-", "")
        start_date, end_date = compute_preset_range(preset, timeline_min_dt, max_dt)
        slider_value = [
            date_to_timeline_idx(start_date, timeline_min_dt, max_dt),
            date_to_timeline_idx(end_date, timeline_min_dt, max_dt),
        ]
        return start_date, end_date, slider_value, preset

    if trigger == "timeline-slider":
        if not slider_range or len(slider_range) != 2:
            return no_update, no_update, no_update, no_update
        start_idx, end_idx = sorted([int(slider_range[0]), int(slider_range[1])])
        start_date = timeline_idx_to_date(start_idx, timeline_min_dt, max_dt)
        end_date = timeline_idx_to_date(end_idx, timeline_min_dt, max_dt)
        return start_date, end_date, [start_idx, end_idx], None

    if trigger in ("start-date", "end-date"):
        if not start_date_str or not end_date_str:
            return no_update, no_update, no_update, no_update
        start_dt = max(timeline_min_dt, min(max_dt, pd.Timestamp(start_date_str)))
        end_dt = max(timeline_min_dt, min(max_dt, pd.Timestamp(end_date_str)))
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
        slider_value = [
            date_to_timeline_idx(start_dt, timeline_min_dt, max_dt),
            date_to_timeline_idx(end_dt, timeline_min_dt, max_dt),
        ]
        return start_dt.date().isoformat(), end_dt.date().isoformat(), slider_value, None

    return no_update, no_update, no_update, no_update


@app.callback(
    [Output(f"preset-{p}", "className") for p in PRESETS],
    Input("active-preset", "data"),
)
def update_preset_button_styles(active_preset):
    return [
        "preset-btn preset-btn-active" if preset == active_preset else "preset-btn"
        for preset in PRESETS
    ]


@app.callback(
    Output("selected-maturities", "data"),
    [Input(f"maturity-{m}", "n_clicks") for m in MATURITY_PRESETS],
    State("selected-maturities", "data"),
    prevent_initial_call=True,
)
def update_selected_maturities(*args):
    selected = list(args[-1] or [])
    trigger = callback_context.triggered_id
    if not trigger:
        return selected

    maturity = str(trigger).replace("maturity-", "")
    if maturity in selected:
        selected.remove(maturity)
    else:
        selected.append(maturity)

    ordered = [m for m in MATURITY_PRESETS if m in selected]
    return ordered or ["10Y"]


@app.callback(
    [Output(f"maturity-{m}", "className") for m in MATURITY_PRESETS],
    Input("selected-maturities", "data"),
)
def update_maturity_button_styles(selected):
    selected_set = set(selected or [])
    return [
        "maturity-btn maturity-btn-active" if maturity in selected_set else "maturity-btn"
        for maturity in MATURITY_PRESETS
    ]


@app.callback(
    Output("refresh-progress-msg", "children"),
    Output("refresh-progress-bar", "value"),
    Output("refresh-progress-wrap", "style"),
    Output("refresh-progress-interval", "disabled"),
    Output("refresh-token", "data"),
    Output("dataset-range-label", "children"),
    Input("refresh-btn", "n_clicks"),
    Input("refresh-progress-interval", "n_intervals"),
    State("refresh-token", "data"),
    prevent_initial_call=True,
)
def handle_refresh(n_clicks: int, _n_intervals: int, token: int):
    trigger = callback_context.triggered_id

    if trigger == "refresh-btn":
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update, no_update
        started = start_refresh_worker()
        if not started:
            return "Refresh already running...", no_update, {"display": "block"}, False, no_update, no_update
        return "Starting refresh...", "0", {"display": "block"}, False, no_update, no_update

    with refresh_lock:
        state = dict(refresh_state)

    if state["running"]:
        return state["message"], str(state["progress"]), {"display": "block"}, False, no_update, no_update

    if state["done"]:
        with refresh_lock:
            refresh_state["done"] = False
        refreshed_min, refreshed_max = get_date_bounds()
        range_text = f"Downloaded range: {refreshed_min.strftime('%b-%Y')} to {refreshed_max.strftime('%b-%Y')}"
        return state["message"], str(state["progress"]), {"display": "none"}, True, (token or 0) + 1, range_text

    return no_update, no_update, {"display": "none"}, True, no_update, no_update


@app.callback(
    Output("indicator-graph", "figure"),
    Output("latest-indicators", "children"),
    Output("freq-label", "children"),
    Output("warning-message", "children"),
    Input("show-yields", "value"),
    Input("selected-maturities", "data"),
    Input("show-spread", "value"),
    Input("show-fed-rate", "value"),
    Input("show-inflation", "value"),
    Input("show-unemployment", "value"),
    Input("show-u6-unemployment", "value"),
    Input("show-unemp-ind", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
    Input("refresh-token", "data"),
)
def update_visuals(
    show_yields_val,
    selected_maturities,
    show_spread_val,
    show_fed_rate_val,
    show_inflation_val,
    show_unemployment_val,
    show_u6_unemployment_val,
    show_unemp_ind_val,
    start_date_str,
    end_date_str,
    _refresh_token,
):
    show_yields = "on" in (show_yields_val or [])
    show_spread = "on" in (show_spread_val or [])
    show_fed_rate = "on" in (show_fed_rate_val or [])
    show_inflation = "on" in (show_inflation_val or [])
    show_unemployment = "on" in (show_unemployment_val or [])
    show_u6_unemployment = "on" in (show_u6_unemployment_val or [])
    show_unemp_ind = "on" in (show_unemp_ind_val or [])

    selected_maturities = selected_maturities or []

    ust_df = load_and_process_csv("ust.csv")
    if ust_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Treasury data found. Ensure ust.csv exists.")
        return fig, [], "Treasury Data: N/A", "No Treasury data found."

    for col in SERIES.values():
        if col in ust_df.columns:
            ust_df[col] = pd.to_numeric(ust_df[col], errors="coerce")

    treasury_dates = ust_df["DATE"]

    fed_df = load_and_process_csv("fedrate.csv")
    if not fed_df.empty:
        fed_df["FED_RATE"] = pd.to_numeric(fed_df["FED_RATE"], errors="coerce")
        fed_monthly = align_to_month_end(fed_df[["DATE", "FED_RATE"]])
        fed_monthly = align_to_treasury_daily_calendar(fed_monthly, ["FED_RATE"], treasury_dates)
    else:
        fed_monthly = pd.DataFrame()

    infl_df = load_and_process_csv("inflation.csv")
    if not infl_df.empty:
        infl_df["PCE_YoY"] = pd.to_numeric(infl_df["PCE_YoY"], errors="coerce")
        infl_monthly = align_to_month_end(infl_df[["DATE", "PCE_YoY"]])
        infl_monthly = align_to_treasury_daily_calendar(infl_monthly, ["PCE_YoY"], treasury_dates)
    else:
        infl_monthly = pd.DataFrame()

    unrate_df = load_and_process_csv("unemployment.csv")
    if not unrate_df.empty:
        if "U6RATE" not in unrate_df.columns:
            unrate_df["U6RATE"] = pd.NA
        for col in ["UNRATE", "U6RATE", "NROU"]:
            unrate_df[col] = pd.to_numeric(unrate_df[col], errors="coerce")
        unrate_df["UNEMP_INDICATOR"] = unrate_df["UNRATE"] - unrate_df["NROU"]
        unrate_monthly = align_to_month_end(unrate_df[["DATE", "UNRATE", "U6RATE", "UNEMP_INDICATOR"]])
        unrate_monthly = align_to_treasury_daily_calendar(
            unrate_monthly,
            ["UNRATE", "U6RATE", "UNEMP_INDICATOR"],
            treasury_dates,
        )
    else:
        unrate_monthly = pd.DataFrame()

    min_date = max(ust_df["DATE"].min().date(), pd.to_datetime("2000-01-01").date())
    max_date = ust_df["DATE"].max().date()

    start_date = pd.to_datetime(start_date_str).date() if start_date_str else max(min_date, pd.to_datetime("2020-01-01").date())
    end_date = pd.to_datetime(end_date_str).date() if end_date_str else max_date

    start_date = max(min_date, min(max_date, start_date))
    end_date = max(min_date, min(max_date, end_date))

    warning = ""
    if start_date > end_date:
        warning = "Start date must be before end date. Resetting to full range."
        start_date, end_date = min_date, max_date

    delta = end_date - start_date
    if delta.days < 183:
        freq = None
        freq_label = "Daily"
    elif delta.days <= 731:
        freq = "W"
        freq_label = "Weekly"
    else:
        freq = "ME"
        freq_label = "Monthly"

    if freq:
        ust_resampled = (
            ust_df[["DATE"] + list(SERIES.values())]
            .set_index("DATE")
            .resample(freq)
            .mean()
            .dropna(how="all")
            .reset_index()
        )
    else:
        ust_resampled = ust_df

    plot_ust = filter_by_date(ust_resampled, start_date, end_date)
    plot_ust["SPREAD_10Y_2Y"] = plot_ust["BC_10YEAR"] - plot_ust["BC_2YEAR"]

    plot_fed = filter_by_date(fed_monthly, start_date, end_date)
    if not plot_fed.empty and len(plot_fed) > 1:
        plot_fed["rate_changed"] = plot_fed["FED_RATE"].diff().abs() > 0.001
        plot_fed.iloc[0, plot_fed.columns.get_loc("rate_changed")] = True
    elif not plot_fed.empty:
        plot_fed["rate_changed"] = True

    plot_infl = filter_by_date(infl_monthly, start_date, end_date)

    if (show_unemployment or show_u6_unemployment or show_unemp_ind) and not unrate_monthly.empty:
        plot_unrate = filter_by_date(unrate_monthly, start_date, end_date)
    else:
        plot_unrate = pd.DataFrame()

    fig = build_figure(
        plot_ust=plot_ust,
        plot_fed=plot_fed,
        plot_infl=plot_infl,
        plot_unrate=plot_unrate,
        selected_maturities=selected_maturities,
        show_yields=show_yields,
        show_spread=show_spread,
        show_fed_rate=show_fed_rate,
        show_inflation=show_inflation,
        show_unemployment=show_unemployment,
        show_u6_unemployment=show_u6_unemployment,
        show_unemp_ind=show_unemp_ind,
    )

    latest_cards = [
        indicator_card("2Y T-bill yield", latest_non_null_value(ust_df, "BC_2YEAR")),
        indicator_card("10Y T-bill yield", latest_non_null_value(ust_df, "BC_10YEAR")),
        indicator_card("Core Inflation", latest_non_null_value(infl_monthly, "PCE_YoY")),
        indicator_card("Fed Rate", latest_non_null_value(fed_monthly, "FED_RATE")),
        indicator_card("U-3 Unemployment rate", latest_non_null_value(unrate_monthly, "UNRATE")),
        indicator_card("U3-NROU", latest_non_null_value(unrate_monthly, "UNEMP_INDICATOR")),
    ]

    return fig, latest_cards, f"Treasury Data: {freq_label}", warning


if __name__ == "__main__":
    app.run(debug=True)
