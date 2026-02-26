import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context, Patch
from dash.exceptions import PreventUpdate
import fastf1
from fastf1.utils import delta_time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import sys
import os
import functools
from datetime import datetime

# --- FastF1 Version Check & Cache Setup ---
required_version = (3, 3, 3)
if tuple(map(int, fastf1.__version__.split('.'))) < required_version:
    sys.exit(
        f"FastF1 >= {'.'.join(map(str, required_version))} required, "
        f"found {fastf1.__version__}. Run: pip install --upgrade fastf1"
    )

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# --- App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
server = app.server

# --- Constants ---
_current_year = datetime.now().year
_today        = datetime.now()
YEARS         = list(range(2018, _current_year + 1))[::-1]
# F1 season starts in March; default to previous year if we're still in Jan–Feb
DEFAULT_YEAR  = _current_year if _today.month > 2 else _current_year - 1

PLOT_TEMPLATE  = 'plotly_dark'
BG_COLOR       = '#111111'
DRIVER1_COLOR  = 'blue'
DRIVER2_COLOR  = 'red'
DROPDOWN_STYLE = {'color': 'black'}
SESSIONS       = ['FP1', 'FP2', 'FP3', 'Q', 'Sprint Shootout', 'Sprint', 'R']
BRAKE_DASH     = {'throttle': 'dash', 'brake': 'dot'}

# --- Session Cache ---
_session_cache    = {}
_SESSION_CACHE_MAX = 8

def get_session(year, gp, session_type, telemetry=True):
    key = (year, gp, session_type, telemetry)
    if key in _session_cache:
        return _session_cache[key]
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=telemetry, laps=True, weather=False)
        if len(_session_cache) >= _SESSION_CACHE_MAX:
            del _session_cache[next(iter(_session_cache))]
        _session_cache[key] = session
        return session
    except Exception as e:
        print(f"Error loading session {year} {gp} {session_type}: {e}")
        return None

@functools.lru_cache(maxsize=16)
def get_schedule(year):
    return fastf1.get_event_schedule(year, include_testing=False)

# --- Telemetry Helpers ---
def resample_telemetry(target_dist, source_dist, source_vals):
    target_dist  = np.asarray(target_dist)
    source_dist  = np.asarray(source_dist)
    source_vals  = np.asarray(source_vals, dtype=float)
    n = len(target_dist)
    if n == 0:
        return np.array([])
    if len(source_dist) != len(source_vals) or len(source_dist) == 0:
        return np.full(n, np.nan)
    unique_dist, unique_idx = np.unique(source_dist, return_index=True)
    if len(unique_dist) < 2:
        return np.full(n, source_vals[unique_idx[0]] if len(unique_dist) == 1 else np.nan)
    return np.interp(target_dist, unique_dist, source_vals[unique_idx])

# --- Plotting Helpers ---
def _best_label_position(cx, cy, track_x, track_y, offset, n_angles=24):
    """Return (lx, ly) at distance `offset` from (cx, cy) in the direction
    that maximises the minimum clearance from any track point."""
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    best_clearance, best_lx, best_ly = -1.0, cx, cy + offset
    for a in angles:
        lx = cx + np.cos(a) * offset
        ly = cy + np.sin(a) * offset
        clearance = float(np.min(np.hypot(track_x - lx, track_y - ly)))
        if clearance > best_clearance:
            best_clearance, best_lx, best_ly = clearance, lx, ly
    return best_lx, best_ly


def create_empty_figure(message=""):
    fig = go.Figure()
    fig.update_layout(
        template=PLOT_TEMPLATE, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        annotations=[dict(text=message, xref="paper", yref="paper", showarrow=False, font=dict(size=16))]
    )
    return fig

def create_delta_plot(x, y, name1, name2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='cyan'), name=f"Δt ({name2} vs {name1})"))
    if x.size > 0:
        fig.add_shape(type='line', x0=x.min(), y0=0, x1=x.max(), y1=0,
                      line=dict(color='yellow', dash='dash', width=1))
    fig.update_layout(
        title=f"Time Delta: {name2} vs {name1}",
        xaxis_title='Distance (m)', yaxis_title='Time Delta (s)',
        template=PLOT_TEMPLATE, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=40, r=20, t=60, b=60),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.08))
    )
    return fig

def create_telemetry_plot(x, y1, y2, title, name1, name2, line_shape='linear'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=name1,
                             line=dict(color=DRIVER1_COLOR), line_shape=line_shape))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name=name2,
                             line=dict(color=DRIVER2_COLOR), line_shape=line_shape))
    fig.update_layout(
        title=title, xaxis_title='Distance (m)', yaxis_title=title.split(' ')[0],
        template=PLOT_TEMPLATE, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

# --- Layout ---
def _dd(id_, **kw):
    return dcc.Dropdown(id=id_, clearable=False, style=DROPDOWN_STYLE, **kw)

def create_driver_controls():
    return html.Div([
        dbc.Label("Year", className="text-white mt-3"),
        _dd('year-drivers', options=YEARS, value=DEFAULT_YEAR),
        dbc.Label("Grand Prix", className="text-white mt-3"),
        _dd('gp-drivers'),
        dbc.Label("Session", className="text-white mt-3"),
        _dd('session-type-drivers', options=SESSIONS, value='Q'),
        dbc.Label("Driver 1 (Reference)", className="text-white mt-3"),
        _dd('driver1-drivers'),
        dbc.Label("Driver 2 (Comparison)", className="text-white mt-3"),
        _dd('driver2-drivers'),
        dbc.Button('Compare', id='compare-btn-drivers', color='primary', className='mt-3 w-100'),
    ])

def create_year_controls():
    return html.Div([
        dbc.Label("Reference Driver (from Year 1)", className="text-white mt-3"),
        _dd('driver-years'),
        dbc.Label("Year 1 / GP 1 / Session 1", className="text-white mt-3"),
        _dd('year1-years', options=YEARS, value=DEFAULT_YEAR),
        _dd('gp1-years', className="mt-1"),
        _dd('session-type1-years', options=SESSIONS, value='Q', className="mt-1"),
        dbc.Label("Year 2 / GP 2 / Session 2", className="text-white mt-3"),
        _dd('year2-years', options=YEARS, value=DEFAULT_YEAR - 1),
        _dd('gp2-years', className="mt-1"),
        _dd('session-type2-years', options=SESSIONS, value='Q', className="mt-1"),
        dbc.Button('Compare', id='compare-btn-years', color='primary', className='mt-3 w-100'),
    ])

_GRAPH_CONFIG = {'displayModeBar': True, 'scrollZoom': True}

app.layout = dbc.Container([
    html.H2("F1 Telemetry Comparator", className="text-white my-3 text-center"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Comparison Mode", className="text-white"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[
                    {'label': 'Compare Drivers (Same Year)', 'value': 'drivers'},
                    {'label': 'Compare Years (Same Driver)',  'value': 'years'},
                ],
                value='drivers', clearable=False, style=DROPDOWN_STYLE,
            ),
            html.Div(id='drivers-controls-container', children=create_driver_controls()),
            html.Div(id='years-controls-container',  children=create_year_controls(), style={'display': 'none'}),
        ], width=3),
        dbc.Col([
            dcc.Loading(dcc.Graph(id='track-plot',    figure=create_empty_figure("Select options and click 'Compare'"), config=_GRAPH_CONFIG), type="circle"),
            dcc.Loading(dcc.Graph(id='delta-plot',    figure=create_empty_figure(), config=_GRAPH_CONFIG), type="circle"),
            dcc.Loading(dcc.Graph(id='speed-plot',    figure=create_empty_figure(), config=_GRAPH_CONFIG), type="circle"),
            dcc.Loading(dcc.Graph(id='gear-plot',     figure=create_empty_figure(), config=_GRAPH_CONFIG), type="circle"),
            dcc.Loading(dcc.Graph(id='throttle-plot', figure=create_empty_figure(), config=_GRAPH_CONFIG), type="circle"),
            dcc.Store(id='corner-dist-store'),
        ], width=9),
    ]),
], fluid=True, style={'backgroundColor': BG_COLOR, 'padding': '20px'})

# --- Control Callbacks ---
@app.callback(
    [Output('drivers-controls-container', 'style'), Output('years-controls-container', 'style')],
    Input('feature-dropdown', 'value'),
    prevent_initial_call=True,
)
def toggle_control_panels(feature):
    show, hide = {'display': 'block'}, {'display': 'none'}
    return (show, hide) if feature == 'drivers' else (hide, show)

# --- Dynamic Dropdown Callbacks ---
def create_gp_dropdown_callback(output_id, year_id):
    @app.callback([Output(output_id, 'options'), Output(output_id, 'value')], Input(year_id, 'value'))
    def _cb(year):
        if not year:
            return [], None
        options = [{'label': n, 'value': n} for n in get_schedule(year)['EventName']]
        return options, (options[-1]['value'] if options else None)

def create_driver_dropdown_callback(output_id, year_id, gp_id, session_id):
    @app.callback(Output(output_id, 'options'), [Input(year_id, 'value'), Input(gp_id, 'value'), Input(session_id, 'value')])
    def _cb(year, gp, session_type):
        if not all([year, gp, session_type]):
            return []
        try:
            session = get_session(year, gp, session_type, telemetry=False)
            drivers = session.laps['Driver'].unique() if session and session.laps is not None else []
            return [{'label': d, 'value': d} for d in sorted(drivers)]
        except Exception as e:
            print(f"Could not load drivers for {year} {gp} {session_type}: {e}")
            return []

create_gp_dropdown_callback('gp-drivers',  'year-drivers')
create_gp_dropdown_callback('gp1-years',   'year1-years')
create_gp_dropdown_callback('gp2-years',   'year2-years')
create_driver_dropdown_callback('driver1-drivers', 'year-drivers',  'gp-drivers',  'session-type-drivers')
create_driver_dropdown_callback('driver2-drivers', 'year-drivers',  'gp-drivers',  'session-type-drivers')
create_driver_dropdown_callback('driver-years',    'year1-years',   'gp1-years',   'session-type1-years')

# --- Main Comparison Callback ---
@app.callback(
    [Output('track-plot', 'figure'), Output('delta-plot', 'figure'), Output('speed-plot', 'figure'),
     Output('gear-plot', 'figure'),  Output('throttle-plot', 'figure'), Output('corner-dist-store', 'data')],
    [Input('compare-btn-drivers', 'n_clicks'), Input('compare-btn-years', 'n_clicks')],
    [State('feature-dropdown', 'value'),
     State('year-drivers', 'value'), State('gp-drivers', 'value'), State('session-type-drivers', 'value'),
     State('driver1-drivers', 'value'), State('driver2-drivers', 'value'),
     State('driver-years', 'value'),
     State('year1-years', 'value'), State('gp1-years', 'value'), State('session-type1-years', 'value'),
     State('year2-years', 'value'), State('gp2-years', 'value'), State('session-type2-years', 'value')],
    prevent_initial_call=True,
)
def compare(drv_clicks, yrs_clicks, feature,
            y_d, gp_d, s_d, drv1, drv2,
            drv_y, y1, gp1, s1, y2, gp2, s2):
    triggered_id = callback_context.triggered_id
    if triggered_id not in ('compare-btn-drivers', 'compare-btn-years'):
        raise PreventUpdate

    try:
        if feature == 'drivers':
            if not all([y_d, gp_d, s_d, drv1, drv2]):
                raise ValueError("Please select all driver options.")
            session = get_session(y_d, gp_d, s_d)
            if session is None:
                raise ValueError(f"Failed to load session: {y_d} {gp_d} {s_d}")
            lap1 = session.laps.pick_driver(drv1).pick_fastest()
            lap2 = session.laps.pick_driver(drv2).pick_fastest()
            if lap1 is None: raise ValueError(f"No lap data for {drv1}.")
            if lap2 is None: raise ValueError(f"No lap data for {drv2}.")
            label1, label2 = drv1, drv2
        else:
            if not all([drv_y, y1, gp1, s1, y2, gp2, s2]):
                raise ValueError("Please select all year options.")
            s1_obj = get_session(y1, gp1, s1)
            s2_obj = get_session(y2, gp1, s1)
            if s1_obj is None: raise ValueError(f"Failed to load session: {y1} {gp1} {s1}")
            if s2_obj is None: raise ValueError(f"Failed to load session: {y2} {gp1} {s1}")
            lap1 = s1_obj.laps.pick_driver(drv_y).pick_fastest()
            lap2 = s2_obj.laps.pick_driver(drv_y).pick_fastest()
            if lap1 is None: raise ValueError(f"No lap data for {drv_y} in {y1} {gp1}.")
            if lap2 is None: raise ValueError(f"No lap data for {drv_y} in {y2} {gp1}.")
            label1, label2 = f"{drv_y} ({y1})", f"{drv_y} ({y2})"

        if pd.isna(lap1.LapTime) or pd.isna(lap2.LapTime):
            raise ValueError("A selected lap has no time set.")

        # --- Racing Line ---
        ref_merged = pd.merge_asof(
            lap1.get_car_data().add_distance().sort_values('Time'),
            lap1.get_pos_data().sort_values('Time'),
            on='Time', direction='nearest',
        ).dropna(subset=['X', 'Y', 'Distance']).reset_index(drop=True)
        cmp_merged = pd.merge_asof(
            lap2.get_car_data().add_distance().sort_values('Time'),
            lap2.get_pos_data().sort_values('Time'),
            on='Time', direction='nearest',
        ).dropna(subset=['X', 'Y', 'Distance']).reset_index(drop=True)

        track_fig = go.Figure()
        for name, data, color in [(label1, ref_merged, DRIVER1_COLOR), (label2, cmp_merged, DRIVER2_COLOR)]:
            is_brake    = data['Brake'] > 0
            starts      = data.index[~is_brake.shift(fill_value=False) & is_brake]
            ends        = data.index[is_brake.shift(fill_value=False) & ~is_brake]
            starts_set  = set(starts.tolist())
            bounds      = sorted(set([0] + starts.tolist() + ends.tolist() + [len(data) - 1]))

            hover = 'Dist: %{customdata[0]:.0f}m<br>Speed: %{customdata[1]:.0f} km/h<br>Gear: %{customdata[2]:.0f}<extra>%{fullData.name}</extra>'
            if len(bounds) > 1:
                for i in range(len(bounds) - 1):
                    start, end = bounds[i], bounds[i + 1]
                    if start >= end:
                        continue
                    seg = data.iloc[start:end + 1]
                    dash = BRAKE_DASH['brake' if start in starts_set else 'throttle']
                    track_fig.add_trace(go.Scatter(
                        x=seg['X'], y=seg['Y'], mode='lines',
                        line=dict(color=color, width=3, dash=dash),
                        name=f"{name} Line", legendgroup=name, showlegend=(i == 0),
                        customdata=np.column_stack([seg['Distance'], seg['Speed'], seg['nGear']]),
                        hovertemplate=hover,
                    ))
            else:
                track_fig.add_trace(go.Scatter(
                    x=data['X'], y=data['Y'], mode='lines',
                    line=dict(color=color, width=3, dash='dash'),
                    name=f"{name} Line", legendgroup=name, showlegend=True,
                    customdata=np.column_stack([data['Distance'], data['Speed'], data['nGear']]),
                    hovertemplate=hover,
                ))

            track_fig.add_trace(go.Scatter(
                x=data.loc[starts, 'X'], y=data.loc[starts, 'Y'], mode='markers',
                marker=dict(symbol='circle-open', size=11, color=color, line=dict(color=color, width=2.5)),
                showlegend=False, name=f'{name}_bs', legendgroup=name,
            ))
            track_fig.add_trace(go.Scatter(
                x=data.loc[ends, 'X'], y=data.loc[ends, 'Y'], mode='markers',
                marker=dict(symbol='circle', size=11, color=color, line=dict(color='white', width=1.5)),
                showlegend=False, name=f'{name}_be', legendgroup=name,
            ))

        # Lap start marker
        track_fig.add_trace(go.Scatter(
            x=[np.mean([ref_merged['X'].iloc[0], cmp_merged['X'].iloc[0]])],
            y=[np.mean([ref_merged['Y'].iloc[0], cmp_merged['Y'].iloc[0]])],
            mode='markers', marker=dict(symbol='star', size=15, color='yellow'), name="Lap Start",
        ))
        # Legend key
        track_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',   line=dict(color='gray', dash='dot'),  name="Under Braking",          showlegend=True))
        track_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',   line=dict(color='gray', dash='dash'), name="Throttle/Coast",          showlegend=True))
        track_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='circle-open', size=10, color='gray', line=dict(color='gray', width=2.5)), name="Brake Start (driver colour)", showlegend=True))
        track_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='circle',      size=10, color='gray', line=dict(color='white', width=1.5)), name="Brake End (driver colour)",   showlegend=True))

        track_fig.update_layout(
            title_text='Racing Line (Geographic Orientation)',
            template=PLOT_TEMPLATE, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=80, b=40),
            xaxis_title="X (m)", yaxis_title="Y (m)",
            annotations=[dict(
                text="Note: Track shown in actual GPS orientation, not broadcast TV layout",
                xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
                font=dict(size=10, color="lightgray"),
                bgcolor="rgba(0,0,0,0.5)", bordercolor="gray", borderwidth=1,
            )],
        )
        track_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.5)')
        track_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.5)')

        # Turn labels — clickable buttons placed in the clearest space near each corner
        corner_store = None
        try:
            active_session = session if feature == 'drivers' else s1_obj
            circuit_info = active_session.get_circuit_info()
            if circuit_info is not None and getattr(circuit_info, 'corners', None) is not None:
                corners_df = circuit_info.corners

                # Combined track points used for clearance checks
                all_tx = np.concatenate([ref_merged['X'].values, cmp_merged['X'].values])
                all_ty = np.concatenate([ref_merged['Y'].values, cmp_merged['Y'].values])

                # Label offset in data-coords: 6 % of the shorter track dimension, min 150 m
                x_ext = float(all_tx.max() - all_tx.min())
                y_ext = float(all_ty.max() - all_ty.min())
                offset = float(np.clip(min(x_ext, y_ext) * 0.06, 150, 500))

                label_xs, label_ys = [], []
                conn_x,  conn_y   = [], []
                for _, corner in corners_df.iterrows():
                    lx, ly = _best_label_position(
                        corner['X'], corner['Y'], all_tx, all_ty, offset
                    )
                    label_xs.append(lx)
                    label_ys.append(ly)
                    conn_x += [corner['X'], lx, None]
                    conn_y += [corner['Y'], ly, None]

                # Thin connector lines from apex to label
                track_fig.add_trace(go.Scatter(
                    x=conn_x, y=conn_y, mode='lines',
                    line=dict(color='rgba(255,255,255,0.18)', width=0.8),
                    showlegend=False, hoverinfo='skip', name='__corner_lines__',
                ))

                # Visible button-style labels — these ARE clickable (scatter fires clickData).
                # customdata has 4 elements [num, dist, apex_x, apex_y] so we can
                # distinguish these clicks from racing-line clicks (3 elements).
                track_fig.add_trace(go.Scatter(
                    x=label_xs, y=label_ys,
                    mode='markers+text',
                    marker=dict(
                        symbol='square', size=24,
                        color='rgba(20,20,20,0.88)',
                        line=dict(color='rgba(200,200,200,0.55)', width=1.2),
                    ),
                    text=[f"T{int(r['Number'])}" for _, r in corners_df.iterrows()],
                    textfont=dict(size=9, color='white'),
                    textposition='middle center',
                    customdata=[[int(r['Number']), float(r['Distance']),
                                 float(r['X']),    float(r['Y'])]
                                for _, r in corners_df.iterrows()],
                    hovertemplate='<b>T%{customdata[0]}</b> · click to zoom<extra></extra>',
                    showlegend=False,
                    name='__corners__',
                ))

                corner_store = {
                    'corners': sorted(
                        [{'num': int(r['Number']), 'dist': float(r['Distance']),
                          'x': float(r['X']),      'y':    float(r['Y'])}
                         for _, r in corners_df.iterrows()],
                        key=lambda c: c['dist'],
                    ),
                    'max_dist': float(ref_merged['Distance'].max()),
                }
        except Exception as e:
            print(f"Could not add turn labels: {e}")

        # --- Telemetry ---
        delta_s, ref_tel, cmp_tel = delta_time(lap1, lap2)
        common_dist = ref_tel['Distance'].to_numpy()

        delta    = resample_telemetry(common_dist, ref_tel['Distance'], delta_s)
        speed1   = resample_telemetry(common_dist, ref_tel['Distance'], ref_tel['Speed'])
        speed2   = resample_telemetry(common_dist, cmp_tel['Distance'], cmp_tel['Speed'])
        gear1    = resample_telemetry(common_dist, ref_tel['Distance'], ref_tel['nGear'])
        gear2    = resample_telemetry(common_dist, cmp_tel['Distance'], cmp_tel['nGear'])
        throttle1 = resample_telemetry(common_dist, ref_tel['Distance'], ref_tel['Throttle'])
        throttle2 = resample_telemetry(common_dist, cmp_tel['Distance'], cmp_tel['Throttle'])

        delta_fig    = create_delta_plot(common_dist, delta, label1, label2)
        speed_fig    = create_telemetry_plot(common_dist, speed1,    speed2,    "Speed (Km/h)", label1, label2)
        gear_fig     = create_telemetry_plot(common_dist, gear1,     gear2,     "Gear",         label1, label2, line_shape='hv')
        throttle_fig = create_telemetry_plot(common_dist, throttle1, throttle2, "Throttle (%)", label1, label2)

        return track_fig, delta_fig, speed_fig, gear_fig, throttle_fig, corner_store

    except Exception as e:
        print(f"Comparison error: {e}")
        empty = create_empty_figure(f"Error: {e}")
        return empty, create_empty_figure(), create_empty_figure(), create_empty_figure(), create_empty_figure(), None


# --- Linked Zoom: drag-zoom on any telemetry plot syncs all others ---
_TELEMETRY_PLOTS = ['delta-plot', 'speed-plot', 'gear-plot', 'throttle-plot']

def _xaxis_patch(**kwargs):
    p = Patch()
    for k, v in kwargs.items():
        p['layout']['xaxis'][k] = v
    return p

@app.callback(
    [Output(pid, 'figure', allow_duplicate=True) for pid in _TELEMETRY_PLOTS],
    [Input(pid, 'relayoutData') for pid in _TELEMETRY_PLOTS],
    prevent_initial_call=True,
)
def sync_telemetry_zoom(delta_rl, speed_rl, gear_rl, throttle_rl):
    triggered_id = callback_context.triggered_id
    if not triggered_id:
        raise PreventUpdate

    rl = dict(zip(_TELEMETRY_PLOTS, [delta_rl, speed_rl, gear_rl, throttle_rl])).get(triggered_id) or {}

    if 'xaxis.range[0]' in rl:
        x0, x1 = rl['xaxis.range[0]'], rl['xaxis.range[1]']
        return [no_update if pid == triggered_id else _xaxis_patch(range=[x0, x1], autorange=False)
                for pid in _TELEMETRY_PLOTS]

    if 'xaxis.autorange' in rl or 'autosize' in rl:
        return [no_update if pid == triggered_id else _xaxis_patch(autorange=True)
                for pid in _TELEMETRY_PLOTS]

    raise PreventUpdate


# --- Turn Click: clicking a corner button zooms track map + all telemetry ---
@app.callback(
    [Output('track-plot', 'figure', allow_duplicate=True)] +
    [Output(pid, 'figure', allow_duplicate=True) for pid in _TELEMETRY_PLOTS],
    Input('track-plot', 'clickData'),
    State('corner-dist-store', 'data'),
    prevent_initial_call=True,
)
def zoom_to_turn(click_data, corner_data):
    if not click_data or not corner_data:
        raise PreventUpdate

    point = click_data['points'][0]
    cd = point.get('customdata')

    # Button labels carry 4-element customdata [num, dist, apex_x, apex_y].
    # Racing-line traces carry [distance, speed, gear] (3 elements) — ignore those.
    if not cd or len(cd) != 4:
        raise PreventUpdate

    _, clicked_dist, apex_x, apex_y = cd[0], cd[1], cd[2], cd[3]
    corners  = corner_data['corners']
    max_dist = corner_data['max_dist']

    idx = min(range(len(corners)), key=lambda i: abs(corners[i]['dist'] - clicked_dist))

    prev_dist = corners[idx - 1]['dist'] if idx > 0              else 0.0
    next_dist = corners[idx + 1]['dist'] if idx < len(corners)-1 else max_dist
    prev_x    = corners[idx - 1]['x']   if idx > 0              else apex_x
    prev_y    = corners[idx - 1]['y']   if idx > 0              else apex_y
    next_x    = corners[idx + 1]['x']   if idx < len(corners)-1 else apex_x
    next_y    = corners[idx + 1]['y']   if idx < len(corners)-1 else apex_y

    # --- Telemetry: midpoint boundaries, 200 m minimum ---
    t0 = (prev_dist + clicked_dist) / 2
    t1 = (clicked_dist + next_dist) / 2
    if t1 - t0 < 200:
        mid = (t0 + t1) / 2
        t0  = max(0.0, mid - 100)
        t1  = min(max_dist, mid + 100)

    # --- Track map: bounding box of prev / current / next corner apexes + padding ---
    xs = [prev_x, apex_x, next_x]
    ys = [prev_y, apex_y, next_y]
    pad = max((max(xs) - min(xs)) * 0.5, (max(ys) - min(ys)) * 0.5, 300)
    track_patch = Patch()
    track_patch['layout']['xaxis']['range']     = [min(xs) - pad, max(xs) + pad]
    track_patch['layout']['xaxis']['autorange'] = False
    track_patch['layout']['yaxis']['range']     = [min(ys) - pad, max(ys) + pad]
    track_patch['layout']['yaxis']['autorange'] = False

    return [track_patch] + [_xaxis_patch(range=[t0, t1], autorange=False)
                             for _ in _TELEMETRY_PLOTS]


if __name__ == '__main__':
    app.run(debug=True)
