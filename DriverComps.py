import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context
import fastf1
from fastf1.utils import delta_time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import sys
import os

# --- FastF1 Version Check & Cache Setup ---
required_version = (3, 3, 3)
if tuple(map(int, fastf1.__version__.split('.'))) < required_version:
    sys.exit(
        f"FastF1 version >= {'.'.join(map(str, required_version))} required, "
        f"found {fastf1.__version__}. Please upgrade with 'pip install --upgrade fastf1'."
    )

CACHE_DIR = '/app/cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

# --- App Setup ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
server = app.server

# --- Constants & Helpers ---
YEARS = list(range(2018, 2026))[::-1]
DEFAULT_YEAR = 2024
PLOT_TEMPLATE = 'plotly_dark'
BG_COLOR = '#111111'
DRIVER1_COLOR = 'blue'
DRIVER2_COLOR = 'red'
DROPDOWN_STYLE = {'color': 'black'}

def get_session(year, gp, session_type):
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=True, laps=True, weather=False)
        return session
    except Exception as e:
        print(f"Error loading session for {year} {gp} {session_type}: {e}")
        return None

def create_empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        template=PLOT_TEMPLATE, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        annotations=[dict(text=message, xref="paper", yref="paper", showarrow=False, font=dict(size=16))]
    )
    return fig

# --- RESTORED Resampling Helper from original code ---
def resample_telemetry(target_dist_axis, source_dist_axis, source_values_axis, source_name="data"):
    L_target = len(target_dist_axis)
    if L_target == 0: return np.array([])
    if len(source_values_axis) == L_target and np.array_equal(source_dist_axis, target_dist_axis): return source_values_axis
    if len(source_dist_axis) == 0 or len(source_values_axis) == 0: return np.full(L_target, np.nan)
    if len(source_dist_axis) != len(source_values_axis): return np.full(L_target, np.nan)
    unique_dist, unique_idx = np.unique(source_dist_axis, return_index=True)
    if len(unique_dist) < 2:
        if len(unique_dist) == 1 and L_target > 0: return np.full(L_target, source_values_axis[unique_idx[0]])
        return np.full(L_target, np.nan)
    return np.interp(target_dist_axis, unique_dist, source_values_axis[unique_idx])

# --- Control Creation Functions ---
def create_driver_controls():
    return html.Div([
        dbc.Label("Year", className="text-white mt-3"),
        dcc.Dropdown(id='year-drivers', options=YEARS, value=DEFAULT_YEAR, clearable=False, style=DROPDOWN_STYLE),
        dbc.Label("Grand Prix", className="text-white mt-3"),
        dcc.Dropdown(id='gp-drivers', clearable=False, style=DROPDOWN_STYLE),
        dbc.Label("Session", className="text-white mt-3"),
        dcc.Dropdown(id='session-type-drivers', options=['FP1','FP2','FP3','Q','R'], value='Q', clearable=False, style=DROPDOWN_STYLE),
        dbc.Label("Driver 1 (Reference)", className="text-white mt-3"),
        dcc.Dropdown(id='driver1-drivers', clearable=False, style=DROPDOWN_STYLE),
        dbc.Label("Driver 2 (Comparison)", className="text-white mt-3"),
        dcc.Dropdown(id='driver2-drivers', clearable=False, style=DROPDOWN_STYLE),
        dbc.Button('Compare', id='compare-btn-drivers', color='primary', className='mt-3 w-100')
    ])

def create_year_controls():
    return html.Div([
        dbc.Label("Reference Driver (from Year 1)", className="text-white mt-3"),
        dcc.Dropdown(id='driver-years', clearable=False, style=DROPDOWN_STYLE),
        dbc.Label("Year 1 / GP 1 / Session 1", className="text-white mt-3"),
        dcc.Dropdown(id='year1-years', options=YEARS, value=DEFAULT_YEAR, clearable=False, style=DROPDOWN_STYLE),
        dcc.Dropdown(id='gp1-years', className="mt-1", clearable=False, style=DROPDOWN_STYLE),
        dcc.Dropdown(id='session-type1-years', options=['FP1','FP2','FP3','Q','R'], value='Q', className="mt-1", clearable=False, style=DROPDOWN_STYLE),
        dbc.Label("Year 2 / GP 2 / Session 2", className="text-white mt-3"),
        dcc.Dropdown(id='year2-years', options=YEARS, value=DEFAULT_YEAR - 1, clearable=False, style=DROPDOWN_STYLE),
        dcc.Dropdown(id='gp2-years', className="mt-1", clearable=False, style=DROPDOWN_STYLE),
        dcc.Dropdown(id='session-type2-years', options=['FP1','FP2','FP3','Q','R'], value='Q', className="mt-1", clearable=False, style=DROPDOWN_STYLE),
        dbc.Button('Compare', id='compare-btn-years', color='primary', className='mt-3 w-100')
    ])

# --- Layout ---
app.layout = dbc.Container(
    [
        html.H2("F1 Telemetry Comparator", className="text-white my-3 text-center"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Comparison Mode", className="text-white"),
                        dcc.Dropdown(id='feature-dropdown', options=[{'label': 'Compare Drivers (Same Year)', 'value': 'drivers'}, {'label': 'Compare Years (Same Driver)', 'value': 'years'}], value='drivers', clearable=False, style=DROPDOWN_STYLE),
                        html.Div(id='drivers-controls-container', children=create_driver_controls(), style={'display': 'block'}),
                        html.Div(id='years-controls-container', children=create_year_controls(), style={'display': 'none'}),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id='track-plot', config={'displayModeBar': True}), type="circle"),
                        dcc.Loading(dcc.Graph(id='delta-plot', config={'displayModeBar': True}), type="circle"),
                        dcc.Loading(dcc.Graph(id='speed-plot', config={'displayModeBar': True}), type="circle"),
                        dcc.Loading(dcc.Graph(id='gear-plot', config={'displayModeBar': True}), type="circle"),
                        dcc.Store(id='telemetry-data-store'),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
    style={'backgroundColor': BG_COLOR, 'padding': '20px'},
)

# --- Control Callbacks ---
@app.callback([Output('drivers-controls-container', 'style'), Output('years-controls-container', 'style')], Input('feature-dropdown', 'value'))
def toggle_control_panels(feature):
    return ({'display': 'block'}, {'display': 'none'}) if feature == 'drivers' else ({'display': 'none'}, {'display': 'block'})

# --- Dynamic Dropdown Callbacks ---
def create_gp_dropdown_callback(output_id, year_id):
    @app.callback(Output(output_id, 'options'), Input(year_id, 'value'))
    def update_gp_dropdown(year):
        if not year: return []
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        return [{'label': row['EventName'], 'value': row['EventName']} for index, row in schedule.iterrows()]

def create_driver_dropdown_callback(output_id, year_id, gp_id, session_id):
    @app.callback(Output(output_id, 'options'), [Input(year_id, 'value'), Input(gp_id, 'value'), Input(session_id, 'value')])
    def update_driver_dropdown(year, gp, session_type):
        if not all([year, gp, session_type]): return []
        try:
            session = get_session(year, gp, session_type)
            drivers = session.laps['Driver'].unique() if session and session.laps is not None else []
            return [{'label': driver, 'value': driver} for driver in sorted(drivers)]
        except Exception as e:
            print(f"Could not load drivers for {year} {gp} {session_type}: {e}")
            return []

create_gp_dropdown_callback('gp-drivers', 'year-drivers')
create_gp_dropdown_callback('gp1-years', 'year1-years')
create_gp_dropdown_callback('gp2-years', 'year2-years')
create_driver_dropdown_callback('driver1-drivers', 'year-drivers', 'gp-drivers', 'session-type-drivers')
create_driver_dropdown_callback('driver2-drivers', 'year-drivers', 'gp-drivers', 'session-type-drivers')
create_driver_dropdown_callback('driver-years', 'year1-years', 'gp1-years', 'session-type1-years')

# --- Main Callback ---
@app.callback(
    [Output('track-plot', 'figure'), Output('delta-plot', 'figure'), Output('speed-plot', 'figure'), Output('gear-plot', 'figure'), Output('telemetry-data-store', 'data')],
    [Input('compare-btn-drivers', 'n_clicks'), Input('compare-btn-years', 'n_clicks'), Input('track-plot', 'relayoutData')],
    [State('feature-dropdown', 'value'),
     State('year-drivers', 'value'), State('gp-drivers', 'value'), State('session-type-drivers', 'value'), State('driver1-drivers', 'value'), State('driver2-drivers', 'value'),
     State('driver-years', 'value'), State('year1-years', 'value'), State('gp1-years', 'value'), State('session-type1-years', 'value'),
     State('year2-years', 'value'), State('gp2-years', 'value'), State('session-type2-years', 'value'),
     State('telemetry-data-store', 'data')]
)
def compare_and_zoom(drv_clicks, yrs_clicks, relayout, feature, y_d, gp_d, s_d, drv1, drv2, drv_y, y1, gp1, s1, y2, gp2, s2, telemetry_data):
    ctx = callback_context
    triggered_id = ctx.triggered_id.split('.')[0] if ctx.triggered_id else None

    # --- CONSERVATIVE ZOOM LOGIC - MAINTAINS DELTA ACCURACY ---
    if triggered_id == 'track-plot' and relayout and telemetry_data:
        is_autoscale = 'xaxis.autorange' in relayout
        if is_autoscale:
            dist, delta = np.array(telemetry_data['common_dist']), np.array(telemetry_data['delta'])
            speed1, speed2 = np.array(telemetry_data['speed1']), np.array(telemetry_data['speed2'])
            gear1, gear2 = np.array(telemetry_data['gear1']), np.array(telemetry_data['gear2'])
            title_suffix = ""
        elif 'xaxis.range[0]' in relayout:
            map_x, map_y, map_dist = np.array(telemetry_data['ref_x']), np.array(telemetry_data['ref_y']), np.array(telemetry_data['ref_dist'])
            x_min, x_max, y_min, y_max = relayout['xaxis.range[0]'], relayout['xaxis.range[1]'], relayout['yaxis.range[0]'], relayout['yaxis.range[1]']
            
            # Find points within the zoom box
            mask = (map_x >= x_min) & (map_x <= x_max) & (map_y >= y_min) & (map_y <= y_max)
            points_in_zoom = np.sum(mask)
            
            if points_in_zoom >= 2:
                # Sufficient points - use them directly
                min_dist, max_dist = map_dist[mask].min(), map_dist[mask].max()
                title_suffix = " (Zoomed)"
            elif points_in_zoom == 1:
                # Single point - add minimal padding
                center_dist = map_dist[mask][0]
                padding = 15  # 15 meters each side
                min_dist = max(0, center_dist - padding)
                max_dist = min(map_dist.max(), center_dist + padding)
                title_suffix = " (Point Focus)"
            else:
                # No points in zoom - find closest point and add minimal context
                center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
                distances_to_center = np.sqrt((map_x - center_x)**2 + (map_y - center_y)**2)
                closest_idx = np.argmin(distances_to_center)
                closest_dist = map_dist[closest_idx]
                
                # Minimal expansion - just enough to show some context
                padding = 20  # 20 meters each side
                min_dist = max(0, closest_dist - padding)
                max_dist = min(map_dist.max(), closest_dist + padding)
                title_suffix = " (Nearest Point)"
            
            # Apply the distance filter to telemetry data
            full_dist = np.array(telemetry_data['common_dist'])
            zoom_mask = (full_dist >= min_dist) & (full_dist <= max_dist)
            
            if np.sum(zoom_mask) == 0:
                return no_update, create_empty_figure("No telemetry in range"), create_empty_figure(""), create_empty_figure(""), no_update
            
            dist = full_dist[zoom_mask]
            delta = np.array(telemetry_data['delta'])[zoom_mask]
            speed1, speed2 = np.array(telemetry_data['speed1'])[zoom_mask], np.array(telemetry_data['speed2'])[zoom_mask]
            gear1, gear2 = np.array(telemetry_data['gear1'])[zoom_mask], np.array(telemetry_data['gear2'])[zoom_mask]
            
        else: 
            return no_update, no_update, no_update, no_update, no_update

        delta_fig = create_delta_plot(dist, delta, telemetry_data['l1_name'], telemetry_data['l2_name'], title_suffix)
        speed_fig = create_telemetry_plot(dist, speed1, speed2, "Speed (Km/h)", telemetry_data['l1_name'], telemetry_data['l2_name'], title_suffix)
        gear_fig = create_telemetry_plot(dist, gear1, gear2, "Gear", telemetry_data['l1_name'], telemetry_data['l2_name'], title_suffix, line_shape='hv')
        return no_update, delta_fig, speed_fig, gear_fig, no_update

    # --- Comparison Logic ---
    if triggered_id not in ['compare-btn-drivers', 'compare-btn-years']:
        return create_empty_figure("Select options and click 'Compare'"), create_empty_figure(""), create_empty_figure(""), create_empty_figure(""), no_update

    try:
        if feature == 'drivers':
            if not all([y_d, gp_d, s_d, drv1, drv2]): raise ValueError("Please select all driver options.")
            session, lap1, lap2 = get_session(y_d, gp_d, s_d), None, None
            lap1, lap2 = session.laps.pick_driver(drv1).pick_fastest(), session.laps.pick_driver(drv2).pick_fastest()
            label1, label2 = drv1, drv2
        else:
            if not all([drv_y, y1, gp1, s1, y2, gp2, s2]): raise ValueError("Please select all year options.")
            # Use same GP and session for year comparison
            gp_to_use, session_to_use = gp1, s1
            s1_obj, s2_obj = get_session(y1, gp_to_use, session_to_use), get_session(y2, gp_to_use, session_to_use)
            lap1, lap2 = s1_obj.laps.pick_driver(drv_y).pick_fastest(), s2_obj.laps.pick_driver(drv_y).pick_fastest()
            label1, label2 = f"{drv_y} ({y1})", f"{drv_y} ({y2})"
        
        if pd.isna(lap1.LapTime) or pd.isna(lap2.LapTime): raise ValueError("A selected lap has no time set.")
        
        # --- RACING LINE PLOT ---
        track_fig = go.Figure()
        dmap = {'throttle': 'dash', 'brake': 'dot'}
        ref_merged = pd.merge_asof(lap1.get_car_data().add_distance(), lap1.get_pos_data(), on='Time', direction='nearest').dropna(subset=['X','Y','Distance'])
        cmp_merged = pd.merge_asof(lap2.get_car_data().add_distance(), lap2.get_pos_data(), on='Time', direction='nearest').dropna(subset=['X','Y','Distance'])

        for name, data, color in [(label1, ref_merged, DRIVER1_COLOR), (label2, cmp_merged, DRIVER2_COLOR)]:
            is_brake, prev_is_brake = data['Brake'] > 0, data['Brake'].shift(fill_value=False)
            starts, ends = data.index[~prev_is_brake & is_brake], data.index[prev_is_brake & ~is_brake]
            bounds = sorted(list(set([0] + starts.tolist() + ends.tolist() + [len(data)-1])))
            
            if len(bounds) > 1:
                for i in range(len(bounds) - 1):
                    start, end = bounds[i], bounds[i+1]
                    if start >= end: continue
                    seg_type = 'brake' if start in starts.values else 'throttle'
                    track_fig.add_trace(go.Scatter(x=data['X'].iloc[start:end+1], y=data['Y'].iloc[start:end+1], mode='lines', line=dict(color=color, width=3, dash=dmap[seg_type]), name=f"{name} Line", legendgroup=name, showlegend=(i==0)))
            else:
                 track_fig.add_trace(go.Scatter(x=data['X'], y=data['Y'], mode='lines', line=dict(color=color, width=3, dash='dash'), name=f"{name} Line", legendgroup=name, showlegend=True))
            
            track_fig.add_trace(go.Scatter(x=data.loc[starts, 'X'], y=data.loc[starts, 'Y'], mode='markers', marker=dict(symbol='circle-open', size=10, color=color), showlegend=False, name=f'{name}_bs', legendgroup=name))
            track_fig.add_trace(go.Scatter(x=data.loc[ends, 'X'], y=data.loc[ends, 'Y'], mode='markers', marker=dict(symbol='circle', size=10, color=color), showlegend=False, name=f'{name}_be', legendgroup=name))
        
        track_fig.add_trace(go.Scatter(x=[np.mean([ref_merged['X'].iloc[0], cmp_merged['X'].iloc[0]])], y=[np.mean([ref_merged['Y'].iloc[0], cmp_merged['Y'].iloc[0]])], mode='markers', marker=dict(symbol='star', size=15, color='yellow'), name="Lap Start"))
        
        # Add legend explanations
        track_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='gray', dash='dot'), name="Under Braking", showlegend=True))
        track_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='gray', dash='dash'), name="Throttle/Coast", showlegend=True))
        track_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='circle-open', size=10, color='gray'), name="Brake Start", showlegend=True))
        track_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='circle', size=10, color='gray'), name="Brake End", showlegend=True))
        
        track_fig.update_layout(
            title_text='Racing Line (Geographic Orientation)', 
            template=PLOT_TEMPLATE, 
            paper_bgcolor=BG_COLOR, 
            plot_bgcolor=BG_COLOR, 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
            margin=dict(l=40, r=40, t=80, b=40), 
            xaxis_title="X (m)", 
            yaxis_title="Y (m)",
            annotations=[
                dict(
                    text="Note: Track shown in actual GPS orientation, not broadcast TV layout",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10, color="lightgray"),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="gray",
                    borderwidth=1
                )
            ]
        )
        track_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.5)')
        track_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.5)')
        
        # --- TELEMETRY --- FIXED: lap1 first, lap2 second
        delta_s, ref_tel, cmp_tel = delta_time(lap1, lap2)
        common_dist = ref_tel['Distance'].to_numpy()
        
        delta = resample_telemetry(common_dist, ref_tel['Distance'], delta_s, "Delta")
        speed1, gear1 = resample_telemetry(common_dist, ref_tel['Distance'], ref_tel['Speed'], f"{label1} Spd"), resample_telemetry(common_dist, ref_tel['Distance'], ref_tel['nGear'], f"{label1} Gear")
        speed2, gear2 = resample_telemetry(common_dist, cmp_tel['Distance'], cmp_tel['Speed'], f"{label2} Spd"), resample_telemetry(common_dist, cmp_tel['Distance'], cmp_tel['nGear'], f"{label2} Gear")
        
        delta_fig, speed_fig, gear_fig = create_delta_plot(common_dist, delta, label1, label2), create_telemetry_plot(common_dist, speed1, speed2, "Speed (Km/h)", label1, label2), create_telemetry_plot(common_dist, gear1, gear2, "Gear", label1, label2, line_shape='hv')
        
        telemetry_store = {'ref_x': ref_merged['X'].tolist(), 'ref_y': ref_merged['Y'].tolist(), 'ref_dist': ref_merged['Distance'].tolist(), 'common_dist': common_dist.tolist(), 'delta': delta.tolist(), 'speed1': speed1.tolist(), 'gear1': gear1.tolist(), 'speed2': speed2.tolist(), 'gear2': gear2.tolist(), 'l1_name': label1, 'l2_name': label2}
        return track_fig, delta_fig, speed_fig, gear_fig, telemetry_store
    except Exception as e:
        print(f"Error processing comparison: {e}")
        return create_empty_figure(f"Error: {e}"), create_empty_figure(""), create_empty_figure(""), create_empty_figure(""), no_update

# --- Plotting Helper Functions ---
def create_delta_plot(x, y, name1, name2, title_suffix=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='cyan'), name=f"Δt ({name2} vs {name1})"))
    if x.size > 0: fig.add_shape(type='line', x0=np.min(x), y0=0, x1=np.max(x), y1=0, line=dict(color='yellow', dash='dash', width=1))
    fig.update_layout(title=f"Time Delta: {name2} vs {name1}{title_suffix}", xaxis_title='Distance (m)', yaxis_title='Time Delta (s)', template=PLOT_TEMPLATE, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR, margin=dict(l=40, r=20, t=60, b=40))
    return fig

def create_telemetry_plot(x, y1, y2, title, name1, name2, title_suffix="", line_shape='linear'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=name1, line=dict(color=DRIVER1_COLOR), line_shape=line_shape))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name=name2, line=dict(color=DRIVER2_COLOR), line_shape=line_shape))
    fig.update_layout(title=f"{title}{title_suffix}", xaxis_title='Distance (m)', yaxis_title=title.split(' ')[0], template=PLOT_TEMPLATE, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=20, t=60, b=40))
    return fig

if __name__ == '__main__':
    app.run(debug=True)
