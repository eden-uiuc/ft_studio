import pandas as pd
import numpy as np
import plotly.graph_objects as go
import zarr
import string

import flowtangent as ft

from functools import lru_cache
from plotly.subplots import make_subplots
from nicegui import ui
from utils.state import app_state, theme_config
from components.context_menus import setup_context_menu
from scipy.spatial import distance

PCA_DATA = pd.read_parquet("/home/jordan/dev/datasets/XFOIL_4D_PCA.parquet")
ZARR_DB = ft.utils.data.XArrayDataset("/home/jordan/dev/datasets/XFOIL_4D.zarr")


@lru_cache(maxsize=2048)
def surf_lookup(name:str):

    safe_name = name.replace("-", "_").replace(".", "_")
    if safe_name and safe_name[0] in string.digits:
        safe_name = f"af_{safe_name}"

    af = getattr(ft.airfoils, safe_name)

    x_af = af.x_coordinates
    y_af = af.y_coordinates

    raw = af.raw_coordinates
    x_raw = raw[:, 0]
    y_raw = raw[:, 1]

    return x_af, y_af, x_raw, y_raw

# --- 1. Data Mocking Layer ---
airfoil_df = pd.DataFrame({
    'name': PCA_DATA['name'],
    'pca_1': PCA_DATA['pca_1'],
    'pca_2': PCA_DATA['pca_2'],
    'pca_3': PCA_DATA['pca_3'],
    # Mocking single-point aerodynamic values for the manifold color mapping
    'CL': np.random.uniform(0.1, 1.5, 1400),
    'CD': np.random.uniform(0.005, 0.05, 1400),
    'CM': np.random.uniform(-0.1, 0.05, 1400),
    'Alpha': np.random.uniform(-5, 15, 1400),
    'Thickness': np.random.uniform(0.05, 0.25, 1400)
})

latent_state = {
    'selected_airfoil': "naca0024",
    'sphereize': True,
    'color_metric': 'Z-Radius',
    'flight_state': {'alpha': 2.0, 're': 1000000, 'mach': 0.1},
    'show_raw_pts': False,
    'show_interp_pts': False,
    # Add default plot configurations:
    'polar_1': {'x': 'alpha', 'y': 'cl', 'z': 're', 'mode': '2D Lines'},
    'polar_2': {'x': 'cd', 'y': 'cl', 'z': 're', 'mode': '2D Lines'},
    'polar_3': {'x': 'alpha', 'y': 'cm', 'z': 're', 'mode': '2D Lines'},
    'opt_objective': 'CL / CD',
    'opt_direction': 'Maximize',
    'constraints': [{'var': 'CD', 'op': '<', 'val': 0.025}],
    'subspace_opacity': 0.1,  # How faint filtered points appear'
}

def get_constraint_mask(df, constraints):
    """Safely evaluates the dynamic constraints against the DataFrame."""
    mask = pd.Series(True, index=df.index)
    for c in constraints:
        var, op, val = c.get('var'), c.get('op'), c.get('val')
        if var not in df.columns: continue
        
        if op == '<': mask &= (df[var] < val)
        elif op == '<=': mask &= (df[var] <= val)
        elif op == '==': mask &= (df[var] == val)
        elif op == '>=': mask &= (df[var] >= val)
        elif op == '>': mask &= (df[var] > val)
    return mask

def get_representative_indices(re_array, mach_array, flap_array, target_var, num_samples=5):
    """Finds the Sobol indices that best isolate a sweep of the target_var."""
    
    # 1. Min-Max Normalize so distance weighting is equal
    def normalize(arr):
        ptp = np.ptp(arr)
        return (arr - np.min(arr)) / ptp if ptp > 0 else np.zeros_like(arr)
        
    n_re = normalize(re_array)
    n_mach = normalize(mach_array)
    n_flap = normalize(flap_array)
    
    points = np.column_stack((n_re, n_mach, n_flap))
    
    # 2. Get nominal (median) values for the "locked" variables
    med_re, med_mach, med_flap = np.median(n_re), np.median(n_mach), np.median(n_flap)
    
    # 3. Generate our ideal targets
    targets = []
    sweep = np.linspace(0, 1, num_samples)
    
    for val in sweep:
        if target_var == 're':
            targets.append([val, med_mach, med_flap])
        elif target_var == 'mach':
            targets.append([med_re, val, med_flap])
        elif target_var == 'flap':  # Assuming you have flap as a UI option
            targets.append([med_re, med_mach, val])
        else:
            # Fallback if z_var isn't a Sobol parameter (e.g. they chose CL as Z)
            # Just return evenly spaced indices
            return np.linspace(0, len(re_array)-1, num_samples, dtype=int)
            
    # 4. Find the closest actual Sobol sample to each ideal target
    dists = distance.cdist(targets, points, metric='euclidean')
    best_indices = np.argmin(dists, axis=1)
    
    return best_indices

def create_polar_plot(x_var, y_var, z_var, mode, data, is_dark):
    template = 'plotly_dark' if is_dark else 'plotly_white'
    labels = {'alpha': 'Alpha', 'cl': 'CL', 'cd': 'CD', 'cm': 'CM', 're': 'Re'}
    fig = go.Figure()

    # Extract the 2D matrices
    X_full = np.asarray(data[x_var])
    Y_full = np.asarray(data[y_var])
    Z_full = np.asarray(data[z_var])
    Alpha_full = np.asarray(data['alpha'])

    sobol_re = np.asarray(data['re'])
    sobol_re = sobol_re[:, 0] if sobol_re.ndim > 1 else sobol_re
    
    sobol_mach = np.asarray(data.get('mach', np.zeros_like(sobol_re)))
    sobol_mach = sobol_mach[:, 0] if sobol_mach.ndim > 1 else sobol_mach
    
    sobol_flap = np.asarray(data.get('flap', np.zeros_like(sobol_re)))
    sobol_flap = sobol_flap[:, 0] if sobol_flap.ndim > 1 else sobol_flap

    idx = get_representative_indices(sobol_re, sobol_mach, sobol_flap, target_var=z_var, num_samples=5)

    X = X_full[idx]
    Y = Y_full[idx]
    Z_param = Z_full[idx]

    alpha_1d = Alpha_full[0] if Alpha_full.ndim > 1 else Alpha_full

    sort_idx = np.argsort(alpha_1d)
    X_sorted = X[:, sort_idx]
    Y_sorted = Y[:, sort_idx]
    Z_sorted = Z_param[:, sort_idx]

    if mode == '2D Lines':
        colors = ['#4c02a1', '#9c179e', '#d8576b', '#fd9f3e', '#f0f921'] if is_dark else \
                 ['#93c5fd', '#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8']
        
        # Loop over the rows (which correspond to different values of the Z-parameter)
        for i in range(X_sorted.shape[0]):
            z_val = Z_sorted[i, 0]
            c = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=X_sorted[i], y=Y_sorted[i], mode='lines', marker_size=3, line_color=c,
                name=f"{labels[z_var]}={z_val:.1e}",
                connectgaps=True
            ))
            
        fig.update_layout(
            template=template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=35, r=10, t=10, b=25), height=180,
            # xaxis_title=labels[x_var], yaxis_title=labels[y_var],
            showlegend=False # Hiding legend to save sidebar space; colors imply the sweep
        )
        
    elif mode == '3D Surface':
        clean_Y = pd.DataFrame(Y_sorted).interpolate(axis=1, limit_direction='both').values
        fig.add_trace(go.Surface(
            x=X_sorted, y=Z_sorted, z=clean_Y, 
            colorscale='Plasma' if is_dark else 'Blues', showscale=False
        ))
        
        clean_axis = dict(showbackground=False, gridcolor='rgba(128,128,128,0.2)')
        fig.update_layout(
            template=template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0), height=180,
            scene=dict(
                # Force Plotly to abandon the perfect cube and stretch the floor to fit the UI
                aspectmode='manual',
                aspectratio=dict(x=2.5, y=2.5, z=1), 
                xaxis_title=labels[x_var], yaxis_title=labels[z_var], zaxis_title=labels[y_var],
                xaxis=clean_axis, yaxis=clean_axis, zaxis=clean_axis
            )
        )
        
    return fig

@lru_cache(maxsize=2048)
def fetch_af_data(airfoil_name):

    x, y, raw_x, raw_y = surf_lookup(airfoil_name)
    
    name_idxs = np.where(ZARR_DB.foil_name == airfoil_name)

    ALPHAS_POS = np.concatenate([
        np.arange(0, 8, 1.0),       # [0, 1, 2 ... 7]
        np.arange(8, 15.25, 0.25)   # [8, 8.25 ... 15.0]
    ])

    # Start just below 0, go down sparsely
    ALPHAS_NEG = np.arange(-1, -6, -1.0) # [-1, -2, -3, -4, -5]

    # The combined array for Zarr sizing and index matching
    alpha_1d = np.concatenate([ALPHAS_POS, ALPHAS_NEG])
    
    re_1d = ZARR_DB.Re[name_idxs]
    
    cl = ZARR_DB.CL[name_idxs]
    cd = ZARR_DB.CD[name_idxs]
    cm = ZARR_DB.CM[name_idxs]
    
    # Create 2D Matrices (Rows = Re, Cols = Alpha)
    A, R = np.meshgrid(alpha_1d, re_1d)
    
    return {
        # Geometry (1D)
        'x': x, 'y': y,
        'raw_x': raw_x, 'raw_y': raw_y,
        
        # Aerodynamics (2D Matrices)
        'alpha': A, 
        're': R,
        # Mock physics equations that react to both Alpha and Re
        'cl': cl,
        'cd': cd,
        'cm': cm
    }

# --- 2. Event Handlers ---
def on_scatter_click(e):
    if 'points' in e.args:
        clicked_name = e.args['points'][0]['customdata'][0]
        latent_state['selected_airfoil'] = clicked_name
        left_inspector_pane.refresh()

# --- 3. UI Components ---

@ui.refreshable
def left_inspector_pane():
    is_dark = app_state.get('is_dark', False)
    template = 'plotly_dark' if is_dark else 'plotly_white'
    
    # Match the hangar's side panel styling
    sidebar_bg = 'bg-neutral-900' if is_dark else 'bg-gray-200'
    sidebar_border = 'border-neutral-800' if is_dark else 'border-gray-300'
    text_color = 'text-gray-200' if is_dark else 'text-gray-800'
    
    # Exact hex colors for the "floating box" Plotly background
    canvas_hex = '#000000' if is_dark else '#ffffff'
    
    airfoil_id = latent_state['selected_airfoil']
    data = fetch_af_data(airfoil_id)

    def update_selection(e):
        latent_state['selected_airfoil'] = e.value
        left_inspector_pane.refresh() # Only refresh the inspector to show the new airfoil!
    
    with ui.column().classes(f'w-full h-full flex-nowrap p-4 border-r {sidebar_bg} {sidebar_border} {text_color}'):
        # ui.label(f"{airfoil_id}").classes('text-xl font-bold border-b border-gray-600/50 w-full pb-2 shrink-0')
        with ui.row().classes('w-full items-center justify-between border-b border-gray-600/50 pb-2 shrink-0 flex-nowrap gap-2'):
            
            # The flex-grow min-w-0 wrapper forces flexbox to allow this div to shrink indefinitely
            with ui.element('div').classes('flex-grow min-w-0'):
                ui.select(
                    options=airfoil_df['name'].tolist(),
                    with_input=True, 
                    label='Search Airfoil',
                    value=latent_state['selected_airfoil'],
                    on_change=update_selection
                ).classes('w-full').props('dense') # Added 'dense' to keep it visually compact

            # Buttons remain protected by shrink-0
            with ui.row().classes('gap-1 shrink-0'):
                ui.button('Import', color='primary', icon='upload', on_click=lambda: ui.notify('Import dummy clicked')) \
                    .props('size="sm"').classes('px-2')
                ui.button('Export', color='primary', icon='download', on_click=lambda: ui.notify('Export dummy clicked')) \
                    .props('size="sm"').classes('px-2')
    
        
        # --- Geometry Viewer ---
        geom_fig = go.Figure()
        
        # Base filled outline
        geom_fig.add_trace(go.Scatter(x=data['x'], y=data['y'], fill='toself', mode='lines', line_color='#3b82f6', name='Shape'))
        
        # Optional Overlays
        if latent_state.get('show_raw_pts'):
            geom_fig.add_trace(go.Scatter(x=data['raw_x'], y=data['raw_y'], mode='markers', marker=dict(size=4, color='#ef4444'), name='Raw'))
        if latent_state.get('show_interp_pts'):
            geom_fig.add_trace(go.Scatter(x=data['x'], y=data['y'], mode='markers', marker=dict(size=2, color='#fbbf24'), name='Interp'))
        
        clean_axis = dict(showgrid=False, zeroline=False, showticklabels=False, title='')
        geom_fig.update_layout(
            template=template, paper_bgcolor=canvas_hex, plot_bgcolor=canvas_hex,
            margin=dict(l=0, r=35, t=35, b=0),
            xaxis=clean_axis,
            yaxis=dict(**clean_axis, scaleanchor="x", scaleratio=1),
            height=100,
            showlegend=False
        )
        
        # Wrap plot in a shadow container for the floating box effect
        with ui.element('div').classes('w-full rounded-none overflow-hidden shadow-lg border border-gray-600/20'):
            ui.plotly(geom_fig).classes('w-full shrink-0')
            
        # Checkbox Row
        with ui.row().classes('w-full items-center justify-center gap-4 -mt-4 mb-2'):
            ui.checkbox('Raw Pts.', value=latent_state['show_raw_pts'], 
                        on_change=lambda e: [latent_state.update({'show_raw_pts': e.value}), left_inspector_pane.refresh()]).props('dense size="sm"')
            ui.checkbox('Interp. Pts.', value=latent_state['show_interp_pts'],
                        on_change=lambda e: [latent_state.update({'show_interp_pts': e.value}), left_inspector_pane.refresh()]).props('dense size="sm"')

        # ui.separator().classes('my-4 opacity-50')
        ui.label(f"Polar Data").classes('text-lg font-bold border-b border-gray-600/50 w-full pb-2 shrink-0')
        
        # --- Dynamic Polars ---
        polar_vars = {'alpha': 'Alpha', 'cl': 'CL', 'cd': 'CD', 'cm': 'CM', 're': 'Re'}
        
        # Master event binder for all 4 controls
        def bind_events(p_el, xs, ys, zs, ms, state_key):
            def update(*args):
                latent_state[state_key].update({'x': xs.value, 'y': ys.value, 'z': zs.value, 'mode': ms.value})
                p_el.update_figure(create_polar_plot(xs.value, ys.value, zs.value, ms.value, data, is_dark))
            
            xs.on_value_change(update)
            ys.on_value_change(update)
            zs.on_value_change(update)
            ms.on_value_change(update)

        # ui.separator().classes('mt-4 mb-2 opacity-30')
        
        for i in range(1, 4):
            state_key = f'polar_{i}'
            p_state = latent_state[state_key]
            
            with ui.column().classes('w-full gap-0 mb-4'):
                with ui.row().classes('w-full items-center justify-between gap-1 px-1 mb-1'):
                    x_sel = ui.select(polar_vars, value=p_state['x'], label='X').classes('w-[18%]').props('dense options-dense')
                    y_sel = ui.select(polar_vars, value=p_state['y'], label='Y').classes('w-[18%]').props('dense options-dense')
                    z_sel = ui.select(polar_vars, value=p_state['z'], label='Z').classes('w-[18%]').props('dense options-dense')
                    
                    # Vertical Separator
                    ui.separator().props('vertical').classes('h-6 mx-1 opacity-30')
                    
                    m_sel = ui.select(['2D Lines', '3D Surface'], value=p_state['mode'], label='Mode').classes('w-[32%]').props('dense options-dense')
                
                # The Plot Canvas
                plot_el = ui.plotly(create_polar_plot(p_state['x'], p_state['y'], p_state['z'], p_state['mode'], data, is_dark)) \
                            .classes('w-full shrink-0 mb-4')
                
                bind_events(plot_el, x_sel, y_sel, z_sel, m_sel, state_key)


@ui.refreshable
def constraint_builder_ui():
    """Dynamically builds the rows of constraints."""
    vars_list = ['CL', 'CD', 'CM', 'Alpha', 'Thickness'] # Pull from dataset in production
    
    with ui.column().classes('w-full gap-2 mt-2'):
        for i, c in enumerate(latent_state['constraints']):
            with ui.row().classes('w-full items-center flex-nowrap gap-1'):
                # Variable
                ui.select(vars_list, value=c['var'], on_change=lambda e, idx=i: latent_state['constraints'][idx].update({'var': e.value})) \
                    .props('dense options-dense').classes('flex-grow w-1/3')
                
                # Operator
                ui.select(['<', '<=', '==', '>=', '>'], value=c['op'], on_change=lambda e, idx=i: latent_state['constraints'][idx].update({'op': e.value})) \
                    .props('dense options-dense').classes('w-16 shrink-0')
                
                # Value
                ui.number(value=c['val'], on_change=lambda e, idx=i: latent_state['constraints'][idx].update({'val': e.value})) \
                    .props('dense').classes('w-20 shrink-0')
                
                # Delete Button
                ui.button(icon='close', color='dark blue', on_click=lambda idx=i: [latent_state['constraints'].pop(idx), constraint_builder_ui.refresh()]) \
                    .props('flat round dense').classes('shrink-0')
                
        with ui.row().classes('w-full gap-2 mt-1 flex-nowrap'):
            ui.button('Add', icon='add', color='primary',
                      on_click=lambda: [latent_state['constraints'].append({'var': 'CL', 'op': '>', 'val': 0.0}), constraint_builder_ui.refresh()]) \
                .classes('flex-1').props('size="sm"')
            
            ui.button('Apply', icon='check_circle', color='primary', 
                      on_click=center_manifold_pane.refresh) \
                .classes('flex-1').props('size="sm"')


def right_tools_pane():
    is_dark = app_state.get('is_dark', False)
    
    sidebar_bg = 'bg-neutral-900' if is_dark else 'bg-gray-200'
    sidebar_border = 'border-neutral-800' if is_dark else 'border-gray-300'
    text_color = 'text-gray-200' if is_dark else 'text-gray-800'
    card_bg = 'bg-slate-800' if is_dark else 'bg-white'
    
    with ui.column().classes(f'w-full h-full p-4 gap-4 overflow-y-auto border-l {sidebar_bg} {sidebar_border} {text_color}'):
        ui.label('Design Space Control').classes('text-xl font-bold border-b border-gray-600/50 w-full pb-2')
        
        # --- 1. OPTIMIZATION OBJECTIVE ---
        with ui.expansion('Optimization Objective', icon='auto_graph').classes(f'w-full {card_bg} rounded shadow-sm').props('default-opened'):
            with ui.column().classes('w-full p-2 gap-2'):
                with ui.row().classes('w-full flex-nowrap gap-2 items-center'):
                    ui.select(['Maximize', 'Minimize', 'Target'], value=latent_state['opt_direction'], 
                              on_change=lambda e: latent_state.update({'opt_direction': e.value})) \
                        .classes('w-1/3 shrink-0').props('dense options-dense')
                    
                    ui.input('Expression (e.g. CL**1.5 / CD)', value=latent_state['opt_objective'], 
                             on_change=lambda e: latent_state.update({'opt_objective': e.value})) \
                        .classes('flex-grow').props('dense')
                
                ui.button('Run Optimizer', color='primary').classes('w-full mt-1')

        # --- 2. SUBSPACE CONSTRAINTS ---
        with ui.expansion('Subspace Constraints', icon='filter_alt').classes(f'w-full {card_bg} rounded shadow-sm').props('default-opened'):
            with ui.column().classes('w-full p-2 gap-2'):
                
                # Visual Masking Slider
                with ui.row().classes('w-full items-center justify-between gap-2 px-1'):
                    ui.label('Subspace Opacity:').classes('text-xs opacity-70')
                    # The on_change here would eventually trigger a refresh of your center_manifold_pane
                    ui.slider(min=0.0, max=1.0, step=0.05, value=latent_state['subspace_opacity'], 
                              on_change=lambda e: latent_state.update({'subspace_opacity': e.value})) \
                        .classes('flex-grow')
                
                ui.separator().classes('mt-1 mb-0 opacity-30')
                
                # Dynamic Rules
                constraint_builder_ui()

        ui.separator().classes('my-0 opacity-50')
        
        # --- 3. FLOWTANGENT ASSISTANT CHAT ---
        with ui.row().classes('w-full items-center justify-between mb-0 mt-1'):
            ui.label('FlowTangent Assistant').classes('text-lg font-bold')
            ui.icon('psychology', size='sm', color='primary')
        
        chat_border = 'border-neutral-800' if is_dark else 'border-gray-300'
        with ui.column().classes(f'w-full h-80 border rounded-lg flex-nowrap gap-0 overflow-hidden bg-transparent shrink-0 {chat_border}'):
            with ui.column().classes('w-full flex-grow p-3 overflow-y-auto gap-2'):
                ui.chat_message('Hello! I am your FlowTangent agent.', name='FT', stamp='System Ready').props('bg-color=primary text-color=white')
                
            with ui.row().classes(f'w-full shrink-0 p-2 border-t items-center flex-nowrap bg-transparent {chat_border}'):
                ui.input(placeholder='Ask FlowTangent...').props('dense outlined').classes('flex-grow text-sm')
                ui.button(icon='send', color='primary').props('round flat dense').classes('ml-2')


# --- 4. Main View Layout ---
@ui.refreshable
def center_manifold_pane():
    is_dark = app_state.get('is_dark', False)
    template = 'plotly_dark' if is_dark else 'plotly_white'
    canvas_bg = 'bg-black text-gray-200' if is_dark else 'bg-white text-gray-800'
    bg_color = 'rgba(0,0,0,0)'

    # Handlers for updates
    def update_selection(e):
        latent_state['selected_airfoil'] = e.value
        left_inspector_pane.refresh() 

    def update_manifold():
        # In a real app, this would query your dataset for the new Alpha/Re/Mach values 
        # before refreshing the plot to update the colors.
        center_manifold_pane.refresh()

    with ui.column().classes(f"w-full h-full p-4 gap-2 flex-nowrap {canvas_bg}"):

        if latent_state.get('sphereize'):
            # Proper Sphereizing (Whitening) for PCA:
            # Equalize the variance along all axes by dividing by their standard deviations
            x_c = airfoil_df['pca_1'] - airfoil_df['pca_1'].mean()
            y_c = airfoil_df['pca_2'] - airfoil_df['pca_2'].mean()
            z_c = airfoil_df['pca_3'] - airfoil_df['pca_3'].mean()
            
            airfoil_df['plot_x'] = x_c / x_c.std()
            airfoil_df['plot_y'] = y_c / y_c.std()
            airfoil_df['plot_z'] = z_c / z_c.std()
            
        else:
            airfoil_df['plot_x'] = airfoil_df['pca_1']
            airfoil_df['plot_y'] = airfoil_df['pca_2']
            airfoil_df['plot_z'] = airfoil_df['pca_3']
        
        axis_lim = max(
            airfoil_df['plot_x'].abs().max(),
            airfoil_df['plot_y'].abs().max(),
            airfoil_df['plot_z'].abs().max()) * 1.1

        airfoil_df['Z-Radius'] = np.sqrt(airfoil_df['plot_x']**2 + airfoil_df['plot_y']**2 + airfoil_df['plot_z']**2)
        
        # --- 2. FILTERING ---

        # MIDDLE: 3D Manifold Plotly Canvas
        mask = get_constraint_mask(airfoil_df, latent_state['constraints'])
        df_in = airfoil_df[mask]
        df_out = airfoil_df[~mask]

        metric = latent_state['color_metric']
        cmin, cmax = airfoil_df[metric].min(), airfoil_df[metric].max()
        colorscale = 'Plasma' if is_dark else 'ice'
        
        scatter_fig = go.Figure()

        # Trace 1: In-Bounds (Full Opacity)
        if len(df_in) > 0:
            scatter_fig.add_trace(go.Scatter3d(
                x=df_in['plot_x'], y=df_in['plot_y'], z=df_in['plot_z'],
                text=df_in['name'],
                mode='markers', opacity=1.0,
                customdata=np.stack((df_in['name'],), axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>PCA 1: %{x:.2f}<br>PCA 2: %{y:.2f}<br>PCA 3: %{z:.2f}<extra></extra>",
                marker=dict(
                    size=3, color=df_in[metric], colorscale=colorscale, cmin=cmin, cmax=cmax,
                    showscale=True, # Show colorbar only on the primary trace
                    colorbar=dict(orientation='h', y=-0.05, thickness=12, len=1.0, title_side='top')
                )
            ))

        # Trace 2: Out-of-Bounds (Faded)
        fade_opacity = latent_state.get('subspace_opacity', 0.1)
        if len(df_out) > 0 and fade_opacity > 0:
            scatter_fig.add_trace(go.Scatter3d(
                x=df_out['plot_x'], y=df_out['plot_y'], z=df_out['plot_z'],
                text=df_in['name'],
                mode='markers', opacity=fade_opacity,
                customdata=np.stack((df_out['name'],), axis=-1),
                hovertemplate="<b>%{customdata[0]}</b> (Out of Bounds)<extra></extra>",
                marker=dict(size=3, color=df_out[metric], colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=False)
            ))

        scatter_fig.add_trace(go.Scatter3d(
            x=[-3, 3, None, 0, 0, None, 0, 0],
            y=[0, 0, None, -3, 3, None, 0, 0],
            z=[0, 0, None, 0, 0, None, -3, 3],
            mode='lines',
            line=dict(color='rgba(128,128,128,0.4)', width=4),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Plotly Magic: Hide the 3D bounding box, grid, axes, and hover spikes completely
        range_lim = min(axis_lim, 3.0)
        clean_axis = dict(
            showbackground=False, showgrid=False, showline=False,
            zeroline=False, showticklabels=False, title='', showspikes=False,
            range=[-axis_lim, axis_lim]
            )
        
        scatter_fig.update_layout(
            showlegend=False,
            template=template, paper_bgcolor=bg_color, plot_bgcolor=bg_color,
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(xaxis=clean_axis, yaxis=clean_axis, zaxis=clean_axis),
            uirevision=f"camera_{latent_state.get('sphereize')}"
        )
        
        # The flexible void space
        with ui.element('div').classes('w-full flex-grow relative'):
            plot = ui.plotly(scatter_fig).classes('w-full h-full absolute inset-0')
            
            # Safe click handler
            def handle_click(e):
                points = e.args.get('points', [])
                if not points: 
                    return # Background click, do nothing
                    
                pt = points[0]
                # Safely try customdata first, fallback to text
                clicked_name = pt.get('customdata', [None])[0] or pt.get('text')
                
                if clicked_name:
                    # Trigger the existing update_selection function safely
                    update_selection(type('obj', (object,), {'value': clicked_name}))

            plot.on('plotly_click', handle_click)

        # BOTTOM: Flight State & Metric Controls
        # Pinned to the bottom, horizontal layout
        with ui.row().classes('w-full shrink-0 items-end justify-between bg-black/5 p-2 rounded-lg gap-2'):
            
            # Left side of bottom bar: Flight State Variables
            with ui.row().classes('gap-2 items-center'):
                ui.label("Flight State:").classes('font-bold text-sm opacity-70')
                ui.number('Alpha (deg)', value=latent_state['flight_state']['alpha'], format='%.1f', 
                          on_change=lambda e: latent_state['flight_state'].update({'alpha': e.value})).classes('w-24').props('dense')
                ui.number('Re (10^6)', value=latent_state['flight_state']['re']/1e6, format='%.1f',
                          on_change=lambda e: latent_state['flight_state'].update({'re': e.value*1e6})).classes('w-24').props('dense')
                ui.number('Mach', value=latent_state['flight_state']['mach'], format='%.2f',
                          on_change=lambda e: latent_state['flight_state'].update({'mach': e.value})).classes('w-24').props('dense')
                
                # Apply button to prevent refreshing the plot on every single keystroke
                ui.button(icon='refresh', on_click=update_manifold).props('flat round dense').classes('ml-1 text-primary')

            # Right side of bottom bar: Color scale metric selector
            with ui.row().classes('items-center gap-4'):
                ui.checkbox('Sphereize', value=latent_state['sphereize'], 
                            on_change=lambda e: [latent_state.update({'sphereize': e.value}), center_manifold_pane.refresh()]) \
                    .props('dense size="sm"')
                    
                ui.select(['Z-Radius', 'CL', 'CD', 'CM'], label='Color By', value=latent_state['color_metric'], 
                          on_change=lambda e: [latent_state.update({'color_metric': e.value}), update_manifold()]) \
                    .classes('w-32').props('dense')

# --- 3. Main View Layout ---
def design_space_ui():
    with ui.column().classes('w-full h-full p-0 gap-0'):
        with ui.splitter(value=350, limits=(350, 600)).classes('w-full h-full').props('unit="px"') as outer_split:
            
            with outer_split.before:
                left_inspector_pane() # Your airfoil geometry & polars
                
            with outer_split.after:
                with ui.splitter(value=400, limits=(300, 800), reverse=True).classes('w-full h-full').props('unit="px"') as inner_split:
                    
                    with inner_split.before:
                        center_manifold_pane() # The new interactive void
                        
                    with inner_split.after:
                        right_tools_pane() # Assistant & Optimization