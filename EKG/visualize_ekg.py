import wfdb
import holoviews as hv
from holoviews import opts
import panel as pn
import os
import glob
import numpy as np

# Set holoviews backend to bokeh
hv.extension('bokeh')
pn.extension()

def get_available_records(data_dir='data'):
    """
    Scans the data directory for all available .hea records.
    """
    hea_files = glob.glob(os.path.join(data_dir, "*.hea"))
    records = sorted([os.path.basename(f).replace('.hea', '') for f in hea_files])
    return records

def load_record_data(record_name, data_dir='data'):
    """
    Loads record metadata and signals using wfdb.
    """
    record_path = os.path.join(data_dir, record_name)
    if not os.path.exists(f"{record_path}.hea"):
        return None
    return wfdb.rdrecord(record_path)

# Initialize widgets
records = get_available_records()
patient_select = pn.widgets.Select(name='Patient (Record)', options=records, value=records[0] if records else None)

# Dynamic lead selection depends on the record
lead_select = pn.widgets.Select(name='Lead', options=[])

def update_lead_options(event):
    record_name = event.new
    record = load_record_data(record_name)
    if record:
        lead_select.options = record.sig_name
        lead_select.value = record.sig_name[0]

patient_select.param.watch(update_lead_options, 'value')

# Trigger initial lead options
if records:
    initial_rec = load_record_data(records[0])
    if initial_rec:
        lead_select.options = initial_rec.sig_name
        lead_select.value = initial_rec.sig_name[0]

@pn.depends(patient_select.param.value, lead_select.param.value)
def create_plot(record_name, lead_name):
    if not record_name or not lead_name:
        return hv.Text(0, 0, "Select a record and lead")
    
    record = load_record_data(record_name)
    if not record or lead_name not in record.sig_name:
        return hv.Text(0, 0, "Error loading data")

    lead_idx = record.sig_name.index(lead_name)
    
    # Snippet for performance
    max_samples = 5000
    fs = record.fs
    time = np.arange(min(record.sig_len, max_samples)) / fs
    data = record.p_signal[:max_samples, lead_idx]

    # Create curve
    curve = hv.Curve((time, data), kdims='Time (s)', vdims='Amplitude (mV)', label=f"{record_name} - {lead_name}")
    
    # Add ECG grid lines
    # Major grid lines every 0.2s and 0.5mV
    # Minor grid lines every 0.04s and 0.1mV (standard EKG paper)
    grid_opts = opts.Curve(
        width=900, height=400, 
        tools=['hover', 'wheel_zoom', 'box_zoom', 'reset'],
        active_tools=['wheel_zoom'],
        show_grid=True,
        gridstyle={'grid_line_color': '#ffcccc', 'grid_line_width': 1, 'minor_grid_line_color': '#ffeeee', 'minor_grid_line_dash': 'solid'},
        yticks=10,
        xticks=10
    )
    
    return curve.opts(grid_opts)

# Create Dashboard Layout
dashboard = pn.Column(
    pn.Row(patient_select, lead_select),
    create_plot,
    styles=dict(background='#f0f0f0', padding='10px')
).servable(title="Dynamic ECG Dashboard")

if __name__ == "__main__":
    import sys
    print("\n" + "="*50)
    print("ECG DYNAMIC DASHBOARD")
    print("="*50)
    print("To run the live dashboard, use the following command:")
    print(f"\n/home/olmozavala/uv/envs/eoasweb/bin/panel serve {sys.argv[0]} --show")
    print("="*50 + "\n")
    
    # Still keep the HTML generation for convenience if run directly
    output_file = 'interactive_ecg_dashboard.html'
    print(f"Generating static snapshot to {output_file}...")
    dashboard.save(output_file, embed=True)
