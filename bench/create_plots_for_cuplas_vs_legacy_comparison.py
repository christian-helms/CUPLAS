import json
import plotly.graph_objects as go
import math

with open("bench/cuplas_quality_over_size.json", "r") as f:
    cuplas_data_quality = json.load(f)

with open("bench/legacy_plas_quality_over_size.json", "r") as f:
    legacy_data_quality = json.load(f)

with open("bench/cuplas_runtime_over_size.json", "r") as f:
    cuplas_data_runtime = json.load(f)

with open("bench/legacy_plas_runtime_over_size.json", "r") as f:
    legacy_data_runtime = json.load(f)

sizes = [item["H"] for item in cuplas_data_quality]

plas_quality = [item["score"] for item in cuplas_data_quality]
legacy_quality = [item["score"] for item in legacy_data_quality]

plas_runtime = [item["time"] for item in cuplas_data_runtime]
legacy_runtime = [item["time"] for item in legacy_data_runtime]

# Create first plot: Quality comparison
fig_quality = go.Figure()

fig_quality.add_trace(
    go.Scatter(
        x=sizes,
        y=plas_quality,
        mode='lines+markers',
        name='CUPLAS',
        line=dict(color='blue', width=2),
        marker=dict(size=8, symbol='circle')
    )
)

fig_quality.add_trace(
    go.Scatter(
        x=sizes,
        y=legacy_quality,
        mode='lines+markers',
        name='Legacy PLAS',
        line=dict(color='red', width=2),
        marker=dict(size=8, symbol='square')
    )
)

# Set x-axis as log2 scale
fig_quality.update_xaxes(
    title_text='Size (H)',
    title_font=dict(size=16),  # Increase axis title font size
    type='log',
    dtick=math.log10(2),  # Set tick spacing to log2
    tickvals=sizes,  # Use actual size values for tick positions
    ticktext=[str(size) for size in sizes],  # Display actual size values
    tickfont=dict(size=14)  # Increase tick font size
)

# Set y-axis as log10 scale
fig_quality.update_yaxes(
    title_text='Quality Score (ANL2)',
    title_font=dict(size=16),  # Increase axis title font size
    type='log',
    tickfont=dict(size=14)  # Increase tick font size
)

fig_quality.update_layout(
    template='plotly_white',
    legend=dict(
        x=0.99,
        y=0.99,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.8)',
        font=dict(size=18)  # Increase font size by 50% (default is 12)
    ),
    width=900,
    height=600,
    margin=dict(t=30)  # Reduce top margin since there's no title
)

# Create second plot: Runtime comparison
fig_runtime = go.Figure()

fig_runtime.add_trace(
    go.Scatter(
        x=sizes,
        y=plas_runtime,
        mode='lines+markers',
        name='CUPLAS',
        line=dict(color='blue', width=2),
        marker=dict(size=8, symbol='circle')
    )
)

fig_runtime.add_trace(
    go.Scatter(
        x=sizes,
        y=legacy_runtime,
        mode='lines+markers',
        name='Legacy PLAS',
        line=dict(color='red', width=2),
        marker=dict(size=8, symbol='square')
    )
)

# Set x-axis as log2 scale
fig_runtime.update_xaxes(
    title_text='Size (H)',
    title_font=dict(size=16),  # Increase axis title font size
    type='log',
    dtick=math.log10(2),  # Set tick spacing to log2
    tickvals=sizes,  # Use actual size values for tick positions
    ticktext=[str(size) for size in sizes],  # Display actual size values
    tickfont=dict(size=14)  # Increase tick font size
)

# Set y-axis as log10 scale
fig_runtime.update_yaxes(
    title_text='Runtime (seconds)',
    title_font=dict(size=16),  # Increase axis title font size
    type='log',
    tickfont=dict(size=14)  # Increase tick font size
)

fig_runtime.update_layout(
    template='plotly_white',
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor='rgba(255, 255, 255, 0.8)',
        font=dict(size=18)  # Increase font size by 50% (default is 12)
    ),
    width=900,
    height=600,
    margin=dict(t=30)  # Reduce top margin since there's no title
)

# Save the plots as SVG files
fig_quality.write_image("bench/plas_quality_comparison.svg")
fig_runtime.write_image("bench/plas_runtime_comparison.svg")

# Optionally, show the plots if running in an interactive environment
# fig_quality.show()
# fig_runtime.show()
