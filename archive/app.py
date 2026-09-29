import datetime
import pandas as pd
import requests
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Set the "today" date as 2025-01-31.
TODAY = datetime.date(2025, 1, 30)
today_str = TODAY.isoformat()

# URL pattern for fetching historical data.
BASE_URL = "https://kalshi-public-docs.s3.amazonaws.com/reporting/market_data_{}.json"

# Correct API URL for retrieving market details.
DESCRIPTION_URL_TEMPLATE = "https://api.elections.kalshi.com/trade-api/v2/markets/{}"

# Load today's data.
try:
    df_today = pd.read_json(BASE_URL.format(today_str))
except Exception as e:
    raise RuntimeError(f"Could not load today's data from {BASE_URL.format(today_str)}: {e}")

# Filter for active markets with block_volume > 10,000.
active_markets_df = df_today[(df_today['status'] == 'active') & (df_today['daily_volume'] > 10000)]

# Use unique values from the "report_ticker" column to populate the dropdown.
unique_markets = active_markets_df['report_ticker'].unique()
dropdown_options = [{'label': market, 'value': market} for market in unique_markets]

# Initialize the Dash app.
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H3("Kalshi Market Contracts High/Low History"),
    dcc.Dropdown(
        id='market-dropdown',
        options=dropdown_options,
        placeholder='Select a market...',
    ),
    # A Div to display the market title.
    html.Div(id='market-description', style={'marginTop': 20, 'fontStyle': 'italic'}),
    dcc.Graph(id='market-graph')
])

@app.callback(
    [Output('market-graph', 'figure'),
     Output('market-description', 'children')],
    [Input('market-dropdown', 'value')]
)
def update_graph(selected_market):
    # If no market is selected, return an empty figure and no description.
    if not selected_market:
        return go.Figure(), ""
    
    # Retrieve the market title using the new API endpoint.
    title = "Title not available."
    try:
        resp = requests.get(DESCRIPTION_URL_TEMPLATE.format(selected_market))
        if resp.status_code == 200:
            data = resp.json()
            # Access the title from response["market"]["title"]
            title = data.get("market", {}).get("title", title)
        else:
            title = f"Error fetching title (status {resp.status_code})."
    except Exception as e:
        title = f"Error fetching title: {e}"
    
    # Dictionary to hold time series data for each contract (ticker_name).
    contract_data = {}
    
    # Iterate over the past 60 days.
    for i in range(60):
        current_date = TODAY - datetime.timedelta(days=i)
        date_str = current_date.isoformat()
        url = BASE_URL.format(date_str)
        try:
            df_day = pd.read_json(url)
        except Exception:
            # If data for this day cannot be loaded, skip it.
            continue

        # Filter rows for the selected market using the report_ticker column.
        day_data = df_day[df_day['report_ticker'] == selected_market]
        if day_data.empty:
            continue

        # Process each contract row for the day.
        for _, row in day_data.iterrows():
            contract = row['ticker_name']
            if contract not in contract_data:
                contract_data[contract] = {'dates': [], 'highs': [], 'lows': []}
            contract_data[contract]['dates'].append(date_str)
            contract_data[contract]['highs'].append(row['high'])
            contract_data[contract]['lows'].append(row['low'])
    
    # Reverse each contract’s lists so that the oldest date appears first.
    for contract in contract_data:
        contract_data[contract]['dates'].reverse()
        contract_data[contract]['highs'].reverse()
        contract_data[contract]['lows'].reverse()
    
    # Build the Plotly figure with traces for each contract.
    fig = go.Figure()
    for contract, data in contract_data.items():
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['highs'],
            mode='lines+markers',
            name=f"{contract} High"
        ))
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['lows'],
            mode='lines+markers',
            name=f"{contract} Low"
        ))
    
    fig.update_layout(
        title=f"High and Low Prices for Contracts in Market: {selected_market}",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    
    description_markup = f"**Market Title:** {title}"
    return fig, description_markup

if __name__ == '__main__':
    app.run_server(debug=True)
