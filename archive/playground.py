import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
series = "KXBTCD"

url = f"{BASE_URL}/events"
params = {"series_ticker": series, "limit": 200}

events = []
first_cursor = None

s = requests.Session()
s.headers.update({"accept": "application/json"})
session = s

while True:
    response = session.get(url, params=params).json()
    events.extend(response.get("events", []))
    cursor = response.get("cursor")

    if not cursor:
        break
    if first_cursor is None:
        first_cursor = cursor
    elif cursor == first_cursor:
        break

    params["cursor"] = cursor

filtered = []
for event in self.get_events():
    dt = self.parse_event_date(event["event_ticker"])
    if dt and start_date <= dt <= end_date:
        filtered.append(event)
print(filtered)
