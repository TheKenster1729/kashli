import numpy as np
import pandas as pd
from utils import Client
import requests
from pprint import pprint

class Arbys:
    def __init__(self):
        pass

    def events_request(self, cursor, original_cursor = None, events = []):
        url = "https://api.elections.kalshi.com/trade-api/v2/events"
        headers = {"accept": "application/json"}
        params = {"status": ["open"], "cursor": cursor}
        response = requests.get(url, headers=headers, params=params).json()
        new_cursor = response["cursor"]
        new_events = response["events"]
        events = events + new_events

        if new_cursor == original_cursor:
            return events
        else:
            return self.events_request(new_cursor, original_cursor, events)

    def parse_event(self, event):
        name = event["title"]
        liq = event["liquidity"]

        return name, liq

    def parse_events(self, events):
        names = []
        for event in events:
            name, liq = self.parse_event(event)
            if liq > 0:
                names.append(name)
        return names

    def get_all_events(self):
        url = "https://api.elections.kalshi.com/trade-api/v2/events"
        headers = {"accept": "application/json"}
        params = {"status": ["open"]}
        response = requests.get(url, headers=headers, params=params).json()
        events = self.events_request(response["cursor"], response["cursor"])

        return events
    
    def get_contract_info(self, id):
        url = f"https://api.elections.kalshi.com/trade-api/v2/events/{id}"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers).json()

        return response

    def arb_opportunity(self, id):
        contract_info = self.get_contract_info(id)
        markets_info = contract_info["markets"]
        no_ask = []
        if len(markets_info) > 1:
            for market in markets_info:
                no_ask.append(market["no_ask"])

        if len(no_ask) - sum(no_ask) > 1:
            return markets_info

    def find_arb(self):
        events = self.get_all_events()
        for event in events:
            event_id = event["event_ticker"]
            event_name = event["title"]
            arb_opportunity = self.arb_opportunity(event_id)
            if arb_opportunity:
                print(event_name)

Arbys().find_arb()