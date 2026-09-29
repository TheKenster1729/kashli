from clients import KalshiHttpClient, Environment
from dotenv import load_dotenv
import os
from cryptography.hazmat.primitives import serialization
import asyncio

class Client:
    def __init__(self, env = "DEMO"):
        self.env_name = env
        if self.env_name == "DEMO":
            self.env = Environment.DEMO
        elif self.env_name == "PROD":
            self.env = Environment.PROD
        else:
            raise ValueError(f"Invalid environment: {self.env_name}")

        load_dotenv()
        self.key_id = os.getenv(f"{self.env_name}_KEYID")
        self.keyfile = os.getenv(f"{self.env_name}_KEYFILE")
        with open(self.keyfile, "rb") as key_file:
            self.private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None
            )
    
    def get_client(self):
        return KalshiHttpClient(self.key_id, self.private_key, self.env)