from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
print("OPENAI_API_KEY present?", bool(os.getenv("OPENAI_API_KEY")))

client = OpenAI()
print("Client created OK")
