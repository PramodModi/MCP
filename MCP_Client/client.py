import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

async def main():
    print("Hello MCP Client")


if __name__=="__main__":
    asyncio.run(main())