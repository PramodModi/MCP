from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Weather")

@mcp.tool()
def get_weather(city: str) -> str:
    """Get weather for a city"""
    return f"Weather for {city} is sunny"

if __name__ == "__main__":
    mcp.run(transport='sse')