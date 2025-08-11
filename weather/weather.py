from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("weather")

# Constatnts
NWAS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

async def make_nws_request(url: str) -> dict[str, Any]:
    """Make a request to the NWS API with proper error handling"""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise Exception(f"NWS request failed: {e}")
            return None     

def format_alerts(feature: dict) -> str:
    """Format the alerts into a readable string"""
    props = feature["properties"]
    return f"""
        Event: {props.get('event', 'Unknown')}
        Area: {props.get('areaDes', 'Unknown')}
        Severity: {props.get('severity', 'Unknown')}
        Description: {props.get('description', 'Unknown')}
        Instructions: {props.get('instruction', 'Unknown')}
        """

@mcp.tool()
async def get_alerts(state:str)->str:
    """ Get weather alerts for a specific state
        Args:
            state (str) : Two letters US state code to get alerts for. (e.g. CA, TX)
        Returns:
            str: Formatted string of alerts for the specified state
    """
    url = f"{NWAS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)
    if not data or "features" not in data:
        return "No alerts found for the specified state"
    if not data["features"]:
        return "No alerts found for the specified state"
    return "\n".join(format_alerts(feature) for feature in data["features"])

@mcp.tool()
async def get_forecast(lattitude:float, longitude:float)->str:
    """ Get weather forecast for a specific state
        Args:
            lattitude (float) : Lattitude of the location
            longitude (float) : Longitude of the location
        Returns:
            str: Formatted string of weather forecast for the specified location
    """
    points_url = f"{NWAS_API_BASE}/points/{lattitude},{longitude}"
    points_data = await make_nws_request(points_url)
    if not points_data or "properties" not in points_data:
        return "No forecast found for the specified state"
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)
    if not forecast_data or "properties" not in forecast_data:
        return "No forecast found for the specified state"
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:
        forecast = f"""{period['name']}
        Temparature = {period['temperature']} {period['temperatureUnit']}
        Wind Speed = {period['windSpeed']}
        Wind Direction = {period['windDirection']}
        forecast = {period['detailedForecast']}
        Chance of precipitation = {period['probabilityOfPrecipitation']['value']}%"""
        forecasts.append(forecast)

    return "\n".join(forecasts)


if __name__ == "__main__":
    mcp.run(transport = 'stdio')

        
        

    
            
    
