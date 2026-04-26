from mcp.server.fastmcp import FastMCP
from random import choice
mcp = FastMCP("Random Name")

@mcp.tool()
def get_random_name(names: list = None)->str:
    """
    Gets a random peoples names. The names are stored in 
    a local array.
    args: names: the user can pass in a list of names 
    to choose from, or it will default to predefined list.
    """
    if names and isinstance(names, list):
        return choice(names)
    else:
        default_names = ["Aline", "Bob", "Charlie", "Diana", "Eve", "Grace", "Jack"]
        return choice(default_names)

@mcp.tool()
def get_my_info()->dict:
    """
    Gets information about me, like name, location.
    """
    return {"Name" : "Pramod", "Location": "Bangalore"}

if __name__ == "__main__":
    mcp.run()