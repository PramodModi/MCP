import asyncio
from dotenv import load_dotenv
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mistralai import ChatMistralAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(api_key=mistral_api_key)

stdio_server_params = StdioServerParameters(
    command = "python",
    args = ["/Users/pramodmodi/Documents/Learning/git/MCP/MCP_Client/servers/math_server.py"]
)

async def main():
   async with stdio_client(stdio_server_params) as (read, write):
    async with ClientSession(read_stream=read, write_stream=write) as session:
        await session.initialize()
        #print("Session initialized ")
        # tools = await session.list_tools() Thiss gives the tool from mcp server.
        # print("Tools: ", tools)
        # Tools from mcp server neeeds to be converted into langchain tools.
        tools = await load_mcp_tools(session)
        #print("Tools: ", tools)
        ## Create Host that is react agent here
        agent = create_react_agent(llm, tools)
        
        result = await agent.ainvoke({"messages": [HumanMessage(content="What is 52 + 2 *3")]})
        print(result["messages"][-1].content)

        

if __name__=="__main__":
    asyncio.run(main())