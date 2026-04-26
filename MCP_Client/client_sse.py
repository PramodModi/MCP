import asyncio
import os
from dotenv import load_dotenv

from mcp.client.sse import sse_client      
from mcp.client.session import ClientSession 

from langchain_mistralai import ChatMistralAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Load env
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# LangChain LLM
llm = ChatMistralAI(api_key=mistral_api_key)


async def main():
    # Connect to MCP server over SSE
    async with sse_client("http://127.0.0.1:8000/") as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("Connected to MCP server via SSE")

            # Load MCP tools and wrap them for LangChain
            tools = await load_mcp_tools(session)
            print(f"Loaded {len(tools)} tools from MCP server")

            # Create an agent (ReAct pattern) with tools
            agent = create_react_agent(llm, tools)

            # Run a sample query using LangGraph agent
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="How is the weather in Bangalore?")]}
            )
            print("🤖 Agent Response:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
