
import os
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search(query: str) -> str:
    """Search for job openings."""
    return str(tavily.search(query=query, max_results=3))


def main():

    model = ChatOllama(model="qwen2.5:3b", temperature=0)

    agent = create_agent(
        model=model,
        tools=[search],
        system_prompt="Find relevant jobs. Do not invent results."
    )

    result = agent.invoke({
        "messages": [
            HumanMessage(
                content="Find 3 current Software Developer fresher jobs in India. Show only company name and job role."
            )
        ]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
