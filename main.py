
import os
from typing import List

from dotenv import load_dotenv
from tavily import TavilyClient
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# Pydantic schemas
class Source(BaseModel):
    url: str = Field(description="URL of the job opening")


class AgentResponse(BaseModel):
    answer: List[str] = Field(
        description="Company name and job role"
    )

    sources: List[Source] = Field(
        description="Sources used to find job openings"
    )


# Load environment variables
load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError("TAVILY_API_KEY not found in .env file")

tavily = TavilyClient(api_key=api_key)


# Search tool
@tool
def search(query: str) -> str:
    """Search the internet for current job openings."""
    return str(tavily.search(query=query, max_results=3))


def main():

    # Initialize Ollama model
    model = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    # Add structured output
    structured_model = model.with_structured_output(AgentResponse)

    # Search using Tavily
    search_results = search.invoke(
        {"query": "3 current Software Developer fresher jobs in India"}
    )

    # Generate structured response
    result = structured_model.invoke(
        [
            HumanMessage(
                content=f"""
                Find 3 current Software Developer fresher jobs in India.

                Use these search results:
                {search_results}

                Return company names and job roles in answer.
                Return source URLs in sources.
                Do not invent information.
                """
            )
        ]
    )

    # Display results
    print("\nJob Openings:\n")

    for job in result.answer:
        print("-", job)

    print("\nSources:\n")

    for source in result.sources:
        print("-", source.url)


if __name__ == "__main__":
    main()
