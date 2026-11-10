
import os
from dotenv import load_dotenv
from langsmith import traceable

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen2.5:3b"


# Tool 1: Get product price
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""

    print(f">> Executing get_product_price: {product}")

    prices = {
        "laptop": 999.99,
        "phone": 599.99,
        "tablet": 399.99
    }

    return prices.get(product.lower(), 0.0)


# Tool 2: Apply discount
@tool
def apply_discount(price: float, discount: str) -> float:
    """Apply a discount to the price."""

    print(f">> Executing apply_discount: {price}, {discount}")

    discount_percentage = {
        "bronze": 5,
        "silver": 10,
        "gold": 15
    }

    percentage = discount_percentage.get(discount.lower(), 0)

    return round(price * (1 - percentage / 100), 2)


# Agent function
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    # Available tools
    tools = [get_product_price, apply_discount]

    # Dictionary to find tools by name
    tools_dict = {t.name: t for t in tools}

    # Initialize model
    llm = ChatOllama(model=MODEL, temperature=0)

    # Connect tools with model
    llm_with_tools = llm.bind_tools(tools)

    print("Question:", question)
    print("=" * 60)

    # Conversation messages
    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that answers questions "
                "about product prices and discounts. "
                "Use get_product_price to find product prices. "
                "Use apply_discount to calculate discounts. "
                "You can call multiple tools when required. "
                "After getting the tool results, give a final answer."
            )
        ),
        HumanMessage(content=question)
    ]

    # Manual agent loop
    for iteration in range(MAX_ITERATIONS):

        print(f"\nIteration {iteration + 1}:")

        # Ask model what to do
        response = llm_with_tools.invoke(messages)

        print("Response:", response.content)

        # Save AI response
        messages.append(response)

        # Check whether AI wants to call a tool
        if response.tool_calls:

            for tool_call in response.tool_calls:

                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                tool_call_id = tool_call["id"]

                print("Tool:", tool_name)
                print("Input:", tool_input)

                # Find and execute the tool
                if tool_name in tools_dict:

                    tool_func = tools_dict[tool_name]

                    tool_output = tool_func.invoke(tool_input)

                    print("Tool output:", tool_output)

                    # Send tool result back to AI
                    messages.append(
                        ToolMessage(
                            content=str(tool_output),
                            tool_call_id=tool_call_id
                        )
                    )

                else:
                    print(f"Tool {tool_name} not found.")
                    return

        else:
            # No more tool calls: final answer
            print("\nFinal Answer:", response.content)
            break


if __name__ == "__main__":

    question = (
        "What is the price of a laptop "
        "after applying a silver discount?"
    )

    run_agent(question)
