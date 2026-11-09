from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from langsmith import traceable

@traceable
def test_tracing():
    return "LangSmith tracing test successful!"

print(test_tracing())
def main():
    print("hello, World!")
    information="""
        Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is an industrialist, technologist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire (in terms of US dollars) in June 2026; as of October 2026, Forbes estimates his net worth to be $1.1 trillion.

        Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal. Musk also became an American citizen in 2002.
        """
    summary_template="""
        Summarize the following information in a 3 shorts points :{information}"""
    summary_prompt_template=PromptTemplate(
        input_variables=["information"],template=summary_template
    )
    llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain=summary_prompt_template | llm
    response=chain.invoke({"information":information})
    print(response.content)
    
if __name__ == "__main__":
    main()
