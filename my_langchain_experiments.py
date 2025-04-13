"""
Langchain Experiments
This file contains various experiments with Langchain, including:
1. Simple Chains
2. Sequential Chains
3. Tools and Utilities
4. Prompt Templates
5. Experimental Features
"""

# Import required libraries
import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains import SimpleSequentialChain
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
import google.generativeai as genai

# Load environment variables
load_dotenv('conf/.env', override=True)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# ============================================================================
# SIMPLE CHAINS
# ============================================================================

def virus_description_chain():
    """
    Creates a simple chain that generates descriptions of viruses in different languages.
    Uses a prompt template with the Google Generative AI model.
    """
    # Define the prompt template
    template = '''
    You are an experienced virologist
    Write a few sentences about the following virus "{virus}" in {language}
    '''
    prompt_template = PromptTemplate.from_template(template=template)
    
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
    
    # Create the chain using the pipe operator
    chain = prompt_template | llm | StrOutputParser()
    
    # Example usage
    output = chain.invoke({'virus': 'HIV', 'language': 'Spanish'})
    print("Virus Description Chain Output:")
    print(output)
    print("\n" + "="*50 + "\n")

def capital_cities_chain():
    """
    Creates a chain that provides information about capital cities and tourist attractions.
    """
    # Define the prompt template
    template = 'What is the capital of {country}?. List the top 3 places to visit in that city. Use bullet points'
    prompt_template = PromptTemplate.from_template(template=template)
    
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
    
    # Create the chain
    chain = prompt_template | llm | StrOutputParser()
    
    # Example usage
    output = chain.invoke({'country': 'India'})
    print("Capital Cities Chain Output:")
    print(output)
    print("\n" + "="*50 + "\n")

# ============================================================================
# SEQUENTIAL CHAINS
# ============================================================================

def python_function_chain():
    """
    Creates a sequential chain that:
    1. Generates a Python function based on a concept
    2. Provides a detailed description of the generated function
    """
    # First chain: Generate Python function
    llm1 = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
    template1 = '''
    You are an experienced data scientist with expertise in python programming. 
    Write a function that implements the concept of {concept}
    '''
    prompt_template1 = PromptTemplate.from_template(template=template1)
    chain1 = prompt_template1 | llm1 | StrOutputParser()
    
    # Second chain: Describe the function
    llm2 = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0)
    template2 = '''
    Describe the following python function in detail also detail internal details 
    of python implementations and the performance of the code

    {function}
    '''
    prompt_template2 = PromptTemplate.from_template(template=template2)
    chain2 = prompt_template2 | llm2 | StrOutputParser()
    
    # Create sequential chain
    seq_chain = chain1 | chain2
    
    # Example usage
    output = seq_chain.invoke('linear regression')
    print("Python Function Chain Output:")
    print(output)
    print("\n" + "="*50 + "\n")

# ============================================================================
# TOOLS AND UTILITIES
# ============================================================================

def duckduckgo_search_example():
    """
    Demonstrates the use of DuckDuckGo search tools in Langchain.
    """
    # Basic search
    search = DuckDuckGoSearchRun()
    output = search.invoke('What is the best translation of Gita in Hindi and by whom.')
    print("DuckDuckGo Search Output:")
    print(output)
    print("\n" + "="*50 + "\n")
    
    # Search with results
    search2 = DuckDuckGoSearchResults()
    output2 = search2.run('What is the best translation of Gita in Hindi and by whom?')
    print("DuckDuckGo Search Results Output:")
    print(output2)
    print("\n" + "="*50 + "\n")
    
    # Search with API wrapper
    wrapper = DuckDuckGoSearchAPIWrapper(
        region='de-de',
        max_results=3,
        safesearch='moderate'
    )

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def json_format_example():
    """
    Demonstrates the use of chat prompt templates with JSON formatting.
    """
    # Create a chat template that enforces JSON output
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content='You respond only in json format'),
        HumanMessagePromptTemplate.from_template('Top {n} countries in {area} by population.')
    ])
    
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    
    # Format messages and get response
    messages = chat_template.format_messages(n='10', area='Europe')
    output = llm.invoke(messages)
    print("JSON Format Example Output:")
    print(output.content)
    print("\n" + "="*50 + "\n")

def system_message_example():
    """
    Demonstrates the use of system messages with the language model.
    """
    # Create messages with system and human content
    messages = [
        SystemMessage(content="You are a physicist and respond in hindi"),
        HumanMessage(content="explain quantum mechanics in one sentence")
    ]
    
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    
    # Get response
    output = llm.invoke(messages)
    print("System Message Example Output:")
    print(output.content)
    print("\n" + "="*50 + "\n")

# ============================================================================
# EXPERIMENTAL FEATURES
# ============================================================================

def python_repl_example():
    """
    Demonstrates the use of Python REPL utility from langchain_experimental.
    """
    # Initialize Python REPL
    python_repl = PythonREPL()
    
    # Run a simple Python command
    result = python_repl.run('print([n for n in range(1,100) if n%13==0])')
    print("Python REPL Example Output:")
    print(result)
    print("\n" + "="*50 + "\n")

def python_agent_example():
    """
    Demonstrates the use of Python agent from langchain_experimental.
    """
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0)
    
    # Create a Python agent
    agent = create_python_agent(llm=llm, tool=PythonREPLTool(), verbose=True)
    
    # Run the agent with a calculation task
    result = agent.invoke('Calculate 1.15 to the power 7.25')
    print("Python Agent Example Output:")
    print(result)
    print("\n" + "="*50 + "\n")

def list_google_models():
    """
    Lists all available Google Generative AI models.
    """
    print("Available Google Generative AI Models:")
    for model in genai.list_models():
        print(model.name)
    print("\n" + "="*50 + "\n")

def main():
    """
    Main function to run all examples.
    """
    print("Running Langchain Experiments...\n")
    
    # Run simple chains
    print("1. Simple Chains")
    virus_description_chain()
    capital_cities_chain()
    
    # Run sequential chains
    print("2. Sequential Chains")
    python_function_chain()
    
    # Run tools examples
    print("3. Tools and Utilities")
    duckduckgo_search_example()
    
    # Run prompt template examples
    print("4. Prompt Templates")
    json_format_example()
    system_message_example()
    
    # Run experimental features
    print("5. Experimental Features")
    python_repl_example()
    python_agent_example()
    list_google_models()

if __name__ == "__main__":
    main() 