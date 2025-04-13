#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Langchain Chains Examples
This file contains examples of simple and sequential chains using Langchain with Google Generative AI.
"""

# ===== IMPORTS AND SETUP =====
import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SimpleSequentialChain

# Load environment variables from .env file
load_dotenv('conf/.env', override=True)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
print(f"API Key loaded: {GOOGLE_API_KEY[:10]}...")

# ===== SIMPLE CHAIN EXAMPLES =====

# Example 1: Virus Description Chain
# This chain generates a description of a virus in a specified language
def virus_description_example():
    print("\n===== VIRUS DESCRIPTION EXAMPLE =====")
    
    # Define the prompt template for virus descriptions
    virus_template = '''
    You are an experienced virologist
    Write a few sentences about the following virus "{virus}" in {language}
    '''
    
    # Create the prompt template
    prompt_template = PromptTemplate.from_template(template=virus_template)
    
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
    
    # Create a pipeline using the pipe (|) operator
    chain = prompt_template | llm | StrOutputParser()
    
    # Invoke the chain with specific parameters
    output = chain.invoke({'virus': 'HIV', 'language': 'Spanish'})
    
    # Print the result
    print(f"Output type: {type(output)}")
    print(f"Output: {output}")
    
    return output

# Example 2: Capital City Information Chain
# This chain provides information about a capital city and tourist attractions
def capital_city_example():
    print("\n===== CAPITAL CITY EXAMPLE =====")
    
    # Define the prompt template for capital city information
    template = 'What is the capital of {country}?. List the top 3 places to visit in that city. Use bullet points'
    
    # Create the prompt template
    prompt_template = PromptTemplate.from_template(template=template)
    
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
    
    # Create a pipeline using the pipe (|) operator
    chain = prompt_template | llm | StrOutputParser()
    
    # Get country input from user (for demonstration, we'll use a fixed value)
    country = "India"  # In the notebook, this was: country = input('Enter Country: ')
    
    # Invoke the chain with the country parameter
    output = chain.invoke({'country': country})
    
    # Print the result
    print(f"Output type: {type(output)}")
    print(f"Output: {output}")
    
    return output

# ===== SEQUENTIAL CHAIN EXAMPLES =====

# Example 3: Python Function Generation and Description
# This sequential chain first generates a Python function and then describes it in detail
def sequential_chain_example():
    print("\n===== SEQUENTIAL CHAIN EXAMPLE =====")
    
    # Initialize the first language model for function generation
    llm1 = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
    
    # Define the first prompt template for generating Python functions
    template1 = '''
    You are an experienced data scientist with expertise in python programming. Write a function that implements the concept of {concept}
    '''
    
    # Create the first prompt template
    prompt_template1 = PromptTemplate.from_template(template=template1)
    
    # Create the first chain
    chain1 = prompt_template1 | llm1 | StrOutputParser()
    
    # Initialize the second language model for function description
    llm2 = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0)
    
    # Define the second prompt template for describing Python functions
    template2 = '''
    Describe the following python function in detail also detail internal details of python implementations and the performance of the code

    {function}
    '''
    
    # Create the second prompt template
    prompt_template2 = PromptTemplate.from_template(template=template2)
    
    # Create the second chain
    chain2 = prompt_template2 | llm2 | StrOutputParser()
    
    # Create a sequential chain by combining the two chains
    seq_chain = chain1 | chain2
    
    # Invoke the sequential chain with a concept
    output = seq_chain.invoke('linear regression')
    
    # Print the result
    print(f"Output type: {type(output)}")
    print(f"Output: {output}")
    
    return output

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Run the examples
    virus_description_example()
    capital_city_example()
    sequential_chain_example() 