
Here are some examples of how to use LangChain, a popular Python framework for developing applications using large language models (LLMs) like OpenAI's GPT:

### 1. Basic Setup
```python
# Install the required package
!pip install langchain openai

# Import necessary modules
from langchain.llms import OpenAI

# Set up your OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'

# Initialize an OpenAI LLM
llm = OpenAI(model_name="gpt-3.5-turbo")

# Use the LLM to generate a response
prompt = "Explain the LangChain framework in simple terms."
response = llm(prompt)
print(response)
```
### 2. Chains Example
LangChain supports building chains where the output of one model can be fed as input to the next model. Here's an example:

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Define a prompt template
template = "Translate this English sentence to French: {sentence}"
prompt = PromptTemplate(template=template, input_variables=["sentence"])

# Create a chain with LLM
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Run the chain with input
result = llm_chain.run("I love programming.")
print(result)  # This will print the translated text in French
```
### 3. Using Tools and Agents
LangChain allows you to integrate external tools or APIs with your chain.

```python
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

# Load tools (like Python REPL or calculator)
tools = load_tools(["serpapi", "llm-math"])

# Initialize an agent
agent = initialize_agent(
    tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Run the agent to answer a question using external tools
response = agent.run("What is the square root of 256?")
print(response)  # Outputs: 16
```
### 4. Memory Example
You can use memory to keep track of previous interactions:

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize memory
memory = ConversationBufferMemory()

# Create a conversation chain
conversation = ConversationChain(llm=llm, memory=memory)

# Engage in a conversation
response_1 = conversation.run("Hi, my name is John.")
response_2 = conversation.run("What's my name?")
print(response_2)  # The model will remember and say "Your name is John."
```
### 5. Custom Prompting
LangChain allows the creation of custom prompt templates for more control:

```python
from langchain.prompts import PromptTemplate

# Define a custom template
template = """You are a helpful assistant.
Respond to this in a friendly way: {question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Use the prompt with the model
response = llm(prompt.format(question="What is LangChain?"))
print(response)
```
### 6. Multi-step Chain
You can also chain multiple LLM responses together in a multi-step chain:

```python
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

# Define two prompt templates
first_template = PromptTemplate(
    template="Summarize the following text: {text}", input_variables=["text"]
)
second_template = PromptTemplate(
    template="Translate the summary to Spanish: {summary}", input_variables=["summary"]
)

# Create two chains
first_chain = LLMChain(llm=llm, prompt=first_template)
second_chain = LLMChain(llm=llm, prompt=second_template)

# Combine the two chains in a sequence
overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain])

# Run the multi-step chain
text = "LangChain is a framework for developing applications powered by language models."
result = overall_chain.run(text)
print(result)  # This will output the summarized text in Spanish
```
These examples should give you a good starting point for building more complex applications using LangChain. The framework allows you to create sophisticated pipelines by chaining together prompts, models, and external tools!
