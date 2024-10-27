import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM with Groq API
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv('GROQ_API_KEY'),  
    model_name="llama-3.1-70b-versatile"
)

# Load data from the provided job listing URL
loader = WebBaseLoader("https://merojob.com/senior-laravel-developer-34/")
page_data = loader.load().pop().page_content

# Limit the web text to 10,000 words
words = page_data.split()
web_text = ' '.join(words[:10000])  

# Create the prompt template
prompt_extract = PromptTemplate.from_template(
    """
    I will give you the job listing text, please extract the following sections:
      - "About the job"
      - "Role Description"
      - "Qualifications"
      - "Experience"
      - "Skills required" (ignore "Skills missing" messages)
      - "Job Description"
      - "Responsibilities"
      - "Job time"
      - and other key points the reader should note.

    Here is the job listing text:

    {web_text}
    """
)

# Combine the prompt and the LLM in a chain
chain_extract = prompt_extract | llm

# Run the chain
res = chain_extract.invoke({'web_text': web_text})

# Print the result
print(res)
