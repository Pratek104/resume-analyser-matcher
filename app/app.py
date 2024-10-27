import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


load_dotenv()


llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name="llama-3.1-70b-versatile"
)

st.title("Resume and Job Listing Matcher")

# Section for uploading resume PDF
st.header("Upload Your Resume (PDF)")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text
    words = full_text.split()
    limited_text = ' '.join(words[:10000])  



    # Section for entering the job listing URL
    st.header("Enter Job Listing URL")
    job_url = st.text_input("Job Listing URL", "")

    if job_url:
        # Load data from the provided job listing URL
        loader = WebBaseLoader(job_url)
        page_data = loader.load().pop().page_content
        words = page_data.split()
        web_text = ' '.join(words[:10000])  

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

        # Extract job information
        chain_extract_job = LLMChain(
            llm=llm,
            prompt=prompt_extract
        )
        job_info = chain_extract_job.run({
            'web_text': web_text
        })

        # Set up a prompt for resume and job listing comparison
        prompt_compare = PromptTemplate.from_template(
            """
            I will provide two texts:

            1. The job listing from the company's job portal:
            "{job_info}"

            2. My resume:
            "{limited_text}"

            Context:
            - Please compare the job description and my resume.
            - Focus on the following sections:
                - Qualifications
                - Skills
                - Experience

            Task:
            - Provide a detailed comparison of how well my resume matches the job requirements.
            - Rate the match as a percentage based on the alignment of skills, qualifications, and experience.
            -Also suggest some improvement tips.
            -Suggest best job for their resume
            -If something is missing in their resume the give some advise
            """
        )

        # Perform the comparison
        chain_compare = LLMChain(
            llm=llm,
            prompt=prompt_compare
        )
        comparison_result = chain_compare.run({
            'job_info': job_info,
            'limited_text': limited_text
        })

        # Display the comparison result
        st.subheader("Resume and Job Listing Match Result")
        st.code(comparison_result)
