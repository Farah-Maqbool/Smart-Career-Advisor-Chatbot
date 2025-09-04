import streamlit as st
from backend import retrieve_chunks
import google.generativeai as ai
import streamlit as st
import google.generativeai as ai
import os
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# Try secrets first, then fallback to .env
api_key = st.secrets.get("API_Key", os.getenv("API_Key"))


config = {
            "temperature": 0.5,
            "response_mime_type": "text/plain"
        }
model = ai.GenerativeModel('gemini-2.5-flash-preview-04-17',generation_config=config)

st.title("Smart Career Advisor Chatbot")

query = st.text_input("Enter your question")

if query:
    with st.spinner("Searching..."):
        context_chunks = retrieve_chunks(query)
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are a professional career advisor.

Use the context provided below to help answer the user's question clearly and helpfully.

If the user mentions a career role (like Data Scientist, UX Designer, etc.), 
search for relevant jobs or internship opportunities online and include direct links.

Return your answer in this format:
1. A clear, short career suggestion or answer.
2. A list of job or internship links (if found).


Context:{context}
Question:{query}
answer:
"""

        response = model.generate_content(prompt)
        st.success("Answer:")
        st.write(response.text)

