import streamlit as st
from backend import retrieve_chunks
import google.generativeai as ai

ai.configure(api_key="AIzaSyAYnk8L9S2ahnE0GXggVMEOcRMUJHHsQkI")
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
You are a professional career advisor. Use the context provided below to answer the user's question clearly and helpfully.

If the context is not sufficient to fully answer the question, use your own knowledge to provide the best possible advice.

Context:{context}
Question:{query}
answer:
"""

        response = model.generate_content(prompt)
        st.success("Answer:")
        st.write(response.text)
