import streamlit as st
from backend import retrieve_chunks
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# Get API key
api_key = os.getenv("API_KEY")


# Configure Gemini API
genai.configure(api_key=api_key)

# --- Select the correct working model ---
MODEL_NAME = "models/gemini-2.5-flash"  # ✅ Available in your account

# Optional config
config = {
    "temperature": 0.5,
    "response_mime_type": "text/plain"
}

# Initialize model
model = genai.GenerativeModel(MODEL_NAME, generation_config=config)

# Streamlit UI
st.title("💼 Smart Career Advisor Chatbot")

query = st.text_input("🔍 Ask your career-related question:")

if query:
    with st.spinner("🔎 Analyzing your query and searching for relevant insights..."):
        context_chunks = retrieve_chunks(query)
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are a professional career advisor.

Use the context provided below to help answer the user's question clearly and helpfully.

If the user mentions a career role (like Data Scientist, UX Designer, etc.),
suggest relevant career paths or internship opportunities.

Return your answer in this format:
1. A clear, short career suggestion or answer.
2. A list of job or internship links (if relevant).

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            response = model.generate_content(prompt)
            st.success("✅ Answer:")
            st.write(response.text)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
