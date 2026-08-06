import json
import os
import streamlit as st
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# 1. API Key Handling
# ------------------------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not api_key:
        st.warning("Please provide a Gemini API Key in Streamlit Secrets or via the sidebar.")
        st.stop()

# ------------------------------------------------------------------
# 2. Define Schema
# ------------------------------------------------------------------
class Flashcard(BaseModel):
    question: str = Field(description="A concise question covering a key concept or fact from the notes.")
    answer: str = Field(description="A clear, direct answer to the question.")
    category: str = Field(description="Topic or subject area (e.g., Definitions, Process, Hardware).")

class FlashcardDeck(BaseModel):
    deck_name: str = Field(description="A suitable title for the flashcard deck.")
    cards: List[Flashcard] = Field(description="List of extracted flashcards.")

# ------------------------------------------------------------------
# 3. Initialize LLM & Chain
# ------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    google_api_key=api_key
)

structured_llm = llm.with_structured_output(FlashcardDeck)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert tutor. Analyze the provided study notes and create "
        "high-quality flashcards to help a student master the material."
    )),
    ("human", "Study Notes:\n\n{notes}")
])

chain = prompt_template | structured_llm

# ------------------------------------------------------------------
# 4. Streamlit App Interface
# ------------------------------------------------------------------
st.title("⚡ AI Flashcard Generator")

notes_input = st.text_area("Paste your study notes here:", height=200)

if st.button("Generate Flashcards") and notes_input:
    with st.spinner("Analyzing notes and generating cards..."):
        try:
            deck = chain.invoke({"notes": notes_input})
            st.success(f"Generated {len(deck.cards)} flashcards for '{deck.deck_name}'!")
            
            for idx, card in enumerate(deck.cards, 1):
                with st.expander(f"Card {idx}: {card.question} ({card.category})"):
                    st.write(f"**Answer:** {card.answer}")
        except Exception as e:
            st.error(f"Error generating flashcards: {e}")
