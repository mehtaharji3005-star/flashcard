import json
import os
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# 1. Define the Schema for Flashcards
# ------------------------------------------------------------------
class Flashcard(BaseModel):
    """Schema for an individual study flashcard."""
    question: str = Field(description="A concise question covering a key concept or fact from the notes.")
    answer: str = Field(description="A clear, direct answer to the question.")
    category: str = Field(description="Topic or subject area (e.g., Definitions, Process, Hardware).")

class FlashcardDeck(BaseModel):
    """Schema for a collection of flashcards."""
    deck_name: str = Field(description="A suitable title for the flashcard deck.")
    cards: List[Flashcard] = Field(description="List of extracted flashcards.")

# ------------------------------------------------------------------
# 2. Initialize the Gemini LLM
# ------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2, # Lower temperature for factual accuracy
)

# Enforce structured output matching our Pydantic schema
structured_llm = llm.with_structured_output(FlashcardDeck)

# ------------------------------------------------------------------
# 3. Create the Prompt Template
# ------------------------------------------------------------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert tutor. Analyze the provided study notes and create "
        "high-quality flashcards to help a student master the material. "
        "Focus on key definitions, core concepts, processes, and important facts."
    )),
    ("human", "Study Notes:\n\n{notes}")
])

# Combine into an LCEL (LangChain Expression Language) chain
chain = prompt_template | structured_llm

# ------------------------------------------------------------------
# 4. Generate Flashcards Function
# ------------------------------------------------------------------
def generate_flashcards(notes_text: str) -> FlashcardDeck:
    """Generates structured flashcards from raw note text."""
    result = chain.invoke({"notes": notes_text})
    return result

# ------------------------------------------------------------------
# 5. Example Run
# ------------------------------------------------------------------
if __name__ == "__main__":
    sample_notes = """
    Operating Systems Concept Notes:
    Virtual Memory is a storage allocation scheme in which secondary memory can be addressed as though it were part of main memory.
    The addresses a program may use to reference memory are called virtual addresses.
    Demand Paging is a method of virtual memory management where pages are loaded only when they are demanded by execution, not in advance.
    A Page Fault occurs when a program attempts to access data or code that is in its address space, but is not currently located in RAM.
    """

    print("Generating flashcards...\n")
    deck: FlashcardDeck = generate_flashcards(sample_notes)

    # Print summary
    print(f"=== Deck: {deck.deck_name} ({len(deck.cards)} cards) ===\n")
    
    for idx, card in enumerate(deck.cards, 1):
        print(f"Card {idx} [{card.category}]")
        print(f"  Q: {card.question}")
        print(f"  A: {card.answer}")
        print("-" * 40)

    # Optional: Save deck to JSON file for a front-end UI
    with open("flashcards.json", "w", encoding="utf-8") as f:
        json.dump(deck.model_dump(), f, indent=2)
    print("\nSaved to flashcards.json!")
