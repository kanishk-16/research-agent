"""
Phase 1 — Foundation
Goal: topic -> one web search -> one LLM summary with one source.
This is the CLI deliverable for Week 1. Every later phase builds on this.
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    print("ERROR: Missing API key(s). Check your .env file has GROQ_API_KEY and TAVILY_API_KEY set.")
    sys.exit(1)

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def search_web(topic: str):
    """Run one Tavily search and return the top result."""
    try:
        response = tavily_client.search(query=topic, search_depth="basic", max_results=1)
        results = response.get("results", [])
        if not results:
            return None
        return results[0]  # {title, url, content, ...}
    except Exception as e:
        print(f"[search_web] Search failed: {e}")
        return None


def summarize_with_llm(topic: str, source_content: str, source_url: str):
    """One LLM call: summarize the source content into a paragraph."""
    prompt = (
        f"Topic: {topic}\n\n"
        f"Source content:\n{source_content}\n\n"
        f"Write a single, factual paragraph summarizing what this source says "
        f"about the topic. Do not add outside knowledge."
    )
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[summarize_with_llm] LLM call failed: {e}")
        return None


def run(topic: str):
    print(f"\nResearching: {topic}\n")

    result = search_web(topic)
    if result is None:
        print("No search results found. Try a different topic or check your Tavily key.")
        return

    source_url = result.get("url", "unknown")
    source_content = result.get("content", "")

    summary = summarize_with_llm(topic, source_content, source_url)
    if summary is None:
        print("Summary generation failed. Check your Groq key/rate limits.")
        return

    print("SUMMARY")
    print("-" * 60)
    print(summary)
    print("-" * 60)
    print(f"Source: {source_url}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py "your research topic"')
        sys.exit(1)

    topic_input = " ".join(sys.argv[1:])
    run(topic_input)
