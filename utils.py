import os
import json
import re
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
from openai import OpenAI

# --------API Key and Model Placeholders-------
API_KEY = ""
MODEL = ""
client = ""


# ------- Utilities ---------
def ask_llm(prompt, temperature=0.2):
    """Send prompt to the LLM and return the text reply."""
    print(
        f"\n[DEBUG] Sending prompt to LLM (temperature={temperature}):\n{prompt[:300]}...\n"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    reply = resp.choices[0].message.content
    print(f"[DEBUG]LLM raw reply (first 300 chars):\n{reply[:300]}...\n")
    return reply


def clean_json(text: str) -> dict:
    """
    Extract JSON object from raw LLM text and parse it. Finds the first '{' and last '}' to retrieve JSON.
    """
    s = text.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("Could not locate JSON braces", s, 0)
    candidate = s[start : end + 1]
    candidate = candidate.replace(",}", "}").replace(",]", "]")
    return json.loads(candidate)


# --------- Wikipedia Search -----------
def wiki_search(query: str) -> str:
    """Wikipedia search and return the summary of the page hits.

    Flow:
    1. Search for the query and take the first result ("first page hit").
       Example: query = "Egypt", search returns ["Egypt"], use "Egypt".
    2. Try to fetch the page summary.
    3. If the query is ambiguous (DisambiguationError):
         - Example: query = "Mercury" can mean the planet or the element.
         - Wikipedia returns options like ["Mercury (planet)", "Mercury (element)"...].
         - Fall back to the first suggested option (e.g. "Mercury (planet)").
    4. If the page does not exist (PageError) or another error occurs:
         - Return an empty string

    """
    print(f"[DEBUG] Searching Wikipedia for: {query}")
    try:
        titles = wikipedia.search(query, results=1)
        print(f"[DEBUG] Search results: {titles}")
        if not titles:
            return ""
        try:
            snippet = wikipedia.summary(titles[0], auto_suggest=False)
            print(
                f"[DEBUG] Retrieved full summary for: {titles[0]} (chars={len(snippet)})"
            )
            return snippet
        except DisambiguationError as e:
            if e.options:
                print(f"[DEBUG] DisambiguationError, trying: {e.options[0]}")
                return wikipedia.summary(e.options[0], auto_suggest=False)
            else:
                return ""
        except PageError:
            return ""
    except Exception as e:
        print(f"[DEBUG] Wikipedia search failed: {e}")
        return ""


def suggest_wiki_queries(topic: str, question: str) -> list[str]:
    """Ask the LLM to propose up to 2 short wikipedia search queries for a given topic and question."""
    prompt = f"""
    Return ONLY a single JSON object like:
    {{"queries": ["...", "..."]}}

    Generate up to 2 concise Wikipedia search queries that would help
    verify an answer for this interview.

    Rules:
    - Keep each query <= 5 words
    - Use simple encyclopaedic terms
    - If a standard Wikipedia page title exists use it
    - Queries should be directly relevant to the topic/question

    Topic: {topic}
    Question: {question}
    """
    raw = ask_llm(prompt, temperature=0.2)
    print(f"[DEBUG] Raw wikipedis query suggestion (first 300):\n{raw[:300]}\n")
    try:
        data = clean_json(raw)
        queries = data.get("queries", [])
        seen, out = set(), []
        for q in queries:
            q_clean = (q or "").strip()
            q_norm = " ".join(q_clean.lower().split())  # lowercase, remove extra spaces
            q_norm = q_norm.rstrip(".")  # remove trailing fullstop
            if q_clean and q_norm not in seen:
                seen.add(q_norm)
                out.append(q_clean)
        return out[:2]
    except Exception as e:
        print(f"[DEBUG] Could not parse wiki query JSON: {e}")
        return []


def build_wiki_context(topic: str, question: str) -> str:
    """Build wikipedia context block.
    Flow:
    - Ask the LLM to suggest up to 2 short wikipedia search queries with suggest_wiki_queries
    - Search each query and fetch the Wikipedia summary.
    - Return them joined together as one context string.
    """
    context_parts, tried = [], set()

    for q in suggest_wiki_queries(topic, question):
        key = (q or "").strip().lower()
        # Skip if query is empty or already tried
        if not key or key in tried:
            continue
        # Track seen query
        tried.add(key)
        summary = wiki_search(q)
        if summary:
            print(
                f"[DEBUG]Wikipedia hit for query: {q} (chars={len(summary)})"
            )  # Check hit and length of summary
            context_parts.append(summary)
        else:
            print(f"[DEBUG] Wikipedia miss for query: {q}")

    return ("\n\n").join(context_parts).strip()


def get_subtopics(topic: str) -> list[str]:
    """Ask the LLM for 3 concise subtopics under the main topic."""
    prompt = f"""
    Return ONLY one JSON object:
    {{"subtopics": ["...", "...", "..."]}}

    Give 3 concise subtopics that help verify expertise under this topic.
    Keep each subtopic <= 5 words.

    Topic: {topic}
    """
    raw = ask_llm(prompt, temperature=0.2)
    try:
        data = clean_json(raw)
        subs = [s.strip() for s in data.get("subtopics", []) if s.strip()]
        return subs[:3] if subs else [topic]
    except Exception:
        return [topic]


# ------- Question Logic --------
def ask_question(
    topic: str,
    subtopic: str,
    last_scores: dict | None,
    prev_q: str = "",
    prev_a: str = "",
) -> tuple[str, str]:
    """Generate the next interview question.
    - If no scores yet: seed question
    - If low correctness: checkpoint question
    - If low specificity: probe question"""
    if not last_scores:
        style = "seed"
        prompt = (
            f"You are interviewing about {topic} - subtopic: {subtopic}.\n"
            f"This is the FIRST question for this subtopic.\n"
            f"Ask ONE clear question that tests knowledge in this area.\n"
            f"It should be checkable and relevant to this subtopic.\n"
            f"Return ONLY the question text, without answers, hints, or commentary. Question:"
        )
    else:
        c = int(last_scores.get("correctness", 0))
        s = int(last_scores.get("specificity", 0))
        if c <= 1:
            style = "checkpoint"
            # Reasoning: Candidate had a low correctness score so verify their knowledge by asking a foundational fact
            prompt = (
                f"Interview on {topic} - subtopic: {subtopic}.\n"
                f"Previous Question: {prev_q}\n"
                f"Previous Answer: {prev_a}\n"
                f"Ask ONE short checkpoint question to verify a key definition or basic fact.\n"
                f"Return ONLY the question text, without answers, hints, or commentary. Question:"
            )
        elif s <= 1:
            style = "probe"
            # Reasoning: Candidate provided a vague answer so verify their knowledge by asking a question that enforces specificity
            prompt = (
                f"Interview on {topic} - subtopic: {subtopic}.\n"
                f"Previous Question: {prev_q}\n"
                f"Previous Answer: {prev_a}\n"
                f"Ask ONE follow-up question that forces the candidate to give a specific example or detail.\n"
                f"Return ONLY the question text, without answers, hints, or commentary. Question:"
            )
        else:
            style = "next"
            # Reasoning: Candidate answered sufficiently, moving on to new question under subtopic
            prompt = (
                f"Interview on {topic} - subtopic: {subtopic}.\n"
                f"Previous Question: {prev_q}\n"
                f"Previous Answer: {prev_a}\n"
                f"Ask ONE different question that is clear and tests knowledge in the subtopic area.\n"
                f"It should be checkable and relevant to this subtopic.\n"
                f"No commentary. Question:"
            )

    print(f"[DEBUG] Question style: {style} ({subtopic})")
    q = ask_llm(prompt, temperature=0.2)
    q = q.replace("Question:", "").replace("QUESTION:", "").strip()
    # Remove anything after 'Answer:'
    q = re.split(r"(?i)\banswer\s*:", q)[0].strip()
    return q, style


def still_poor_after_checkpoint(history: list[dict]) -> bool:
    """Return True if a checkpoint question was asked but correctness never reached >= 2."""
    asked = any(h["style"] == "checkpoint" for h in history)
    max_c = max(
        (int(h["scores"].get("correctness", 0)) for h in history if h.get("scores")),
        default=0,
    )
    return asked and max_c < 2


def still_poor_after_probe(history: list[dict]) -> bool:
    """Return True if a probe question was asked but specificity never reached >= 2."""
    asked = any(h["style"] == "probe" for h in history)
    max_s = max(
        (int(h["scores"].get("specificity", 0)) for h in history if h.get("scores")),
        default=0,
    )
    return asked and max_s < 2
