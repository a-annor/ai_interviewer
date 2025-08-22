import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# interview functions
import utils as interview

load_dotenv()
API_KEY = os.getenv("API_KEY", "")
MODEL = "deepseek/deepseek-r1:free"

st.set_page_config(page_title="AI Interviewer", page_icon="📋", layout="wide")
st.title("AI Interviewer")


if not API_KEY:
    st.error("Missing API key. Add API_KEY to .env and refresh.")
    st.stop()

# Add client and LLM model to interview module
interview.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
interview.MODEL = MODEL

# ------ Sidebar -----
st.sidebar.header("Interview Settings")

user_topic = st.sidebar.text_input("Enter interview topic", key="topic_input")

max_subtopics = st.sidebar.slider(
    "Max subtopics", 1, 5, 2, help="Number of subtopics to cover."
)

user_subtopics = [
    st.sidebar.text_input(
        f"Subtopic {i+1}",
        "",
        key=f"subtopic_{i+1}",
        help="Optional: leave all subtopics blank to autogenerate with LLM",
    )
    for i in range(max_subtopics)
]
user_subtopics = [s.strip() for s in user_subtopics if s.strip()]

# Valid if either:
# All subtopics are blank (LLM autogenerate) OR
# Exactly max_subtopics are filled (manual input)
valid_subtopics = len(user_subtopics) == 0 or len(user_subtopics) == max_subtopics

if not valid_subtopics:
    st.sidebar.warning(
        f"Enter exactly {max_subtopics} subtopics, "
        f"or leave all blank to autogenerate."
    )
# Button to start interview (disabled unless inputs are valid)
start_button = st.sidebar.button(
    "Start/Restart interview",
    type="primary",
    disabled=not valid_subtopics,
)


# ---- Session state ------
def init_state():
    st.session_state.topic = (user_topic or "").strip()
    st.session_state.subtopics = []
    st.session_state.sub_idx = 0
    st.session_state.turns_in_sub = 0
    st.session_state.last_scores = None
    st.session_state.history_sub = []
    st.session_state.full_history = []
    st.session_state.poor_areas = []
    st.session_state.question = ""
    st.session_state.style = ""
    st.session_state.awaiting_answer = False
    st.session_state.finished = False


if "topic" not in st.session_state:
    init_state()

if start_button:
    init_state()
    if not st.session_state.topic:
        st.warning(
            "Enter a topic and press 'Start/Restart interview'. Fill all subtopics for manual input or leave blank to autogenerate."
        )
        st.stop()
    else:
        try:
            if user_subtopics:
                # Use manual input
                st.session_state.subtopics = user_subtopics
            else:
                # Autogenerate
                st.session_state.subtopics = interview.get_subtopics(
                    st.session_state.topic
                )[:max_subtopics]
        except Exception as e:
            st.error(f"Failed to get subtopics: {e}")
        st.rerun()


# ------ Main flow ----
if st.session_state.finished:
    # Interview is done
    st.success("Interview complete.")
else:
    topic = st.session_state.topic
    subtopics = st.session_state.subtopics

    # In case no subtopics set on rerun
    if not subtopics:
        st.info(
            "Enter a topic and press 'Start/Restart interview'. Fill all subtopics for manual input or leave blank to autogenerate."
        )
        st.stop()

    # Display all subtopics
    st.subheader("All subtopics for this interview")
    for i, s in enumerate(subtopics, 1):
        st.write(f"{i}. {s}")

    st.divider()

    # Current subtopic
    sub = subtopics[st.session_state.sub_idx]

    st.subheader(f"Subtopic {st.session_state.sub_idx + 1} of {len(subtopics)}: {sub}")

    # If no current question, generate one
    if not st.session_state.question or not st.session_state.awaiting_answer:
        try:
            prev_q = (
                st.session_state.history_sub[-1]["question"]
                if st.session_state.history_sub
                else ""
            )
            prev_a = (
                st.session_state.history_sub[-1]["answer"]
                if st.session_state.history_sub
                else ""
            )
            q, style = interview.ask_question(
                topic, sub, st.session_state.last_scores, prev_q, prev_a
            )
        except Exception as e:
            st.error(f"Failed to generate question: {e}")
            st.stop()
        st.session_state.question = q
        st.session_state.style = style
        st.session_state.awaiting_answer = True

    # Show current question
    st.markdown("#### Question")
    st.write(st.session_state.question)
    # Answer Input
    with st.form("answer_form", clear_on_submit=True):
        answer = st.text_area(
            "Your answer", height=160, placeholder="Type your answer here…"
        )
        submitted = st.form_submit_button("Submit answer")

    if submitted:
        st.session_state.turns_in_sub += 1
        with st.spinner("Processing your answer..."):
            if not answer.strip():
                # Empty answer then record zeros, stay on subtopic
                scores = {"correctness": 0, "specificity": 0}
                entry = {
                    "subtopic": sub,
                    "style": st.session_state.style,
                    "question": st.session_state.question,
                    "answer": "",
                    "scores": scores,
                    "evidence": {"verdict": "insufficient", "reason": "no answer"},
                }
                st.session_state.history_sub.append(entry)
                st.session_state.full_history.append(entry)
                st.session_state.last_scores = scores
                st.session_state.awaiting_answer = False
                st.rerun()

            # For non empty answer build context and grade
            context = ""
            try:
                # Build Wikipedia context
                context = interview.build_wiki_context(topic, st.session_state.question)
            except Exception as e:
                st.warning(f"Wikipedia context unavailable: {e}")

            # Ask LLM to grade the answer
            grading_prompt = f"""
            You are grading an interview answer.

            Provide ONLY a JSON object:
            {{"correctness": <0-3>, "specificity": <0-3>, "evidence": {{"verdict": "supported" | "contradicted" | "insufficient", "reason": "short sentence"}}}}

            Scoring criteria:
            - correctness:
            0 = wrong/irrelevant
            1 = mostly wrong or missing key points
            2 = mostly correct but incomplete or vague
            3 = fully correct, complete, and precise
            - specificity:
            0 = vague/general
            1 = somewhat specific, lacks detail
            2 = specific but not thorough
            3 = highly precise with clear details/examples

            Rules:
            - Use CONTEXT to inform correctness:
            - If CONTEXT explicitly refutes a claim in the ANSWER, verdict="contradicted" and correctness <= 1.
            - If CONTEXT supports key claims, verdict="supported" and correctness >= 2 (unless other major errors).
            - If CONTEXT is unrelated/insufficient, verdict="insufficient". Do NOT penalise correctness for irrelevance, judge correctness from general knowledge.

            Topic: {topic}
            Question: {st.session_state.question}
            Context: {context if context else "[no context found]"}
            Answer: {answer}
            """
            try:
                raw = interview.ask_llm(grading_prompt, temperature=0.0)
                result = interview.clean_json(raw)
            except Exception:
                result = {
                    "correctness": 0,
                    "specificity": 0,
                    "evidence": {
                        "verdict": "insufficient",
                        "reason": "Could not parse JSON",
                    },
                }

            # Show scores
            c, s = int(result.get("correctness", 0)), int(result.get("specificity", 0))
            ev = result.get("evidence", {})
            col1, col2 = st.columns(2)
            col1.metric("Correctness (0–3)", c)
            col2.metric("Specificity (0–3)", s)
            if ev:
                st.caption(f"Evidence: {ev.get('verdict')} - {ev.get('reason')}")

            # Record answer in session history
            entry = {
                "subtopic": sub,
                "style": st.session_state.style,
                "question": st.session_state.question,
                "answer": answer,
                "scores": result,
                "evidence": ev,
            }
            st.session_state.history_sub.append(entry)
            st.session_state.full_history.append(entry)
            st.session_state.last_scores = result
            st.session_state.awaiting_answer = False

            # Advance to next subtopic?
            moved = False
            last_subtopic = st.session_state.sub_idx >= len(subtopics) - 1

            if st.session_state.turns_in_sub >= 2:
                info_statement = "Subtopic turn limit reached."
                if not last_subtopic:
                    info_statement += " Advancing to next subtopic."
                st.info(info_statement)
                moved = True

            if moved:
                # Flag poor areas for this subtopic
                if interview.still_poor_after_checkpoint(
                    st.session_state.history_sub
                ) or interview.still_poor_after_probe(st.session_state.history_sub):
                    reasons = []
                    if interview.still_poor_after_checkpoint(
                        st.session_state.history_sub
                    ):
                        reasons.append(
                            "correctness stayed < 2 after a checkpoint question"
                        )
                    if interview.still_poor_after_probe(st.session_state.history_sub):
                        reasons.append("specificity stayed < 2 after a probe question")
                    st.session_state.poor_areas.append(
                        {
                            "subtopic": sub,
                            "reasons": reasons,
                            "turns": list(st.session_state.history_sub),
                        }
                    )

                # Advance to next subtopic or finish
                st.session_state.sub_idx += 1
                st.session_state.turns_in_sub = 0
                st.session_state.last_scores = None
                st.session_state.history_sub = []
                st.session_state.question = ""
                st.session_state.style = ""
                if st.session_state.sub_idx >= len(subtopics):
                    st.session_state.finished = True

            st.rerun()

# ------- Final report --------
if st.session_state.get("finished", False):
    st.header("Final report")

    # Questions and answers in expanders
    st.subheader("Interview Record")
    for i, entry in enumerate(st.session_state.full_history, 1):
        with st.expander(f"Q{i}: {entry['question']}"):
            st.markdown(f"**Answer:** {entry['answer'] or '(no answer)'}")
            s = entry.get("scores", {})
            ev = entry.get("evidence", {})
            st.write(f"- Correctness: {s.get('correctness', 0)}")
            st.write(f"- Specificity: {s.get('specificity', 0)}")
            if ev:
                st.caption(f"Evidence: {ev.get('verdict')} - {ev.get('reason')}")
    # Show subtopics where weaknesses were detected
    st.subheader("Poor areas")
    if not st.session_state.poor_areas:
        st.write("No persistent issues detected.")
    else:
        for area in st.session_state.poor_areas:
            with st.expander(f"{area['subtopic']} - {', '.join(area['reasons'])}"):
                st.json(area)

    # Summary of Strengths and Weaknesses
    full_history = st.session_state.full_history
    try:
        summary_prompt = f"""
        Return ONLY this template (plain text, no extra headings):

        Strengths:
        - <up to 3 short bullets grounded in the records>
        Weaknesses:
        - <up to 3 short bullets grounded in the records>

        Records (use scores exactly as is, do not change numbers):
        {json.dumps(full_history, indent=2)}
        """
        summary = interview.ask_llm(summary_prompt, temperature=0.2)
        summary = summary.replace("Weaknesses:", "\n\nWeaknesses:")
    except Exception as e:
        summary = f"(Could not generate summary: {e})"

    st.subheader("Strengths & Weaknesses")
    st.markdown(summary)
