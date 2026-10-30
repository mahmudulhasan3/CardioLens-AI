import os
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="CardioLens AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Custom styling --------------------

st.markdown(
    """
    <style>
    :root {
        --cl-primary: #2878d0;
        --cl-border: rgba(128, 128, 128, .22);
        --cl-muted: #7b8492;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.7rem;
        padding-bottom: 7rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--cl-border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    h1 {
        letter-spacing: -0.04em;
        padding-bottom: .15rem;
    }

    h2, h3 {
        letter-spacing: -0.025em;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed var(--cl-border);
        border-radius: 16px;
        padding: .35rem;
        background: rgba(128, 128, 128, .025);
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border-color: var(--cl-border) !important;
    }

    div.stButton > button {
        border-radius: 11px;
        min-height: 2.65rem;
        font-weight: 550;
        transition: border-color .15s ease, transform .15s ease;
    }

    div.stButton > button:hover {
        border-color: var(--cl-primary);
        transform: translateY(-1px);
    }

    div.stButton > button[kind="primary"] {
        border: 0;
        box-shadow: 0 5px 16px rgba(40, 120, 208, .16);
    }

    /* Chat messages stay inside the available screen width. */
    [data-testid="stChatMessage"] {
        box-sizing: border-box;
        width: 100%;
        max-width: 100%;
        min-width: 0;
        padding: .9rem 1rem;
        margin: .55rem 0;
        border: 1px solid var(--cl-border);
        border-radius: 16px;
        background: rgba(128, 128, 128, .025);
        overflow-wrap: anywhere;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        min-width: 0;
        overflow-wrap: anywhere;
        word-break: normal;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        max-width: 100%;
        overflow-wrap: anywhere;
    }

    /* User message aligned to the right without negative margins. */
    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) {
        flex-direction: row-reverse;
        margin-left: auto;
        width: fit-content;
        max-width: min(82%, 900px);
        background: rgba(40, 120, 208, .075);
        border-color: rgba(40, 120, 208, .20);
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) [data-testid="stChatMessageContent"] {
        text-align: left;
    }

    /* Assistant messages remain on the left. */
    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarAssistant"]
    ) {
        margin-right: auto;
        max-width: 100%;
    }

    /* Chat input styling. */
    [data-testid="stChatInput"] {
        border-radius: 16px;
        border-color: var(--cl-border);
        box-shadow: 0 8px 28px rgba(0, 0, 0, .07);
    }

    [data-testid="stChatInput"] textarea {
        line-height: 1.5;
    }

    [data-testid="stAlert"] {
        border-radius: 13px;
    }

    [data-testid="stProgressBar"] {
        margin-top: .2rem;
    }

    @media (max-width: 760px) {
        .block-container {
            padding: 1rem .8rem 7rem;
        }

        [data-testid="stChatMessage"] {
            padding: .75rem;
        }

        [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageAvatarUser"]
        ) {
            max-width: 94%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Configuration --------------------

LEVEL_COLORS = {
    "high": "blue",
    "moderate": "orange",
    "low": "gray",
}

SUGGESTIONS = {
    "English": [
        "What does this result mean?",
        "How reliable is this model?",
        "What should I do next?",
    ],
    "Bangla": [
        "এই ফলাফলের মানে কী?",
        "এই model কতটা নির্ভরযোগ্য?",
        "এরপর আমার কী করা উচিত?",
    ],
}

SHARE_FOOTER = "\n\n— CardioLens AI " "(educational prototype, not a medical diagnosis)"

ACTION_BAR = components.declare_component(
    "action_bar",
    path=str(Path(__file__).parent / "action_bar"),
)

# -------------------- Session state --------------------

DEFAULTS = {
    "session_id": None,
    "result": None,
    "messages": [],
    "file_key": None,
    "language": "English",
    "uploader_key": 0,
    "ratings": {},
    "seen_events": {},
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -------------------- Helper functions --------------------


def fmt_pct(p):
    """Avoid displaying model scores as absolute certainty."""
    if p >= 0.99:
        return ">99%"

    if p < 0.01:
        return "<1%"

    return f"{p:.0%}"


def reset_analysis():
    """Clear the current ECG analysis and start a fresh session."""
    st.session_state.session_id = None
    st.session_state.result = None
    st.session_state.messages = []
    st.session_state.file_key = None
    st.session_state.ratings = {}
    st.session_state.seen_events = {}
    st.session_state.uploader_key += 1


def call_chat(question):
    """Call /chat and return (success, answer_or_error)."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "session_id": st.session_state.session_id,
                "question": question,
            },
            timeout=60,
        )

    except requests.RequestException:
        return (
            False,
            "Could not reach the server. Check that the API is running.",
        )

    if response.status_code == 200:
        try:
            return True, response.json()["answer"]

        except (ValueError, KeyError):
            return False, "The server returned an unexpected response."

    try:
        return False, str(response.json().get("detail", response.text))

    except ValueError:
        return False, response.text


def run_analysis(file, language):
    """Send the ECG to /predict and fetch its initial explanation."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={
                "file": (
                    file.name,
                    file.getvalue(),
                    file.type,
                )
            },
            data={"language": language},
            timeout=60,
        )

    except requests.RequestException:
        return "Could not reach the server. Is FastAPI running?"

    if response.status_code != 200:
        try:
            return str(
                response.json().get(
                    "detail",
                    "Something went wrong.",
                )
            )

        except ValueError:
            return "Something went wrong. The server returned an invalid response."

    try:
        data = response.json()
        session_id = data["session_id"]

    except (ValueError, KeyError):
        return "The prediction response is missing required fields."

    st.session_state.result = data
    st.session_state.session_id = session_id
    st.session_state.file_key = (file.name, file.size)
    st.session_state.language = language
    st.session_state.ratings = {}
    st.session_state.seen_events = {}

    ok, first = call_chat(None)

    if ok:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": first,
            }
        ]
    else:
        st.session_state.messages = []

    return None


def send_feedback(idx, rating):
    """Send a rating to the API."""
    try:
        requests.post(
            f"{API_URL}/feedback",
            json={
                "session_id": st.session_state.session_id,
                "message_index": idx,
                "rating": rating,
            },
            timeout=10,
        )

        st.toast("Thanks for your feedback!")

    except requests.RequestException:
        # Feedback is optional.
        pass


def render_actions(idx, msg, is_last, file):
    """Render the answer toolbar and handle its events."""
    key = f"bar_{st.session_state.session_id}_{idx}"

    event = ACTION_BAR(
        text=msg["content"],
        footer=SHARE_FOOTER,
        can_retry=is_last,
        rating=st.session_state.ratings.get(key),
        key=key,
        default=None,
    )

    if not event:
        return

    event_id = event.get("n")

    if st.session_state.seen_events.get(key) == event_id:
        return

    st.session_state.seen_events[key] = event_id

    if event.get("action") == "rate":
        rating = event.get("value")
        st.session_state.ratings[key] = rating

        if rating:
            send_feedback(idx, rating)

    elif event.get("action") == "retry":
        if file is None:
            st.warning("Please upload the ECG image again before retrying.")
            return

        with st.spinner("Analyzing the ECG again..."):
            error = run_analysis(
                file,
                st.session_state.language,
            )

        if error:
            st.error(error)
        else:
            st.rerun()


# -------------------- Header --------------------

st.title("🫀 CardioLens AI")

st.caption("ECG image analysis with an AI-generated explanation")

st.warning(
    "**Educational prototype only - not a medical device or diagnosis.** "
    "A qualified healthcare professional must interpret an ECG."
)


# -------------------- Sidebar --------------------

with st.sidebar:
    st.header("⚙️ Settings")

    language = st.selectbox(
        "Language for the first explanation",
        ["English", "Bangla"],
        index=(0 if st.session_state.language == "English" else 1),
    )

    st.caption("You can ask follow-up questions")

    st.divider()

    if st.button(
        "＋ New analysis",
        use_container_width=True,
    ):
        reset_analysis()
        st.rerun()

    st.caption("CardioLens AI · Project prototype")


# -------------------- ECG upload --------------------

st.subheader("1. Upload an ECG")

uploaded = st.file_uploader(
    "Choose an ECG image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}",
    help=(
        "Use a clear ECG image. The model output is experimental "
        "and must not guide medical decisions."
    ),
)

current_key = (uploaded.name, uploaded.size) if uploaded else None

left, right = st.columns(
    [1.05, 0.95],
    gap="large",
)


# -------------------- ECG preview --------------------

with left:
    with st.container(border=True):
        st.markdown("#### ECG image")

        if uploaded is not None:
            st.image(
                uploaded,
                use_container_width=True,
            )

            st.caption(f"{uploaded.name} · {uploaded.size / 1024:.1f} KB")

            if st.button(
                "Analyze ECG",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Analyzing ECG image..."):
                    error = run_analysis(
                        uploaded,
                        language,
                    )

                if error:
                    st.error(error)
                else:
                    st.rerun()

        else:
            st.markdown("Upload an image to preview it here and start the analysis.")

            st.caption("Supported formats: JPG, JPEG, PNG")


# -------------------- Prediction result --------------------

with right:
    with st.container(border=True):
        st.markdown("#### Analysis result")

        result = st.session_state.result

        if result and st.session_state.file_key == current_key:
            level = result.get("confidence_level", "low")
            color = LEVEL_COLORS.get(level, "gray")

            st.caption("Model prediction · Not a diagnosis")

            st.markdown(f"## {result.get('predicted_class', 'Unavailable')}")

            confidence = float(result.get("confidence", 0))

            st.markdown(
                f"Model certainty: :{color}[**{level.upper()}**] "
                f"({fmt_pct(confidence)})"
            )

            if level == "low":
                st.info(
                    "The model is uncertain about this image. "
                    "Interpret this result with extra caution."
                )

            probabilities = result.get("probabilities", {})

            if probabilities:
                st.caption("Model score by class · Not a medical probability")

                for name, probability in sorted(
                    probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    score = min(
                        max(float(probability), 0.0),
                        1.0,
                    )

                    st.progress(
                        score,
                        text=f"{name} · {fmt_pct(score)}",
                    )

        else:
            st.markdown("Your prediction will appear here after analysis.")

            st.caption(
                "The displayed scores describe model output, " "not clinical risk."
            )


# -------------------- Chat --------------------

if st.session_state.session_id and st.session_state.file_key == current_key:
    st.divider()

    st.subheader("2. Ask about this result")

    st.caption("Ask a follow-up question about the model output or its limitations.")

    if not st.session_state.messages:
        st.warning("The initial explanation could not be generated.")

        if st.button(
            "Try generating the explanation again",
            type="primary",
        ):
            ok, answer = call_chat(None)

            if ok:
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                ]
                st.rerun()
            else:
                st.error(answer)

    else:
        messages = st.session_state.messages

        # Render existing conversation history.
        for idx, msg in enumerate(messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant":
                    render_actions(
                        idx,
                        msg,
                        idx == len(messages) - 1,
                        uploaded,
                    )

        # Suggested questions.
        st.markdown("**Suggested questions**")

        chosen = None

        suggestions = SUGGESTIONS[st.session_state.language]

        suggestion_cols = st.columns(
            len(suggestions),
            gap="small",
        )

        for idx, (col, suggestion) in enumerate(zip(suggestion_cols, suggestions)):
            if col.button(
                suggestion,
                key=f"sugg_{st.session_state.session_id}_{idx}",
                use_container_width=True,
            ):
                chosen = suggestion

        # Native Streamlit chat input stays at the bottom of the page.
        typed = st.chat_input("Ask about this ECG")

        question = chosen or typed

        if question:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.spinner("Preparing an answer..."):
                ok, answer = call_chat(question)

            if ok:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                st.rerun()

            else:
                # Remove the unanswered user turn on API failure.
                st.session_state.messages.pop()
                st.error(answer)
