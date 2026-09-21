import os
import streamlit as st

from rag import ask_question, generate_mock_test


CHROMA_PATH = "chroma_db"


st.set_page_config(
    page_title="Samacheer Exam Prep AI",
    page_icon="📚",
    layout="wide"
)


# -----------------------------
# CSS
# -----------------------------

st.markdown(
    """
    <style>

    .title {
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
    }

    .subtitle {
        font-size: 18px;
        color: #ffffff;
    }

    .answer-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }

    .answer-box p,
    .answer-box li,
    .answer-box h1,
    .answer-box h2,
    .answer-box h3,
    .answer-box h4,
    .answer-box strong,
    .answer-box em {
        color: #000000 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="title">📚 Samacheer Exam Prep AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Local AI-powered textbook assistant</div>',
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("📚 Textbooks")

    if os.path.exists("data"):

        files = [
            file
            for file in os.listdir("data")
            if file.lower().endswith((".pdf", ".docx"))
        ]

        if files:

            st.success(f"{len(files)} textbook(s) found")

            for file in files:
                st.write("📖", file)

        else:
            st.warning("No textbooks found.")

    else:
        st.warning("Create a data folder and add textbooks.")

    st.divider()

    st.header("🤖 AI Model")
    st.info("Running locally with Ollama")
    st.caption("Model: llama3.2:latest")
    st.caption("No API key required.")


# -----------------------------
# Check ChromaDB
# -----------------------------

if not os.path.exists(CHROMA_PATH):

    st.warning("⚠️ Textbook database not found.")

    st.markdown(
        """
        ### First-time setup

        1. Put your PDF/DOCX textbooks inside the `data` folder.
        2. Run `python ingest.py`.
        3. Restart Streamlit.
        """
    )

    st.stop()


# -----------------------------
# Main navigation
# -----------------------------

st.header("Choose a Feature")

feature = st.radio(
    "Select",
    [
        "💬 Ask Your Textbook",
        "🎯 Mock Test"
    ],
    horizontal=True
)


# =========================================================
# ASK YOUR TEXTBOOK
# =========================================================

if feature == "💬 Ask Your Textbook":

    st.subheader("💬 Ask Your Textbook")

    question = st.text_area(
        "What would you like to know?",
        placeholder="Example: Explain photosynthesis in simple words.",
        height=120
    )

    if st.button("🔎 Ask AI", type="primary"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner(
                "Searching textbook and generating answer..."
            ):

                try:

                    result = ask_question(question)

                    st.subheader("🤖 Answer")

                    st.markdown(
                        '<div class="answer-box">',
                        unsafe_allow_html=True
                    )

                    st.markdown(result["answer"])

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

                    if result["sources"]:

                        st.subheader("📖 Sources")

                        for source in result["sources"]:
                            st.write(f"• {source}")

                except Exception as error:

                    st.error("Something went wrong.")
                    st.code(str(error))


# =========================================================
# MOCK TEST
# =========================================================

else:

    st.subheader("🎯 Textbook-Based Mock Test")

    st.info(
        "Questions are generated from the retrieved textbook content "
        "using your local Llama model."
    )

    subject = st.text_input(
        "📚 Enter subject",
        placeholder="Example: Science"
    )

    chapter = st.text_input(
        "📖 Enter chapter",
        placeholder="Example: Electricity"
    )

    st.markdown("### 📝 Question Pattern")

    col1, col2 = st.columns(2)

    with col1:

        mcq_count = st.number_input(
            "MCQ questions — 1 mark",
            min_value=0,
            max_value=20,
            value=5,
            step=1
        )

        two_mark_count = st.number_input(
            "2-mark questions",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

    with col2:

        three_mark_count = st.number_input(
            "3-mark questions",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

        five_mark_count = st.number_input(
            "5-mark questions",
            min_value=0,
            max_value=5,
            value=1,
            step=1
        )

    total_questions = (
        mcq_count
        + two_mark_count
        + three_mark_count
        + five_mark_count
    )

    total_marks = (
        mcq_count
        + two_mark_count * 2
        + three_mark_count * 3
        + five_mark_count * 5
    )

    st.write(f"**Total questions:** {total_questions}")
    st.write(f"**Total marks:** {total_marks}")

    if st.button("⚙️ Generate Mock Test", type="primary"):

        if not subject.strip() or not chapter.strip():

            st.warning(
                "Please enter both subject and chapter."
            )

        elif total_questions == 0:

            st.warning(
                "Please select at least one question."
            )

        else:

            with st.spinner(
                "Retrieving textbook content and generating mock test..."
            ):

                try:

                    generated_test = generate_mock_test(
                        subject=subject,
                        chapter=chapter,
                        mcq_count=mcq_count,
                        two_mark_count=two_mark_count,
                        three_mark_count=three_mark_count,
                        five_mark_count=five_mark_count
                    )

                    st.session_state["mock_test"] = generated_test
                    st.session_state["mock_submitted"] = False

                    # Reset previous answers
                    st.session_state["mock_answers"] = {}

                    st.success("Mock test generated successfully!")

                except Exception as error:

                    st.error("Could not generate the mock test.")
                    st.code(str(error))


    # -----------------------------------------------------
    # Display generated test
    # -----------------------------------------------------

    if "mock_test" in st.session_state:

        test = st.session_state["mock_test"]

        st.divider()

        st.header(
            f"🎯 {test.get('subject', subject)} — "
            f"{test.get('chapter', chapter)}"
        )

        st.write(
            "Answer all questions and click **Submit Test** "
            "at the bottom."
        )

        question_number = 1

        # -----------------------------
        # MCQs
        # -----------------------------

        mcqs = test.get("mcqs", [])

        if mcqs:

            st.subheader("Part A — Multiple Choice Questions")
            st.caption(f"{len(mcqs)} × 1 = {len(mcqs)} marks")

            for index, question in enumerate(mcqs):

                st.markdown(
                    f"**{question_number}. "
                    f"{question.get('question', '')}**"
                )

                options = question.get("options", [])

                answer_key = f"mcq_{index}"

                selected_answer = st.radio(
                    "Choose an answer:",
                    options,
                    key=answer_key,
                    index=None
                )

                st.session_state["mock_answers"][answer_key] = (
                    selected_answer
                )

                question_number += 1

        # -----------------------------
        # 2-mark questions
        # -----------------------------

        two_mark_questions = test.get(
            "two_mark_questions",
            []
        )

        if two_mark_questions:

            st.subheader("Part B — 2-Mark Questions")
            st.caption(
                f"{len(two_mark_questions)} × 2 = "
                f"{len(two_mark_questions) * 2} marks"
            )

            for index, question in enumerate(two_mark_questions):

                st.markdown(
                    f"**{question_number}. "
                    f"{question.get('question', '')}**"
                )

                answer_key = f"two_mark_{index}"

                student_answer = st.text_area(
                    "Your answer:",
                    key=answer_key,
                    height=100
                )

                st.session_state["mock_answers"][answer_key] = (
                    student_answer
                )

                question_number += 1

        # -----------------------------
        # 3-mark questions
        # -----------------------------

        three_mark_questions = test.get(
            "three_mark_questions",
            []
        )

        if three_mark_questions:

            st.subheader("Part C — 3-Mark Questions")
            st.caption(
                f"{len(three_mark_questions)} × 3 = "
                f"{len(three_mark_questions) * 3} marks"
            )

            for index, question in enumerate(three_mark_questions):

                st.markdown(
                    f"**{question_number}. "
                    f"{question.get('question', '')}**"
                )

                answer_key = f"three_mark_{index}"

                student_answer = st.text_area(
                    "Your answer:",
                    key=answer_key,
                    height=120
                )

                st.session_state["mock_answers"][answer_key] = (
                    student_answer
                )

                question_number += 1

        # -----------------------------
        # 5-mark questions
        # -----------------------------

        five_mark_questions = test.get(
            "five_mark_questions",
            []
        )

        if five_mark_questions:

            st.subheader("Part D — 5-Mark Questions")
            st.caption(
                f"{len(five_mark_questions)} × 5 = "
                f"{len(five_mark_questions) * 5} marks"
            )

            for index, question in enumerate(five_mark_questions):

                st.markdown(
                    f"**{question_number}. "
                    f"{question.get('question', '')}**"
                )

                answer_key = f"five_mark_{index}"

                student_answer = st.text_area(
                    "Your answer:",
                    key=answer_key,
                    height=180
                )

                st.session_state["mock_answers"][answer_key] = (
                    student_answer
                )

                question_number += 1

        st.divider()

        # -----------------------------
        # Submit and score
        # -----------------------------

        if st.button("📊 Submit Test", type="primary"):

            score = 0
            maximum_mcq_score = len(mcqs)

            mcq_results = []

            for index, question in enumerate(mcqs):

                answer_key = f"mcq_{index}"

                selected_answer = st.session_state[
                    "mock_answers"
                ].get(answer_key)

                correct_answer = question.get(
                    "correct_answer",
                    ""
                )

                is_correct = (
                    selected_answer == correct_answer
                )

                if is_correct:
                    score += 1

                mcq_results.append(
                    {
                        "question": question.get(
                            "question",
                            ""
                        ),
                        "selected_answer": selected_answer,
                        "correct_answer": correct_answer,
                        "is_correct": is_correct,
                        "explanation": question.get(
                            "explanation",
                            ""
                        )
                    }
                )

            st.session_state["mock_submitted"] = True
            st.session_state["mock_score"] = score
            st.session_state["mcq_results"] = mcq_results

        # -----------------------------
        # Results
        # -----------------------------

        if st.session_state.get("mock_submitted", False):

            st.divider()
            st.header("📊 Mock Test Result")

            st.metric(
                "MCQ Score",
                f"{st.session_state['mock_score']} / "
                f"{maximum_mcq_score}"
            )

            st.subheader("MCQ Review")

            for index, result in enumerate(
                st.session_state["mcq_results"],
                start=1
            ):

                if result["is_correct"]:
                    st.success(
                        f"Question {index}: Correct"
                    )
                else:
                    st.error(
                        f"Question {index}: Incorrect"
                    )

                st.write(
                    f"**Your answer:** "
                    f"{result['selected_answer'] or 'Not answered'}"
                )

                st.write(
                    f"**Correct answer:** "
                    f"{result['correct_answer']}"
                )

                st.write(
                    f"**Explanation:** "
                    f"{result['explanation']}"
                )

            st.info(
                "Written-answer questions are displayed with their "
                "answer keys. Automatic written-answer evaluation can "
                "be added as the next step."
            )

            st.subheader("Written-Answer Keys")

            written_sections = [
                (
                    "2-Mark Answer Keys",
                    test.get("two_mark_questions", [])
                ),
                (
                    "3-Mark Answer Keys",
                    test.get("three_mark_questions", [])
                ),
                (
                    "5-Mark Answer Keys",
                    test.get("five_mark_questions", [])
                )
            ]

            for section_title, questions in written_sections:

                if questions:

                    st.markdown(f"### {section_title}")

                    for index, question in enumerate(
                        questions,
                        start=1
                    ):

                        st.write(
                            f"**{index}. "
                            f"{question.get('question', '')}**"
                        )

                        st.write(
                            "**Expected answer:** "
                            f"{question.get('answer_key', '')}"
                        )

                        st.write(
                            "**Explanation:** "
                            f"{question.get('explanation', '')}"
                        )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "🦙 Ollama + 🤗 HuggingFace + "
    "🗄️ ChromaDB + 🎈 Streamlit"
)