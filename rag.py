import json
import re

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CHROMA_PATH = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:latest"


# -----------------------------
# Vector database
# -----------------------------

def get_vector_database():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return db


# -----------------------------
# Retriever
# -----------------------------

def get_retriever():

    db = get_vector_database()

    return db.as_retriever(
        search_kwargs={"k": 15}
    )


# -----------------------------
# LLM
# -----------------------------

def get_llm():

    return OllamaLLM(
        model=LLM_MODEL,
        temperature=0.1
    )


# -----------------------------
# Normal textbook question answering
# -----------------------------

def ask_question(question):

    retriever = get_retriever()
    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a Samacheer textbook assistant.

Answer the question using ONLY the textbook context below.

If the answer is not available in the context, clearly say:
"The textbook context does not contain enough information."

TEXTBOOK CONTEXT:
{context}

QUESTION:
{question}

Give a clear, student-friendly answer.
"""

    llm = get_llm()
    answer = llm.invoke(prompt)

    sources = []

    for document in documents:

        source = document.metadata.get(
            "source_file",
            document.metadata.get("source", "Unknown")
        )

        if source not in sources:
            sources.append(source)

    return {
        "answer": answer,
        "sources": sources
    }


# -----------------------------
# Extract JSON from LLM output
# -----------------------------

def extract_json(text):

    text = text.strip()

    # Remove Markdown code fences if present
    text = re.sub(
        r"```json|```",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()

    # Try direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding a JSON object inside the response
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        json_text = text[start:end + 1]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "The model did not return valid JSON. Please try again."
    )


# -----------------------------
# Generate realistic mock test
# -----------------------------

def generate_mock_test(subject, chapter, mcq_count, two_mark_count, three_mark_count, five_mark_count):
    retriever = get_retriever()

    search_query = f"""
    {subject} {chapter}
    Important textbook concepts, definitions, laws, formulas, examples,
    applications, differences, classifications, explanations,
    processes and exam-relevant questions
    """

    docs = retriever.invoke(search_query)

    # Remove duplicate chunks
    unique_docs = []
    seen = set()

    for doc in docs:
        text = doc.page_content.strip()

        if text and text not in seen:
            seen.add(text)
            unique_docs.append(doc)

    context = "\n\n--- TEXTBOOK SECTION ---\n\n".join(
        doc.page_content for doc in unique_docs
    )

    total_questions = (
        mcq_count
        + two_mark_count
        + three_mark_count
        + five_mark_count
    )

    prompt = f"""
You are an experienced school examination question paper setter.

Create a realistic mock examination using ONLY the textbook content provided below.

SUBJECT:
{subject}

CHAPTER:
{chapter}

NUMBER OF QUESTIONS REQUIRED:

MCQs: {mcq_count}
2-mark questions: {two_mark_count}
3-mark questions: {three_mark_count}
5-mark questions: {five_mark_count}

TOTAL QUESTIONS:
{total_questions}

VERY IMPORTANT RULES:

1. Use ONLY information present in the textbook context.
2. Do NOT use outside knowledge.
3. Do NOT invent facts, formulas, definitions or examples.
4. Every question must be different.
5. NEVER repeat the same question.
6. NEVER ask the same concept in two different sections.
7. Do NOT create two questions that are merely rewordings of each other.
8. Before writing questions, mentally identify different concepts available in the textbook.
9. Distribute questions across different concepts whenever possible.
10. A concept used for an MCQ should preferably NOT be used again in 2M, 3M or 5M.
11. Make the questions look like realistic school examination questions.
12. Do not simply copy complete sentences from the textbook.

QUESTION TYPES:

MCQs:
- Exactly {mcq_count}
- 1 mark each
- Four options: A, B, C and D
- Exactly ONE correct answer
- Test facts, definitions, concepts, formulas or identification

2-MARK:
- Exactly {two_mark_count}
- Short-answer questions
- Should require around two important points or a short explanation

3-MARK:
- Exactly {three_mark_count}
- Should require a detailed explanation, comparison,
  relationship, process, application or multiple points

5-MARK:
- Exactly {five_mark_count}
- Should require a comprehensive answer,
  derivation, explanation, classification, process or application
- Do NOT make these simply longer versions of 2-mark questions

OUTPUT FORMAT:

Return ONLY valid JSON.

Do not write:
- Markdown
- ```json
- explanations outside JSON
- introductory text

Use exactly this structure:

{{
  "mcqs": [
    {{
      "question": "Question",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "answer": "A",
      "explanation": "Explanation based on textbook"
    }}
  ],

  "two_mark": [
    {{
      "question": "Question",
      "answer": "Expected answer"
    }}
  ],

  "three_mark": [
    {{
      "question": "Question",
      "answer": "Expected answer"
    }}
  ],

  "five_mark": [
    {{
      "question": "Question",
      "answer": "Expected answer"
    }}
  ]
}}

FINAL CHECK BEFORE RETURNING JSON:

MCQs must contain exactly {mcq_count} questions.

two_mark must contain exactly {two_mark_count} questions.

three_mark must contain exactly {three_mark_count} questions.

five_mark must contain exactly {five_mark_count} questions.

TOTAL must equal {total_questions}.

Check ALL questions against each other.

If two questions test the same concept, replace one with a different concept from the textbook.

TEXTBOOK CONTEXT:

{context}
"""

    llm = get_llm()

    # Try up to 2 times if the model produces invalid counts
    for attempt in range(2):

        response = llm.invoke(prompt)

        try:
            test = extract_json(response)

            # Make sure all required sections exist
            if not all(
                key in test
                for key in ["mcqs", "two_mark", "three_mark", "five_mark"]
            ):
                raise ValueError("Missing question section.")

            # Validate counts
            if len(test["mcqs"]) != mcq_count:
                raise ValueError(
                    f"Incorrect number of MCQs generated. "
                    f"Expected {mcq_count}, got {len(test['mcqs'])}."
                )

            if len(test["two_mark"]) != two_mark_count:
                raise ValueError(
                    f"Incorrect number of 2-mark questions generated. "
                    f"Expected {two_mark_count}, got {len(test['two_mark'])}."
                )

            if len(test["three_mark"]) != three_mark_count:
                raise ValueError(
                    f"Incorrect number of 3-mark questions generated. "
                    f"Expected {three_mark_count}, got {len(test['three_mark'])}."
                )

            if len(test["five_mark"]) != five_mark_count:
                raise ValueError(
                    f"Incorrect number of 5-mark questions generated. "
                    f"Expected {five_mark_count}, got {len(test['five_mark'])}."
                )

            # Check exact duplicate questions
            all_questions = []

            for section in [
                "mcqs",
                "two_mark",
                "three_mark",
                "five_mark"
            ]:
                for item in test[section]:
                    q = item["question"].strip().lower()

                    # Normalize whitespace
                    q = re.sub(r"\s+", " ", q)

                    if q in all_questions:
                        raise ValueError(
                            "Duplicate question generated."
                        )

                    all_questions.append(q)

            return test

        except Exception as e:

            # First failure → ask model again
            if attempt == 0:
                prompt += f"""

IMPORTANT CORRECTION:

Your previous response failed validation because:

{str(e)}

Generate the COMPLETE mock test again.

Do NOT return only the missing questions.
Return ALL sections again.

This time strictly follow the required question counts.
"""
                continue

            # Second failure → show useful error
            raise ValueError(str(e))

    raise ValueError("Could not generate mock test.")