import json
import re

from openai import OpenAI

from src.retriever import (
    PRODUCTS,
    detect_products,
    flatten_document_results,
    load_product_catalog,
    retrieve_exhaustive,
    retrieve_targeted,
)

LLM_MODEL = "qwen3:1.7b"

TARGETED = "targeted"
EXHAUSTIVE = "exhaustive"
RETRIEVAL_MODES = (TARGETED, EXHAUSTIVE)

# Statuswerte für dokumentbezogene Befunde
SUPPORTED = "supported"
NEGATIVE = "negative"
INSUFFICIENT = "insufficient_evidence"
INVALID = "invalid_response"
FINDING_STATUSES = (SUPPORTED, NEGATIVE, INSUFFICIENT)

llm_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


SYSTEM_PROMPT = """
You are a knowledge assistant.

Answer using only the provided context.
If the answer is not in the context, say:
"I cannot answer this based on the provided documents."

Do not invent facts or use external knowledge.
Do not generate source citations in your answer.
The application displays retrieved sources separately.
"""

DOCUMENT_PROMPT = """
You evaluate ONE product document against a user question.
Use only the excerpts from this document. Do not use external knowledge.

Respond with a single JSON object and nothing else:
{
  "product_type": "<kind of product, as the document overview describes it>",
  "in_scope": true | false,
  "status": "supported" | "negative" | "insufficient_evidence",
  "answer": "<one or two sentences answering the question for this product>",
  "evidence": "<short verbatim quote from the excerpts backing the answer, or empty>"
}

Determine "product_type" from the document itself, not from the question.
"in_scope" is false only if the question asks about a specific kind of product
and this product is clearly a different kind; otherwise it is true.

Status rules:
- "supported": the excerpts explicitly contain information that answers the
  question for this product (e.g. the requested value or feature is stated).
- "negative": the excerpts EXPLICITLY state that the product does not meet the
  question (e.g. a stated value that falls outside the requested range, or an
  explicit "not supported"). Quote that statement as evidence.
- "insufficient_evidence": the excerpts do not mention the information.
  Absence of a mention is NOT a negative finding.
"""


def retrieve(question, n_results=5):
    """Retrieve relevant chunks from one or multiple product documents."""
    return retrieve_targeted(question, n_results=n_results)


def format_context(chunks):
    return "\n\n".join(
        f"[{chunk['source']}, page {chunk['page']}, chunk {chunk['chunk_index']}]\n"
        f"{chunk['text']}"
        for chunk in chunks
    )


def call_llm(system_prompt, user_content, json_output=False):
    kwargs = {}

    if json_output:
        kwargs["response_format"] = {"type": "json_object"}

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_content}\n/no_think"}
        ],
        temperature=0,
        **kwargs
    )

    return response.choices[0].message.content or ""


def parse_finding(raw):
    """Parse a document-level LLM response into a validated finding dict.

    Malformed responses become INVALID rather than being guessed. Negative or
    supported findings without quoted evidence are downgraded to
    INSUFFICIENT, so missing evidence is never reported as a negative result.
    """
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None

    if not isinstance(data, dict):
        return {
            "status": INVALID,
            "product_type": "unknown",
            "in_scope": True,
            "answer": "The model response could not be parsed.",
            "evidence": ""
        }

    status = str(data.get("status", "")).strip().lower()
    evidence = str(data.get("evidence") or "").strip()

    if status not in FINDING_STATUSES:
        status = INVALID
    elif status in (SUPPORTED, NEGATIVE) and not evidence:
        status = INSUFFICIENT

    # Nur ein eindeutiges "false" schließt ein Produkt aus
    in_scope = str(data.get("in_scope", True)).strip().lower() not in ("false", "no")

    return {
        "status": status,
        "product_type": str(data.get("product_type") or "unknown").strip(),
        "in_scope": in_scope,
        "answer": str(data.get("answer") or "").strip(),
        "evidence": evidence
    }


def evaluate_document(question, document):
    """Ask the LLM to answer the question using only one document's chunks."""
    finding = {"model": document["model"], "source": document["source"]}

    if not document["chunks"]:
        finding.update({
            "status": INSUFFICIENT,
            "product_type": "unknown",
            "in_scope": True,
            "answer": "No content was retrieved from this document.",
            "evidence": ""
        })
        return finding

    user_content = (
        f"Product: {document['model']} (file: {document['source']})\n\n"
        f"Document overview:\n{document.get('overview') or '(not available)'}\n\n"
        f"Excerpts:\n{format_context(document['chunks'])}\n\n"
        f"Question:\n{question}"
    )

    finding.update(parse_finding(call_llm(DOCUMENT_PROMPT, user_content, json_output=True)))
    return finding


def combine_findings(findings):
    """Build the final answer from document-level findings, grouped by status."""
    if not findings:
        return "No documents were available to examine."

    sections = [
        (SUPPORTED, "Supported by the documents"),
        (NEGATIVE, "Explicitly negative"),
        (INSUFFICIENT, "Insufficient evidence"),
        (INVALID, "Could not be evaluated (malformed model response)"),
    ]

    lines = [
        f"Examined {len(findings)} document(s) individually. "
        "Only the most relevant passages of each document were checked, "
        "so details elsewhere in a document may have been missed."
    ]

    in_scope = [f for f in findings if f.get("in_scope", True)]
    out_of_scope = [f for f in findings if not f.get("in_scope", True)]

    def format_finding(finding):
        line = f"- **{finding['model']}** ({finding['product_type']}): {finding['answer']}"

        if finding["evidence"]:
            line += f' — _"{finding["evidence"]}"_'

        return line

    for status, title in sections:
        group = [f for f in in_scope if f["status"] == status]

        if group:
            lines.append(f"\n**{title}:**")
            lines.extend(format_finding(f) for f in group)

    if out_of_scope:
        lines.append("\n**Not the kind of product asked about:**")
        lines.extend(
            f"- **{f['model']}** ({f['product_type']})"
            for f in out_of_scope
        )

    return "\n".join(lines)


def answer_exhaustive(question, selection=None, n_results=5):
    """Evaluate every selected document separately and combine the findings.

    Returns (answer, chunks, findings).
    """
    documents = retrieve_exhaustive(question, selection=selection, n_results=n_results)
    findings = [evaluate_document(question, document) for document in documents]

    return combine_findings(findings), flatten_document_results(documents), findings


def answer_targeted(question, n_results=5):
    chunks = retrieve_targeted(question, n_results=n_results)

    if not chunks:
        return "I cannot answer this based on the provided documents.", chunks

    answer = call_llm(
        SYSTEM_PROMPT,
        f"Context:\n{format_context(chunks)}\n\nQuestion:\n{question}"
    )

    return answer, chunks


def ask_rag(question, mode=TARGETED, selection=None):
    """Retrieve context and generate an answer.

    mode: "targeted" (default) or "exhaustive". `selection` optionally limits
    exhaustive mode to the given model names or source filenames.
    Returns (answer, chunks).
    """
    if mode == TARGETED:
        return answer_targeted(question)

    if mode == EXHAUSTIVE:
        answer, chunks, _ = answer_exhaustive(question, selection=selection)
        return answer, chunks

    raise ValueError(f"Unknown retrieval mode: {mode!r}. Use one of {RETRIEVAL_MODES}.")


if __name__ == "__main__":
    question = input("Ask a question: ")
    mode = input(f"Mode {RETRIEVAL_MODES} [targeted]: ").strip() or TARGETED

    answer, chunks = ask_rag(question, mode=mode)

    print("\n--- Answer ---")
    print(answer)

    print("\n--- Retrieved Sources ---")

    for chunk in chunks:
        print(
            f"{chunk['source']} | "
            f"Page {chunk['page']} | "
            f"Chunk {chunk['chunk_index']} | "
            f"Distance: {chunk['distance']:.4f}"
        )
