from typing import List, Dict
from llama_cpp import Llama


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path="./data/models/qwen2-1_5b-instruct-q4_k_m.gguf",
            n_ctx=1024,
            n_threads=4,
            verbose=False,
        )
    return _llm


def build_prompt(query: str, contexts: List[Dict]) -> str:
    ctx_texts = []
    for i, c in enumerate(contexts, 1):
        ctx_texts.append(f"[{i}] {c.get('text_preview', '')}")

    context_block = "\n\n".join(ctx_texts)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context_block}

Question:
{query}

Answer:
""".strip()
    return prompt


def generate_answer(prompt: str) -> str:
    llm = get_llm()
    out = llm(
        prompt,
        max_tokens=256,
        temperature=0.2,
        stop=["<|im_end|>"],
    )
    return out["choices"][0]["text"].strip()


def run_generation(query: str, search_results: List[Dict]) -> str:
    prompt = build_prompt(query, search_results)
    return generate_answer(prompt)
