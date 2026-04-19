from .llm import generate_answer as llm_generate


def generate_answer(query: str, context: str) -> str:
    try:
        return llm_generate(query, context)
    except Exception as e:
        print("[LLM ROUTER ERROR]:", e)
        return "System error: unable to generate answer."
