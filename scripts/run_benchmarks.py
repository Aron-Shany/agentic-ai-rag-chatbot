"""Runs benchmark queries and writes docs/benchmark_results.md.
Usage (from repo root): python -m scripts.run_benchmarks
"""
from pathlib import Path

from app.rag_graph import rag_graph

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",
]


def main() -> None:
    lines = ["# Benchmark Results\n"]
    for q in QUERIES:
        r = rag_graph.invoke({"question": q})
        print(f"\nQ: {q}\nA: {r['answer']}\nConfidence: {r['confidence']}")
        lines.append(f"## Q: {q}\n")
        lines.append(f"**Answer:** {r['answer']}\n")
        lines.append(f"**Confidence:** {r['confidence']}\n")
        lines.append("**Retrieved chunks:**\n")
        for c in r["context_chunks"]:
            snippet = c["text"][:200].replace("\n", " ")
            lines.append(f"- (page {c['page']}, score {c['score']}) {snippet}...")
        lines.append("")
    Path("docs/benchmark_results.md").write_text("\n".join(lines))
    print("\nSaved docs/benchmark_results.md")


if __name__ == "__main__":
    main()
