"""Interactive chunk annotation tool for evaluation dataset.

For each evaluation query, retrieves top-K chunks and allows manual labeling
of relevance (0=irrelevant, 1=partially relevant, 2=fully relevant).

Annotations are saved back to the evaluation dataset JSON under
`chunk_annotations` for each query.

Usage:
    python scripts/annotate_chunks.py [--top-k 10] [--query-id eq1]

Requirements:
    - Weaviate running with indexed corpus
    - Evaluation dataset at data/evaluation/rag_evaluation_dataset.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import retrieve
from src.utils.system_checks import is_weaviate_ready


DATASET_PATH = Path("data/evaluation/rag_evaluation_dataset.json")


def load_dataset() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: dict) -> None:
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def annotate_query(query: dict, top_k: int) -> list[dict] | None:
    """Retrieve chunks for a query and prompt for relevance labels."""
    print(f"\n{'='*70}")
    print(f"Query ID: {query['query_id']}")
    print(f"Query: {query['query']}")
    print(f"Expected pages: {query.get('expected_pages', [])}")
    print(f"Expected answer: {query.get('expected_answer_summary', 'N/A')}")
    print(f"{'='*70}")

    results = retrieve(query["query"], top_k=top_k)

    if not results:
        print("  No results retrieved.")
        return None

    annotations = []
    for i, r in enumerate(results):
        print(f"\n--- Chunk {i+1}/{len(results)} ---")
        print(f"  Chunk ID:   {r.chunk_id}")
        print(f"  Page:       {r.page_number}")
        print(f"  Section:    {r.section_title or 'N/A'}")
        print(f"  Similarity: {r.similarity_score:.4f}")
        print("  Text (first 300 chars):")
        print(f"  {r.text[:300]}...")
        print()

        while True:
            label = input("  Relevance [0=irrelevant, 1=partial, 2=relevant, s=skip query]: ")
            label = label.strip().lower()
            if label == "s":
                return None
            if label in ("0", "1", "2"):
                annotations.append({
                    "chunk_id": r.chunk_id,
                    "page_number": r.page_number,
                    "similarity_score": r.similarity_score,
                    "relevance": int(label),
                })
                break
            print("  Invalid input. Enter 0, 1, 2, or s.")

    return annotations


def run_annotation(query_id: str | None = None, top_k: int = 10) -> None:
    """Run interactive annotation session."""
    if not is_weaviate_ready():
        print("ERROR: Weaviate is not running. Start it with: docker compose up -d")
        sys.exit(1)

    data = load_dataset()
    queries = data["queries"]
    annotated_count = 0

    for q in queries:
        # Skip negative queries
        if q.get("difficulty") == "negative":
            continue

        # Filter to specific query if requested
        if query_id and q["query_id"] != query_id:
            continue

        # Skip already annotated queries
        if q.get("chunk_annotations"):
            existing = len(q["chunk_annotations"])
            print(f"\n[SKIP] {q['query_id']} already has {existing} annotations.")
            continue

        annotations = annotate_query(q, top_k)

        if annotations is not None:
            q["chunk_annotations"] = annotations
            save_dataset(data)
            annotated_count += 1
            print(f"\n  Saved {len(annotations)} annotations for {q['query_id']}.")

        cont = input("\nContinue to next query? [Y/n]: ").strip().lower()
        if cont == "n":
            break

    print(f"\nAnnotation complete. {annotated_count} queries annotated.")


def report_chunk_quality(data: dict) -> None:
    """Report chunk quality diagnostics from existing annotations."""
    print("\n" + "=" * 70)
    print("CHUNK QUALITY REPORT")
    print("=" * 70)

    annotated = [q for q in data["queries"] if q.get("chunk_annotations")]
    if not annotated:
        print("No annotated queries found. Run annotation first.")
        return

    total_chunks = 0
    relevant_chunks = 0
    boundary_splits = 0

    for q in annotated:
        anns = q["chunk_annotations"]
        total_chunks += len(anns)
        rel = [a for a in anns if a["relevance"] >= 1]
        relevant_chunks += len(rel)

        # Detect boundary splits: relevant content on expected page but
        # partial relevance suggests the answer is split across chunks
        partial = [a for a in anns if a["relevance"] == 1]
        if len(partial) >= 2:
            boundary_splits += 1

        print(f"\n{q['query_id']} ({q.get('difficulty', '?')}):")
        print(f"  Chunks: {len(anns)} total, {len(rel)} relevant, {len(partial)} partial")
        for a in anns:
            label = {0: "IRRELEVANT", 1: "PARTIAL", 2: "RELEVANT"}[a["relevance"]]
            print(f"    {a['chunk_id']} (p.{a['page_number']}, sim={a['similarity_score']:.4f}) → {label}")

    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"  Annotated queries:  {len(annotated)}")
    print(f"  Total chunks:       {total_chunks}")
    print(f"  Relevant chunks:    {relevant_chunks} ({relevant_chunks/total_chunks*100:.1f}%)")
    print(f"  Boundary splits:    {boundary_splits} queries with 2+ partial chunks")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate chunk relevance for evaluation")
    parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve per query")
    parser.add_argument("--query-id", type=str, default=None, help="Annotate a specific query only")
    parser.add_argument("--report", action="store_true", help="Print chunk quality report from existing annotations")
    args = parser.parse_args()

    if args.report:
        data = load_dataset()
        report_chunk_quality(data)
    else:
        run_annotation(query_id=args.query_id, top_k=args.top_k)
