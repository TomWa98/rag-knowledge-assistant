import json
import unittest
from unittest import mock

import src.rag as rag
import src.retriever as retriever

PRODUCTS = [
    {"model": "CAM-100", "source": "CAM-100.pdf"},
    {"model": "CAM-200", "source": "CAM-200.pdf"},
    {"model": "NVR-900", "source": "NVR-900.pdf"},
]


class FakeCollection:
    """Minimal stand-in for a ChromaDB collection with metadata filtering."""

    def __init__(self, records):
        self.records = records
        self.queries = []

    def count(self):
        return len(self.records)

    def query(self, query_embeddings, n_results, where=None, include=None):
        self.queries.append(where)
        matches = [
            r for r in self.records
            if where is None or r["source"] == where["source"]
        ][:n_results]

        return {
            "documents": [[r["text"] for r in matches]],
            "metadatas": [[
                {"source": r["source"], "page": r["page"], "chunk_index": r["chunk_index"]}
                for r in matches
            ]],
            "distances": [[r["distance"] for r in matches]],
        }

    def get(self, where, include=None):
        conditions = {k: v for c in where["$and"] for k, v in c.items()}
        matches = [
            r for r in self.records
            if all(r[key] == value for key, value in conditions.items())
        ]
        return {"documents": [r["text"] for r in matches]}


def make_records():
    records = []

    for product in PRODUCTS:
        for index in range(3):
            records.append({
                "text": f"{product['model']} text {index}",
                "source": product["source"],
                "page": 1,
                "chunk_index": index,
                "distance": 0.1 * (index + 1),
            })

    return records


def queried_sources(collection):
    return [where["source"] for where in collection.queries if where]


class DetectionTests(unittest.TestCase):
    def test_detects_multiple_models(self):
        found = retriever.detect_products(
            "Compare the resolution of cam-100 and CAM-200.", PRODUCTS
        )
        self.assertEqual([p["model"] for p in found], ["CAM-100", "CAM-200"])

    def test_no_model_mentioned(self):
        self.assertEqual(retriever.detect_products("Which support PoE?", PRODUCTS), [])

    def test_missing_catalog_returns_empty_list(self):
        self.assertEqual(retriever.load_product_catalog("does/not/exist.json"), [])


class TargetedRetrievalTests(unittest.TestCase):
    def test_returns_results_from_both_models(self):
        collection = FakeCollection(make_records())
        chunks = retriever.retrieve_targeted(
            "Compare CAM-100 and NVR-900", n_results=2, products=PRODUCTS,
            collection=collection, query_embedding=[0.0],
        )

        self.assertEqual(queried_sources(collection), ["CAM-100.pdf", "NVR-900.pdf"])
        self.assertEqual({c["source"] for c in chunks}, {"CAM-100.pdf", "NVR-900.pdf"})
        self.assertEqual(len(chunks), 4)

    def test_without_model_searches_globally(self):
        collection = FakeCollection(make_records())
        chunks = retriever.retrieve_targeted(
            "general question", n_results=5, products=PRODUCTS,
            collection=collection, query_embedding=[0.0],
        )

        self.assertEqual(collection.queries, [None])
        self.assertEqual(len(chunks), 5)

    def test_chunk_fields_preserved(self):
        collection = FakeCollection(make_records())
        chunk = retriever.retrieve_targeted(
            "CAM-200", n_results=1, products=PRODUCTS,
            collection=collection, query_embedding=[0.0],
        )[0]

        self.assertEqual(chunk, {
            "text": "CAM-200 text 0", "source": "CAM-200.pdf",
            "page": 1, "chunk_index": 0, "distance": 0.1,
        })


class ExhaustiveRetrievalTests(unittest.TestCase):
    def test_considers_every_document(self):
        collection = FakeCollection(make_records())
        documents = retriever.retrieve_exhaustive(
            "Which support PoE?", products=PRODUCTS,
            collection=collection, query_embedding=[0.0],
        )

        self.assertEqual([d["source"] for d in documents],
                         [p["source"] for p in PRODUCTS])
        self.assertEqual(queried_sources(collection), [p["source"] for p in PRODUCTS])
        for document in documents:
            self.assertTrue(all(c["source"] == document["source"] for c in document["chunks"]))

    def test_selection_limits_documents(self):
        collection = FakeCollection(make_records())
        documents = retriever.retrieve_exhaustive(
            "q", selection=["CAM-200", "nvr-900.pdf", "UNKNOWN"], products=PRODUCTS,
            collection=collection, query_embedding=[0.0],
        )

        self.assertEqual([d["model"] for d in documents], ["CAM-200", "NVR-900"])
        self.assertEqual(queried_sources(collection), ["CAM-200.pdf", "NVR-900.pdf"])

    def test_document_without_chunks_is_still_listed(self):
        records = [r for r in make_records() if r["source"] != "CAM-200.pdf"]
        documents = retriever.retrieve_exhaustive(
            "q", products=PRODUCTS, collection=FakeCollection(records),
            query_embedding=[0.0],
        )

        by_model = {d["model"]: d["chunks"] for d in documents}
        self.assertEqual(by_model["CAM-200"], [])
        self.assertEqual(len(documents), 3)

    def test_overview_is_first_chunk_of_each_document(self):
        documents = retriever.retrieve_exhaustive(
            "q", products=PRODUCTS, collection=FakeCollection(make_records()),
            query_embedding=[0.0],
        )

        self.assertEqual([d["overview"] for d in documents],
                         [f"{p['model']} text 0" for p in PRODUCTS])

    def test_duplicates_removed(self):
        records = make_records()
        records.append(dict(records[0]))
        documents = retriever.retrieve_exhaustive(
            "q", selection=["CAM-100"], n_results=10, products=PRODUCTS,
            collection=FakeCollection(records), query_embedding=[0.0],
        )

        self.assertEqual(len(documents[0]["chunks"]), 3)


class EmptyResultTests(unittest.TestCase):
    def test_empty_collection(self):
        collection = FakeCollection([])

        self.assertEqual(retriever.retrieve_targeted(
            "CAM-100", products=PRODUCTS, collection=collection, query_embedding=[0.0]
        ), [])
        documents = retriever.retrieve_exhaustive(
            "q", products=PRODUCTS, collection=collection, query_embedding=[0.0]
        )
        self.assertTrue(all(d["chunks"] == [] for d in documents))

    def test_empty_catalog(self):
        self.assertEqual(retriever.retrieve_exhaustive(
            "q", products=[], collection=FakeCollection([]), query_embedding=[0.0]
        ), [])

    def test_malformed_query_result(self):
        self.assertEqual(retriever._results_to_chunks({}), [])
        self.assertEqual(retriever._results_to_chunks({"documents": [[]]}), [])
        self.assertEqual(retriever._results_to_chunks(None), [])

    def test_exhaustive_answer_with_empty_collection(self):
        with mock.patch.object(retriever, "get_collection", return_value=FakeCollection([])), \
             mock.patch.object(retriever, "embed_query", return_value=[0.0]), \
             mock.patch.object(retriever, "PRODUCTS", PRODUCTS), \
             mock.patch.object(rag, "call_llm") as llm:
            answer, chunks = rag.ask_rag("q", mode="exhaustive")

        llm.assert_not_called()
        self.assertEqual(chunks, [])
        self.assertIn("Insufficient evidence", answer)
        self.assertNotIn("Explicitly negative", answer)

    def test_targeted_answer_with_empty_collection(self):
        with mock.patch.object(retriever, "get_collection", return_value=FakeCollection([])), \
             mock.patch.object(retriever, "embed_query", return_value=[0.0]), \
             mock.patch.object(rag, "call_llm") as llm:
            answer, chunks = rag.ask_rag("q")

        llm.assert_not_called()
        self.assertEqual(chunks, [])
        self.assertIn("cannot answer", answer)


class FindingParsingTests(unittest.TestCase):
    def test_valid_response(self):
        finding = rag.parse_finding(json.dumps({
            "product_type": "IP camera", "status": "supported",
            "answer": "Yes.", "evidence": "PoE (802.3af)",
        }))
        self.assertEqual(finding["status"], rag.SUPPORTED)
        self.assertEqual(finding["product_type"], "IP camera")

    def test_think_tags_and_code_fences(self):
        raw = '<think>hmm</think>```json\n{"status": "negative", "evidence": "No PoE", "answer": "No."}\n```'
        self.assertEqual(rag.parse_finding(raw)["status"], rag.NEGATIVE)

    def test_negative_without_evidence_is_not_negative(self):
        finding = rag.parse_finding('{"status": "negative", "answer": "Not mentioned."}')
        self.assertEqual(finding["status"], rag.INSUFFICIENT)

    def test_in_scope_defaults_to_true(self):
        self.assertTrue(rag.parse_finding('{"status": "insufficient_evidence"}')["in_scope"])
        self.assertTrue(rag.parse_finding('{"status": "supported", "evidence": "x", "in_scope": "unclear"}')["in_scope"])
        self.assertFalse(rag.parse_finding('{"status": "supported", "evidence": "x", "in_scope": false}')["in_scope"])

    def test_malformed_responses(self):
        for raw in ["", "not json", "{broken", '["list"]', '{"status": "maybe", "evidence": "x"}']:
            self.assertEqual(rag.parse_finding(raw)["status"], rag.INVALID, raw)


class ExhaustivePipelineTests(unittest.TestCase):
    def run_exhaustive(self, responses, selection=None):
        with mock.patch.object(retriever, "get_collection", return_value=FakeCollection(make_records())), \
             mock.patch.object(retriever, "embed_query", return_value=[0.0]), \
             mock.patch.object(retriever, "PRODUCTS", PRODUCTS), \
             mock.patch.object(rag, "call_llm", side_effect=responses) as llm:
            result = rag.answer_exhaustive("Which support feature X?", selection=selection)

        return result, llm

    def test_each_document_evaluated_individually(self):
        responses = [
            '{"product_type": "camera", "status": "supported", "answer": "Yes.", "evidence": "X: yes"}',
            '{"product_type": "camera", "status": "insufficient_evidence", "answer": "Not stated.", "evidence": ""}',
            'garbage',
        ]
        (answer, chunks, findings), llm = self.run_exhaustive(responses)

        self.assertEqual(llm.call_count, 3)
        for call, product in zip(llm.call_args_list, PRODUCTS):
            prompt = call.args[1]
            self.assertIn(product["source"], prompt)
            others = [p["model"] for p in PRODUCTS if p is not product]
            self.assertTrue(all(f"{m} text" not in prompt for m in others))

        self.assertEqual([f["status"] for f in findings],
                         [rag.SUPPORTED, rag.INSUFFICIENT, rag.INVALID])
        self.assertEqual({c["source"] for c in chunks}, {p["source"] for p in PRODUCTS})
        self.assertIn("**CAM-100**", answer)
        self.assertNotIn("Explicitly negative", answer)

    def test_missing_evidence_not_reported_as_negative(self):
        responses = [
            '{"status": "negative", "answer": "Not mentioned.", "evidence": ""}',
        ]
        (answer, _, findings), _ = self.run_exhaustive(responses, selection=["CAM-100"])

        self.assertEqual(findings[0]["status"], rag.INSUFFICIENT)
        self.assertNotIn("Explicitly negative", answer)
        self.assertIn("Insufficient evidence", answer)


    def test_out_of_scope_listed_separately(self):
        responses = [
            '{"product_type": "camera", "in_scope": true, "status": "supported", "answer": "Yes.", "evidence": "X: yes"}',
            '{"product_type": "recorder", "in_scope": false, "status": "supported", "answer": "Yes.", "evidence": "X: yes"}',
        ]
        (answer, _, _), _ = self.run_exhaustive(responses, selection=["CAM-100", "NVR-900"])

        supported, out_of_scope = answer.split("**Not the kind of product asked about:**")
        self.assertIn("CAM-100", supported)
        self.assertNotIn("NVR-900", supported)
        self.assertIn("NVR-900", out_of_scope)


class CompatibilityTests(unittest.TestCase):
    def test_ask_rag_default_is_targeted(self):
        with mock.patch.object(retriever, "get_collection", return_value=FakeCollection(make_records())), \
             mock.patch.object(retriever, "embed_query", return_value=[0.0]), \
             mock.patch.object(retriever, "PRODUCTS", PRODUCTS), \
             mock.patch.object(rag, "call_llm", return_value="Answer") as llm:
            answer, chunks = rag.ask_rag("Compare CAM-100 and CAM-200")

        self.assertEqual(answer, "Answer")
        self.assertEqual(llm.call_count, 1)
        self.assertEqual({c["source"] for c in chunks}, {"CAM-100.pdf", "CAM-200.pdf"})
        self.assertEqual(set(chunks[0]), {"text", "source", "page", "chunk_index", "distance"})

    def test_legacy_rag_functions_still_exported(self):
        self.assertTrue(callable(rag.retrieve))
        self.assertTrue(callable(rag.detect_products))
        self.assertTrue(callable(rag.load_product_catalog))
        self.assertIsInstance(rag.PRODUCTS, list)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            rag.ask_rag("q", mode="fuzzy")


if __name__ == "__main__":
    unittest.main()
