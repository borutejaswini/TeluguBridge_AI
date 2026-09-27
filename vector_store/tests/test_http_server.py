import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import Request, urlopen

import vector_store.server as server_module


class FakeRetriever:
    def retrieve(self, query, **kwargs):
        return {"query": query, "count": 0, "results": [], "faiss_candidates": [], "timings_ms": {}, "options": kwargs}


def test_retrieval_http_contract(monkeypatch):
    monkeypatch.setattr(server_module, "active_retriever", FakeRetriever())
    server = ThreadingHTTPServer(("127.0.0.1", 0), server_module.RetrievalHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base}/health") as response:
            assert json.loads(response.read())["loaded"] is True
        request = Request(
            f"{base}/retrieve",
            data=json.dumps({"query": "stem borer", "include_candidates": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            result = json.loads(response.read())
        assert result["query"] == "stem borer"
        assert result["options"]["include_candidates"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
