# knowledge-rag · RAG Q&A Console for Your Notes

[中文](README.md)

A personal knowledge-base Q&A console over local Markdown / TXT / PDF / Word documents. Ships with five bundled Agent-interview notes as sample corpus and starts with **zero infrastructure**: no MySQL, Redis or Qdrant install required — index and state live in the disposable `data/` directory. A mock model is enabled by default so the full pipeline works without any API key; add your own LLM key in the settings page to switch to real retrieval and generation.

## Highlights

- **Full RAG pipeline**: multi-format parsing → manifest + sha256 source snapshots → chunking & embedding → Qdrant ingestion → cited generation.
- **Hybrid retrieval**: self-built Chinese BM25 (jieba tokenization, sparse-vector dot product) + dense vectors fused with RRF; optional rerank with silent fallback on failure.
- **Zero-infra local mode**: embedded Qdrant + JSON file state, single process; bundled corpus is indexed automatically on first run.
- **Mock model by default**: explore the whole loop without a key; the settings page can save and test real provider connectivity.
- **Retrieval eval**: Recall@k / MRR over a 30-query labeled seed set comparing dense / hybrid / hybrid+rerank, with out-of-corpus refusal tracked separately.
- **Advanced server mode**: MySQL + Redis + Qdrant + async ingestion (transactional outbox + `SKIP LOCKED`) + immutable index generations with dual-CAS publish/rollback + multi-turn conversations (one-command docker compose).

## Quick start

Requires Python 3.12+; [uv](https://docs.astral.sh/uv/) recommended (the script falls back to `venv + pip`). Prebuilt frontend assets are committed, so **Node is not required**.

```bash
git clone <repo-url> && cd knowledge-rag
make run
```

Open <http://127.0.0.1:8000>. The first run indexes the bundled corpus automatically (mock embedding, a few seconds).

- **Use a real model**: Settings page → fill embedding / generation base_url, model, key → save & test → re-submit ingestion in "Knowledge sources".
- **Reset**: `make reset` (same as `./run.sh --reset`) wipes index and runtime settings; next start re-seeds.
- Local mode answers single-turn questions; conversations, async jobs and index publishing belong to server mode.

## Add your own corpus

Drop md / txt / pdf / docx files into `corpus/` (subdirectories act as categories), then scan and include them via "Knowledge sources → Discover local documents" and submit ingestion. Scanning never calls the model; ingestion reuses vectors of unchanged documents by hash.

## Configuration

- Local mode needs no environment variables; runtime data lives in the disposable `data/` directory.
- `IR_`-prefixed env vars or `.env` override settings; see `.env.example` (mainly for server mode).
- Model settings persist in `data/runtime_settings.json` (plaintext, local only).

## Advanced: server mode (full engineering path)

```bash
docker compose -f deploy/compose.yaml --profile infra --profile mock-model --profile app up --build
```

Provides MySQL / Redis / Qdrant, async ingestion jobs (transactional outbox + `SKIP LOCKED` dispatch), index generations with dual-CAS publish/rollback, and versioned conversation/answer storage. See [docs/design.md](docs/design.md) for architecture goals and trade-offs; design goals are not claimed as production acceptance.

## Development

```bash
make setup   # Python + frontend deps
make dev     # Vite hot reload + mock model
make test    # unit tests + SSE tests
uv run python eval/run_retrieval_eval.py   # three-config retrieval comparison
```

## Security note

This tool has **no authentication** and is meant for local self-hosting: services bind to 127.0.0.1 by default. Do not expose them to the public internet without adding your own proxy and auth. Keys and datasources stay on your machine.

## License

MIT
