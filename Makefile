SHELL := /bin/bash
MOCK ?= 0

.PHONY: dev start serve setup build-web migrate manifest-validate test run reset

dev: web/.installed
	IR_QDRANT_MODE=local uv run python scripts/serve.py --local --dev --mock

start: build-web
	IR_QDRANT_MODE=local uv run python scripts/serve.py --local --mock

serve: build-web
	IR_QDRANT_MODE=server uv run python scripts/serve.py $(if $(filter 1,$(MOCK)),--mock,)

setup:
	uv sync
	$(MAKE) web/.installed

build-web: web/.installed
	cd web && npm run build

web/.installed: web/package.json web/package-lock.json
	cd web && npm ci
	@touch web/.installed

migrate:
	uv run alembic upgrade head

manifest-validate:
	uv run interview-rag-manifest validate

test:
	uv run python -m unittest discover -s tests -v
	node --experimental-strip-types --test tests/sse.test.mjs

## 零基础设施一条命令启动（等价 ./run.sh；自动装依赖、复用内置前端产物、首启索引内置语料）
run:
	./run.sh

## 清空 data/ 下可丢弃数据（索引、运行时设置），下次启动重新种子化
reset:
	./run.sh --reset
