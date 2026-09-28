# knowledge-rag · 面试知识库 RAG 问答控制台

[English](README_en.md)

用本地 Markdown / TXT / PDF / Word 文档做知识库问答的个人控制台。仓库内置 5 篇 Agent 面试笔记作为样例语料，**零基础设施一条命令启动**：不需要安装 MySQL、Redis 或 Qdrant，索引与状态全部落在可丢弃的 `data/` 目录。默认 mock 模型即可体验全链路；在设置页填入自己的大模型 API key 后切换为真实检索与生成。

## 特性

- **RAG 全链路**：多格式解析 → manifest 与 sha256 源快照 → 切分与 embedding → Qdrant 向量入库 → 带引用摘录的生成。
- **混合检索**：自研中文 BM25（jieba 分词、稀疏向量点积）+ 稠密向量 RRF 融合，可选 rerank（失败静默降级）。
- **零基础设施 local 模式**：内嵌 Qdrant + JSON 文件态，单进程运行；首启自动索引内置语料。
- **mock 模型默认**：不填 key 也能跑通问答与评估；设置页支持保存并测试连通性。
- **检索评估**：Recall@k / MRR 脚本在 30 条标注种子集上对比 dense / hybrid / hybrid+rerank，语料外问题单独统计拒答正确率。
- **进阶 server 模式**：MySQL + Redis + Qdrant + 异步入库（事务性 outbox + `SKIP LOCKED`）+ 索引代双 CAS 原子发布/回滚 + 多轮会话与答案落盘（docker compose 一键编排）。

## 快速开始

前置：Python 3.12+；推荐安装 [uv](https://docs.astral.sh/uv/)（缺失时脚本自动回退 `venv + pip`）。前端构建产物已随仓库分发，**无需安装 Node**。

```bash
git clone <repo-url> && cd knowledge-rag
make run
```

打开 <http://127.0.0.1:8000>。首次启动会自动索引内置语料（mock embedding，秒级完成）。

- **使用真实模型**：控制台「设置」页填写 embedding / generation 的 base_url、model、key → 保存并测试连通性 → 在「知识来源」重新提交入库，即切换为真实向量与生成。
- **重置**：`make reset`（等价 `./run.sh --reset`）清空索引与运行时设置，下次启动重新种子化。
- local 模式为单轮问答；多轮会话、异步任务与索引发布属于 server 模式（见下）。

## 加入自己的语料

把 md / txt / pdf / docx 文件放入 `corpus/`（可按分类建子目录），在控制台「知识来源 → 发现本地文档」扫描并勾选纳入，再提交入库。扫描不调用模型；入库按 hash 增量复用未变化文档的向量。

## 支持的文档格式

| 格式 | 文字提取 | 说明 |
|---|---|---|
| `.md` | 保留 Markdown 标题切分 | 引用定位到原始行号 |
| `.txt` | UTF-8 / UTF-16 / GB18030 | 超长行拆为 ≤1200 字符片段 |
| `.pdf` | `pypdf` 文字层逐页切分 | 引用带页码；扫描页需先 OCR |
| `.docx` | `python-docx` 段落与表格 | 引用为提取文本行号 |
| `.doc` | macOS `textutil` 或 `antiword` | 缺工具时页面标记不可纳入 |

单文件上限 5 MiB、提取文本上限 200 万字符、PDF 最多 500 页；损坏或加密文件入库失败并显示原因，不发布部分索引。

## 配置

- local 模式无需任何环境变量；运行时数据在可丢弃的 `data/` 目录（索引、状态、模型设置）。
- `IR_` 前缀环境变量或 `.env` 可覆盖配置，样例见 `.env.example`（主要服务 server 模式）。
- 模型设置持久化在 `data/runtime_settings.json`（明文、仅本机使用）。

## 进阶：server 模式（完整工程链路）

```bash
docker compose -f deploy/compose.yaml --profile infra --profile mock-model --profile app up --build
```

提供 MySQL / Redis / Qdrant、异步入库任务（事务性 outbox + `SKIP LOCKED` 分发）、索引 generation 与双 CAS 发布/回滚、多轮会话与答案版本化落盘。架构目标与取舍见 [docs/design.md](docs/design.md)；设计目标不等于已完成的生产验收。注意：local 与 server 的索引存储不互通，切换后需重新入库。

## 开发

```bash
make setup   # Python + 前端依赖
make dev     # Vite 热更新 + mock 模型（5173）
make test    # 单元测试 + SSE 测试
uv run python eval/run_retrieval_eval.py   # 三组检索配置对比评估
```

## 排障速查

| 现象 | 处理 |
|---|---|
| 8000/8001 端口占用 | 在原终端停止旧进程；脚本不会自动杀进程 |
| 首启未自动入库 | 看启动日志「首启种子」提示；真实 provider 不可用时可在 UI 配置后手动入库 |
| hash 不一致 | 源文件改动后在「知识来源」刷新摘要再入库 |
| 页面提示 local 模式限制 | 会话/任务/版本属 server 模式，local 为单轮问答 |

## 安全说明

本工具**无鉴权**，定位为本地自托管：服务默认绑定 127.0.0.1，请勿直接暴露到公网；如需对外请先自行加反向代理与认证。数据源与模型 key 仅存本机。

## License

MIT
