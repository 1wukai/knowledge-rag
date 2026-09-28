# Stage 1：Embedding 与向量检索

> 目标：理解「文本如何变成向量」「怎么度量相似」「向量库和索引怎么工作」。这是 RAG 的检索底座，面试必考。

## 1. 什么是 Embedding

**Embedding = 把文本（或其他对象）映射成一个高维稠密向量，且语义相近的文本向量也相近。**

- 你输入一句话，embedding 模型输出一个固定长度的浮点数组，比如 1536 维：`[0.012, -0.033, 0.28, ...]`
- 关键性质：**"苹果手机怎么样" 和 "iPhone 好用吗"** 虽然字面完全不同，但向量距离很近；而 **"苹果手机" 和 "苹果水果"** 字面相近，向量却被拉远（因为语义不同）

<a id="vector-shape"></a>

### 问答：向量是什么形式

在文本 Embedding 场景中，一个向量通常就是**固定维数的浮点数组**。程序里可以用 Go 的 `[]float32` 表示，HTTP JSON 中通常表现为数字数组，索引内部可能保存浮点数或量化后的编码。

```go
// 人工构造的 3 维教学数据，不是模型实际输出。
embedding := []float32{0.8, 0.6, 0.0}
fmt.Println(len(embedding)) // 3：维数
```

例如，一条知识库记录可以表示为：

```json
{
  "id": 101,
  "content": "MySQL 复合索引的使用方法",
  "source": "mysql-index.md",
  "embedding": [0.8, 0.6, 0.0]
}
```

- **维数**是数组长度；1536 维意味着有 1536 个坐标，不是文本有 1536 个 token。
- **坐标值**可以为正、负或零；它们不是词的编号，也不一定是概率。
- **单个坐标通常没有稳定、可直接命名的业务含义**，应把整组数作为分布式表示来理解。后文的「隐式特征」类比不意味着可以把每维解释成一个标签。
- **向量与原文分别保留**：检索先用向量找相关记录，再通过 ID 取原文。检索链路不能靠向量可靠还原原文；同时也不能把向量视为匿名化或加密后的安全数据。

后端类比：原文是业务记录，向量是为相似度检索准备的派生字段，ANN 索引是加速访问的索引结构。这三个东西职责不同。

以未压缩的 `float32` 为例，1536 维向量的原始数值占用 `1536 × 4 = 6144` 字节，即 6 KiB。实际存储还包括 ID、原文、元数据、索引结构和副本；量化又会改变向量的表示与空间成本。

面试口述：**文本向量在工程上就是固定长度的数值数组，用整体的方向或距离表达学习到的相关性。维数、token 数、原文字节数是三个不同概念。**

### 为什么相近的文本，向量也相近（知其所以然）

这是 embedding 最核心的性质，分三层理解：

**第一层：向量 = 语义特征的「坐标」**

向量的每一维不是有名字的标签，而是模型学到的某种**隐式语义特征**的强度（如「主题偏科技」「语气在求助」「涉及手机」……）。一句话被编码成 1536 个特征分，语义相近的句子，这些特征分的取值模式接近，向量自然就接近。

后端类比：**embedding 模型像一个「自动打标签器」**——把每句话映射成 1536 个隐式标签的分值。语义相近的句子，标签分值接近。向量之间的「距离」其实就是「特征分差异的总和」。

**第二层：这个性质是「训练目标逼出来的」，不是巧合**

模型不是天生懂语义，是损失函数直接优化出来的。主流训练方法是**对比学习（contrastive learning）**：

1. 训练数据组织成「正样本对/负样本对」：`(锚点, 正样本)` 语义相近，`(锚点, 负样本)` 语义不相近
2. 损失函数（InfoNCE）的目标就一句话：**让正样本对的相似度尽量大、负样本对的相似度尽量小**
3. 数学上，损失里的打分就是两个向量的内积（相似度），softmax 后要求「正样本概率最高」——等价于逼模型把语义相近的句子映射到相邻位置

所以「语义相近 → 向量相近」是训练目标**定义**出来的，不是模型的天然能力。

**第三层：为什么能泛化到没见过的新句子**

训练数据不可能穷尽所有句子对。但模型为了完成「拉近正样本、推远负样本」，被迫学到的是**「哪些语言特征决定语义相似」**（同义改写、主题、意图），而不是死记具体句子。所以遇到新句子，它也能提取出正确的特征、落到正确的区域。

**一个关键认知**：这个性质是「学到的近似」，不是绝对真理。模型没见过的垂直领域（行业黑话、专有名词），embedding 也可能失效——这也是为什么有时要微调或用领域专用模型。

后端类比：**embedding 就像一个「语义哈希」**——把非结构化的文本，变成可以做数学运算（距离计算）的结构化数据。有了向量，文本检索就变成了「向量空间里的最近邻查找」，这就能用数据库、索引、缓存那套工程手段来优化了。

### 为什么 RAG 需要 embedding

用户问的是「怎么给 iPhone 换电池」，文档里写的却是「更换 iPhone 电池的步骤」——措辞不同但语义相关。Embedding 为这种检索提供向量表示。数据库也能支持全文检索等能力，`LIKE` 本身是模式匹配；RAG 可以采用关键词、向量或混合检索，使用 RAG 不意味着必须使用 Embedding。

## 2. 相似度度量（必考，要能手写）

### 余弦相似度（Cosine Similarity）—— 最常用

衡量两个向量的**方向**是否一致，与长度无关。取值范围 [-1, 1]，越接近 1 越相似。

```text
cos(a, b) = (a·b) / (|a| × |b|)
```

### 点积（Dot Product）

余弦相似度的分子部分。当两个向量**都已归一化（模长为 1）时，点积 == 余弦相似度**。是否自动归一化取决于模型和索引配置，接入时需要确认。

### 欧氏距离（Euclidean Distance）—— 距离越小越相似

衡量绝对距离，受向量长度影响。

```text
dist(a, b) = √(Σ(a_i - b_i)²)
```

### Go 实现

下面是推导公式用的函数片段，假定输入等长且数值有限；零向量返回 0 是示例约定，数学上的余弦对零向量未定义。第 4 节的[完整检索程序](#vector-search-go)会显式拒绝维数错误、零向量和非法数值。

```go
package main

import "math"

// 余弦相似度，越接近 1 越相似
func CosineSimilarity(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// 点积
func DotProduct(a, b []float64) float64 {
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// 欧氏距离，越小越相似
func EuclideanDistance(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}
```

### 怎么选

| 场景 | 推荐 |
|------|------|
| 通用语义检索（默认） | 余弦相似度 |
| 已归一化向量（多数模型默认 L2 归一化） | 点积（等价于余弦，更快） |
| 需要区分「长度/强度」的场景（如图像特征） | 欧氏距离 |

面试要点：**「余弦相似度和欧氏距离的区别？」** → 答：余弦只看方向（归一化后与长度无关），欧氏看绝对距离受长度影响；文本语义检索通常用余弦，因为文本向量长度往往不代表语义强弱。

## 3. 主流 Embedding 模型

| 模型 | 维度 | 特点 |
|------|------|------|
| OpenAI text-embedding-3-small | 1536 | 便宜、通用，API 调用 |
| OpenAI text-embedding-3-large | 3072 | 更准，可降维（Matryoshka） |
| BGE（BAAI/bge-large-zh） | 1024 | 中文效果好，可本地部署（开源） |
| m3e（中文社区模型） | 768 | 中文轻量，本地部署 |

选型要点：
- 中文场景优先 BGE / m3e（本地部署可控成本）
- 快速验证用 OpenAI 的 API
- **维度越高通常越准，但存储和计算成本越高**（向量库里维度是固定的，换模型要重建索引）
  - 为什么：维度越高，能编码的语义特征越细、容量越大，检索更准；但边际收益递减、存储和计算成本线性增长

面试要点：**「换 embedding 模型要注意什么？」** → 答：维度可能不同，向量不可混用，需要重新 embedding 全部数据并重建索引。

## 4. 向量数据库

### 存向量与加速检索是两件事

- 向量可以通过数组、二进制字段或专用向量类型保存；能保存不等于能高效检索。
- 精确扫描 N 条 d 维向量，需要约 `O(N × d)` 的距离计算。小数据集、过滤后候选很少或需要精确基线时，这种方式有价值。
- 当数据规模、维数和并发使精确扫描超出延迟预算时，可以用**近似最近邻（ANN）索引**减少搜索工作；具体收益需要测量。

### pgvector —— 你的 MySQL 背景最容易上手

PostgreSQL 的向量扩展，能直接在 SQL 里做向量检索。你熟 SQL，这个零门槛。

```sql
-- 启用扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 建表：embedding 维度要和模型一致
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)
);

-- 插入（向量用字符串表示）
INSERT INTO documents (content, embedding)
VALUES ('如何更换 iPhone 电池', '[0.012, -0.033, ...]');

-- 余弦相似度检索 top-5：<=> 是余弦「距离」，1 - 距离 = 相似度
SELECT id, content, 1 - (embedding <=> '[0.01, ...]') AS similarity
FROM documents
ORDER BY embedding <=> '[0.01, ...]'
LIMIT 5;

-- 建 HNSW 索引加速
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
```

### 其他主流向量库

| 向量库 | 定位 | 特点 |
|--------|------|------|
| **pgvector** | PG 扩展 | 和业务数据同库，事务一致，运维简单 |
| **Milvus** | 专用向量库 | 分布式、十亿级、功能全，运维重 |
| **Qdrant** | 专用向量库 | Rust 实现，性能好，带过滤和 payload |
| **Weaviate** | 专用向量库 | 内置混合检索、模块化 |

选型建议（面试常见）：
- 数据量不大（< 千万）、要和业务数据 join → **pgvector**
- 十亿级、高吞吐、独立检索服务 → **Milvus / Qdrant**
- 需要开箱即用的混合检索 → Weaviate

面试要点：**「向量库和传统数据库的本质区别？」** → 答：传统库做精确匹配/范围查询，向量库做「语义相似度」的近似最近邻检索，核心是 ANN 索引 + 距离计算，而不是 B+Tree。

<a id="qdrant-comparison"></a>

### 问答：Qdrant 是什么，和 Faiss、Milvus、Weaviate、pgvector 怎么比

**是什么**：Qdrant 是 Rust 编写的开源向量数据库，以独立服务方式运行（HTTP + gRPC API），保存向量和 payload（任意 JSON 元数据），基于 HNSW 做 ANN 检索，检索、过滤、持久化都由服务自身管理。

**核心特点（为什么选它）**：

- **可过滤的 HNSW**：payload 过滤条件直接参与图索引的搜索过程，而不是「先 ANN 再过滤」（可能过滤完不够 k 条）或「先过滤再暴力扫」（候选多时慢）。多租户、按标签限定范围的检索，召回和延迟更稳定。
- **量化压缩**：支持 scalar / product / binary 量化，用可控的精度损失换内存，相当于把第 5 节的 PQ 思路做成了开箱即用的产品能力。
- **两种运行形态、同一套 API**：服务端模式（Docker/K8s，支持分布式分片与副本）；Python 客户端的**本地内嵌模式**（`QdrantClient(path=...)`，数据落在本地目录，无需启动任何服务）。开发调试和生产部署可以无缝切换。
- **工程能力**：快照备份、collection alias（重建索引后别名一键切换、不停机）、mTLS 等。

**与同类产品对比**：

| 比较项 | Faiss | pgvector | Qdrant | Milvus | Weaviate |
|--------|-------|----------|--------|--------|----------|
| 形态 | 应用内库 | PostgreSQL 扩展 | 独立服务 / 客户端内嵌 | 独立服务（分布式架构） | 独立服务 |
| 实现语言 | C++（Python 绑定） | C | Rust | Go / C++ | Go |
| 数据管理 | 应用自己负责持久化、原文与 ID 映射 | 与业务数据同库，SQL + 事务 | 向量 + payload 由服务 API 管理 | 存算分离（对象存储 + 计算节点） | 向量 + 对象，内置向量化模块 |
| 过滤检索 | 需自行设计（IDSelector 等） | SQL WHERE 天然组合 | 可过滤 HNSW | 分区 / 标量字段过滤 | 内置过滤 + BM25 混合检索 |
| 规模定位 | 单机库，上限取决于应用工程 | 中小规模，随 PG 实例 | 单机到集群，千万～十亿级 | 十亿级大集群、高吞吐 | 中小规模，开箱即用 |
| 运维成本 | 无服务运维，但要自建服务层 | 最低（复用现有 PG） | 低（单二进制 / 单容器） | 高（依赖 etcd、对象存储、消息队列等多组件） | 中 |
| 典型选型理由 | 索引实验、嵌入自研服务 | 数据量不大、要和业务数据 join | 要专用向量服务又不想重运维 | 超大规模、独立检索基础设施 | 要内置向量化与混合检索 |

一句话记忆：**Faiss 是「检索引擎库」，pgvector 是「数据库顺手加的向量能力」，Qdrant / Weaviate 是「轻中量级专用向量服务」，Milvus 是「重量级分布式向量基础设施」。**

后端类比：Qdrant 与 Milvus 的取舍，类似「单容器部署的 Redis」与「完整 Kafka 集群」的区别——都能解决问题，差在运维组件数量、规模上限和团队投入。

**本仓库项目的实际用法**：`projects/interview-rag` 开发态用客户端内嵌模式 `QdrantClient(path="data/qdrant_local")`，零外部依赖即可跑通全链路；多进程 / 容器部署时切到服务端模式 `QdrantClient(url=...)`（compose 中运行 `qdrant/qdrant` 镜像，挂载数据卷）。切换只改配置（`qdrant_mode: local | server`），业务代码不变，见 `src/interview_rag/storage/vectorstore.py`——这正是「同一 API、两种形态」带来的工程收益。

面试口述：**Qdrant 是 Rust 编写的开源向量数据库，以服务形式保存向量和 payload，基于 HNSW 做 ANN 检索，亮点是过滤条件直接参与索引搜索、且支持客户端内嵌与服务端两种形态。对比来看：pgvector 胜在与业务数据同库、运维最省，但规模受 PG 限制；Milvus 面向十亿级分布式场景，代价是多组件重运维；Faiss 只是库，持久化、过滤和服务化都要自己补；Weaviate 强在内置向量化模块和混合检索。我们项目用 Qdrant 内嵌模式做本地开发、服务端模式做容器部署，同一套 API 无缝切换。**

自检问题：

1. 「先 ANN 再过滤」为什么可能返回不足 k 条？可过滤 HNSW 是怎么在图遍历时处理过滤条件的？
2. scalar 量化和 binary 量化分别适合什么场景？精度损失的量级差异是什么？
3. 什么时候 pgvector 是比专用向量库更好的选择？什么时候必须换专用向量库？

<a id="faiss-es-storage-search"></a>

### 问答：Faiss、Elasticsearch 如何存向量和匹配用户输入

先把流程拆成两条链路：

```text
入库：文档 → 切块 → Embedding 编码 → 保存向量、ID、原文和元数据 → 建索引
查询：用户问题 → 查询编码 → 向量检索 top-k → 取回原文 → 可选 rerank → 组织答案
```

查询与文档必须使用**兼容的同一套 Embedding 模型与编码配置**，维数相同只是必要条件；不同模型输出同样维数也不能直接混用。某些检索模型对 query/document 要使用不同的规定前缀或编码入口，应遵循模型要求。[Elasticsearch kNN 文档](https://www.elastic.co/docs/solutions/search/vector/knn)

#### Faiss：应用内的向量检索库

Faiss 提供索引、向量添加和相似度搜索能力。它不是开箱即用的完整数据库服务：原文、业务元数据、权限、服务接口和更新一致性通常由应用及外部存储负责。

可以把它类比为嵌入应用的一套检索引擎：索引保存向量或编码，检索返回 ID 和分数，应用根据 ID 到 MySQL、文件或内存映射取原文。

下面采用 Faiss 官方 Python 接口展示原生调用，便于对应 API；Go 版本的检索原理程序见本节末尾。运行此片段需要已安装 `faiss-cpu` 和 `numpy`。

```python
import faiss
import numpy as np

# 人工构造的向量，用于看清排序过程；实际应由 Embedding 模型生成。
docs = ["MySQL 复合索引的使用方法", "使用 EXPLAIN 分析慢查询", "苹果的保存方法"]
vectors = np.array([[0.8, 0.6, 0], [0.6, 0.8, 0], [0, 0, 1]], dtype="float32")
query = np.array([[1, 0, 0]], dtype="float32")

faiss.normalize_L2(vectors)
faiss.normalize_L2(query)
index = faiss.IndexFlatIP(3)  # 精确内积搜索；双方归一化后等于余弦相似度
index.add(vectors)           # 此处使用添加顺序作为 ID：0、1、2
scores, ids = index.search(query, 2)
for pos, score in zip(ids[0], scores[0]):
    if pos >= 0:             # 不足 k 个结果时可能有 -1 占位
        print(pos, round(float(score), 2), docs[pos])

faiss.write_index(index, "demo.faiss")
restored = faiss.read_index("demo.faiss")
print(restored.ntotal)
```

预期前两条是复合索引与 EXPLAIN，分数约为 0.8、0.6。`IndexFlatIP` 是精确扫描；若改用 HNSW、IVF 等索引，则引入不同的训练要求、内存开销和召回率取舍。浮点矩阵输入及 `add/search` 的用法见 [Faiss 入门](https://github.com/facebookresearch/faiss/wiki/Getting-started)，归一化内积与余弦的关系见 [Faiss 距离说明](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)。

`write_index/read_index` 用于索引持久化，`docs` 这样的原文映射仍要单独持久化并保持 ID 对应关系；重新加载索引不会自动恢复你外部保存的业务数据。[Faiss 索引读写](https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning)

#### Elasticsearch：文档与向量一起保存

一篇 ES 文档可以同时保存 `content`、`source`、`tenant_id` 和 `embedding`。其中 `dense_vector` 描述向量字段，索引配置决定如何执行 kNN 搜索。

以下是适用于 Elasticsearch 8.19 的 Dev Tools 请求示例，使用外部生成向量的方式；服务需自行准备。本例明确选择普通 `hnsw`，避免依赖不同版本的默认量化配置。数据仍为人工教学向量。[8.19 kNN 文档](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/knn-search.html)

```http
PUT interview_demo
{
  "mappings": {
    "properties": {
      "content": {"type": "text"},
      "source": {"type": "keyword"},
      "tenant_id": {"type": "keyword"},
      "embedding": {
        "type": "dense_vector",
        "dims": 3,
        "index": true,
        "similarity": "cosine",
        "index_options": {"type": "hnsw"}
      }
    }
  }
}

POST interview_demo/_bulk?refresh=wait_for
{"index":{"_id":"101"}}
{"content":"MySQL 复合索引的使用方法","source":"mysql-index.md","tenant_id":"demo","embedding":[0.8,0.6,0]}
{"index":{"_id":"102"}}
{"content":"使用 EXPLAIN 分析慢查询","source":"mysql-explain.md","tenant_id":"demo","embedding":[0.6,0.8,0]}
{"index":{"_id":"103"}}
{"content":"苹果的保存方法","source":"fruit.md","tenant_id":"demo","embedding":[0,0,1]}

POST interview_demo/_search
{
  "size": 2,
  "_source": ["content", "source"],
  "knn": {
    "field": "embedding",
    "query_vector": [1,0,0],
    "k": 2,
    "num_candidates": 20,
    "filter": {"term": {"tenant_id": "demo"}}
  }
}
```

Bulk 请求使用 NDJSON，最后一条数据后也需要换行。这里的 `[1,0,0]` 代替用户问题的真实编码结果；实际由应用先调用 Embedding 模型，再发送查询向量。

- `k` 是需要找的近邻数量，`size` 控制最终返回条数；这里都设为 2。
- `num_candidates` 是每个分片搜索时考虑的候选规模，不是 Embedding 维数；在本例 HNSW 路径中增大它通常会增加搜索工作量，并可能提高召回。
- `filter` 在 kNN 内限定候选范围；实际租户身份应由后端鉴权确定，不能信任客户端任意指定租户值。
- ES 对纯 cosine kNN 的 `_score` 使用 `(1 + cosine) / 2`。本例前两条预计约为 0.9、0.8，而不是 Faiss 的 0.8、0.6；混合检索、boost 等还会改变最终分数。分数不是正确概率。[向量字段与分数定义](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector)

| 比较项 | Faiss | Elasticsearch |
|--------|-------|---------------|
| 使用方式 | 库调用，服务封装由应用承担 | 通过 HTTP 调用搜索服务 |
| 原文与元数据 | 通常另存，维护 ID 映射 | 可以与向量保存于同一文档 |
| 向量持久化 | 索引文件读写，更新机制由应用组织 | 由 ES 的文档、索引和分片机制管理 |
| 业务过滤与关键词检索 | 需要按索引能力和业务需求另行设计 | 可使用字段过滤、全文检索并组合检索策略 |
| 主要取舍 | 便于控制索引，但要补服务与数据管理 | 能整合文档搜索能力，但需承担 ES 运维和资源成本 |

后端面试口述：**先把文档编码成向量并建立索引，再把用户问题编码到兼容空间中，按相似度搜索 top-k。Faiss 通常返回 ID 后由应用取原文，Elasticsearch 可以直接返回匹配文档。语义相关性来自模型表示，检索库负责数值搜索。**

<a id="vector-search-go"></a>

#### Go：可运行的“向量 → 排序 → 原文”示例

下面使用标准库精确扫描，展示与上述相同的数值搜索过程。保存为 `main.go` 后执行 `GO111MODULE=off go run main.go`；人工向量只演示计算，不代表模型能达到这些分数。

<!-- runnable: vector-search -->
```go
package main

import (
	"fmt"
	"math"
	"sort"
)

type Document struct {
	ID     int
	Text   string
	Vector []float32
}

func cosine(a, b []float32) (float64, error) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, fmt.Errorf("向量必须非空且等维")
	}
	var dot, aa, bb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) {
			return 0, fmt.Errorf("向量包含非法数值")
		}
		dot += x * y
		aa += x * x
		bb += y * y
	}
	if aa == 0 || bb == 0 {
		return 0, fmt.Errorf("零向量没有余弦相似度")
	}
	return dot / (math.Sqrt(aa) * math.Sqrt(bb)), nil
}

func main() {
	docs := []Document{
		{101, "MySQL 复合索引的使用方法", []float32{0.8, 0.6, 0}},
		{102, "使用 EXPLAIN 分析慢查询", []float32{0.6, 0.8, 0}},
		{103, "苹果的保存方法", []float32{0, 0, 1}},
	}
	query := []float32{1, 0, 0}
	type Hit struct {
		Doc   Document
		Score float64
	}
	hits := make([]Hit, 0, len(docs))
	for _, doc := range docs {
		score, err := cosine(query, doc.Vector)
		if err != nil {
			panic(err)
		}
		hits = append(hits, Hit{doc, score})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score == hits[j].Score {
			return hits[i].Doc.ID < hits[j].Doc.ID
		}
		return hits[i].Score > hits[j].Score
	})
	k := 2
	if k > len(hits) {
		k = len(hits)
	}
	for _, hit := range hits[:k] {
		fmt.Printf("id=%d cosine=%.2f %s\n", hit.Doc.ID, hit.Score, hit.Doc.Text)
	}
}
```

输出前两条的 ID 为 101、102，余弦相似度分别约为 0.80、0.60。扫描成本约 `O(N × d)`，这里全排序另需 `O(N log N)`；如果只保留 top-k，可用大小为 k 的堆降低排序工作量，ANN 则进一步改变候选搜索方式。

## 5. 近似最近邻（ANN）索引 —— 面试重点

精确最近邻（暴力扫）是 O(N)，数据量大不可行，所以用 ANN 用「一点精度换巨大速度」。

### HNSW（Hierarchical Navigable Small World）—— 最常用

**核心思想：多层跳表 + 小世界图。**

- 把向量建成一张图，每个点连若干「邻居」（邻居的选择基于距离）
- 分层：上层节点少、跳跃大（像高速公路），下层节点多、精确（像街道）
- 检索时从上层快速逼近目标区域，再到下层精细搜索

类比：**HNSW 就像「导航从高速公路下到街道」**——先走高速快速接近目的地，再走小路精确定位。

- 特点：查询快、召回率高，但**内存占用大**（要存整张图）
- 参数：`M`（每个节点连多少邻居）、`ef_search`（搜索时考察的候选数，越大越准越慢）

### IVF（Inverted File Index）

**核心思想：先聚类，再只搜最近的几个簇。**

- 用 KMeans 把所有向量聚成 N 个簇（N 通常取数据量的平方根量级）
- 查询时：先算查询向量离哪个簇中心近，只搜最接近的 `nprobe` 个簇
- 特点：内存省，但召回率依赖簇划分质量（簇中心边缘的点容易漏）

类比：**IVF 就像图书馆分区**——书按主题分到不同书架，找书只翻相关的几个书架，而不是全馆扫描。

### PQ（Product Quantization）—— 压缩

**核心思想：把高维向量切成小段，每段用码本（codebook）里的最近码字代替**，大幅压缩存储。常和 IVF 组合（IVF-PQ）。

- 效果：内存占用能降一个数量级，代价是精度损失
- 类比：有损压缩（类似 JPEG）

### 三者对比（面试常问）

| 维度 | HNSW | IVF | PQ |
|------|------|-----|-----|
| 解决什么 | 检索速度 | 检索速度 | 存储压缩 |
| 查询速度 | 快 | 较快 | 需配合 IVF |
| 内存占用 | 大（存全图） | 中 | 小 |
| 召回率 | 高 | 中 | 有损 |
| 关系 | 独立使用 | 常与 PQ 组合 | 常附在 IVF 上 |

核心 trade-off 一句话：**用内存换速度（HNSW）、用精度换速度（IVF）、用精度换存储（PQ）**。

面试要点：**「ANN 和精确检索的 trade-off？」** → 答：ANN 用可控的召回率损失，把 O(N) 暴力搜索降到近似对数级，核心指标是「召回率（recall）vs 延迟（latency）」，通过调 ef_search/nprobe 等参数在两者间权衡。

## 6. 混合检索（Hybrid Search）：向量 + 关键词

纯向量检索有盲区：对**专有名词、缩写、精确匹配**（如「订单号 ABC-123」「型号 X200」）效果差，因为 embedding 模型没见过这些词。

**混合检索 = 向量检索（语义）+ BM25（关键词）**，两者结果加权融合。

- **BM25**：经典的全文检索打分算法（你熟 Elasticsearch 的话，它就是背后的相关性打分），擅长精确词匹配
  - 为什么擅长精确词：BM25 基于**词频（TF）× 逆文档频率（IDF）**——词在文档里出现越多、且这个词越稀有（区分度高），得分越高。所以它对「精确出现的词」很敏感，但对「同义改写」（换了个词）无能为力，恰好和向量检索互补
- 融合方式：把两个通道的 top-k 结果做 **RRF（Reciprocal Rank Fusion）**——按排名倒数加权合并，简单有效
  - 为什么用「排名倒数」而不是原始分数：向量相似度和 BM25 分数的**量纲、分布都不同，不能直接相加比较**；排名是统一的尺度，倒数加权能让「排在前面的文档」权重更高

```text
RRF_score(doc) = Σ 1 / (k + rank_in_channel)
```

面试要点：**「为什么需要混合检索？单用向量有什么问题？」** → 答：向量擅长语义但弱于精确词匹配（专有名词、缩写、ID），BM25 补上这块，两者互补；融合常用 RRF。

## 7. 面试高频题速查

1. **Embedding 是什么？为什么语义相近向量就近？** → 见第 1 节
2. **余弦相似度和欧氏距离区别？** → 见第 2 节
3. **手写余弦相似度** → 见第 2 节 Go 代码
4. **向量库为什么比暴力扫快？** → ANN 索引，见第 5 节
5. **HNSW 原理？** → 分层小世界图，见第 5 节
6. **IVF 原理？** → 先聚类再只搜近簇，见第 5 节
7. **PQ 是什么？** → 有损压缩，见第 5 节
8. **pgvector 和 Milvus 怎么选？** → 见第 4 节表格
9. **换 embedding 模型要注意什么？** → 维度变化需重算+重建索引
10. **为什么需要混合检索？** → 向量弱于精确词匹配，BM25 互补
11. **Faiss 和 Elasticsearch 怎么存向量、返回原文？** → 见第 4 节问答，区分向量索引、原文与 ID 映射
12. **向量维数相同就能比较吗？** → 还必须来自兼容的模型与编码配置；维数一致是必要条件
13. **Qdrant 和 Faiss / Milvus / Weaviate / pgvector 怎么选？** → 见第 4 节问答：库、PG 扩展、轻中量级服务、分布式基础设施的分层对比
