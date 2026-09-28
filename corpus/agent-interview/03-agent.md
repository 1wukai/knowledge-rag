# Stage 3：Agent

> 目标：理解 Agent 的核心机制（ReAct、Function Calling、Memory），并会用 Go 实现一个最小 Agent。这是和「纯 RAG」拉开差距的地方，也是面试区分度最高的部分。

## 1. Agent 是什么

**Agent = 能自主使用「工具」、规划「多步行动」来完成任务的 LLM 应用。**

和普通 LLM 应用的区别：

| | 普通 LLM | RAG | Agent |
|--|----------|-----|-------|
| 能力 | 一问一答 | 检索+回答 | **用工具 + 多步规划 + 自我修正** |
| 输出 | 文本 | 基于检索的文本 | 行动序列（调用工具/继续思考） |
| 典型场景 | 聊天 | 知识问答 | 查询数据库、执行代码、订机票 |

核心思想：**LLM 是「大脑」，工具是「手脚」**。大脑决定做什么，手脚去执行，执行结果再喂回大脑，循环直到任务完成。

## 2. ReAct：Agent 的核心范式（必考）

**ReAct = Reasoning + Acting**（推理 + 行动）。模型不是一步出答案，而是在「思考→行动→观察」之间循环：

```text
循环：
  思考(Thought)：我该做什么？
  行动(Action)：调用某个工具
  观察(Observation)：工具返回结果
  思考：结果够了吗？够了就回答，不够就继续行动
```

举例（回答「上个月销售额最高的客户是谁」）：

```text
Thought: 需要查数据库，但我不知道表结构，先查表结构
Action:  get_table_schema("orders")
Observation: orders 表有 id, customer_id, amount, created_at 字段
Thought: 用 SQL 查上个月销售额 top1 的客户
Action:  run_sql("SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at >= '2026-08-01' GROUP BY customer_id ORDER BY total DESC LIMIT 1")
Observation: customer_id=1024, total=889900
Thought: 拿到结果了，可以回答了
Answer: 上个月销售额最高的客户是编号 1024，销售额 889900 元
```

### 为什么 ReAct 比「一步到位」更好

- 模型可以先「侦察」（查 schema、查表）再行动，减少瞎猜
- 工具执行失败（如 SQL 报错）可以把错误喂回模型，让它**自我修正**（self-correction）

面试要点：**「Agent 和普通 LLM 调用的本质区别？」** → 答：Agent 引入「工具调用 + 多轮循环 + 观察反馈」，LLM 从「单次生成答案」变成「循环决策的控制器」。

## 3. Function Calling / Tool Use 机制（必考，会写代码）

### 原理

1. 你告诉模型「有哪些工具、每个工具干什么、参数长什么样」（用 JSON Schema 描述）
2. 模型根据用户问题，**不直接回答，而是返回一个结构化的「工具调用」**（工具名 + 参数）
3. **你的代码**去真正执行这个工具
4. 把执行结果作为新消息回传给模型
5. 模型再决定：继续调工具，还是最终回答

**关键认知：模型不执行任何代码/查询，它只「建议」调用什么工具，执行永远由你的代码完成。** 这是安全边界，也是面试常考的认知点。

**为什么参数要用 JSON Schema 描述**：模型要「按这个 schema 生成参数」，后端才能稳定地反序列化和校验，避免模型乱编参数格式导致解析失败。它本质是 Stage 0 提过的「结构化输出」的规范版。

### 数据结构

请求里带上 tools（工具定义）：

```go
type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Parameters  map[string]any `json:"parameters"` // JSON Schema
	} `json:"function"`
}
```

响应里不再是纯文本 content，而是 tool_calls：

```go
type ToolCall struct {
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"` // JSON 字符串，需要解析
	} `json:"function"`
}
```

### Go 实现一个最小 Agent 循环

```go
package main

import (
	"encoding/json"
	"fmt"
)

// 工具注册表：工具名 -> 实际执行函数
type ToolFunc func(args string) (string, error)

var tools = map[string]ToolFunc{
	"get_table_schema": func(args string) (string, error) {
		// 实际查 MySQL 的 information_schema
		return "orders: id, customer_id, amount, created_at", nil
	},
	"run_sql": func(args string) (string, error) {
		// 实际执行 SQL（生产环境要做白名单校验！）
		return "customer_id=1024, total=889900", nil
	},
}

// Agent 循环（核心）
func agentLoop(chatFn func(messages []Message) (*LLMResponse, error), userQuestion string) (string, error) {
	messages := []Message{{Role: "user", Content: userQuestion}}

	for i := 0; i < 10; i++ { // 最多循环 10 轮，防止死循环
		resp, err := chatFn(messages)
		if err != nil {
			return "", err
		}

		// 没有工具调用 = 模型给出最终答案
		if len(resp.ToolCalls) == 0 {
			return resp.Content, nil
		}

		// 把模型的工具调用作为 assistant 消息追加
		messages = append(messages, Message{Role: "assistant", ToolCalls: resp.ToolCalls})

		// 逐个执行工具，结果作为 tool 消息回传
		for _, tc := range resp.ToolCalls {
			fn, ok := tools[tc.Function.Name]
			if !ok {
				return "", fmt.Errorf("unknown tool: %s", tc.Function.Name)
			}
			result, err := fn(tc.Function.Arguments)
			if err != nil {
				result = "ERROR: " + err.Error() // 把错误回传给模型，让它自我修正
			}
			messages = append(messages, Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Content:    result,
			})
		}
	}

	return "", fmt.Errorf("agent exceeded max iterations")
}
```

### 几个工程细节（面试加分）

- **循环上限**：必须有 max iterations，否则模型可能陷入死循环（反复调同一个工具）
- **错误回传**：工具报错要把错误信息回传给模型，让它自我修正，而不是直接失败
- **工具白名单**：像 `run_sql` 这类工具必须校验参数（只读 SQL、防注入），这是安全底线
- **成本控制**：每轮循环都消耗 token，循环次数直接决定成本

## 4. 规划（Planning）—— Agent 的进阶能力

简单 Agent 是「一步一想」，复杂任务需要先「规划再执行」。

- **Plan-and-Execute**：先让模型把大任务拆成步骤清单，再逐步执行
- **反思（Reflection）**：执行完让模型自我检查「做对了吗」，不对就修正
- 这些是 AutoGPT、LangGraph 等框架的核心思想，面试能说清「规划=拆解步骤，反思=自我检查」即可

## 5. Memory（记忆）

Agent 要处理多轮、跨会话的任务，就需要记忆。

| 类型 | 存储 | 用途 | 类比 |
|------|------|------|------|
| **短期记忆** | 上下文窗口（messages 列表） | 当前对话 | 进程内存 |
| **长期记忆** | 向量库 / 结构化存储 | 跨会话的用户偏好、历史 | 数据库 |

- 短期记忆的问题：**上下文窗口有限**，对话长了要「遗忘/摘要压缩」（summarization）
- 长期记忆：本质还是 RAG 那套——把历史交互 embedding 存起来，需要时检索

## 6. 多 Agent 与框架

### 多 Agent 模式

多个 Agent 分工协作，比如「规划者 Agent 拆任务 → 执行者 Agent 干活 → 审查者 Agent 检查」。

- 好处：职责单一、各自 prompt 更聚焦、质量更高
- 代价：通信和协调复杂、token 成本翻倍

### 主流框架

| 框架 | 特点 |
|------|------|
| **LangChain / LangGraph** | 生态最大，LangGraph 用「图」建模 Agent 状态机，适合复杂流程 |
| **AutoGen**（微软） | 多 Agent 对话式协作 |
| **自研（Go）** | 其实 Agent 核心就是「一个 for 循环 + 工具注册表」，简单场景不必上框架 |

面试要点：**「为什么很多团队自己写 Agent 循环而不上 LangChain？」** → 答：Agent 核心逻辑（循环+工具调用）并不复杂，框架带来额外抽象和黑盒；团队要可控性、可调试性、低依赖时倾向自研。能说出这个观点，说明你有工程判断力。

<a id="framework-platform-selection"></a>

### 问答：LangChain、LangGraph、LangSmith、Coze、Dify 详解与区别

先分层，这五个不是同一层面的竞品：**LangChain / LangGraph 是代码框架（SDK），LangSmith 是可观测与评估平台，Coze / Dify 是应用搭建平台（低代码）**。前三个是 LangChain 公司的三条产品线；Coze（扣子）属于字节跳动；Dify 是开源社区产品。

#### LangChain（代码框架）

- **是什么**：起步最早、生态最大的 LLM 应用开发框架（Python / JS）。把模型调用、Prompt 模板、文档加载器、切分器、向量库接入、工具定义抽象成标准组件，用链式表达式（LCEL）把组件串成 pipeline。
- **能做什么**：快速搭 RAG、问答、摘要、翻译等流水线式应用；几百个现成集成（各家模型、向量库、数据源）是它最大的资产。
- **适合**：原型验证、需要大量第三方集成的场景。
- **trade-off**：抽象层多、调用栈深，出问题时定位困难；简单场景下其行为和性能可能不如直接调 SDK。这正是上面「很多团队自研」的原因。

#### LangGraph（编排框架）

- **是什么**：同一家公司的 Agent 工作流框架，用**图 / 状态机**建模：节点是步骤（调模型、执行工具、条件判断），边是转移逻辑，一个共享 state 在图中流转。
- **和 LangChain 的区别**：LangChain 的链适合线性 / DAG 的 pipeline；LangGraph 支持**循环、条件分支、并行、checkpoint 持久化、human-in-the-loop**，面向需要多轮决策、可中断恢复的复杂 Agent。它可以脱离 LangChain 单独使用。
- **适合**：生产级 Agent、多 Agent 协作、需要断点恢复或人工审批介入的流程。
- **后端类比**：相当于给 Agent 循环引入带持久化状态的工作流引擎（Temporal / 状态机的思路）。第 3 节的 `agentLoop` 就是它的最简特例——单节点、固定转移、无持久化。

#### LangSmith（可观测与评估平台）

- **是什么**：LLMOps 平台。trace 记录每次调用的全链路（prompt、模型输出、工具调用、延迟、token 消耗），另提供数据集管理、自动评估（含 LLM-as-judge）、线上监控和 Prompt playground。
- **关键认知**：**它不参与应用运行**，只负责观测和评估；不用 LangChain / LangGraph 也能接入。它就是 `04-engineering.md` 里「评估集 + LLM-as-judge + 可观测」的平台化实现。
- **适合**：调试 Agent / RAG 链路、做回归评估、监控生产质量与成本。
- **trade-off**：商业闭源（SaaS 为主，企业版可自托管），trace 数据出域需要评估合规。

#### Coze / 扣子（低代码 Bot 平台）

- **是什么**：字节跳动的 Agent / Bot 搭建平台。在网页上通过拖拽完成工作流编排、插件（工具）配置、知识库（内置 RAG）、变量与数据库配置，搭好后一键发布到抖音、飞书、微信公众号、客服等渠道。国内版 coze.cn，海外版 coze.com；核心引擎 Coze Studio 已开源。
- **能做什么**：不写代码交付客服 bot、资讯查询 bot、内容生成流程。
- **适合**：运营 / 产品角色快速搭建对话应用；深度绑定字节系渠道和模型生态的场景。
- **trade-off**：托管在平台上，深度定制受限（复杂控制流、私有协议、数据合规都受约束）；模型与渠道选择受平台生态限制。

#### Dify（开源 LLM 应用平台）

- **是什么**：开源、可自托管的 LLM 应用开发平台。一套 Web UI 覆盖三块能力：**画布式工作流编排、内置 RAG pipeline（文档导入 → 切分 → 向量检索）、Agent（工具调用）**，外加多厂商模型管理、API 发布、日志与标注反馈。
- **和 Coze 的区别**：开源可私有化部署（数据不出域）；更面向开发者——搭好的应用直接发布为 API，可嵌入自有系统。
- **和 LangChain 的区别**：不写代码，靠平台抽象完成编排；代价是能力边界由平台决定，复杂逻辑最终还是要落回代码。
- **适合**：中小团队快速搭内部「AI 应用工厂」、对数据合规有自托管要求的场景。

#### 汇总对比

| | LangChain | LangGraph | LangSmith | Coze（扣子） | Dify |
|--|-----------|-----------|-----------|--------------|------|
| 层面 | 代码框架 / SDK | 编排框架（图 / 状态机） | 可观测 + 评估平台 | 低代码托管平台 | 开源可自托管平台 |
| 使用者 | 开发者 | 开发者 | 开发 / 测试 / 运维 | 运营、产品、轻开发 | 中小团队、开发者 |
| 是否写代码 | 是 | 是 | 只接 SDK / API | 否 | 否（产出 API 供集成） |
| 部署形态 | 嵌入你的应用 | 嵌入你的应用 | SaaS（企业版可自托管） | SaaS 托管（Studio 已开源） | 自托管或 SaaS |
| 典型场景 | 快速原型 RAG / 链式应用 | 复杂 Agent、多 Agent、human-in-the-loop | trace、评估集、线上监控 | 面向 C 端渠道的 bot | 企业内部 AI 应用平台 |
| 开源 | 是 | 是 | 否 | Coze Studio 开源 | 是 |

选型一句话：**要代码级掌控力，用 LangChain / LangGraph，配 LangSmith（或自建 trace）做可观测；要不写代码快速交付应用，选 Coze（渠道生态）或 Dify（自托管 + API 化）；分层不同，不是二选一。**

**与本仓库实践的关系**：`interview-rag` 和 `chatbi` 都选择自研核心链路（Python 服务 + 显式的解析 / 检索 / 生成 / 评估编排），没有引入上述框架或平台——目的是让每一步机制可见、可控、可评估，这本身就是可在面试中陈述的选型判断（对应上面「为什么自研 Agent 循环」的论点）。

面试口述：**这五个要分三层讲。LangChain 是组件化 LLM 开发框架，解决「快速接模型、接数据源」；LangGraph 是用图 / 状态机做 Agent 编排的框架，解决「循环、分支、持久化、人工介入」这类复杂控制流，两者代码级互补。LangSmith 是 trace 与评估平台，不参与运行，解决可观测和质量回归。Coze 和 Dify 是低代码应用平台：Coze 托管、绑定字节渠道生态；Dify 开源、可自托管、应用发布为 API。选型时先看团队要代码掌控还是快速交付，再看数据合规是否要求自托管，可观测层无论哪条路线都要补。**

自检问题：

1. LangGraph 相比 LangChain 的链式表达，多了哪些控制流能力？为什么 Agent 场景必须有「循环」？
2. 如果公司要求数据不出域，Coze、Dify、LangSmith 的选择分别受什么影响？
3. 自研 Agent 循环（如第 3 节的实现）相比上框架，放弃了什么、换来了什么？

## 7. Agent 的常见坑

1. **死循环**：模型反复调同一个工具不收敛 → max iterations + 检测重复调用
2. **幻觉工具调用**：调一个不存在的工具/参数错误 → 校验 + 把错误回传
3. **成本失控**：循环 + 长上下文 → 限制轮数、压缩历史
4. **安全风险**：工具执行了危险操作 → 工具白名单、权限校验、只读默认
5. **上下文爆炸**：工具返回结果太长 → 截断/摘要工具输出

## 8. 面试高频题速查

1. **Agent 和普通 LLM/RAG 的区别？** → 工具调用 + 多步循环 + 自我修正
2. **ReAct 是什么？** → Reasoning+Acting，思考-行动-观察循环
3. **Function Calling 流程？模型真的执行工具吗？** → 模型只返回调用指令，代码执行
4. **Agent 死循环怎么办？** → max iterations、检测重复调用
5. **工具报错怎么处理？** → 错误回传给模型自我修正
6. **短期记忆和长期记忆区别？** → 上下文 vs 向量库/存储
7. **上下文窗口满了怎么办？** → 摘要压缩、滑动窗口、长期记忆外置
8. **多 Agent 什么时候用？** → 任务可分解、需要职责隔离时；成本翻倍需权衡
9. **LangChain 该不该用？** → 见第 6 节
10. **Agent 怎么控制成本？** → 限制循环轮数、精简工具输出、压缩历史
11. **LangChain / LangGraph / LangSmith / Coze / Dify 有什么区别？** → 见第 6 节问答：代码框架、编排框架、可观测平台、低代码平台的分层对比
