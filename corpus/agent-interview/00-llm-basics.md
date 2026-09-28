# Stage 0：LLM 基础

> 目标：理解大语言模型「是什么、怎么生成文本、怎么通过 API 调用」。这是 RAG 和 Agent 的地基，很多面试题（如「什么是幻觉」「temperature 有什么用」）都源于这里。

## 1. 语言模型到底在做什么

**本质：预测下一个 token 的概率分布。**

给你一段文本，模型输出的不是「一句话」，而是「下一个 token 的概率分布」。所谓「生成」，就是反复做这件事：

1. 输入 `"今天天气真"`，模型算出下一个 token 的概率：`好(0.6)、差(0.2)、的(0.1)...`
2. 按一定策略采样出一个 token（比如 `好`），拼到末尾
3. 输入变成 `"今天天气真好"`，再预测下一个 token
4. 循环，直到输出终止符（EOS）或达到 max_tokens

这就是 **自回归生成（autoregressive generation）**。它解释了 LLM 的几个核心特性：

- **生成是逐个 token 串行进行的** → 所以很慢，所以需要流式输出（先出一个字是一个字）
- **一步错，后面全错**（错误会累积到上下文里）→ 这就是为什么「幻觉」很难根治
- **上下文窗口（context window）** 指模型一次能处理的最大 token 数，超了就要截断或报错

### Token 与 Tokenization

模型不直接处理「字」，而是处理 **token**（词元）。tokenization 是把文本切成 token 的过程，主流算法是 **BPE（Byte Pair Encoding）**。

- 为什么用 BPE 而不是「整个词」：语言的词是无穷的（词表会爆炸），且总有没见过的词（未登录词 OOV）。BPE 用「子词」折中——高频词整体一个 token，低频/新词拆成子词组合，词表可控又能覆盖任意文本。

- 一个 token 大约等于：英文 0.75 个词 / 中文 1~2 个字（粗算：1 token ≈ 1.5 汉字）
- 例：`"人工智能"` 可能被切成 `[人工, 智能]` 两个 token，也可能整体一个 token
- **为什么重要**：调用 API 计费按 token 算、上下文窗口按 token 算、切分不好会导致中文「被切碎」影响 embedding 质量（Stage 1 会再讲）

面试常见问题：**「一个 token 等于多少个字？」** → 答：不固定，取决于语言和模型的分词器，粗略估计中文 1~2 字/token，英文约 4 字符/token。

## 2. Transformer 核心思想（面试要能讲框架，不需手写）

Transformer 是几乎所有现代 LLM 的底座。你只需要理解两个关键点：

### (1) 自注意力（Self-Attention）

每个 token 都要「看」其他所有 token，算出「我应该关注谁」。

每个 token 会生成三个向量：
- **Q（Query，查询）**：我在找什么
- **K（Key，键）**：我有什么
- **V（Value，值）**：我的实际内容

注意力分数 = Q 和 K 的点积，归一化（softmax）后作为权重，对 V 加权求和。

```text
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

除以 `√d_k`（维度开根号）是为了防止点积过大导致 softmax 梯度消失——这叫 **缩放点积注意力（scaled dot-product attention）**。

通俗理解：把「注意力」类比成数据库查询——**Query 是你的查询条件，Key 是索引，Value 是要取的数据**。模型用 Q 去匹配 K，按匹配程度取对应的 V。你作为后端，这个类比应该很亲切。

### (2) 位置编码（Positional Encoding）

注意力机制本身**不知道 token 的顺序**（"我爱你" 和 "你爱我" 的注意力矩阵可能一样）。所以需要额外注入位置信息，告诉模型「谁在前谁在后」。

### 为什么有「上下文窗口」限制

自注意力要算**每个 token 和其他所有 token** 之间的关系，复杂度是 **O(n²)**（n = 输入 token 数）。所以输入长度不能无限增长——这既是算力的限制，也是「上下文窗口」存在的根本原因。这也解释了为什么长文档必须切块（Stage 2），以及为什么「超长上下文」是各家模型的卖点。

### 为什么需要多层堆叠 + 多头注意力

- **多头（multi-head）**：多组 QKV，让模型从不同角度（语法、语义、指代）同时关注
- **多层**：底层学语法/局部特征，高层学语义/全局推理

面试要点：能说出「QKV 三个向量、缩放点积、softmax 加权求和、位置编码解决顺序问题」就够了。

## 3. 推理参数（必考，要能说清每个参数的影响）

### temperature（温度）

控制随机性。原理：把 logits（原始分数）除以 T 再 softmax。

- `T → 0`：概率分布变尖锐，几乎总是选最高概率 token → **确定性强，适合代码/翻译/抽取**
- `T → 大`：分布变平滑，低概率 token 也有机会被选 → **创造性高，适合写作/头脑风暴**

```text
P(token_i) = softmax(logits_i / T)
```

### top_p（核采样 nucleus sampling）

只从「累积概率达到 p 的最小 token 集合」里采样。`top_p=0.9` 表示只考虑概率最高的、加起来占 90% 的那些 token。

### top_k

只从概率最高的 k 个 token 里采样。

### max_tokens（输出长度上限）

限制生成的最大 token 数，超过就截断。它是控制成本和时间的重要手段。

### 三者怎么配合

temperature 管「分布形状」，top_p/top_k 管「候选范围」。调参顺序通常是：先定 top_p（比如 0.9），再调 temperature。

面试常见问题：**「temperature=0 是不是就完全确定、没有随机性？」** → 答：理论上分布趋近 one-hot，但受浮点精度和实现细节影响，仍可能有极小随机性，所以工程上要「确定性输出」通常还需固定 seed 或多次采样取多数。

### 补充：什么是「固定 seed」和「多次采样取多数」

**固定 seed（随机种子）**

LLM 采样时也依赖随机数。随机数不是真的「随机」，而是由随机数生成器（PRNG）产生——给它一个初始值（种子 seed），它会产出一串确定的随机数序列。

- 同一个 seed → 同一串随机数 → 同样的采样结果 → 输出可复现
- 不同 seed → 不同随机数序列 → 输出不同

Go 里的对应概念（你应该很熟）：

```go
import "math/rand"

r := rand.New(rand.NewSource(42)) // 固定 seed=42
r.Intn(100) // 每次运行都得到相同的序列
```

LLM API 通常不直接暴露 seed 参数（部分模型支持），更多用于本地推理（如 vLLM、transformers 库）做可复现实验。工程上「确定性输出」更常见的做法是：**temperature=0 + 贪婪解码（top_k=1）**，从源头消除采样随机性。

**多次采样取多数（self-consistency，自洽性）**

当任务需要一定创造性、不得不采样时，单次采样可能出错。做法是：让模型对同一问题**独立采样多次**（如 5 次），统计各答案出现次数，取出现最多的作为最终答案。

- 类比：分布式系统里的「多数派投票」（quorum）——单个节点可能出错，多数一致才可信
- 为什么有效：模型出错是随机的、分散的，而正确答案往往在多次采样中重复出现
- 代价：调用次数 ×N，成本和时间翻倍，只在「可靠性要求高、且值得花钱」的场景用

一句话总结：**固定 seed 是「让随机变可复现」，多次采样取多数是「用多次随机投票抵消单次误差」**——前者追求确定性，后者在保留随机性的前提下追求稳定性。

## 4. 幻觉（Hallucination）是什么、为什么产生

幻觉 = 模型一本正经地输出错误或编造的内容。

**根本原因**：
1. **本质是概率采样**：模型是在「生成最像正确答案的内容」，而不是「查证正确答案」
2. **知识有截止日期**：训练数据是静态快照，问它训练后发生的事它只能编
3. **没有事实约束**：裸模型没有数据库/检索，只能靠参数记忆，记忆模糊就「合理脑补」

**缓解手段**（这是 RAG 存在的意义，Stage 2 会展开）：
- RAG：把外部知识检索出来喂给模型
- Function Calling / Tool Use：让模型去查真实数据而不是凭记忆
- 降低 temperature、要求模型「不知道就说不知道」

## 5. Prompt 工程基础

### system prompt vs user prompt

- **system**：设定角色、边界、行为规范（如「你是一个严谨的 SQL 专家，只输出 SQL」）
- **user**：具体任务/问题

### few-shot（少样本 / in-context learning）

在 prompt 里给几个「输入→输出」示例，让模型照着格式学。**不需要改模型权重**，所以叫 in-context learning（上下文学习）。

### Chain-of-Thought（CoT，思维链）

让模型「一步步思考」再给答案，显著提升复杂推理题的正确率。最简单做法是加一句 `Let's think step by step`，或给 few-shot 示例时把推理过程也写出来。

### 结构化输出

让模型输出 JSON，便于后端解析。常用手段：明确要求 JSON 格式 + few-shot 给 JSON 示例（Function Calling 就是结构化输出的规范版，Stage 3 细讲）。

## 6. LLM API 调用范式（Go 代码）

### (1) 普通 chat completions

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

func chat(client *http.Client, apiKey, prompt string) (string, error) {
	reqBody := ChatRequest{
		Model: "gpt-4o-mini",
		Messages: []Message{
			{Role: "system", Content: "你是一个严谨的助手。"},
			{Role: "user", Content: prompt},
		},
		Temperature: 0.3,
	}
	body, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest("POST",
		"https://api.openai.com/v1/chat/completions",
		bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Choices []struct {
			Message Message `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}
	return result.Choices[0].Message.Content, nil
}
```

### (2) 流式输出（SSE）

LLM 生成慢，用户体验上必须「边生成边返回」。底层就是 **SSE（Server-Sent Events）**：服务端持续推 `data: {...}\n\n` 数据块，客户端逐行读。每块里 `delta.content` 是增量 token。

```go
import (
	"bufio"
	"strings"
)

// 流式调用：逐个 token 打印
func streamChat(client *http.Client, apiKey, prompt string) error {
	reqBody := map[string]any{
		"model":    "gpt-4o-mini",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": prompt}},
	}
	body, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest("POST",
		"https://api.openai.com/v1/chat/completions",
		strings.NewReader(string(body)))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			fmt.Print(chunk.Choices[0].Delta.Content)
		}
	}
	return scanner.Err()
}
```

面试要点：**「为什么要用 SSE 而不是 WebSocket 或普通 HTTP 长连接？」** → 答：LLM 生成是单向的、服务端到客户端的持续推送，SSE 足够简单（HTTP 之上，自动重连、天然走网关），WebSocket 是双向全双工，这里用不上、成本更高。

### (3) Function Calling（先建立概念，Stage 3 深入）

在请求里告诉模型「有哪些工具可用、参数 schema 是什么」，模型不在文本里直接回答，而是返回一个**结构化的工具调用指令**（函数名 + 参数 JSON），由你的代码去执行、再把结果回传给模型。

```go
// 请求里多传一个 tools 字段
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"` // JSON Schema
}

// 响应里多一个 tool_calls 字段，而不是纯文本 content
```

## 7. 面试高频题速查

1. **LLM 生成文本的基本原理？** → 自回归预测下一个 token，循环采样直到 EOS
2. **什么是 token？为什么重要？** → 分词最小单位，决定计费/上下文/速度
3. **Transformer 里 Q、K、V 是什么？** → 查询/键/值，注意力=softmax(QKᵀ/√d)V
4. **为什么除以 √d_k？** → 防止点积过大导致 softmax 梯度消失
5. **位置编码解决什么问题？** → 注意力机制本身不感知顺序
6. **temperature / top_p / top_k 的区别？** → 见第 3 节
7. **什么是幻觉？为什么产生？怎么缓解？** → 见第 4 节
8. **为什么 LLM 生成慢？** → 自回归串行，一个 token 一个 token 生成
9. **SSE 和 WebSocket 选哪个？为什么？** → 单向推送用 SSE 更简单
10. **few-shot 和微调（fine-tuning）的区别？** → few-shot 不改变权重，靠上下文；微调更新模型权重
