import type {
  AnswerDetail,
  AnswerSummary,
  ChatResponse,
  Citation,
  Conversation,
  CorpusCandidate,
  CorpusDoc,
  CorpusResponse,
  GenerationList,
  IngestionJob,
  Message,
  ProviderTestResult,
  RuntimeSettings,
  SseEvent,
  SystemStatus,
} from "./types";

function errorMessage(body: { detail?: unknown }, status: number) {
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) return "输入不符合要求，请检查表单内容。";
  return `请求失败（HTTP ${status}）`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(errorMessage(body, res.status));
  return body as T;
}
const json = (method: string, body: unknown): RequestInit => ({
  method,
  body: JSON.stringify(body),
});
const id = encodeURIComponent;

export const api = {
  status: () => request<SystemStatus>("/v1/system/status"),
  getSettings: () => request<RuntimeSettings>("/v1/settings"),
  putSettings: (settings: RuntimeSettings) =>
    request<RuntimeSettings>("/v1/settings", json("PUT", settings)),
  testProviders: () =>
    request<ProviderTestResult>("/v1/providers/test", { method: "POST" }),
  getCorpus: () => request<CorpusResponse>("/v1/corpus"),
  getCandidates: () =>
    request<{ candidates: CorpusCandidate[] }>("/v1/corpus/candidates"),
  addDocuments: (paths: string[]) =>
    request<{ added: string[] }>(
      "/v1/corpus/add",
      json("POST", { doc_ids: paths }),
    ),
  updateDocument: (
    doc: string,
    update: {
      title?: string;
      category?: string;
      enabled?: boolean;
      refresh?: boolean;
    },
  ) =>
    request<CorpusDoc>(
      `/v1/corpus/documents/${id(doc)}`,
      json("PATCH", update),
    ),
  submitIngestion: (doc_ids: string[]) =>
    request<IngestionJob>("/v1/ingestions", json("POST", { doc_ids })),
  listJobs: () => request<{ jobs: IngestionJob[] }>("/v1/ingestions?limit=100"),
  getIngestion: (job: string) =>
    request<IngestionJob>(`/v1/ingestions/${id(job)}`),
  generations: () => request<GenerationList>("/v1/index-generations"),
  activate: (
    generation: string,
    action: "activate" | "rollback",
    active: string | null,
    revision: number,
  ) =>
    request<{ generation_id: string; revision: number }>(
      `/v1/index-generations/${id(generation)}/${action}`,
      json("POST", {
        expected_generation_id: active,
        expected_revision: revision,
      }),
    ),
  cleanup: (retain: number) =>
    request<{ deleted_generations: string[]; deleted_collections: string[] }>(
      `/v1/index-generations/cleanup?retain=${retain}`,
      { method: "POST" },
    ),
  listAnswers: (conversation?: string) =>
    request<{ answers: AnswerSummary[] }>(
      `/v1/answers${conversation ? `?conversation_id=${id(conversation)}` : ""}`,
    ),
  answer: (answer: string) =>
    request<AnswerDetail>(`/v1/answers/${id(answer)}`),
  citation: (answer: string, index: number) =>
    request<Citation>(`/v1/answers/${id(answer)}/citations/${index}`),
  listConversations: () =>
    request<{ conversations: Conversation[] }>("/v1/conversations"),
  createConversation: () =>
    request<Conversation>("/v1/conversations", { method: "POST" }),
  listMessages: (conversation: string) =>
    request<{ messages: Message[] }>(
      `/v1/conversations/${id(conversation)}/messages`,
    ),
  chat: (
    question: string,
    top_k: number,
    min_score: number,
    signal?: AbortSignal,
  ) =>
    request<ChatResponse>("/v1/chat", {
      ...json("POST", { question, top_k, min_score, stream: false }),
      signal,
    }),
  chatStream: async (
    question: string,
    opts: { top_k: number; min_score: number; conversation_id: string },
    onEvent: (e: SseEvent) => void,
    signal?: AbortSignal,
  ) => {
    const res = await fetch("/v1/chat", {
      ...json("POST", { question, stream: true, ...opts }),
      headers: { "Content-Type": "application/json" },
      signal,
    });
    if (!res.ok)
      throw new Error(
        errorMessage(await res.json().catch(() => ({})), res.status),
      );
    if (
      !res.body ||
      !res.headers.get("content-type")?.includes("text/event-stream")
    )
      throw new Error("服务未返回流式响应，请检查启动模式。");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let complete = false;
    try {
      while (!complete) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        let boundary: RegExpExecArray | null;
        while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
          const frame = buffer.slice(0, boundary.index);
          buffer = buffer.slice(boundary.index + boundary[0].length);
          let event = "message";
          const data: string[] = [];
          for (const line of frame.split(/\r?\n/)) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            if (line.startsWith("data:"))
              data.push(line.slice(5).replace(/^ /, ""));
          }
          if (!data.length) continue;
          const payload = JSON.parse(data.join("\n")) as Record<
            string,
            unknown
          >;
          if (event === "error")
            throw new Error(String(payload.message ?? "生成失败"));
          onEvent({ event, data: payload });
          if (event === "done") {
            complete = true;
            break;
          }
        }
        if (done) break;
      }
      if (!complete)
        throw new Error("连接提前结束，答案尚未完成。已保留当前收到的内容。");
    } finally {
      await reader.cancel().catch(() => undefined);
      reader.releaseLock();
    }
  },
};
