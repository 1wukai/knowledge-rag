import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api } from "./api";
import type {
  AnswerDetail,
  Citation,
  Conversation,
  CorpusCandidate,
  CorpusDoc,
  CorpusResponse,
  Generation,
  IngestionJob,
  ProviderConfig,
  ProviderTestResult,
  RuntimeSettings,
} from "./types";

type Page = "ask" | "corpus" | "jobs" | "indexes" | "answers" | "settings";
const pages: {
  id: Page;
  label: string;
  title: string;
  subtitle: string;
  icon: string;
}[] = [
  {
    id: "ask",
    label: "知识问答",
    title: "让知识，成为你的答案。",
    subtitle: "从自己的笔记出发，沿着可靠的来源深入思考。",
    icon: "chat",
  },
  {
    id: "corpus",
    label: "知识来源",
    title: "整理每一份知识。",
    subtitle: "纳入本地笔记、管理分类，让值得保存的内容进入知识库。",
    icon: "book",
  },
  {
    id: "jobs",
    label: "入库任务",
    title: "看见知识入库的过程。",
    subtitle: "跟踪构建状态，查看处理结果与失败原因。",
    icon: "activity",
  },
  {
    id: "indexes",
    label: "索引版本",
    title: "每一次更新，都有迹可循。",
    subtitle: "查看已发布版本，按需切换、回滚和清理。",
    icon: "layers",
  },
  {
    id: "answers",
    label: "答案记录",
    title: "回到答案的出处。",
    subtitle: "查阅历史回答、使用的索引版本和保存的引用摘录。",
    icon: "archive",
  },
  {
    id: "settings",
    label: "模型设置",
    title: "配置你的思考伙伴。",
    subtitle: "为检索、回答和重排选择合适的模型。",
    icon: "settings",
  },
];
const terminal = new Set(["succeeded", "failed", "cancelled"]);
const statusLabels: Record<string, string> = {
  queued: "等待处理",
  running: "正在构建",
  validating: "校验中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  ready: "可发布",
  active: "当前使用",
  building: "构建中",
  completed: "已完成",
  refused: "证据不足",
  retry_wait: "等待重试",
};
const short = (value?: string | null) => (value ? value.slice(0, 10) : "—");
const date = (value?: string | null) =>
  value
    ? new Date(
        /[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`,
      ).toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      })
    : "—";
const message = (e: unknown) =>
  e instanceof Error ? e.message : "操作失败，请重试。";

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, ReactNode> = {
    chat: (
      <>
        <path d="M20 11a8 8 0 0 1-8 8H5l-4 3 2-7a8 8 0 1 1 17-4Z" />
        <path d="M7 9h8M7 13h5" />
      </>
    ),
    book: (
      <>
        <path d="M12 5v16M3 3c4-1 6 0 9 2 3-2 5-3 9-2v16c-4-1-6 0-9 2-3-2-5-3-9-2Z" />
      </>
    ),
    activity: (
      <>
        <path d="M3 12h4l3-8 4 16 3-8h4" />
      </>
    ),
    layers: (
      <>
        <path d="m12 3 10 5-10 5L2 8Zm-9 10 9 5 9-5M3 18l9 5 9-5" />
      </>
    ),
    archive: (
      <>
        <rect x="4" y="5" width="16" height="16" rx="2" />
        <path d="M8 2h8M8 10h8M8 14h8M8 18h4" />
      </>
    ),
    settings: (
      <>
        <path d="M4 7h16M4 17h16" />
        <circle cx="9" cy="7" r="3" />
        <circle cx="16" cy="17" r="3" />
      </>
    ),
    arrow: (
      <>
        <path d="m12 5 7 7-7 7M5 12h14" />
      </>
    ),
    plus: <path d="M12 5v14M5 12h14" />,
    refresh: (
      <>
        <path d="M20 7V2l-4 4a8 8 0 1 0 4 10M20 7h-5" />
      </>
    ),
    sun: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1 1M18 18l1 1M5 19l1-1M18 6l1-1" />
      </>
    ),
    close: <path d="m6 6 12 12M6 18 18 6" />,
    search: (
      <>
        <circle cx="10" cy="10" r="6" />
        <path d="m15 15 6 6" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] ?? paths.book}
    </svg>
  );
}
function Notice({
  children,
  error = false,
}: {
  children?: ReactNode;
  error?: boolean;
}) {
  return children ? (
    <div
      className={`notice ${error ? "error" : ""}`}
      role={error ? "alert" : "status"}
    >
      {children}
    </div>
  ) : null;
}
function Badge({ status }: { status: string }) {
  return (
    <span className={`badge ${status}`}>{statusLabels[status] ?? status}</span>
  );
}
function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <Icon name="book" size={32} />
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
function ModeHint() {
  return (
    <Empty title="在完整服务模式下使用">
      停止当前进程后运行 <code>make serve</code>
      ，即可使用会话、任务、索引版本与答案记录。
    </Empty>
  );
}
function useResource<T>(loader: () => Promise<T>, interval = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const sequence = useRef(0);
  const load = useCallback(async () => {
    const current = ++sequence.current;
    setLoading(true);
    try {
      const value = await loader();
      if (current === sequence.current) {
        setData(value);
        setError("");
      }
    } catch (e) {
      if (current === sequence.current) setError(message(e));
    } finally {
      if (current === sequence.current) setLoading(false);
    }
  }, [loader]);
  useEffect(() => {
    void load();
    return () => {
      sequence.current++;
    };
  }, [load]);
  useEffect(() => {
    if (!interval) return;
    const timer = window.setInterval(() => void load(), interval);
    return () => clearInterval(timer);
  }, [load, interval]);
  return { data, error, loading, load };
}
function Modal({
  title,
  children,
  onClose,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  return (
    <dialog
      ref={ref}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      aria-label={title}
    >
      <div className="dialog-head">
        <h2>{title}</h2>
        <button className="icon-button" onClick={onClose} aria-label="关闭">
          <Icon name="close" />
        </button>
      </div>
      {children}
    </dialog>
  );
}
function Citations({ citations }: { citations: Citation[] }) {
  return (
    <div className="citations">
      {citations.map((c) => (
        <details className="citation" key={c.index}>
          <summary>
            <span className="citation-num">
              {String(c.index).padStart(2, "0")}
            </span>
            <span>
              {c.title}
              <small>{c.heading.join(" / ")}</small>
            </span>
            <span className="citation-lines">
              行 {c.start_line}–{c.end_line}
            </span>
          </summary>
          <p className="preserve">{c.snippet}</p>
          <small>
            已保存的引用摘录 · <code>{c.source_hash}</code>
          </small>
        </details>
      ))}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("ask");
  const [traceId, setTraceId] = useState<string | null>(null);
  const [chatBusy, setChatBusy] = useState(false);
  const [theme, setTheme] = useState(
    () => localStorage.getItem("ir-theme") ?? "light",
  );
  const system = useResource(api.status);
  const corpus = useResource(api.getCorpus);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ir-theme", theme);
  }, [theme]);
  const selected = pages.find((p) => p.id === page)!;
  const server = system.data?.mode === "server";
  function go(next: Page) {
    setPage(next);
    if (next !== "answers") setTraceId(null);
    void corpus.load();
  }
  function trace(id: string) {
    setTraceId(id);
    setPage("answers");
  }
  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">
        跳到主要内容
      </a>
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            if (!chatBusy) go("ask");
          }}
        >
          <span className="brand-mark">
            <Icon name="book" size={25} />
          </span>
          <span>
            知识书房<small>KNOWLEDGE DESK</small>
          </span>
        </a>
        <div className="nav-caption">
          我的工作空间 <span>01</span>
        </div>
        <nav aria-label="主导航">
          {pages.map((p) => (
            <button
              key={p.id}
              onClick={() => go(p.id)}
              disabled={chatBusy && page !== p.id}
              aria-current={page === p.id ? "page" : undefined}
              className={page === p.id ? "selected" : ""}
            >
              <Icon name={p.icon} />
              {p.label}
              {page === p.id && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="eyebrow">YOUR NOTES, CONNECTED.</span>
          <p>
            把零散的记录，
            <br />
            读成自己的知识。
          </p>
          <div className="fine-line" />
          <small>本地知识库 · 个人工作空间</small>
        </div>
        <div className="sidebar-footer">
          <span>
            <i className={`dot ${system.data ? "online" : ""}`} />
            {system.data
              ? server
                ? "完整服务模式"
                : "轻量调试模式"
              : "连接中"}
          </span>
          <button
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
            className="theme-button"
            aria-label="切换明暗主题"
          >
            <Icon name="sun" size={18} />
          </button>
        </div>
      </aside>
      <main id="main" className="workspace">
        <header className="topbar">
          <div>
            工作空间 <span>/</span> <strong>{selected.label}</strong>
          </div>
          <div className="topbar-status">
            <span className="dot online" />
            {corpus.data?.index
              ? `${corpus.data.index.documents.length} 篇已入库 · ${corpus.data.index.chunk_count} 个片段`
              : "等待知识入库"}
            <button
              className="icon-button"
              onClick={() => {
                void system.load();
                void corpus.load();
              }}
              aria-label="刷新服务状态"
            >
              <Icon name="refresh" size={16} />
            </button>
          </div>
        </header>
        <div className={`page-content ${page === "ask" ? "chat-page" : ""}`}>
          <div className="page-heading">
            <div>
              <span className="eyebrow">
                {String(pages.indexOf(selected) + 1).padStart(2, "0")} /{" "}
                {page === "ask" ? "ASK YOUR KNOWLEDGE" : "KNOWLEDGE WORKSPACE"}
              </span>
              <h1>{selected.title}</h1>
              <p>{selected.subtitle}</p>
            </div>
            <span className="heading-seal" aria-hidden="true">
              知<br />行
            </span>
          </div>
          <Notice error>{system.error || corpus.error}</Notice>
          {system.error && (
            <button className="button" onClick={() => void system.load()}>
              重新连接
            </button>
          )}
          {system.data && (
            <>
              <div hidden={page !== "ask"}>
                <AskView
                  server={server}
                  corpus={corpus.data}
                  onTrace={trace}
                  onBusy={setChatBusy}
                  onSources={() => go("corpus")}
                />
              </div>
              {page === "corpus" && (
                <CorpusView
                  corpus={corpus.data}
                  refresh={corpus.load}
                  onJobs={() => go("jobs")}
                  server={server}
                />
              )}
              {page === "jobs" && (server ? <JobsView /> : <ModeHint />)}
              {page === "indexes" &&
                (server ? (
                  <IndexesView onChange={corpus.load} />
                ) : (
                  <ModeHint />
                ))}
              {page === "answers" &&
                (server ? <AnswersView answerId={traceId} /> : <ModeHint />)}
              {page === "settings" && <SettingsView />}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

interface Turn {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  answerId?: string;
  refused?: boolean;
  pending?: boolean;
  error?: string;
  saved?: boolean;
}
function AskView({
  server,
  corpus,
  onTrace,
  onBusy,
  onSources,
}: {
  server: boolean;
  corpus: CorpusResponse | null;
  onTrace: (id: string) => void;
  onBusy: (busy: boolean) => void;
  onSources: () => void;
}) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [threshold, setThreshold] = useState(0.4);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  const requestId = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const refresh = useCallback(async () => {
    if (server) setConversations((await api.listConversations()).conversations);
  }, [server]);
  useEffect(() => {
    void refresh().catch((e) => setError(message(e)));
  }, [refresh]);
  useEffect(
    () => () => {
      requestId.current++;
      controller.current?.abort();
    },
    [],
  );
  useEffect(() => {
    onBusy(busy);
  }, [busy, onBusy]);
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "nearest" });
  }, [turns]);
  async function open(id: string) {
    if (busy) return;
    const current = ++requestId.current;
    setBusy(true);
    setError("");
    try {
      if (!id) {
        setConversationId("");
        setTurns([]);
        return;
      }
      const [history, records] = await Promise.all([
        api.listMessages(id),
        api.listAnswers(id),
      ]);
      const answers = await Promise.all(
        records.answers.map((a) => api.answer(a.answer_id)),
      );
      if (requestId.current !== current) return;
      const restored = history.messages.map((m, i): Turn => {
        const matches =
          m.role === "assistant"
            ? answers.filter(
                (a) =>
                  a.question === history.messages[i - 1]?.content &&
                  (a.answer ?? "") === m.content,
              )
            : [];
        const answer = matches.length === 1 ? matches[0] : undefined;
        return {
          id: `${id}-${i}`,
          role: m.role as Turn["role"],
          content: m.content,
          citations: answer?.citations,
          refused: answer?.refused,
          answerId: answer?.answer_id,
          saved: Boolean(answer),
        };
      });
      setConversationId(id);
      setTurns(restored);
    } catch (e) {
      setError(message(e));
    } finally {
      if (current === requestId.current) setBusy(false);
    }
  }
  async function ask() {
    if (!question.trim() || controller.current || busy) return;
    const q = question.trim();
    const turnId = crypto.randomUUID();
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError("");
    setQuestion("");
    setTurns((prev) => [
      ...prev,
      { id: `${turnId}-q`, role: "user", content: q },
      { id: turnId, role: "assistant", content: "", pending: true },
    ]);
    const update = (change: Partial<Turn> | ((t: Turn) => Partial<Turn>)) =>
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId
            ? { ...t, ...(typeof change === "function" ? change(t) : change) }
            : t,
        ),
      );
    try {
      if (server) {
        let id = conversationId;
        if (!id) {
          id = (await api.createConversation()).conversation_id;
          setConversationId(id);
        }
        if (abort.signal.aborted)
          throw new DOMException("Stopped", "AbortError");
        await api.chatStream(
          q,
          { conversation_id: id, top_k: topK, min_score: threshold },
          (e) => {
            if (e.event === "meta")
              update({
                answerId: String(e.data.answer_id),
                refused: Boolean(e.data.refused),
              });
            if (e.event === "delta")
              update((t) => ({
                content: t.content + String(e.data.content ?? ""),
              }));
            if (e.event === "citations")
              update({ citations: e.data.citations as Citation[] });
            if (e.event === "done")
              update({
                pending: false,
                saved: true,
                refused: Boolean(e.data.refused),
              });
          },
          abort.signal,
        );
      } else {
        const response = await api.chat(q, topK, threshold, abort.signal);
        update({
          content: response.answer ?? "",
          citations: response.citations,
          refused: response.refused,
          pending: false,
        });
      }
    } catch (e) {
      update({
        pending: false,
        error: abort.signal.aborted
          ? "已停止接收，当前内容可能不完整，尚未确认保存。"
          : message(e),
      });
    } finally {
      controller.current = null;
      setBusy(false);
      void refresh().catch((e) => setError(message(e)));
      input.current?.focus();
    }
  }
  return (
    <div className="chat-layout">
      <section className="chat-card">
        <div className="chat-toolbar">
          <div className="session-select">
            <Icon name="chat" size={18} />
            <select
              aria-label="选择会话"
              value={conversationId}
              disabled={busy || !server}
              onChange={(e) => void open(e.target.value)}
            >
              <option value="">
                {server ? "新的思考，从这里开始" : "单轮问答 · local"}
              </option>
              {conversations.map((c, i) => (
                <option key={c.conversation_id} value={c.conversation_id}>
                  {c.title === "新会话"
                    ? `会话 ${conversations.length - i} · ${date(c.created_at)}`
                    : c.title}
                </option>
              ))}
            </select>
          </div>
          <button
            className="text-button"
            onClick={() => void open("")}
            disabled={busy}
          >
            <Icon name="plus" size={16} />
            新对话
          </button>
        </div>
        <Notice error>{error}</Notice>
        <div className="chat-scroll" aria-label="对话内容">
          {!turns.length ? (
            <div className="chat-welcome">
              <div className="book-emblem">
                <Icon name="book" size={40} />
                <span className="emblem-orbit" />
              </div>
              <span className="eyebrow">
                A LITTLE CURIOSITY GOES A LONG WAY
              </span>
              <h2>今天，想弄懂什么？</h2>
              <p>
                一个概念，一次追问，一条新的思路。
                <br />
                从你的知识库中找到有出处的回答。
              </p>
              <div className="suggestions">
                {[
                  "RAG 和微调，应该如何选择？",
                  "Embedding 如何衡量语义相似度？",
                  "怎样设计一个可靠的 AI Agent？",
                ].map((q, i) => (
                  <button
                    key={q}
                    onClick={() => {
                      setQuestion(q);
                      input.current?.focus();
                    }}
                  >
                    <span>0{i + 1}</span>
                    {q}
                    <Icon name="arrow" size={16} />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            turns.map((t) => (
              <article className={`turn ${t.role}`} key={t.id}>
                <div className="turn-label">
                  {t.role === "user" ? (
                    "你"
                  ) : (
                    <>
                      <Icon name="book" size={16} />
                      书房助手
                      {t.pending && <span className="thinking">正在思考</span>}
                    </>
                  )}
                </div>
                <div className="turn-content preserve">
                  {t.refused
                    ? "当前知识库的证据不足以回答这个问题。可以补充资料，或换一个更具体的问题。"
                    : t.content || (t.pending ? "正在检索相关资料…" : "")}
                </div>
                <Notice error>{t.error}</Notice>
                {t.citations && <Citations citations={t.citations} />}
                <div className="turn-actions">
                  {t.saved && t.answerId && (
                    <button
                      className="text-button"
                      onClick={() => onTrace(t.answerId!)}
                      disabled={busy}
                    >
                      查看答案记录 <Icon name="arrow" size={14} />
                    </button>
                  )}
                </div>
              </article>
            ))
          )}
          <div ref={bottom} />
        </div>
        <div className="composer">
          <label className="sr-only" htmlFor="question">
            输入问题
          </label>
          <textarea
            id="question"
            ref={input}
            placeholder="向你的知识库提问，或继续追问…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                void ask();
              }
            }}
          />
          <div className="composer-footer">
            <span>
              Enter 发送 <span className="separator">·</span> Shift + Enter 换行
            </span>
            {busy ? (
              <button
                className="button"
                onClick={() => controller.current?.abort()}
              >
                <Icon name="stop" size={16} />
                停止接收
              </button>
            ) : (
              <button
                className="button primary"
                disabled={!question.trim()}
                onClick={() => void ask()}
              >
                发送问题
                <Icon name="arrow" size={17} />
              </button>
            )}
          </div>
        </div>
        <details className="query-options">
          <summary>检索设置</summary>
          <div className="form-row">
            <label>
              证据条数
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) =>
                  setTopK(Math.min(20, Math.max(1, Number(e.target.value))))
                }
              />
            </label>
            <label>
              最低相关度
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={threshold}
                onChange={(e) =>
                  setThreshold(Math.min(1, Math.max(0, Number(e.target.value))))
                }
              />
            </label>
          </div>
          <small>阈值与模型有关；降低阈值可能引入不相关证据。</small>
        </details>
      </section>
      <aside className="knowledge-aside">
        <div className="aside-heading">
          <Icon name="layers" size={16} />
          <span>本次知识范围</span>
        </div>
        <h3>
          你的笔记，
          <br />
          是回答的起点。
        </h3>
        <p>回答来自当前发布的知识索引。引用可展开，便于回看原始章节。</p>
        <div className="source-count">
          <strong>{corpus?.index?.documents.length ?? 0}</strong>
          <span>篇已入库笔记</span>
        </div>
        <ul>
          {(corpus?.index?.documents ?? []).slice(0, 5).map((d) => (
            <li key={d.doc_id}>
              <Icon name="book" size={14} />
              {d.title}
            </li>
          ))}
        </ul>
        <button className="text-button" onClick={onSources} disabled={busy}>
          管理知识来源
          <Icon name="arrow" size={15} />
        </button>
        <div className="aside-tip">
          <span className="eyebrow">READ · ASK · CONNECT</span>
          <p>
            {server
              ? "可以新建会话连续追问。重要结论请结合引用核对。"
              : "当前为单轮调试模式。使用 make serve 开启持久化多轮与流式输出。"}
          </p>
        </div>
      </aside>
    </div>
  );
}

function CorpusView({
  corpus,
  refresh,
  onJobs,
  server,
}: {
  corpus: CorpusResponse | null;
  refresh: () => Promise<void>;
  onJobs: () => void;
  server: boolean;
}) {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("");
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [candidates, setCandidates] = useState<CorpusCandidate[] | null>(null);
  const [candidatePaths, setCandidatePaths] = useState<Set<string>>(new Set());
  const [editing, setEditing] = useState<CorpusDoc | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const docs = corpus?.documents ?? [];
  useEffect(() => {
    setChecked(
      (prev) =>
        new Set(
          docs
            .filter((d) => d.enabled && (prev.has(d.doc_id) || d.indexed))
            .map((d) => d.doc_id),
        ),
    );
  }, [corpus]);
  const filtered = docs.filter(
    (d) =>
      (!category || d.category === category) &&
      `${d.title} ${d.path}`.toLowerCase().includes(search.toLowerCase()),
  );
  const categories = [...new Set(docs.map((d) => d.category).filter(Boolean))];
  function toggle(value: string, setter: typeof setChecked) {
    setter((prev) => {
      const next = new Set(prev);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return next;
    });
  }
  async function perform(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  async function submit() {
    const removed = docs.filter((d) => d.indexed && !checked.has(d.doc_id));
    if (
      removed.length &&
      !window.confirm(
        `本次索引将移除 ${removed.length} 篇未勾选笔记，源文件保持不变。确认提交？`,
      )
    )
      return;
    await perform(async () => {
      const job = await api.submitIngestion([...checked]);
      if (server && job.status !== "succeeded") {
        onJobs();
      } else {
        setNotice(`入库完成，共 ${job.chunk_count ?? "—"} 个片段。`);
        await refresh();
      }
    });
  }
  return (
    <>
      <div className="metric-strip">
        <div>
          <strong>{docs.length}</strong>
          <span>篇知识来源</span>
        </div>
        <div>
          <strong>{categories.length}</strong>
          <span>个分类</span>
        </div>
        <div>
          <strong>{corpus?.index?.chunk_count ?? 0}</strong>
          <span>个已索引片段</span>
        </div>
        <div className="metric-action">
          <button
            className="button"
            disabled={busy}
            onClick={() =>
              void perform(async () => {
                setCandidates((await api.getCandidates()).candidates);
                setCandidatePaths(new Set());
              })
            }
          >
            <Icon name="plus" size={17} />
            发现本地文档
          </button>
        </div>
      </div>
      <section className="panel">
        <div className="panel-toolbar">
          <div className="search-field">
            <Icon name="search" size={17} />
            <input
              aria-label="搜索知识来源"
              placeholder="搜索标题或路径"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <select
            aria-label="筛选分类"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
          >
            <option value="">全部分类</option>
            {categories.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <button
            className="button primary"
            disabled={busy || !checked.size}
            onClick={() => void submit()}
          >
            入库所选 · {checked.size}
          </button>
        </div>
        <Notice error>{error}</Notice>
        <Notice>{notice}</Notice>
        <div className="selection-bar">
          <label>
            <input
              type="checkbox"
              aria-label="全选当前列表"
              checked={
                filtered.some((d) => d.enabled) &&
                filtered
                  .filter((d) => d.enabled)
                  .every((d) => checked.has(d.doc_id))
              }
              onChange={(e) =>
                setChecked((prev) => {
                  const next = new Set(prev);
                  filtered
                    .filter((d) => d.enabled)
                    .forEach((d) =>
                      e.target.checked
                        ? next.add(d.doc_id)
                        : next.delete(d.doc_id),
                    );
                  return next;
                })
              }
            />{" "}
            当前列表
          </label>
          <span>共选中 {checked.size} 篇（包含筛选外的选择）</span>
        </div>
        {!filtered.length ? (
          <Empty title="没有匹配的文档">调整搜索条件，或发现本地文档。</Empty>
        ) : (
          <div className="document-list">
            {filtered.map((d, i) => (
              <div
                className={`document-row ${checked.has(d.doc_id) ? "checked" : ""}`}
                key={d.doc_id}
              >
                <input
                  type="checkbox"
                  aria-label={`选择 ${d.title}`}
                  checked={checked.has(d.doc_id)}
                  disabled={!d.enabled || busy}
                  onChange={() => toggle(d.doc_id, setChecked)}
                />
                <span className="file-mark">{d.path.split(".").pop()?.toUpperCase() || "文档"}</span>
                <div className="document-info">
                  <h3>{d.title}</h3>
                  <code>{d.path}</code>
                  <div className="document-meta">
                    <span>{d.category || "未分类"}</span>
                    <span>{d.enabled ? "已启用" : "已停用"}</span>
                    {d.indexed && <Badge status="active" />}
                  </div>
                </div>
                <span className="row-number">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <button
                  className="text-button"
                  disabled={busy}
                  onClick={() => setEditing({ ...d })}
                >
                  编辑
                </button>
                <button
                  className="text-button"
                  disabled={busy}
                  onClick={() =>
                    void perform(async () => {
                      await api.updateDocument(d.doc_id, { refresh: true });
                      await refresh();
                      setNotice("内容摘要已刷新，重新入库后更新检索内容。");
                    })
                  }
                >
                  刷新摘要
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="panel-footnote">
          修改标题、分类、启用状态或源文件后，需要重新入库。分类筛选用于整理来源，当前问答检索整个已发布知识库。
        </div>
      </section>
      {editing && (
        <Modal
          title="编辑知识来源"
          onClose={() => {
            if (!busy) setEditing(null);
          }}
        >
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void perform(async () => {
                await api.updateDocument(editing.doc_id, {
                  title: editing.title,
                  category: editing.category,
                  enabled: editing.enabled,
                });
                await refresh();
                setEditing(null);
                setNotice("文档设置已保存，重新入库后生效。");
              });
            }}
          >
            <Notice error>{error}</Notice>
            <label className="field">
              标题
              <input
                required
                maxLength={256}
                value={editing.title}
                onChange={(e) =>
                  setEditing({ ...editing, title: e.target.value })
                }
              />
            </label>
            <label className="field">
              分类
              <input
                maxLength={64}
                value={editing.category}
                placeholder="例如：后端开发"
                onChange={(e) =>
                  setEditing({ ...editing, category: e.target.value })
                }
              />
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={editing.enabled}
                onChange={(e) =>
                  setEditing({ ...editing, enabled: e.target.checked })
                }
              />
              启用此文档
            </label>
            <p className="muted">
              停用后仍保留源文件和历史答案，新索引将不再包含它。
            </p>
            <div className="dialog-actions">
              <button
                type="button"
                className="button"
                disabled={busy}
                onClick={() => setEditing(null)}
              >
                取消
              </button>
              <button className="button primary" disabled={busy}>
                保存设置
              </button>
            </div>
          </form>
        </Modal>
      )}
      {candidates && (
        <Modal
          title="发现本地文档"
          onClose={() => {
            if (!busy) setCandidates(null);
          }}
        >
          <p className="muted">
            支持 MD、TXT、PDF、Word（DOCX / DOC），单文件最大 5 MiB。PDF 需要文字层，扫描件请先 OCR；旧 DOC 需要本机转换器。仅纳入你勾选的文件。
          </p>
          <Notice error>{error}</Notice>
          {!candidates.length ? (
            <Empty title="暂未发现新文档">
              将 .md、.txt、.pdf、.docx 或 .doc 文件放到 notes 下的知识目录，再重新扫描。
            </Empty>
          ) : (
            <div className="candidate-list">
              {candidates.map((c) => (
                <label key={c.path}>
                  <input
                    type="checkbox"
                    checked={candidatePaths.has(c.path)}
                    disabled={busy || Boolean(c.error)}
                    onChange={() => toggle(c.path, setCandidatePaths)}
                  />
                  <span>
                    {c.title}
                    <small>
                      {c.format ?? c.path.split(".").pop()?.toUpperCase()} · {c.path} · {c.category}
                      {c.size_bytes !== undefined && ` · ${(c.size_bytes / 1024).toFixed(1)} KiB`}
                    </small>
                    {c.error && <small role="alert">无法纳入：{c.error}</small>}
                  </span>
                </label>
              ))}
            </div>
          )}
          <div className="dialog-actions">
            <button
              className="button"
              disabled={busy}
              onClick={() => setCandidates(null)}
            >
              取消
            </button>
            <button
              className="button primary"
              disabled={busy || !candidatePaths.size}
              onClick={() =>
                void perform(async () => {
                  const result = await api.addDocuments([...candidatePaths]);
                  await refresh();
                  setCandidates(null);
                  setNotice(
                    `已纳入 ${result.added.length} 篇。请勾选后提交入库。`,
                  );
                })
              }
            >
              纳入所选 · {candidatePaths.size}
            </button>
          </div>
        </Modal>
      )}
    </>
  );
}

function JobsView() {
  const jobs = useResource(api.listJobs, 4000);
  const [selected, setSelected] = useState<IngestionJob | null>(null);
  const current =
    jobs.data?.jobs.find((j) => j.job_id === selected?.job_id) ?? selected;
  return (
    <section className="panel">
      <div className="panel-toolbar">
        <div>
          <h2>最近入库任务</h2>
          <p className="muted">每 4 秒刷新 · 最近 100 条</p>
        </div>
        <button
          className="button"
          disabled={jobs.loading}
          onClick={() => void jobs.load()}
        >
          <Icon name="refresh" size={16} />
          刷新
        </button>
      </div>
      <Notice error>{jobs.error}</Notice>
      {!jobs.data ? (
        <Empty title={jobs.loading ? "正在读取任务…" : "无法读取任务"} />
      ) : !jobs.data.jobs.length ? (
        <Empty title="还没有入库任务">
          在知识来源中勾选文档，提交第一个任务。
        </Empty>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>任务</th>
                <th>状态</th>
                <th>片段</th>
                <th>提交时间</th>
                <th>完成时间</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {jobs.data.jobs.map((j) => (
                <tr key={j.job_id}>
                  <td>
                    <code>{short(j.job_id)}</code>
                  </td>
                  <td>
                    <Badge status={j.status} />
                  </td>
                  <td>{j.progress?.chunks ?? "—"}</td>
                  <td>{date(j.created_at)}</td>
                  <td>{date(j.finished_at)}</td>
                  <td>
                    <button
                      className="text-button"
                      onClick={() => setSelected(j)}
                    >
                      查看详情
                      <Icon name="arrow" size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {current && (
        <Modal title="入库任务详情" onClose={() => setSelected(null)}>
          <Badge status={current.status} />
          <dl className="metadata">
            <dt>任务 ID</dt>
            <dd>
              <code>{current.job_id}</code>
            </dd>
            <dt>目标版本</dt>
            <dd>
              <code>{current.target_generation_id ?? "尚未生成"}</code>
            </dd>
            <dt>清单摘要</dt>
            <dd>
              <code>{current.manifest_hash}</code>
            </dd>
            <dt>处理片段</dt>
            <dd>{current.progress?.chunks ?? "尚未报告"}</dd>
            <dt>提交时间</dt>
            <dd>{date(current.created_at)}</dd>
          </dl>
          <Notice error>{current.error}</Notice>
          <p className="muted">
            {terminal.has(current.status)
              ? "任务已结束，可在索引版本页查看已发布版本。"
              : "任务正在后台处理，可关闭此窗口。刷新页面后仍可继续查看。"}
          </p>
          <div className="dialog-actions">
            <button className="button" onClick={() => void jobs.load()}>
              刷新状态
            </button>
          </div>
        </Modal>
      )}
    </section>
  );
}

function IndexesView({ onChange }: { onChange: () => Promise<void> }) {
  const list = useResource(api.generations);
  const [action, setAction] = useState<{
    generation?: Generation;
    type: "activate" | "rollback" | "cleanup";
  } | null>(null);
  const [retain, setRetain] = useState(2);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  async function apply() {
    if (!action || !list.data) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      if (action.type === "cleanup") {
        const r = await api.cleanup(retain);
        setNotice(
          `已清理 ${r.deleted_generations.length} 个旧版本；当前版本保持不变。`,
        );
      } else {
        await api.activate(
          action.generation!.generation_id,
          action.type,
          list.data.active,
          list.data.revision,
        );
        setNotice("当前索引版本已切换，新问题将使用该版本。");
      }
      setAction(null);
      await list.load();
      await onChange();
    } catch (e) {
      setError(message(e));
      await list.load();
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="version-banner">
        <div>
          <span className="eyebrow">PUBLISHED KNOWLEDGE</span>
          <h2>
            {list.data?.active
              ? `当前版本 ${short(list.data.active)}`
              : "尚未发布索引"}
          </h2>
          <p>每个版本保存一次知识索引。切换版本不会改写历史答案。</p>
        </div>
        <div>
          <small>发布修订号</small>
          <strong>{list.data?.revision ?? "—"}</strong>
        </div>
      </div>
      <section className="panel">
        <div className="panel-toolbar">
          <h2>版本档案</h2>
          <div className="actions">
            <button
              className="button"
              disabled={list.loading || busy}
              onClick={() => void list.load()}
            >
              <Icon name="refresh" size={16} />
              刷新
            </button>
            <button
              className="button"
              disabled={!list.data?.generations.length || busy}
              onClick={() => setAction({ type: "cleanup" })}
            >
              清理旧版本
            </button>
          </div>
        </div>
        <Notice error>{list.error || error}</Notice>
        <Notice>{notice}</Notice>
        {!list.data?.generations.length ? (
          <Empty title={list.loading ? "正在读取版本…" : "版本档案还是空的"}>
            提交入库并等待完成后，新版本会出现在这里。
          </Empty>
        ) : (
          <div className="version-list">
            {list.data.generations.map((g) => (
              <article
                className={`version-row ${g.generation_id === list.data?.active ? "current" : ""}`}
                key={g.generation_id}
              >
                <Icon name="layers" size={25} />
                <div className="version-info">
                  <h3>
                    <code>{short(g.generation_id)}</code>
                    <Badge
                      status={
                        g.generation_id === list.data?.active
                          ? "active"
                          : g.status
                      }
                    />
                  </h3>
                  <p>
                    {g.embedding_model} · {g.embedding_dim ?? "—"} 维 ·{" "}
                    {g.chunk_count} 个片段
                  </p>
                  <small>
                    {date(g.created_at)} 创建 · <code>{g.collection}</code>
                  </small>
                </div>
                {g.generation_id !== list.data?.active &&
                  g.status === "ready" && (
                    <div className="actions">
                      <button
                        className="button"
                        disabled={busy}
                        onClick={() =>
                          setAction({
                            generation: g,
                            type: g.activated_at ? "rollback" : "activate",
                          })
                        }
                      >
                        {g.activated_at ? "回滚到此版本" : "发布此版本"}
                      </button>
                    </div>
                  )}
              </article>
            ))}
          </div>
        )}
      </section>
      {action && (
        <Modal
          title={
            action.type === "cleanup" ? "确认清理旧版本" : "确认切换索引版本"
          }
          onClose={() => {
            if (!busy) setAction(null);
          }}
        >
          <Notice error>{error}</Notice>
          {action.type === "cleanup" ? (
            <>
              <p>
                删除超过保留数量的旧索引及版本关联。此操作不可撤销；当前使用的索引、构建中版本和答案记录不会删除。
              </p>
              <label className="field">
                额外保留的可回滚版本数
                <input
                  type="number"
                  min={0}
                  max={100}
                  value={retain}
                  onChange={(e) =>
                    setRetain(
                      Math.min(100, Math.max(0, Number(e.target.value))),
                    )
                  }
                />
              </label>
            </>
          ) : (
            <>
              <p>
                将当前版本 <code>{short(list.data?.active)}</code> 切换为{" "}
                <code>{short(action.generation?.generation_id)}</code>
                。新问题将使用该版本的知识内容。
              </p>
              <p className="muted">
                若其他任务已发布新版本，本次操作会被拒绝，请刷新后重新确认。
              </p>
            </>
          )}
          <div className="dialog-actions">
            <button
              className="button"
              disabled={busy}
              onClick={() => setAction(null)}
            >
              取消
            </button>
            <button
              className={`button ${action.type === "cleanup" ? "danger" : "primary"}`}
              disabled={busy}
              onClick={() => void apply()}
            >
              {busy ? "处理中…" : "确认执行"}
            </button>
          </div>
        </Modal>
      )}
    </>
  );
}

function AnswersView({ answerId }: { answerId: string | null }) {
  const list = useResource(api.listAnswers);
  const [detail, setDetail] = useState<AnswerDetail | null>(null);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const seq = useRef(0);
  const open = useCallback(async (id: string) => {
    const n = ++seq.current;
    setLoading(true);
    setError("");
    setDetail(null);
    try {
      const d = await api.answer(id);
      if (n === seq.current) setDetail(d);
    } catch (e) {
      if (n === seq.current) setError(message(e));
    } finally {
      if (n === seq.current) setLoading(false);
    }
  }, []);
  useEffect(() => {
    if (answerId) void open(answerId);
    return () => {
      seq.current++;
    };
  }, [answerId, open]);
  const filtered =
    list.data?.answers.filter((a) =>
      a.question.toLowerCase().includes(search.toLowerCase()),
    ) ?? [];
  return (
    <section className="panel">
      <div className="panel-toolbar">
        <div className="search-field">
          <Icon name="search" size={17} />
          <input
            aria-label="搜索历史答案"
            placeholder="搜索历史问题"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <button
          className="button"
          disabled={list.loading}
          onClick={() => void list.load()}
        >
          <Icon name="refresh" size={16} />
          刷新
        </button>
      </div>
      <Notice error>{list.error || error}</Notice>
      {!filtered.length ? (
        <Empty title={list.loading ? "正在读取答案…" : "没有匹配的答案记录"}>
          完成一次会话问答后，答案与引用摘录会保存在这里。
        </Empty>
      ) : (
        <div className="answer-list">
          {filtered.map((a) => (
            <button
              className="answer-row"
              key={a.answer_id}
              onClick={() => void open(a.answer_id)}
            >
              <span className="answer-mark">
                <Icon name="chat" />
              </span>
              <span>
                <strong>{a.question}</strong>
                <small>
                  {date(a.created_at)} · {short(a.answer_id)}
                </small>
              </span>
              <Badge status={a.status} />
              <Icon name="arrow" size={17} />
            </button>
          ))}
        </div>
      )}
      {(detail || loading) && (
        <Modal
          title="答案与出处"
          onClose={() => {
            seq.current++;
            setDetail(null);
            setLoading(false);
          }}
        >
          {loading ? (
            <Empty title="正在读取答案…" />
          ) : (
            detail && (
              <>
                <span className="eyebrow">
                  ANSWER ARCHIVE / {short(detail.answer_id)}
                </span>
                <h3 className="archive-question">{detail.question}</h3>
                <Badge status={detail.status} />
                <div className="archive-answer preserve">
                  {detail.answer ||
                    (detail.refused
                      ? "没有足够证据，未生成答案。"
                      : "未保存回答正文。")}
                </div>
                <h3>引用摘录 · {detail.citations.length}</h3>
                <Citations citations={detail.citations} />
                <dl className="metadata">
                  <dt>答案 ID</dt>
                  <dd>
                    <code>{detail.answer_id}</code>
                  </dd>
                  <dt>索引版本</dt>
                  <dd>
                    <code>{detail.generation_id ?? "—"}</code>
                  </dd>
                  <dt>生成模型</dt>
                  <dd>{detail.generation_model ?? "—"}</dd>
                  <dt>向量模型</dt>
                  <dd>{detail.embedding_model ?? "—"}</dd>
                  <dt>相关度 / 重排</dt>
                  <dd>
                    {detail.top_score?.toFixed(4) ?? "—"} /{" "}
                    {detail.reranked ? "已启用" : "未使用"}
                  </dd>
                </dl>
              </>
            )
          )}
        </Modal>
      )}
    </section>
  );
}

const blankProvider = (): ProviderConfig => ({
  base_url: "",
  api_key: "",
  model: "",
});
const providerLabels = {
  embedding: "向量模型",
  generation: "回答模型",
  rerank: "重排模型",
};
function ProviderCard({
  part,
  value,
  onChange,
  result,
  disabled,
}: {
  part: keyof typeof providerLabels;
  value: ProviderConfig;
  onChange: (field: keyof ProviderConfig, value: string) => void;
  result?: { ok: boolean; message: string };
  disabled: boolean;
}) {
  return (
    <fieldset className="provider-card" disabled={disabled}>
      <legend>{providerLabels[part]}</legend>
      <span className="eyebrow">{part.toUpperCase()}</span>
      <p>
        {part === "embedding"
          ? "把文档与问题转化为向量，用于检索。"
          : part === "generation"
            ? "结合问题、历史和检索证据生成回答。"
            : "重新排序检索候选，选出更相关的内容。"}
      </p>
      {(["base_url", "model", "api_key"] as const).map((field) => (
        <label className="field" key={field}>
          {field === "base_url"
            ? "Base URL"
            : field === "model"
              ? "模型名称"
              : "API Key"}
          <input
            required={field !== "api_key"}
            type={field === "api_key" ? "password" : "text"}
            autoComplete={field === "api_key" ? "new-password" : "off"}
            value={value[field]}
            placeholder={
              field === "base_url"
                ? "https://…/v1"
                : field === "model"
                  ? "填写服务商提供的模型标识"
                  : "输入密钥；mock 可留空"
            }
            onChange={(e) => onChange(field, e.target.value)}
          />
        </label>
      ))}
      {result && <Notice error={!result.ok}>{result.message}</Notice>}
    </fieldset>
  );
}
function SettingsView() {
  const resource = useResource(api.getSettings);
  const [settings, setSettings] = useState<RuntimeSettings | null>(null);
  const [rerankEnabled, setRerankEnabled] = useState(false);
  const [test, setTest] = useState<ProviderTestResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  useEffect(() => {
    if (resource.data) {
      setSettings(resource.data);
      setRerankEnabled(Boolean(resource.data.rerank?.base_url));
    }
  }, [resource.data]);
  async function save(testAfter: boolean) {
    if (!settings) return;
    setBusy(true);
    setError("");
    setNotice("");
    setTest(null);
    try {
      const payload = {
        ...settings,
        rerank: rerankEnabled ? settings.rerank : null,
      };
      if (
        !payload.embedding.base_url.trim() ||
        !payload.embedding.model.trim() ||
        !payload.generation.base_url.trim() ||
        !payload.generation.model.trim() ||
        (rerankEnabled &&
          (!payload.rerank?.base_url.trim() || !payload.rerank.model.trim()))
      )
        throw new Error("请填写已启用模型的 Base URL 和模型名称。");
      await api.putSettings(payload);
      setNotice("设置已保存。更换向量模型后，请重新入库。");
      if (testAfter) {
        setTest(await api.testProviders());
      }
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="settings-intro">
        <Icon name="settings" size={24} />
        <p>
          每种能力独立配置，按需组合。
          <br />
          <small>
            设置保存在本机；连通性测试使用保存后的配置，会向已配置的模型服务发送少量请求。
          </small>
        </p>
      </div>
      <Notice error>{resource.error || error}</Notice>
      <Notice>{notice}</Notice>
      {!settings ? (
        <Empty
          title={resource.loading ? "正在读取模型配置…" : "无法读取模型配置"}
        >
          <button className="button" onClick={() => void resource.load()}>
            重试
          </button>
        </Empty>
      ) : (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void save(false);
          }}
        >
          <div className="provider-grid">
            {(["embedding", "generation"] as const).map((part) => (
              <ProviderCard
                key={part}
                part={part}
                value={settings[part]}
                disabled={busy}
                result={test?.[part]}
                onChange={(field, value) =>
                  setSettings(
                    (s) =>
                      s && { ...s, [part]: { ...s[part], [field]: value } },
                  )
                }
              />
            ))}
          </div>
          <div className="rerank-toggle">
            <div>
              <h3>启用重排</h3>
              <p>可选增强。未启用时使用混合检索的排序结果。</p>
            </div>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={rerankEnabled}
                disabled={busy}
                onChange={(e) => {
                  setRerankEnabled(e.target.checked);
                  if (e.target.checked && !settings.rerank)
                    setSettings({ ...settings, rerank: blankProvider() });
                }}
              />
              <span>{rerankEnabled ? "已启用" : "未启用"}</span>
            </label>
          </div>
          {rerankEnabled && (
            <ProviderCard
              part="rerank"
              value={settings.rerank ?? blankProvider()}
              disabled={busy}
              result={test?.rerank}
              onChange={(field, value) =>
                setSettings(
                  (s) =>
                    s && {
                      ...s,
                      rerank: {
                        ...(s.rerank ?? blankProvider()),
                        [field]: value,
                      },
                    },
                )
              }
            />
          )}
          <div className="settings-footer">
            <small>不同服务商的重排协议可能不同，先测试连通性。</small>
            <div className="actions">
              <button
                className="button"
                type="button"
                disabled={busy}
                onClick={() => void save(true)}
              >
                {busy ? "处理中…" : "保存并测试连通性"}
              </button>
              <button className="button primary" disabled={busy}>
                <Icon name="check" size={16} />
                保存配置
              </button>
            </div>
          </div>
        </form>
      )}
    </>
  );
}
