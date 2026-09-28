import assert from 'node:assert/strict';
import test from 'node:test';
import { api } from '../web/src/api.ts';

const options = { top_k: 3, min_score: 0.4, conversation_id: 'test' };
function response(text, type = 'text/event-stream') {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += 2) controller.enqueue(bytes.slice(i, i + 2));
      controller.close();
    },
  }), { headers: { 'Content-Type': type } });
}
async function withFetch(implementation, run) {
  const original = globalThis.fetch;
  globalThis.fetch = implementation;
  try { await run(); } finally { globalThis.fetch = original; }
}

test('SSE handles split UTF-8, CRLF, heartbeat and multi-line data', async () => {
  const events = [];
  const text = ': heartbeat\r\n\r\nevent: delta\r\ndata: {\r\ndata: "content":"中文内容"}\r\n\r\nevent: done\r\ndata: {"refused":false}\r\n\r\n';
  await withFetch(async () => response(text), () => api.chatStream('问题', options, e => events.push(e)));
  assert.deepEqual(events.map(e => e.event), ['delta', 'done']);
  assert.equal(events[0].data.content, '中文内容');
});

test('SSE rejects truncated connection without done', async () => {
  await withFetch(async () => response('event: delta\ndata: {"content":"partial"}\n\n'), async () => {
    await assert.rejects(api.chatStream('问题', options, () => {}), /提前结束/);
  });
});

test('SSE forwards server error rather than reporting success', async () => {
  await withFetch(async () => response('event: error\ndata: {"message":"模型超时"}\n\n'), async () => {
    await assert.rejects(api.chatStream('问题', options, () => {}), /模型超时/);
  });
});

test('Callback failures are not swallowed as malformed frames', async () => {
  await withFetch(async () => response('event: done\ndata: {}\n\n'), async () => {
    await assert.rejects(api.chatStream('问题', options, () => { throw new Error('consumer failed'); }), /consumer failed/);
  });
});

test('JSON response is not mistaken for streaming completion', async () => {
  await withFetch(async () => response('{"answer":"local"}', 'application/json'), async () => {
    await assert.rejects(api.chatStream('问题', options, () => {}), /未返回流式响应/);
  });
});

test('Abort signal propagates to fetch', async () => {
  const abort = new AbortController();
  abort.abort();
  await withFetch(async (_url, init) => { assert.equal(init.signal, abort.signal); throw new DOMException('Aborted', 'AbortError'); }, async () => {
    await assert.rejects(api.chatStream('问题', options, () => {}, abort.signal), { name: 'AbortError' });
  });
});
