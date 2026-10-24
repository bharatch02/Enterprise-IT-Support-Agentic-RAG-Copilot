import assert from "node:assert/strict";
import test from "node:test";

import { fetchJson } from "../static/js/api-client.mjs";

function response({
  status = 200,
  body = "",
  ok = status >= 200 && status < 300,
  requestId = "",
} = {}) {
  return {
    ok,
    status,
    headers: { get: (name) => name.toLowerCase() === "x-request-id" ? requestId : null },
    text: async () => body,
  };
}

test("parses a valid JSON response", async () => {
  const data = await fetchJson("/api/chat", {}, {
    fetchImpl: async () => response({ body: '{"answer":"Hi"}' }),
  });

  assert.deepEqual(data, { answer: "Hi" });
});

test("reports an empty successful response clearly", async () => {
  await assert.rejects(
    fetchJson("/api/chat", {}, { fetchImpl: async () => response() }),
    /empty response/i,
  );
});

test("includes the request ID when the response body is empty", async () => {
  await assert.rejects(
    fetchJson("/api/chat", {}, {
      fetchImpl: async () => response({ requestId: "abc123" }),
    }),
    /Request ID: abc123/,
  );
});

test("reports malformed JSON clearly", async () => {
  await assert.rejects(
    fetchJson("/api/chat", {}, {
      fetchImpl: async () => response({ body: "<html>proxy error</html>" }),
    }),
    /invalid response/i,
  );
});

test("uses a JSON detail for non-success responses", async () => {
  await assert.rejects(
    fetchJson("/api/chat", {}, {
      fetchImpl: async () => response({
        status: 503,
        ok: false,
        body: '{"detail":"Service unavailable"}',
      }),
    }),
    /Service unavailable/,
  );
});

test("reports the status for non-JSON error responses", async () => {
  await assert.rejects(
    fetchJson("/api/chat", {}, {
      fetchImpl: async () => response({
        status: 502,
        ok: false,
        body: "Bad Gateway",
      }),
    }),
    /HTTP 502/,
  );
});

test("explains that an HTTP 502 may indicate an unavailable Render process", async () => {
  await assert.rejects(
    fetchJson(
      "/api/chat",
      {},
      {
        fetchImpl: async () => ({
          ok: false,
          status: 502,
          headers: new Headers(),
          text: async () => "<html>Bad Gateway</html>",
        }),
      },
    ),
    /HTTP 502.*may have restarted or be temporarily unavailable/,
  );
});

test("turns fetch network errors into a useful message", async () => {
  await assert.rejects(
    fetchJson("/api/chat", {}, {
      fetchImpl: async () => {
        throw new TypeError("fetch failed");
      },
    }),
    /Could not reach the support service/i,
  );
});

test("aborts a request after its configured timeout", async () => {
  const slowFetch = (_url, { signal }) => new Promise((resolve, reject) => {
    signal.addEventListener("abort", () => {
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });

  await assert.rejects(
    fetchJson("/api/chat", {}, { timeoutMs: 5, fetchImpl: slowFetch }),
    /timed out after 1 seconds/i,
  );
});

test("caller abort cancels the fetch and reports it as an aborted request", async () => {
  const controller = new AbortController();
  const slowFetch = (_url, { signal }) => new Promise((resolve, reject) => {
    signal.addEventListener("abort", () => {
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });
  const pending = fetchJson("/api/chat", { signal: controller.signal }, {
    timeoutMs: 5000,
    fetchImpl: slowFetch,
  });

  controller.abort();

  await assert.rejects(pending, /request was cancelled/i);
});
