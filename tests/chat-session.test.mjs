import assert from "node:assert/strict";
import test from "node:test";

import { createChatSession } from "../static/js/chat-session.mjs";

test("reset aborts an in-flight request and makes its result stale", () => {
  const session = createChatSession();
  const oldRequest = session.begin();

  assert.ok(oldRequest);
  assert.equal(session.isCurrent(oldRequest), true);

  session.reset();

  assert.equal(oldRequest.controller.signal.aborted, true);
  assert.equal(session.isCurrent(oldRequest), false);
  assert.ok(session.begin(), "a new request can start after reset");
});

test("only one request can be active in a conversation", () => {
  const session = createChatSession();
  const request = session.begin();

  assert.equal(session.begin(), null);
  session.finish(request);
  assert.ok(session.begin());
});
