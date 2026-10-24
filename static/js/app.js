
import { fetchJson } from "./api-client.mjs";
import { createChatSession } from "./chat-session.mjs";

const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const question = document.getElementById("question");
const trace = document.getElementById("trace");
const sourceUsed = document.getElementById("sourceUsed");
const sendBtn = form.querySelector('button[type="submit"]');
let isSending = false;
const session = createChatSession();
const welcomeState = chat.firstElementChild.cloneNode(true);
const newChatBtn = document.getElementById("newChatBtn");

function escapeHtml(s = "") {
  return String(s).replace(/[&<>'"]/g, c => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;"
  })[c]);
}

function formatText(s = "") {
  return escapeHtml(s).replace(/\n/g, "<br>");
}

function scrollChatToBottom() {
  chat.scrollTop = chat.scrollHeight;
}


function addMessage(role, text, source = "", citations = []) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const citeHtml = citations.length
    ? `<div class="citations">
         <strong>Sources</strong><br>
         ${citations.map(c =>
           c.url
             ? `<a href="${escapeHtml(c.url)}"
                   target="_blank"
                   rel="noopener noreferrer">${escapeHtml(c.title)}</a>`
             : escapeHtml(c.title)
         ).join("<br>")}
       </div>`
    : "";

  wrap.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble">
      <div class="message-text">${formatText(text)}</div>
      ${source
        ? `<div class="answer-source">Source: ${escapeHtml(source)}</div>`
        : ""}
      ${citeHtml}
    </div>
  `;

  chat.appendChild(wrap);
  scrollChatToBottom();
}

// Remove only keys used by the former active-chat persistence feature.
try {
  localStorage.removeItem("it_support_copilot_chat_v1");
  localStorage.removeItem("it_support_copilot_source_v1");
} catch (error) {
  console.warn("Could not clear the previous active chat:", error);
}

function startNewChat() {
  session.reset();
  isSending = false;
  sendBtn.disabled = false;
  question.disabled = false;
  question.value = "";
  chat.replaceChildren(welcomeState.cloneNode(true));
  renderTrace([]);
  sourceUsed.textContent = "—";
  try {
    localStorage.removeItem("it_support_copilot_chat_v1");
    localStorage.removeItem("it_support_copilot_source_v1");
  } catch (error) {
    console.warn("Could not clear the previous active chat:", error);
  }
  question.focus();
}

newChatBtn.addEventListener("click", startNewChat);

function renderTrace(items = []) {
  trace.innerHTML = items.length
    ? items.map(x =>
        `<div class="trace-item">${escapeHtml(x)}</div>`
      ).join("")
    : '<div class="empty">No trace.</div>';

  // Keep the latest trace visible inside the trace panel.
  trace.scrollTop = trace.scrollHeight;
}

async function askAgent(q) {
  q = q.trim();

  if (!q) {
    return;
  }

  const request = session.begin();
  if (!request) return;

  isSending = true;
  sendBtn.disabled = true;
  question.disabled = true;

  addMessage("user", q);

  question.value = "";
  renderTrace(["Running LangGraph workflow..."]);
  sourceUsed.textContent = "Running";

  try {
    const data = await fetchJson("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      signal: request.controller.signal,
      body: JSON.stringify({
        question: q
      })
    });

    if (!session.isCurrent(request)) return;

    if (typeof data.answer !== "string") {
      throw new Error("The server response did not include an answer. Please try again.");
    }

    addMessage(
      "assistant",
      data.answer,
      data.source_used,
      data.citations || []
    );

    renderTrace(data.trace || []);
    sourceUsed.textContent = data.source_used || "Unknown";
  } catch (e) {
    if (!session.isCurrent(request)) return;
    addMessage("assistant", `Error: ${e.message}`);
    renderTrace(["Request failed"]);
    sourceUsed.textContent = "Error";
  } finally {
    if (session.isCurrent(request)) {
      session.finish(request);
      isSending = false;
      sendBtn.disabled = false;
      question.disabled = false;
      question.focus();
    }
  }
}

// Clicking Ask Agent submits the form.
form.addEventListener("submit", function (event) {
  event.preventDefault();

  const q = question.value.trim();

  if (q && !isSending) {
    askAgent(q);
  }
});

// Enter sends; Shift + Enter inserts a newline.

question.addEventListener("keydown", function (event) {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();

    if (!isSending && question.value.trim()) {
      form.requestSubmit();
    }
  }
});


// Example questions use the same sending function.
document.querySelectorAll(".example").forEach(button => {
  button.addEventListener("click", () => {
    askAgent(button.textContent.trim());
  });
});

// Document upload modal.
const modal = document.getElementById("uploadModal");

document.getElementById("openUpload").onclick = () => {
  modal.classList.remove("hidden");
};

document.getElementById("closeUpload").onclick = () => {
  modal.classList.add("hidden");
};

document.getElementById("uploadBtn").onclick = async () => {
  const file = document.getElementById("fileInput").files[0];
  const key = document.getElementById("adminKey").value;
  const status = document.getElementById("uploadStatus");
  const uploadBtn = document.getElementById("uploadBtn");

  if (!file) {
    status.textContent = "Choose a file first.";
    return;
  }

  uploadBtn.disabled = true;
  status.textContent = "Indexing document...";

  const fd = new FormData();
  fd.append("file", file);

  try {
    const data = await fetchJson("/api/ingest", {
      method: "POST",
      headers: {
        "X-Admin-Key": key
      },
      body: fd
    });

    status.textContent =
      `Indexed ${data.file}: ${data.chunks} chunks.`;

  } catch (e) {
    status.textContent = `Error: ${e.message}`;

  } finally {
    uploadBtn.disabled = false;
  }
};
