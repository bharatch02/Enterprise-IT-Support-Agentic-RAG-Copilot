
import { fetchJson } from "./api-client.mjs";

const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const question = document.getElementById("question");
const trace = document.getElementById("trace");
const sourceUsed = document.getElementById("sourceUsed");
const sendBtn = form.querySelector('button[type="submit"]');
const CHAT_STORAGE_KEY = "it_support_copilot_chat_v1";
const SOURCE_STORAGE_KEY = "it_support_copilot_source_v1";

let isSending = false;

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


function saveChat() {
  const messages = [...chat.querySelectorAll(".message")].map(message => {
    const bubble = message.querySelector(".bubble");
    const sourceElement = bubble.querySelector(".answer-source");

    const citations = [...bubble.querySelectorAll(".citations a")].map(link => ({
      title: link.textContent,
      url: link.href
    }));

    return {
      role: message.classList.contains("user") ? "user" : "assistant",
      text: bubble.querySelector(".message-text")?.textContent
        ?? bubble.innerText.replace(
          sourceElement?.innerText || "",
          ""
        ).trim(),
      source: sourceElement
        ? sourceElement.textContent.replace(/^Source:\s*/, "")
        : "",
      citations
    };
  });

  try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages));
    localStorage.setItem(SOURCE_STORAGE_KEY, sourceUsed.textContent);
  } catch (error) {
    console.warn("Could not save chat history:", error);
  }
}

function addMessage(role, text, source = "", citations = [], shouldSave = true) {
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

  if (shouldSave) {
    saveChat();
  }
}

function restoreChat() {
  try {
    const saved = localStorage.getItem(CHAT_STORAGE_KEY);

    if (saved === null) {
      return;
    }

    const messages = JSON.parse(saved);

    if (!Array.isArray(messages)) {
      return;
    }

    chat.innerHTML = "";

    messages.forEach(message => {
      if (
        !message ||
        !["user", "assistant"].includes(message.role) ||
        typeof message.text !== "string"
      ) {
        return;
      }

      addMessage(
        message.role,
        message.text,
        message.source || "",
        Array.isArray(message.citations) ? message.citations : [],
        false
      );
    });

    const savedSource = localStorage.getItem(SOURCE_STORAGE_KEY);

    if (savedSource) {
      sourceUsed.textContent = savedSource;
    }

    scrollChatToBottom();
  } catch (error) {
    console.warn("Could not restore chat history:", error);
  }
}

restoreChat();

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

  if (!q || isSending) {
    return;
  }

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
      body: JSON.stringify({
        question: q
      })
    });

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
    saveChat();

  } catch (e) {
    addMessage("assistant", `Error: ${e.message}`);
    renderTrace(["Request failed"]);
    sourceUsed.textContent = "Error";
    saveChat();

  } finally {
    isSending = false;
    sendBtn.disabled = false;
    question.disabled = false;
    question.focus();
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
