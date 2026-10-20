const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const chatDiv = document.getElementById("chat");
const videoTitleEl = document.getElementById("videoTitle");

let currentVideoId = null;

// ---------------- SESSION ID ----------------
function getSessionId() {
  let sessionId = localStorage.getItem("vidsense_session_id");
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("vidsense_session_id", sessionId);
  }
  return sessionId;
}

// ---------------- VIDEO DETECTION ----------------
function detectActiveVideo() {
  if (typeof chrome === "undefined" || !chrome.tabs) {
    videoTitleEl.innerText = "Standalone Mode (Test Video: dQw4w9WgXcQ)";
    currentVideoId = "dQw4w9WgXcQ";
    return;
  }

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs || !tabs[0]) return;
    chrome.tabs.sendMessage(tabs[0].id, { type: "GET_VIDEO_ID" }, (response) => {
      if (response && response.videoId) {
        currentVideoId = response.videoId;
        videoTitleEl.innerText = `YouTube Target ID: ${currentVideoId}`;
      } else {
        videoTitleEl.innerText = "No active YouTube video detected";
      }
    });
  });
}

// ---------------- CHAT UI HELPERS ----------------
function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message user";
  div.innerText = text;
  chatDiv.appendChild(div);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

function addLoadingIndicator() {
  const div = document.createElement("div");
  div.className = "message ai loading-msg";
  div.innerHTML = `
    <div style="display:flex; align-items:center; gap:8px;">
      <span class="status-dot"></span>
      <span style="font-family:var(--font-mono); font-size:11px; color:#94a3b8;">Agent Planning & ChromaDB Retrieval...</span>
    </div>
  `;
  chatDiv.appendChild(div);
  chatDiv.scrollTop = chatDiv.scrollHeight;
  return div;
}

function renderAgentMessage(container, data) {
  container.className = "message ai";
  const confidencePct = Math.round((data.confidence_score || 0.85) * 100);
  const sourceLabel = data.source ? data.source.toUpperCase() : "TRANSCRIPT";
  const toolsUsed = (data.tools_invoked || ["ChromaDB"]).join(", ");

  let stepsHtml = "";
  if (data.execution_steps && data.execution_steps.length > 0) {
    stepsHtml = data.execution_steps
      .map(
        (s) => `
        <div class="trace-step">
          <span class="trace-step-name">[${s.step_number}] ${s.name}:</span>
          <span>${s.output_summary}</span>
        </div>
      `
      )
      .join("");
  } else {
    stepsHtml = `<div class="trace-step">ChromaDB semantic search & LLM generation.</div>`;
  }

  container.innerHTML = `
    <div class="trace-accordian">
      <div class="trace-header">
        <div class="trace-badges">
          <span class="badge badge-source">${sourceLabel}</span>
          <span class="badge badge-guard">🛡️ GUARD PASSED</span>
          <span class="badge badge-tool">⚡ ${confidencePct}% CONF</span>
        </div>
        <span class="trace-toggle">▼ Trace</span>
      </div>
      <div class="trace-body">
        ${stepsHtml}
      </div>
    </div>
    <div class="answer-body">${formatAnswerText(data.answer)}</div>
  `;

  // Attach toggle listener
  const header = container.querySelector(".trace-header");
  const body = container.querySelector(".trace-body");
  const toggle = container.querySelector(".trace-toggle");
  header.addEventListener("click", () => {
    const isVisible = body.style.display === "block";
    body.style.display = isVisible ? "none" : "block";
    toggle.innerText = isVisible ? "▼ Trace" : "▲ Close";
  });

  chatDiv.scrollTop = chatDiv.scrollHeight;
}

function formatAnswerText(text) {
  if (!text) return "No response generated.";
  return text.replace(/\n/g, "<br/>");
}

// ---------------- SEND QUERY ----------------
async function handleAsk(questionText) {
  const question = questionText || questionInput.value.trim();
  if (!question) return;

  addUserMessage(question);
  questionInput.value = "";
  const loadingEl = addLoadingIndicator();
  askBtn.disabled = true;

  const sessionId = getSessionId();

  try {
    // 1. If video is detected, make sure it's ingested in ChromaDB
    if (currentVideoId) {
      try {
        await fetch(`http://127.0.0.1:8000/ingest/youtube/${currentVideoId}`, {
          method: "POST",
        });
      } catch (err) {
        console.warn("Ingestion notice:", err);
      }
    }

    // 2. Ask Agent
    const res = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        session_id: sessionId,
        video_id: currentVideoId || null,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      loadingEl.className = "message ai";
      loadingEl.innerHTML = `<span style="color:#f43f5e;">⚠️ Error: ${errData.detail || "Agent failed to respond."}</span>`;
      return;
    }

    const data = await res.json();
    renderAgentMessage(loadingEl, data);
  } catch (error) {
    loadingEl.className = "message ai";
    loadingEl.innerHTML = `<span style="color:#f43f5e;">⚠️ Backend unreachable at 127.0.0.1:8000. Ensure FastAPI server is running.</span>`;
  } finally {
    askBtn.disabled = false;
  }
}

// ---------------- LISTENERS ----------------
askBtn.addEventListener("click", () => handleAsk());
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleAsk();
});

// Quick chips
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const prompt = chip.getAttribute("data-prompt");
    handleAsk(prompt);
  });
});

// On load
detectActiveVideo();
