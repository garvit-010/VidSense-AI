const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const chatDiv = document.getElementById("chat");

// ---------------- SESSION ID ----------------
function getSessionId() {
  let sessionId = localStorage.getItem("vidsense_session_id");

  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("vidsense_session_id", sessionId);
  }

  return sessionId;
}

// ---------------- CHAT UI ----------------
function addMessage(text, className) {
  const div = document.createElement("div");
  div.className = `message ${className}`;
  div.innerText = text;
  chatDiv.appendChild(div);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

// ---------------- ASK BUTTON ----------------
askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage(question, "user");
  questionInput.value = "";

  const sessionId = getSessionId();

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(
      tabs[0].id,
      { type: "GET_VIDEO_ID" },
      async (response) => {
        const videoId = response?.videoId;

        if (!videoId) {
          addMessage("Not a YouTube video", "ai");
          return;
        }

        await fetch(
          `http://127.0.0.1:8000/ingest/youtube/${videoId}`,
          { method: "POST" }
        );

        addMessage("Thinking...", "ai");

        const res = await fetch(
          "http://127.0.0.1:8000/ask",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              question: question,
              session_id: sessionId
            })
          }
        );

        const data = await res.json();

        chatDiv.lastChild.innerText =
          data.answer || "Something went wrong";
      }
    );
  });
});
