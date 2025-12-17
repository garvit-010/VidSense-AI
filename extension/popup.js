const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const chatDiv = document.getElementById("chat");

function addMessage(text, className) {
  const div = document.createElement("div");
  div.className = `message ${className}`;
  div.innerText = text;
  chatDiv.appendChild(div);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("You: " + question, "user");
  questionInput.value = "";

  // Get video ID from content script
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(
      tabs[0].id,
      { type: "GET_VIDEO_ID" },
      async (response) => {
        const videoId = response?.videoId;
        if (!videoId) {
          addMessage("AI: Not a YouTube video", "ai");
          return;
        }

        // Ingest video (safe to call multiple times)
        await fetch(
          `http://127.0.0.1:8000/ingest/youtube/${videoId}`,
          { method: "POST" }
        );

        // Ask question
        const res = await fetch(
          "http://127.0.0.1:8000/ask",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
          }
        );

        const data = await res.json();
        addMessage("AI: " + data.answer, "ai");
      }
    );
  });
});
