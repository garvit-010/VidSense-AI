function getVideoId() {
  const url = window.location.href;

  // Standard watch URL
  const watchMatch = url.match(/[?&]v=([^&]+)/);
  if (watchMatch) return watchMatch[1];

  // Shorts URL
  const shortsMatch = url.match(/shorts\/([^?]+)/);
  if (shortsMatch) return shortsMatch[1];

  // youtu.be URL
  const shortUrlMatch = url.match(/youtu\.be\/([^?]+)/);
  if (shortUrlMatch) return shortUrlMatch[1];

  return null;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "GET_VIDEO_ID") {
    sendResponse({ videoId: getVideoId() });
  }
});
