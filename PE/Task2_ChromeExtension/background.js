/**
 * TaskFlow Background Service Worker (Manifest V3)
 * Keeps Chrome toolbar extension badge in sync with pending tasks.
 */

// Initial default tasks for new users
const DEFAULT_TASKS = [
  { id: "1", text: "Welcome to TaskFlow! Click the circle to complete", category: "General", completed: false },
  { id: "2", text: "Submit Web Tech Assignment", category: "College", completed: false },
  { id: "3", text: "Build Chrome Extension using AI", category: "Coding", completed: true }
];

// On Extension Installed
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.get(["tasks", "theme"], (res) => {
    if (!res.tasks || res.tasks.length === 0) {
      chrome.storage.sync.set({ tasks: DEFAULT_TASKS, theme: "light" }, () => {
        updateBadge(DEFAULT_TASKS);
      });
    } else {
      updateBadge(res.tasks);
    }
  });
});

// Update Badge count on extension icon
function updateBadge(tasks) {
  try {
    const pendingCount = (tasks || []).filter(t => !t.completed).length;
    const badgeText = pendingCount > 0 ? (pendingCount > 99 ? "99+" : String(pendingCount)) : "";
    
    chrome.action.setBadgeText({ text: badgeText });
    chrome.action.setBadgeBackgroundColor({ color: "#4F46E5" });
  } catch (err) {
    console.error("Badge error:", err);
  }
}

// Sync badge helper
function syncBadge() {
  chrome.storage.sync.get(["tasks"], (res) => {
    updateBadge(res ? res.tasks : []);
  });
}

// Storage change listener
chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === "sync" && changes.tasks) {
    updateBadge(changes.tasks.newValue || []);
  }
});

// Initial badge sync on worker start
syncBadge();
