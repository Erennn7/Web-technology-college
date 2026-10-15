/**
 * TaskFlow - Minimal JavaScript To-Do Extension
 * Manifest V3 Compliant
 */

// Application State
let tasks = [];
let currentFilter = "all";
let selectedCategory = "General";
let currentTheme = "light";

// DOM Elements
const taskInput = document.getElementById("task-input");
const addBtn = document.getElementById("add-btn");
const taskList = document.getElementById("task-list");
const emptyState = document.getElementById("empty-state");
const taskCounter = document.getElementById("task-counter");
const clearBtn = document.getElementById("clear-btn");
const themeBtn = document.getElementById("theme-btn");
const themeMoon = document.getElementById("theme-moon");
const themeSun = document.getElementById("theme-sun");
const categoryPills = document.querySelectorAll(".pill");
const filterTabs = document.querySelectorAll(".tab-btn");

// Storage Helper (Chrome Storage with LocalStorage fallback)
const Storage = {
  isChrome: () => typeof chrome !== "undefined" && chrome.storage && chrome.storage.sync,

  get: (keys) => {
    return new Promise((resolve) => {
      if (Storage.isChrome()) {
        chrome.storage.sync.get(keys, (res) => resolve(res || {}));
      } else {
        const out = {};
        keys.forEach(k => {
          const val = localStorage.getItem(`tf_${k}`);
          try { out[k] = val ? JSON.parse(val) : null; } catch (e) { out[k] = val; }
        });
        resolve(out);
      }
    });
  },

  set: (data) => {
    return new Promise((resolve) => {
      if (Storage.isChrome()) {
        chrome.storage.sync.set(data, () => resolve());
      } else {
        Object.keys(data).forEach(k => localStorage.setItem(`tf_${k}`, JSON.stringify(data[k])));
        resolve();
      }
    });
  }
};

// Initialize App
document.addEventListener("DOMContentLoaded", async () => {
  const data = await Storage.get(["tasks", "theme"]);

  if (Array.isArray(data.tasks)) {
    tasks = data.tasks;
  } else {
    // Initial sample tasks
    tasks = [
      { id: "1", text: "Welcome to TaskFlow! Click the circle to complete", category: "General", completed: false },
      { id: "2", text: "Submit Web Tech Assignment", category: "College", completed: false },
      { id: "3", text: "Build Chrome Extension using AI", category: "Coding", completed: true }
    ];
    await Storage.set({ tasks });
  }

  // Theme setup
  currentTheme = data.theme || "light";
  applyTheme(currentTheme);

  // Setup Event Listeners
  initEvents();

  // Initial Render
  render();
});

// Event Listeners Setup
function initEvents() {
  // Add task on button click or Enter key
  addBtn.addEventListener("click", handleAddTask);
  taskInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleAddTask();
  });

  // Category pill selection
  categoryPills.forEach(pill => {
    pill.addEventListener("click", () => {
      categoryPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      selectedCategory = pill.dataset.cat;
    });
  });

  // Filter tabs (All, Active, Completed)
  filterTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      filterTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentFilter = tab.dataset.filter;
      render();
    });
  });

  // Clear completed tasks
  clearBtn.addEventListener("click", async () => {
    tasks = tasks.filter(t => !t.completed);
    await saveTasks();
    render();
  });

  // Theme toggle
  themeBtn.addEventListener("click", async () => {
    currentTheme = currentTheme === "light" ? "dark" : "light";
    applyTheme(currentTheme);
    await Storage.set({ theme: currentTheme });
  });
}

// Add a new task
async function handleAddTask() {
  const text = taskInput.value.trim();
  if (!text) return;

  const newTask = {
    id: "task_" + Date.now(),
    text,
    category: selectedCategory,
    completed: false,
    createdAt: Date.now()
  };

  tasks.unshift(newTask);
  taskInput.value = "";
  await saveTasks();
  render();
}

// Toggle complete status
async function toggleTask(id) {
  const task = tasks.find(t => t.id === id);
  if (task) {
    task.completed = !task.completed;
    await saveTasks();
    render();
  }
}

// Delete a task
async function deleteTask(id) {
  tasks = tasks.filter(t => t.id !== id);
  await saveTasks();
  render();
}

// Save tasks to storage and update toolbar badge
async function saveTasks() {
  await Storage.set({ tasks });
  updateBadge();
}

// Update Chrome extension toolbar badge
function updateBadge() {
  const count = tasks.filter(t => !t.completed).length;
  if (typeof chrome !== "undefined" && chrome.action && chrome.action.setBadgeText) {
    chrome.action.setBadgeText({ text: count > 0 ? String(count) : "" });
    chrome.action.setBadgeBackgroundColor({ color: "#4F46E5" });
  }
}

// Apply Dark/Light theme
function applyTheme(theme) {
  if (theme === "dark") {
    document.body.classList.add("dark");
    themeMoon.classList.add("hidden");
    themeSun.classList.remove("hidden");
  } else {
    document.body.classList.remove("dark");
    themeMoon.classList.remove("hidden");
    themeSun.classList.add("hidden");
  }
}

// Render task list and counters
function render() {
  // Filter tasks based on current tab
  const filtered = tasks.filter(t => {
    if (currentFilter === "active") return !t.completed;
    if (currentFilter === "completed") return t.completed;
    return true;
  });

  // Clear list
  taskList.innerHTML = "";

  if (filtered.length === 0) {
    emptyState.classList.remove("hidden");
  } else {
    emptyState.classList.add("hidden");

    filtered.forEach(task => {
      const li = document.createElement("li");
      li.className = `task-item ${task.completed ? "completed" : ""}`;

      // Left: Checkbox + Title
      const left = document.createElement("div");
      left.className = "task-left";

      const checkBtn = document.createElement("button");
      checkBtn.className = "checkbox-btn";
      checkBtn.title = task.completed ? "Mark incomplete" : "Mark complete";
      checkBtn.innerHTML = `
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      `;
      checkBtn.addEventListener("click", () => toggleTask(task.id));

      const title = document.createElement("span");
      title.className = "task-title";
      title.textContent = task.text;

      left.appendChild(checkBtn);
      left.appendChild(title);

      // Right: Category Tag + Delete Button
      const right = document.createElement("div");
      right.className = "task-right";

      const tag = document.createElement("span");
      tag.className = "cat-tag";
      tag.textContent = task.category;

      const delBtn = document.createElement("button");
      delBtn.className = "delete-btn";
      delBtn.title = "Delete";
      delBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      `;
      delBtn.addEventListener("click", () => deleteTask(task.id));

      right.appendChild(tag);
      right.appendChild(delBtn);

      li.appendChild(left);
      li.appendChild(right);
      taskList.appendChild(li);
    });
  }

  // Update Remaining Tasks Counter
  const activeCount = tasks.filter(t => !t.completed).length;
  taskCounter.textContent = `${activeCount} left`;
}
