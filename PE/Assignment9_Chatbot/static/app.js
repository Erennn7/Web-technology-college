// Simple & Clean Chat Interface for HealthAware AI
let conversationHistory = [];
let savedApiKey = localStorage.getItem("healthaware_api_key") || "";

const chatMessages = document.getElementById("chatMessages");
const chatContainer = document.getElementById("chatContainer");
const userInput = document.getElementById("userInput");
const btnSend = document.getElementById("btnSend");
const keyModal = document.getElementById("keyModal");
const txtApiKey = document.getElementById("txtApiKey");
const btnApiKey = document.getElementById("btnApiKey");
const apiKeyStatusText = document.getElementById("apiKeyStatusText");
const keyStatusIndicator = document.getElementById("keyStatusIndicator");

document.addEventListener("DOMContentLoaded", () => {
  setupInputListeners();
  updateApiKeyButtonState();
});

function setupInputListeners() {
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
  });
}

// Modal functions
function openKeyModal() {
  txtApiKey.value = savedApiKey;
  updateModalStatus();
  keyModal.classList.add("active");
  txtApiKey.focus();
}

function closeKeyModal() {
  keyModal.classList.remove("active");
}

function closeModalOnBackdrop(e) {
  if (e.target.id === "keyModal") {
    closeKeyModal();
  }
}

function saveApiKey() {
  const val = txtApiKey.value.trim();
  savedApiKey = val;
  if (val) {
    localStorage.setItem("healthaware_api_key", val);
  } else {
    localStorage.removeItem("healthaware_api_key");
  }
  updateApiKeyButtonState();
  closeKeyModal();
}

function clearApiKey() {
  savedApiKey = "";
  txtApiKey.value = "";
  localStorage.removeItem("healthaware_api_key");
  updateApiKeyButtonState();
  closeKeyModal();
}

function updateApiKeyButtonState() {
  if (savedApiKey) {
    btnApiKey.classList.add("has-key");
    apiKeyStatusText.textContent = "API Key (Active)";
  } else {
    btnApiKey.classList.remove("has-key");
    apiKeyStatusText.textContent = "API Key";
  }
}

function updateModalStatus() {
  if (savedApiKey) {
    let providerName = "Unknown Provider";
    if (savedApiKey.startsWith("gsk_")) providerName = "Groq Cloud (Llama 3)";
    else if (savedApiKey.startsWith("AIza")) providerName = "Google Gemini";
    else if (savedApiKey.startsWith("sk-")) providerName = "OpenAI GPT";

    keyStatusIndicator.className = "key-status-indicator active";
    keyStatusIndicator.textContent = `✓ Active Key configured for ${providerName}`;
  } else {
    keyStatusIndicator.className = "key-status-indicator";
    keyStatusIndicator.textContent = "No key configured (using offline simulation engine)";
  }
}

function handleKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleFormSubmit(e);
  }
}

function handleFormSubmit(e) {
  if (e) e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;

  userInput.value = "";
  userInput.style.height = "auto";
  sendMessage(text);
}

let currentPromptMode = "initial";

function setMode(mode) {
  currentPromptMode = mode;
  const btnInit = document.getElementById("btnModeInitial");
  const btnHard = document.getElementById("btnModeHardened");
  if (btnInit && btnHard) {
    btnInit.classList.toggle("active", mode === "initial");
    btnHard.classList.toggle("active", mode === "hardened");
  }
}

function sendQuickPrompt(promptText) {
  sendMessage(promptText);
}

async function sendMessage(text) {
  // Append user message to UI
  appendMessage("user", text);

  // Append temporary loading state
  const loadingId = appendLoading();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        prompt_mode: currentPromptMode,
        provider: "auto",
        api_key: savedApiKey,
        history: conversationHistory
      })
    });

    const data = await res.json();
    removeLoading(loadingId);

    // Append assistant response to UI
    appendMessage("assistant", data.response);

    // Maintain conversation context
    conversationHistory.push({ role: "user", content: text });
    conversationHistory.push({ role: "assistant", content: data.response });

  } catch (err) {
    removeLoading(loadingId);
    appendMessage("assistant", "I am having trouble connecting right now. Please check if the local server is running and try again.");
  }
}

function appendMessage(role, text) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "👤" : "🤖";

  const body = document.createElement("div");
  body.className = "message-body";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;

  body.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(body);

  chatMessages.appendChild(row);
  scrollToBottom();
}

function appendLoading() {
  const id = "loading_" + Date.now();
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.id = id;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = "🤖";

  const body = document.createElement("div");
  body.className = "message-body";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerHTML = "<em>HealthAware AI is typing...</em>";

  body.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(body);

  chatMessages.appendChild(row);
  scrollToBottom();
  return id;
}

function removeLoading(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function clearChat() {
  conversationHistory = [];
  chatMessages.innerHTML = `
    <div class="message-row assistant">
      <div class="message-avatar">🤖</div>
      <div class="message-body">
        <div class="message-bubble">
          <p>Hello! I am <strong>HealthAware AI</strong>, your health and wellness awareness assistant.</p>
          <p>I can help answer questions about healthy lifestyle habits, sleep hygiene, nutrition, everyday self-care comfort measures, and recognizing when to consult a healthcare professional.</p>
          <p>How can I assist you with your health and wellbeing today?</p>
        </div>
        <div class="quick-chips">
          <button class="chip" onclick="sendQuickPrompt('What are evidence-based habits to improve sleep quality?')">Habits for better sleep</button>
          <button class="chip" onclick="sendQuickPrompt('What home comfort measures can I take for a common cold?')">Care for a common cold</button>
          <button class="chip" onclick="sendQuickPrompt('What are effective ways to manage exam stress?')">Managing exam stress</button>
          <button class="chip" onclick="sendQuickPrompt('What does a balanced daily diet look like?')">Balanced daily nutrition</button>
        </div>
      </div>
    </div>
  `;
}
