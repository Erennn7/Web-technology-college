document.addEventListener("DOMContentLoaded", () => {
  // Initialize Lucide icons
  lucide.createIcons();

  // Settings Panel Toggle
  const toggleSettingsBtn = document.getElementById("toggle-settings-btn");
  const settingsPanel = document.getElementById("settings-panel");
  toggleSettingsBtn.addEventListener("click", () => {
    settingsPanel.classList.toggle("open");
  });

  // Slider Displays
  const tempSlider = document.getElementById("global-temp");
  const tempVal = document.getElementById("temp-val");
  tempSlider.addEventListener("input", (e) => tempVal.textContent = e.target.value);

  const topPSlider = document.getElementById("global-top-p");
  const topPVal = document.getElementById("top-p-val");
  topPSlider.addEventListener("input", (e) => topPVal.textContent = e.target.value);

  // Tab Navigation
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.dataset.tab;
      
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(targetTab).classList.add("active");
    });
  });

  // Fetch Prompts for Tab 1
  let promptsData = {};
  async function loadPrompts() {
    try {
      const res = await fetch("/api/prompts");
      promptsData = await res.json();
      updatePromptDisplay("story");
    } catch (err) {
      console.error("Failed to load prompts:", err);
    }
  }

  const promptSelect = document.getElementById("prompt-template-select");
  promptSelect.addEventListener("change", (e) => {
    updatePromptDisplay(e.target.value);
  });

  function updatePromptDisplay(key) {
    if (!promptsData[key]) return;
    const item = promptsData[key];
    const cfg = item.config;

    document.getElementById("pillar-role").textContent = cfg.role;
    document.getElementById("pillar-context").textContent = cfg.context;
    document.getElementById("pillar-constraints").textContent = cfg.constraints;
    document.getElementById("pillar-format").textContent = cfg.output_format;
    document.getElementById("compiled-prompt-code").textContent = item.compiled;
  }

  loadPrompts();

  // Copy Buttons
  document.querySelectorAll(".copy-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.dataset.target;
      const text = document.getElementById(targetId).textContent;
      navigator.clipboard.writeText(text);
      btn.textContent = "Copied!";
      setTimeout(() => {
        btn.innerHTML = `<i data-lucide="copy"></i> Copy`;
        lucide.createIcons();
      }, 1500);
    });
  });

  // Helper for API Request Options
  function getRequestOptions(extraBody = {}) {
    return {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: document.getElementById("provider-select").value,
        api_key: document.getElementById("api-key-input").value.trim() || null,
        temperature: parseFloat(document.getElementById("global-temp").value),
        top_p: parseFloat(document.getElementById("global-top-p").value),
        ...extraBody
      })
    };
  }

  // TAB 2: CONTENT GENERATION
  const genBtn = document.getElementById("generate-content-btn");
  const contentResultCard = document.getElementById("content-result-card");
  const contentOutputText = document.getElementById("content-output-text");
  const contentMetaBadge = document.getElementById("content-meta-badge");

  genBtn.addEventListener("click", async () => {
    const topic = document.getElementById("content-topic-input").value.trim();
    const contentType = document.getElementById("content-type-select").value;
    if (!topic) return;

    genBtn.disabled = true;
    genBtn.innerText = "Generating...";

    try {
      const res = await fetch("/api/generate", getRequestOptions({ topic, content_type: contentType }));
      const data = await res.json();

      contentOutputText.textContent = data.text;
      contentMetaBadge.textContent = `Engine: ${data.meta.provider} | Latency: ${data.meta.latency_seconds}s`;
      contentResultCard.classList.remove("hidden");
    } catch (err) {
      alert("Error generating content");
    } finally {
      genBtn.disabled = false;
      genBtn.innerHTML = `<i data-lucide="play"></i> Generate Content`;
      lucide.createIcons();
    }
  });

  // TAB 3: PODCAST PLANNING
  const podBtn = document.getElementById("generate-podcast-btn");
  const podResultCard = document.getElementById("podcast-result-card");
  const podOutputText = document.getElementById("podcast-output-text");
  const podMetaBadge = document.getElementById("podcast-meta-badge");

  podBtn.addEventListener("click", async () => {
    const topic = document.getElementById("podcast-topic-input").value.trim();
    if (!topic) return;

    podBtn.disabled = true;
    podBtn.innerText = "Generating...";

    try {
      const res = await fetch("/api/generate", getRequestOptions({ topic, content_type: "podcast" }));
      const data = await res.json();

      podOutputText.textContent = data.text;
      podMetaBadge.textContent = `Engine: ${data.meta.provider} | Latency: ${data.meta.latency_seconds}s`;
      podResultCard.classList.remove("hidden");
    } catch (err) {
      alert("Error generating podcast plan");
    } finally {
      podBtn.disabled = false;
      podBtn.innerHTML = `<i data-lucide="mic"></i> Generate Podcast Blueprint`;
      lucide.createIcons();
    }
  });

  // TAB 4: TEXT ANALYSIS
  const analyzeBtn = document.getElementById("analyze-text-btn");
  const analysisCard = document.getElementById("analysis-result-card");

  analyzeBtn.addEventListener("click", async () => {
    const text = document.getElementById("analysis-text-input").value.trim();
    if (!text) return;

    analyzeBtn.disabled = true;
    analyzeBtn.innerText = "Analyzing...";

    try {
      const res = await fetch("/api/analyze", getRequestOptions({ text }));
      const result = await res.json();

      if (result.data && result.data.sentiment) {
        const s = result.data.sentiment;
        document.getElementById("sentiment-label-val").textContent = s.label;
        document.getElementById("sentiment-score-val").textContent = (s.score > 0 ? "+" : "") + s.score.toFixed(2);
        document.getElementById("sentiment-explanation-text").textContent = s.explanation;

        const kws = result.data.keywords || [];
        document.getElementById("keywords-count-val").textContent = kws.length;

        // Render CSS Bar Chart for Keywords
        const chartContainer = document.getElementById("keywords-chart-container");
        chartContainer.innerHTML = "";

        kws.forEach(item => {
          const pct = Math.min(100, Math.max(0, item.relevance * 100));
          const el = document.createElement("div");
          el.className = "keyword-bar-item";
          el.innerHTML = `
            <div class="keyword-bar-header">
              <span>${item.keyword}</span>
              <span>${item.relevance.toFixed(2)}</span>
            </div>
            <div class="keyword-bar-track">
              <div class="keyword-bar-fill" style="width: ${pct}%;"></div>
            </div>
          `;
          chartContainer.appendChild(el);
        });
      }

      analysisCard.classList.remove("hidden");
    } catch (err) {
      alert("Error analyzing text");
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = `<i data-lucide="search"></i> Analyze Text`;
      lucide.createIcons();
    }
  });

  // TAB 5: PARAMETER EXPERIMENTATION
  const expBtn = document.getElementById("run-exp-btn");

  expBtn.addEventListener("click", async () => {
    const prompt = document.getElementById("exp-prompt-input").value.trim();
    if (!prompt) return;

    expBtn.disabled = true;
    expBtn.innerText = "Comparing...";

    try {
      const res = await fetch("/api/experiment", getRequestOptions({ prompt }));
      const data = await res.json();

      document.getElementById("exp-res-1").textContent = data.profile1.text;
      document.getElementById("exp-res-2").textContent = data.profile2.text;
      document.getElementById("exp-res-3").textContent = data.profile3.text;
    } catch (err) {
      alert("Error running parameter comparison");
    } finally {
      expBtn.disabled = false;
      expBtn.innerHTML = `<i data-lucide="sliders"></i> Run Parameter Comparison`;
      lucide.createIcons();
    }
  });
});
