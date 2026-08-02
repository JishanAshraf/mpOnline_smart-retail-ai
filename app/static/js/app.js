/**
 * Smart Retail AI Platform Dashboard Client JS
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  loadDashboardStats();
  initSentimentAnalyzer();
  initChatbot();
  initVisionUploaders();
});

/* --- Tab Switching Logic --- */
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");

      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(targetTab).classList.add("active");
    });
  });
}

/* --- Dashboard Executive Analytics --- */
async function loadDashboardStats() {
  try {
    const res = await fetch("/dashboard/stats");
    if (!res.ok) return;
    const data = await res.json();

    // 1. Visits Metric
    const visitsEl = document.getElementById("metric-visits");
    if (visitsEl) visitsEl.textContent = data.total_visits.toLocaleString();

    // 2. Sentiment Counts & Progress Bars
    const totalSentiments = (data.sentiment_counts.positive || 0) + 
                            (data.sentiment_counts.negative || 0) + 
                            (data.sentiment_counts.neutral || 0);

    if (totalSentiments > 0) {
      const posPct = Math.round((data.sentiment_counts.positive / totalSentiments) * 100);
      const neuPct = Math.round((data.sentiment_counts.neutral / totalSentiments) * 100);
      const negPct = Math.round((data.sentiment_counts.negative / totalSentiments) * 100);

      document.getElementById("pos-count").textContent = `${data.sentiment_counts.positive} (${posPct}%)`;
      document.getElementById("neu-count").textContent = `${data.sentiment_counts.neutral} (${neuPct}%)`;
      document.getElementById("neg-count").textContent = `${data.sentiment_counts.negative} (${negPct}%)`;

      document.getElementById("bar-pos").style.width = `${posPct}%`;
      document.getElementById("bar-neu").style.width = `${neuPct}%`;
      document.getElementById("bar-neg").style.width = `${negPct}%`;

      document.getElementById("metric-pos-ratio").textContent = `${posPct}% Positive`;
    }

    // 3. Top Intents List
    const intentsList = document.getElementById("top-intents-list");
    if (intentsList && data.top_intents) {
      intentsList.innerHTML = data.top_intents.map(item => `
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
          <span style="font-weight: 600; font-size: 0.9rem;"># ${item.intent}</span>
          <span class="badge badge-primary">${item.count} inquiries</span>
        </div>
      `).join("");
    }
  } catch (err) {
    console.error("Error loading dashboard stats:", err);
  }
}

/* --- Sentiment Analysis Workbench --- */
function initSentimentAnalyzer() {
  const input = document.getElementById("sentiment-input");
  const btn = document.getElementById("btn-analyze-sentiment");
  const resultBox = document.getElementById("sentiment-result");

  // Sample Chips
  document.querySelectorAll(".sentiment-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      input.value = chip.getAttribute("data-text");
      analyzeSentiment();
    });
  });

  if (btn) btn.addEventListener("click", analyzeSentiment);

  async function analyzeSentiment() {
    const text = input.value.trim();
    if (!text) return;

    btn.disabled = true;
    btn.textContent = "Analyzing...";

    try {
      const res = await fetch("/analyze-sentiment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      if (!res.ok) throw new Error("API error");
      const data = await res.json();

      let badgeClass = "badge-neutral";
      if (data.sentiment === "positive") badgeClass = "badge-positive";
      if (data.sentiment === "negative") badgeClass = "badge-negative";

      resultBox.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
          <span class="badge ${badgeClass}">${data.sentiment.toUpperCase()}</span>
          <span style="font-size: 0.85rem; font-weight: 700; color: var(--text-muted);">Confidence: ${(data.confidence * 100).toFixed(0)}%</span>
        </div>
        <p style="font-size: 0.9rem; color: var(--text-dim);">"${data.text}"</p>
      `;
      resultBox.classList.add("active");
    } catch (err) {
      alert("Failed to analyze sentiment. Please try again.");
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span>⚡</span> Analyze Sentiment`;
    }
  }
}

/* --- Interactive FAQ Chatbot --- */
function initChatbot() {
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("btn-send-chat");
  const messagesContainer = document.getElementById("chat-messages");

  // Sample Chat Chips
  document.querySelectorAll(".chat-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      input.value = chip.getAttribute("data-text");
      sendChatMessage();
    });
  });

  if (sendBtn) sendBtn.addEventListener("click", sendChatMessage);
  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendChatMessage();
    });
  }

  async function sendChatMessage() {
    const message = input.value.trim();
    if (!message) return;

    // Append User Message
    appendMessage(message, "user");
    input.value = "";

    try {
      const res = await fetch("/chatbot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, user_id: "demo_user" })
      });

      if (!res.ok) throw new Error("Chatbot API error");
      const data = await res.json();

      // Append Bot Response
      appendMessage(data.message, "bot", data.intent, data.match_type);
    } catch (err) {
      appendMessage("Sorry, I encountered an error connecting to customer support.", "bot", "error", "failed");
    }
  }

  function appendMessage(text, sender, intent = null, matchType = null) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${sender}`;

    const icon = sender === "bot" ? "🤖" : "👤";
    const metaHTML = intent ? `
      <div class="message-meta">
        <span class="badge badge-primary" style="font-size: 0.65rem;">Intent: ${intent}</span>
        <span>${matchType}</span>
      </div>
    ` : "";

    msgDiv.innerHTML = `
      <div class="message-avatar">${icon}</div>
      <div>
        <div class="message-content">${escapeHTML(text)}</div>
        ${metaHTML}
      </div>
    `;

    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
      tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
    );
  }
}

/* --- Drag & Drop Vision Uploaders --- */
function initVisionUploaders() {
  setupUploader("face-dropzone", "face-file-input", "face-preview", "face-result", "/recognize-face", parseFaceResult);
  setupUploader("product-dropzone", "product-file-input", "product-preview", "product-result", "/classify-product", parseProductResult);
}

function setupUploader(dropzoneId, inputId, previewId, resultId, endpoint, resultHandler) {
  const dropzone = document.getElementById(dropzoneId);
  const fileInput = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  const resultBox = document.getElementById(resultId);

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFile(fileInput.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  async function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file.");
      return;
    }

    // Live Thumbnail Preview
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Send API Upload
    const formData = new FormData();
    formData.append("file", file);

    try {
      resultBox.style.display = "block";
      resultBox.innerHTML = `<p style="font-size: 0.85rem; color: var(--text-muted);">AI Model Processing...</p>`;

      const res = await fetch(endpoint, {
        method: "POST",
        body: formData
      });

      if (!res.ok) throw new Error("Vision API Error");
      const data = await res.json();
      resultBox.innerHTML = resultHandler(data);
    } catch (err) {
      resultBox.innerHTML = `<p style="color: var(--danger); font-size: 0.85rem;">Failed to process image.</p>`;
    }
  }
}

function parseFaceResult(data) {
  const isRecognized = data.status === "recognized";
  const badgeClass = isRecognized ? "badge-positive" : "badge-neutral";
  const statusLabel = isRecognized ? "RECOGNIZED" : "NO FACE DETECTED";
  const idLabel = isRecognized ? `ID: ${data.customer_id}` : "Unrecognized Visitor";
  
  return `
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <span class="badge ${badgeClass}">${statusLabel}</span>
      <span style="font-size: 0.85rem; font-weight: 700; color: var(--text-main);">${(data.confidence * 100).toFixed(0)}% Match</span>
    </div>
    <p style="margin-top: 0.5rem; font-size: 0.9rem; font-weight: 700; color: #a5b4fc;">${idLabel}</p>
  `;
}

function parseProductResult(data) {
  return `
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <span class="badge badge-primary">${data.category}</span>
      <span style="font-size: 0.85rem; font-weight: 700; color: var(--text-main);">${(data.confidence * 100).toFixed(0)}% Confidence</span>
    </div>
    <p style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-dim);">MobileNetV2 Inference Complete</p>
  `;
}
