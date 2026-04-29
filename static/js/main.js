/* ════════════════════════════════════════════════════════════════
   VISRA — CBIR Frontend Controller
   ════════════════════════════════════════════════════════════════ */

// ── Class colours matching CSS variables ────────────────────────
const CLASS_COLORS = ["#00e5ff","#a78bfa","#34d399","#fb923c","#f472b6"];

// ── DOM refs ─────────────────────────────────────────────────────
const uploadForm   = document.getElementById("upload-form");
const fileInput    = document.getElementById("file-input");
const dropZone     = document.getElementById("drop-zone");
const dzDefault    = document.getElementById("dz-default");
const dzPreview    = document.getElementById("dz-preview");
const previewImg   = document.getElementById("preview-img");
const previewName  = document.getElementById("preview-name");
const browseBtn    = document.getElementById("browse-btn");
const clearBtn     = document.getElementById("clear-btn");
const searchBtn    = document.getElementById("search-btn");
const topkSelect   = document.getElementById("topk-select");

const resultsArea    = document.getElementById("results-area");
const loadingState   = document.getElementById("loading-state");
const resultsContent = document.getElementById("results-content");
const errorState     = document.getElementById("error-state");
const errorMsg       = document.getElementById("error-msg");
const retryBtn       = document.getElementById("retry-btn");
const newSearchBtn   = document.getElementById("new-search-btn");

const resultQueryImg = document.getElementById("result-query-img");
const queryMeta      = document.getElementById("query-meta");
const resultsMeta    = document.getElementById("results-meta");
const resultsGrid    = document.getElementById("results-grid");
const matchBadge     = document.getElementById("match-count-badge");
const distBars       = document.getElementById("dist-bars");

const cardTemplate   = document.getElementById("result-card-tpl");

// Sort tabs
const sortTabs = document.querySelectorAll(".sort-tab");

// ── State ────────────────────────────────────────────────────────
let currentResults = [];
let currentSort    = "rank";
let selectedFile   = null;

// ── File Input ───────────────────────────────────────────────────
browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", (e) => {
  if (!e.target.closest(".btn-clear") && !e.target.closest(".dz-browse")) {
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

clearBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearSelection();
});

// ── Drag & Drop ───────────────────────────────────────────────────
["dragenter","dragover"].forEach(evt =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.add("drag-over");
  })
);
["dragleave","drop"].forEach(evt =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.remove("drag-over");
  })
);
dropZone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  const allowed = ["image/jpeg","image/png","image/bmp","image/webp","image/tiff"];
  if (!allowed.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|bmp|webp|tiff)$/i)) {
    alert("Unsupported file type. Please upload JPG, PNG, BMP, WEBP, or TIFF.");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    alert("File too large. Maximum 16 MB.");
    return;
  }
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src   = e.target.result;
    previewName.textContent = file.name + " · " + formatBytes(file.size);
  };
  reader.readAsDataURL(file);

  dzDefault.hidden = true;
  dzPreview.hidden = false;
  searchBtn.disabled = false;
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  dzDefault.hidden = false;
  dzPreview.hidden = true;
  searchBtn.disabled = true;
  hideResults();
}

// ── Submit ────────────────────────────────────────────────────────
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  showLoading();

  const fd = new FormData();
  fd.append("query_image", selectedFile);
  fd.append("top_k", topkSelect.value);

  try {
    const res  = await fetch("/retrieve", { method:"POST", body: fd });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || `Server error ${res.status}`);
      return;
    }

    currentResults = data.results;
    renderResults(data);
    showResults();
  } catch (err) {
    showError("Network error: " + err.message);
  }
});

// ── Render ────────────────────────────────────────────────────────
function renderResults(data) {
  // Query image
  resultQueryImg.src = data.query_image;
  queryMeta.innerHTML = `
    <div>File: <span style="color:var(--text-1)">${selectedFile.name}</span></div>
    <div>Size: <span style="color:var(--text-1)">${formatBytes(selectedFile.size)}</span></div>
    <div>Dims: <span style="color:var(--text-1)">${data.query_size[0]} × ${data.query_size[1]} px</span></div>
    <div>DB: <span style="color:var(--text-1)">${data.total_db} images searched</span></div>
  `;

  // Meta banner
  const topScore = data.results[0]?.score ?? 0;
  resultsMeta.innerHTML = `Top similarity: <strong style="color:var(--accent)">${(topScore*100).toFixed(1)}%</strong>`;

  matchBadge.textContent = data.results.length;

  renderGrid(data.results);
  renderDistribution(data.results);
}

function renderGrid(results) {
  resultsGrid.innerHTML = "";

  results.forEach((r, i) => {
    const node  = cardTemplate.content.cloneNode(true);
    const card  = node.querySelector(".result-card");
    card.style.animationDelay = `${i * 0.045}s`;

    // Rank
    node.querySelector(".rc-rank-badge").textContent = `#${r.rank}`;

    // Image
    const imgEl  = node.querySelector(".rc-img");
    const noImg  = node.querySelector(".rc-no-img");
    if (r.has_image && r.image_data) {
      imgEl.src = r.image_data;
      imgEl.alt = r.class_name;
      noImg.hidden = true;
    } else {
      imgEl.hidden = true;
    }

    // Class
    const color  = CLASS_COLORS[r.label % CLASS_COLORS.length];
    const dot    = node.querySelector(".rc-class-dot");
    dot.style.background = color;
    dot.style.boxShadow  = `0 0 6px ${color}`;
    node.querySelector(".rc-class-name").textContent = r.class_name;
    node.querySelector(".rc-class-name").style.color = color;

    // Score
    const pct   = Math.max(0, Math.min(1, r.score));
    const scoreVal = node.querySelector(".rc-score-val");
    scoreVal.textContent = (pct * 100).toFixed(1) + "%";
    scoreVal.className = "rc-score-val " +
      (pct > 0.8 ? "score-high" : pct > 0.6 ? "score-med" : "score-low");

    // Score bar
    const bar = node.querySelector(".rc-score-bar");
    // Set initial width 0 for animation
    bar.style.width = "0%";
    bar.style.background = color;
    setTimeout(() => { bar.style.width = (pct * 100).toFixed(1) + "%"; }, 50 + i * 40);

    resultsGrid.appendChild(node);
  });
}

function renderDistribution(results) {
  // Count per class
  const counts = {};
  results.forEach(r => {
    const key = `${r.label}|||${r.class_name}`;
    counts[key] = (counts[key] || 0) + 1;
  });

  const total = results.length;
  distBars.innerHTML = "";

  Object.entries(counts)
    .sort((a,b) => b[1] - a[1])
    .forEach(([key, count]) => {
      const [label, name] = key.split("|||");
      const color = CLASS_COLORS[parseInt(label) % CLASS_COLORS.length];
      const pct   = ((count / total) * 100).toFixed(0);

      const row = document.createElement("div");
      row.className = "dist-row";
      row.innerHTML = `
        <span class="dist-name">${name}</span>
        <div class="dist-bar-wrap">
          <div class="dist-bar" style="width:0%;background:${color};opacity:.7"
               data-target="${pct}%"></div>
        </div>
        <span class="dist-count">${count} / ${total}</span>
        <span class="dist-pct">${pct}%</span>
      `;
      distBars.appendChild(row);
    });

  // Animate bars
  requestAnimationFrame(() => {
    distBars.querySelectorAll(".dist-bar").forEach(bar => {
      bar.style.width = bar.dataset.target;
    });
  });
}

// ── Sort ──────────────────────────────────────────────────────────
sortTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    sortTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    currentSort = tab.dataset.sort;

    let sorted = [...currentResults];
    if (currentSort === "class") {
      sorted.sort((a,b) => a.label - b.label || a.rank - b.rank);
    } else {
      sorted.sort((a,b) => a.rank - b.rank);
    }
    renderGrid(sorted);
  });
});

// ── New Search ────────────────────────────────────────────────────
newSearchBtn.addEventListener("click", () => {
  clearSelection();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

retryBtn.addEventListener("click", () => {
  hideResults();
});

// ── UI State helpers ──────────────────────────────────────────────
function showLoading() {
  resultsArea.hidden = false;
  loadingState.hidden   = false;
  resultsContent.hidden = true;
  errorState.hidden     = true;
}
function showResults() {
  loadingState.hidden   = true;
  resultsContent.hidden = false;
  errorState.hidden     = true;
  resultsArea.hidden    = false;
  resultsContent.scrollIntoView({ behavior:"smooth", block:"start" });
}
function showError(msg) {
  loadingState.hidden   = true;
  resultsContent.hidden = true;
  errorState.hidden     = false;
  resultsArea.hidden    = false;
  errorMsg.textContent  = msg;
}
function hideResults() {
  resultsArea.hidden    = true;
  loadingState.hidden   = true;
  resultsContent.hidden = true;
  errorState.hidden     = true;
}

// ── Util ──────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024)       return bytes + " B";
  if (bytes < 1024*1024)  return (bytes/1024).toFixed(1) + " KB";
  return (bytes/(1024*1024)).toFixed(2) + " MB";
}
