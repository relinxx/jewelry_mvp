const form = document.querySelector("#searchForm");
const imageInput = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const previewFrame = document.querySelector("#previewFrame");
const previewImage = document.querySelector("#previewImage");
const clearImage = document.querySelector("#clearImage");
const limitInput = document.querySelector("#limitInput");
const limitOutput = document.querySelector("#limitOutput");
const resultsGrid = document.querySelector("#resultsGrid");
const resultsTitle = document.querySelector("#resultsTitle");
const messageState = document.querySelector("#messageState");
const resetResults = document.querySelector("#resetResults");
const healthStatus = document.querySelector("#healthStatus");
const resultTemplate = document.querySelector("#resultTemplate");

let previewUrl = null;

function setHealth(state, label) {
  healthStatus.classList.remove("ok", "error");
  healthStatus.classList.add(state);
  healthStatus.querySelector("span:last-child").textContent = label;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");

    if (!response.ok) {
      throw new Error("Health check failed");
    }

    setHealth("ok", "API ready");
  } catch {
    setHealth("error", "API offline");
  }
}

function setMessage(message, type = "idle") {
  messageState.hidden = false;
  messageState.classList.toggle("error", type === "error");
  messageState.querySelector("p").textContent = message;
}

function clearResults() {
  resultsGrid.replaceChildren();
}

function resetSearchState() {
  clearResults();
  resultsTitle.textContent = "Ready to search";
  setMessage("Upload a jewelry image to compare it with the catalog.");
}

function setPreview(file) {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewFrame.hidden = false;
}

function clearPreview() {
  imageInput.value = "";
  previewFrame.hidden = true;
  previewImage.removeAttribute("src");

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function setSelectedFile(file) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  setPreview(file);
}

function renderResults(results) {
  clearResults();
  messageState.hidden = true;

  for (const result of results) {
    const node = resultTemplate.content.cloneNode(true);
    const image = node.querySelector(".result-image");

    image.src = result.image_url;
    image.alt = `${result.category} ${result.km_code}`;
    node.querySelector(".result-category").textContent = result.category;
    node.querySelector(".result-code").textContent = result.km_code;
    node.querySelector(".result-score").textContent = Number(result.similarity).toFixed(4);

    const downloadBtn = node.querySelector(".result-download-btn");
    if (downloadBtn) {
      downloadBtn.href = result.image_url;
      downloadBtn.download = `${result.km_code}.jpg`;
    }

    resultsGrid.append(node);
  }
}

async function submitSearch(event) {
  event.preventDefault();

  const file = imageInput.files[0];

  if (!file) {
    setMessage("Choose a jewelry image before searching.", "error");
    return;
  }

  const formData = new FormData(form);
  const button = form.querySelector(".search-button");

  button.disabled = true;
  resultsTitle.textContent = "Searching catalog";
  clearResults();
  setMessage("Generating embedding and ranking catalog items.");

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(payload.detail || "Search failed.");
    }

    if (!payload.results.length) {
      resultsTitle.textContent = "No matches returned";
      setMessage("The catalog search completed without returning items.");
      return;
    }

    resultsTitle.textContent = `${payload.results.length} matches for ${payload.query.filename}`;
    renderResults(payload.results);
  } catch (error) {
    resultsTitle.textContent = "Search unavailable";
    setMessage(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];

  if (file) {
    setPreview(file);
  }
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");

  const file = event.dataTransfer.files[0];

  if (file) {
    setSelectedFile(file);
  }
});

clearImage.addEventListener("click", clearPreview);

limitInput.addEventListener("input", () => {
  limitOutput.textContent = limitInput.value;
});

resetResults.addEventListener("click", resetSearchState);
form.addEventListener("submit", submitSearch);
checkHealth();
