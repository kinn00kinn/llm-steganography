"use strict";

const RESULTS_URL = "./pages/data/phase-results.json";

const state = {
  document: null,
  selectedPhase: 1,
  selectedSample: 0,
};

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = String(text);
  return element;
}

function formatNumber(value) {
  return new Intl.NumberFormat("ja-JP").format(value);
}

function statusLabel(status) {
  return {
    completed: "completed",
    next: "next",
    planned: "planned",
  }[status] ?? status;
}

function renderSummary(resultDocument) {
  document.querySelectorAll("[data-summary]").forEach((element) => {
    const key = element.dataset.summary;
    element.textContent = formatNumber(resultDocument.summary[key]);
  });
}

function renderPhaseRail(phases) {
  const rail = document.querySelector("#phase-rail");
  rail.replaceChildren();

  phases.forEach((phase) => {
    const button = createElement("button", "phase-button");
    button.type = "button";
    button.dataset.phase = String(phase.id);
    button.dataset.status = phase.status;
    button.setAttribute("role", "listitem");
    button.setAttribute("aria-selected", String(phase.id === state.selectedPhase));
    button.setAttribute("aria-label", `Phase ${phase.id}: ${phase.name}, ${statusLabel(phase.status)}`);
    button.append(
      createElement("span", "", String(phase.id).padStart(2, "0")),
      createElement("small", "", statusLabel(phase.status)),
    );
    button.addEventListener("click", () => {
      state.selectedPhase = phase.id;
      renderPhaseRail(phases);
      renderPhaseDetail(phase);
    });
    rail.append(button);
  });
}

function renderPhaseDetail(phase) {
  const detail = document.querySelector("#phase-detail");
  const copy = createElement("div", "phase-copy");
  const status = createElement("span", `status-pill ${phase.status}`, statusLabel(phase.status));
  const heading = createElement("h3", "", `Phase ${phase.id} — ${phase.name}`);
  const summary = createElement("p", "", phase.summary);
  const criterion = createElement("div", "criterion");
  criterion.append(
    createElement("small", "", "EXIT CRITERION"),
    createElement("span", "", phase.exit_criterion),
  );
  copy.append(status, heading, summary, criterion);

  if (phase.commit_url || phase.pull_request_url) {
    const links = createElement("div", "source-links");
    if (phase.pull_request_url) {
      const pullRequest = createElement("a", "", "Pull Request ↗");
      pullRequest.href = phase.pull_request_url;
      links.append(pullRequest);
    }
    if (phase.commit_url) {
      const commit = createElement("a", "", `${phase.commit.slice(0, 7)} ↗`);
      commit.href = phase.commit_url;
      links.append(commit);
    }
    copy.append(links);
  }

  const proof = createElement("div", "phase-proof");
  const proofGrid = createElement("div", "proof-grid");
  phase.evidence.forEach((item) => {
    const card = createElement("div", "proof-item");
    card.append(createElement("small", "", item.label), createElement("strong", "", item.value));
    proofGrid.append(card);
  });

  if (phase.evidence.length === 0) {
    const card = createElement("div", "proof-item");
    card.append(
      createElement("small", "", "EVIDENCE"),
      createElement("strong", "", "Not produced yet"),
    );
    proofGrid.append(card);
  }
  proof.append(proofGrid);

  if (phase.artifacts.length > 0) {
    proof.append(createElement("h4", "artifact-heading", "SOURCE ARTIFACTS"));
    const list = createElement("ul", "artifact-list");
    phase.artifacts.forEach((artifact) => {
      const item = createElement("li", "");
      const link = createElement("a", "");
      link.href = artifact.url;
      link.append(createElement("span", "", artifact.label), createElement("code", "", artifact.path));
      item.append(link);
      list.append(item);
    });
    proof.append(list);
  } else {
    proof.append(
      createElement(
        "p",
        "sample-note",
        "実装・test・検証結果が揃うまで、このフェーズをcompletedにはしません。",
      ),
    );
  }

  detail.replaceChildren(copy, proof);
}

function renderSampleList(samples) {
  const list = document.querySelector("#sample-list");
  list.replaceChildren();
  samples.forEach((sample, index) => {
    const button = createElement("button", "sample-button");
    button.type = "button";
    button.setAttribute("aria-selected", String(index === state.selectedSample));
    button.append(
      createElement("span", "", String(index + 1).padStart(2, "0")),
      createElement("span", ""),
    );
    button.lastElementChild.append(
      createElement("strong", "", sample.label),
      createElement("small", "", `${sample.metrics.frame_bits} bits / ${sample.compression}`),
    );
    button.addEventListener("click", () => {
      state.selectedSample = index;
      renderSampleList(samples);
      renderSample(sample);
    });
    list.append(button);
  });
}

function textStage(label, value) {
  const stage = createElement("div", "text-stage");
  stage.append(createElement("small", "", label), createElement("p", "", value || "(empty)"));
  return stage;
}

function metricStage(label, bytes, bits, maximum) {
  const stage = createElement("div", "metric-stage");
  const fill = createElement("span", "metric-fill");
  const percentage = maximum === 0 ? 0 : Math.max(4, (bytes / maximum) * 100);
  fill.style.setProperty("--metric-width", `${Math.min(percentage, 100)}%`);
  stage.append(
    createElement("small", "", label),
    createElement("strong", "", `${formatNumber(bytes)} bytes`),
    createElement("em", "", `${formatNumber(bits)} bits`),
    fill,
  );
  return stage;
}

function renderSample(sample) {
  const detail = document.querySelector("#sample-detail");
  const titleRow = createElement("div", "sample-title-row");
  titleRow.append(createElement("h3", "", sample.label));
  const badges = createElement("div", "sample-badges");
  badges.append(
    createElement(
      "span",
      "roundtrip-badge",
      sample.exact_match ? "exact round-trip" : "round-trip failed",
    ),
    createElement("span", "compression-badge", sample.compression),
  );
  titleRow.append(badges);

  const flow = createElement("div", "text-flow");
  flow.append(
    textStage("01 / SYNTHETIC INPUT", sample.synthetic_secret),
    createElement("span", "flow-arrow", "→"),
    textStage("02 / NFC NORMALIZED", sample.normalized_text),
    createElement("span", "flow-arrow", "→"),
    textStage("03 / RESTORED", sample.restored_text),
  );

  const metrics = sample.metrics;
  const maximum = Math.max(metrics.raw_bytes, metrics.stored_bytes, metrics.frame_bytes);
  const pipeline = createElement("div", "metric-pipeline");
  pipeline.append(
    metricStage("RAW UTF-8", metrics.raw_bytes, metrics.raw_bits, maximum),
    metricStage(`STORED / ${sample.compression}`, metrics.stored_bytes, metrics.stored_bits, maximum),
    metricStage("VERSIONED FRAME", metrics.frame_bytes, metrics.frame_bits, maximum),
  );

  const frame = createElement("details", "frame-details");
  const summary = createElement(
    "summary",
    "",
    `Frame hexを見る — ${metrics.frame_bytes} bytes / header 10 bytes`,
  );
  frame.append(summary, createElement("code", "", sample.frame_hex));

  const note = createElement(
    "p",
    "sample-note",
    sample.normalization_changed
      ? `NFCにより ${metrics.input_code_points} → ${metrics.normalized_code_points} code pointsへ変化。`
      : `NFC後 ${metrics.normalized_code_points} code points。圧縮で ${metrics.bytes_saved} bytes削減。`,
  );

  detail.replaceChildren(titleRow, flow, pipeline, frame, note);
}

function render(resultDocument) {
  state.document = resultDocument;
  renderSummary(resultDocument);
  renderPhaseRail(resultDocument.phases);
  renderPhaseDetail(resultDocument.phases[state.selectedPhase]);
  renderSampleList(resultDocument.samples);
  renderSample(resultDocument.samples[state.selectedSample]);
}

async function loadResults() {
  try {
    const response = await fetch(RESULTS_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const resultDocument = await response.json();
    render(resultDocument);
  } catch (error) {
    console.error("Failed to load static phase results", error);
    document.querySelector("#load-error").hidden = false;
  }
}

loadResults();
