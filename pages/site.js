"use strict";

const RESULTS_URL = "./pages/data/phase-results.json";

const state = {
  selectedPhase: null,
  selectedSample: 0,
};

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = String(text);
  return element;
}

function createExternalLink(label, url) {
  const link = createElement("a", "", label);
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function formatNumber(value) {
  return new Intl.NumberFormat("ja-JP").format(value);
}

function statusLabel(status) {
  return {
    completed: "完了",
    next: "次に検証",
    research: "先行試作",
    planned: "予定",
  }[status] ?? status;
}

function renderOverview(resultDocument) {
  const { phases, summary } = resultDocument;
  const completed = phases.filter((phase) => phase.status === "completed");
  const next = phases.find((phase) => phase.status === "next");
  const lastCompleted = completed.at(-1);

  document.querySelectorAll("[data-summary]").forEach((element) => {
    const key = element.dataset.summary;
    const value = key === "total_phases" ? phases.length : summary[key];
    element.textContent = formatNumber(value);
  });

  document.querySelector("#progress-fill").style.width = `${(completed.length / phases.length) * 100}%`;
  document.querySelector("#last-completed").textContent = lastCompleted
    ? `Phase ${lastCompleted.id} / ${lastCompleted.name}`
    : "未完了";
  document.querySelector("#next-phase").textContent = next
    ? `Phase ${next.id} / ${next.name}`
    : "すべて完了";
}

function makePhaseButton(phase, phases) {
  const button = createElement("button", "phase-button");
  button.type = "button";
  button.dataset.phase = String(phase.id);
  button.setAttribute("aria-pressed", String(phase.id === state.selectedPhase));
  button.setAttribute("aria-label", `Phase ${phase.id}: ${phase.name}、${statusLabel(phase.status)}`);
  button.append(
    createElement("span", "phase-number", String(phase.id).padStart(2, "0")),
    createElement("span", "phase-name", phase.name),
    createElement("span", "phase-state", statusLabel(phase.status)),
  );
  button.addEventListener("click", () => {
    state.selectedPhase = phase.id;
    renderPhaseLists(phases);
    renderPhaseDetail(phase);
  });
  return button;
}

function renderPhaseLists(phases) {
  const activeList = document.querySelector("#active-phase-list");
  const plannedList = document.querySelector("#planned-phase-list");
  const active = phases.filter((phase) => phase.status !== "planned");
  const planned = phases.filter((phase) => phase.status === "planned");

  activeList.replaceChildren(...active.map((phase) => makePhaseButton(phase, phases)));
  plannedList.replaceChildren(...planned.map((phase) => makePhaseButton(phase, phases)));
  document.querySelector("#planned-count").textContent = `(${planned.length})`;
}

function renderPhaseDetail(phase) {
  const detail = document.querySelector("#phase-detail");
  const header = createElement("div", "phase-detail-header");
  const title = createElement("div", "");
  title.append(
    createElement("span", "phase-number", `PHASE ${String(phase.id).padStart(2, "0")}`),
    createElement("h3", "", phase.name),
  );
  header.append(title, createElement("span", `status-pill ${phase.status}`, statusLabel(phase.status)));

  const summary = createElement("p", "phase-summary", phase.summary);
  const criterion = createElement("div", "criterion");
  criterion.append(
    createElement("small", "", "完了条件（満たすまで完了扱いにしません）"),
    createElement("p", "", phase.exit_criterion),
  );

  detail.replaceChildren(header, summary, criterion);

  if (phase.evidence.length > 0) {
    const evidence = createElement("dl", "evidence-list");
    phase.evidence.forEach((item) => {
      const row = createElement("div", "");
      row.append(createElement("dt", "", item.label), createElement("dd", "", item.value));
      evidence.append(row);
    });
    detail.append(evidence);
  }

  if (phase.artifacts.length > 0 || phase.pull_request_url || phase.commit_url) {
    const source = createElement("div", "source-block");
    source.append(createElement("h4", "", "実装・テストの根拠"));
    const links = createElement("div", "source-links");

    phase.artifacts.forEach((artifact) => {
      links.append(createExternalLink(`${artifact.label} ↗`, artifact.url));
    });
    if (phase.pull_request_url) {
      links.append(createExternalLink("Pull Request（変更の審査） ↗", phase.pull_request_url));
    }
    if (phase.commit_url) {
      links.append(createExternalLink(`${phase.commit.slice(0, 7)} ↗`, phase.commit_url));
    }
    source.append(links);
    detail.append(source);
  } else {
    detail.append(
      createElement(
        "p",
        "phase-empty",
        "実装・テスト・検証結果はまだありません。",
      ),
    );
  }
}

function renderSampleList(samples) {
  const list = document.querySelector("#sample-list");
  list.replaceChildren();

  samples.forEach((sample, index) => {
    const button = createElement("button", "sample-button", sample.label);
    button.type = "button";
    button.setAttribute("aria-pressed", String(index === state.selectedSample));
    button.addEventListener("click", () => {
      state.selectedSample = index;
      renderSampleList(samples);
      renderSample(sample);
    });
    list.append(button);
  });
}

function makeTextBox(label, value) {
  const box = createElement("div", "text-box");
  box.append(createElement("small", "", label), createElement("p", "", value || "(empty)"));
  return box;
}

function renderSample(sample) {
  const detail = document.querySelector("#sample-detail");
  const header = createElement("div", "sample-title-row");
  header.append(
    createElement("h3", "", sample.label),
    createElement(
      "span",
      `status-pill ${sample.exact_match ? "completed" : "planned"}`,
      sample.exact_match ? "完全一致" : "復元失敗",
    ),
  );

  const flow = createElement("div", "sample-flow");
  flow.append(
    makeTextBox("公開用の秘密文", sample.synthetic_secret),
    createElement("span", "flow-arrow", "→"),
    makeTextBox("復元結果", sample.restored_text),
  );

  const metrics = createElement("dl", "sample-metrics");
  const metricValues = [
    ["暗号方式", sample.secure_metrics.algorithm],
    [
      "秘密文の大きさ",
      `${formatNumber(sample.metrics.input_code_points)}文字 / ${formatNumber(sample.metrics.raw_bytes)} bytes`,
    ],
    [
      "暗号化後の大きさ",
      `${formatNumber(sample.secure_metrics.frame_bytes)} bytes（固定の付加分 ${formatNumber(sample.secure_metrics.overhead_bytes)} bytes）`,
    ],
  ];
  metricValues.forEach(([label, value]) => {
    const item = createElement("div", "");
    item.append(createElement("dt", "", label), createElement("dd", "", value));
    metrics.append(item);
  });

  detail.replaceChildren(header, flow);

  detail.insertBefore(
    createElement("p", "sample-description", sample.description),
    flow,
  );

  if (sample.normalization_changed) {
    const note = createElement("div", "sample-note");
    note.append(
      createElement("small", "", "NFC正規化後"),
      createElement("p", "", sample.normalized_text),
    );
    detail.append(note);
  }

  detail.append(metrics);

  const frame = createElement("details", "frame-details");
  frame.append(
    createElement(
      "summary",
      "",
      `開発者向け：暗号化前の内部フレーム（圧縮方式 ${sample.compression}）`,
    ),
    createElement("code", "", sample.frame_hex),
  );
  detail.append(frame);
}

function renderComparison(comparison) {
  const container = document.querySelector("#comparison");
  if (!container) return;

  if (comparison.status === "validated") {
    container.classList.remove("pending-card");
    const pill = createElement("span", "status-pill completed", "検証済み");
    const title = createElement("h3", "", comparison.title);
    const desc = createElement("p", "", comparison.reason);
    
    const view = createElement("div", "comparison-view");
    
    const control = createElement("div", "comparison-box");
    control.append(
      createElement("h4", "", "通常生成（秘密なし）"),
      createElement("p", "", comparison.control_text),
    );
    
    const stego = createElement("div", "comparison-box");
    stego.append(
      createElement("h4", "", "埋め込み生成（秘密あり）"),
      createElement("p", "", comparison.stego_text),
    );
    
    view.append(control, stego);
    container.replaceChildren(pill, title, desc, view);
    return;
  }

  container.classList.add("pending-card");
  const requirements = createElement("div", "future-comparison");
  comparison.required_evidence.forEach((item) =>
    requirements.append(createElement("span", "", item)),
  );
  container.replaceChildren(
    createElement("span", "status-pill research", "実モデル検証待ち"),
    createElement("h3", "", comparison.title),
    createElement("p", "", comparison.reason),
    requirements,
  );
}

function render(resultDocument) {
  state.selectedPhase = resultDocument.project.last_completed_phase;
  renderOverview(resultDocument);
  renderPhaseLists(resultDocument.phases);
  const selectedPhase = resultDocument.phases.find(
    (phase) => phase.id === state.selectedPhase,
  );
  renderPhaseDetail(selectedPhase);
  renderSampleList(resultDocument.samples);
  renderSample(resultDocument.samples[state.selectedSample]);
  renderComparison(resultDocument.comparison);
}

async function loadResults() {
  try {
    const response = await fetch(RESULTS_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    console.error("Failed to load static phase results", error);
    document.querySelector("#load-error").hidden = false;
  }
}

loadResults();
