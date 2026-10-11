// The browser sends the resolution text; the server owns preprocessing and the models.
// Two steps inside «Extraer»: load a resolution, then review the proposal paragraph by paragraph.
const $ = (selector) => document.querySelector(selector);
const textInput = $("#resolution-text");
const analyzeButton = $("#extract-button");

const METHOD_ORDER = ["beto", "tfidf", "rule"];
const METHOD_LABELS = { beto: "BETO con ajuste fino", tfidf: "TF-IDF + regresión logística", rule: "Regla de sección" };
const METHOD_HELP = {
  beto: "El más preciso, pero más lento en este equipo.",
  tfidf: "Rápido y sin dependencias adicionales.",
  rule: "Elige todo «Fundamentos jurídicos». Sirve como referencia.",
};
const SECTION_NAMES = {
  encabezado: "Encabezado", antecedentes: "Antecedentes", argumentos: "Argumentos", problema_juridico: "Problema jurídico",
  fundamentos: "Fundamentos jurídicos", por_tanto: "Por tanto", otro: "Otra sección",
};
const REQUEST_TIMEOUT = 120000;

const state = {
  ready: false,
  limits: { min_characters: 1000, max_characters: 400000, min_paragraphs: 5 },
  methods: [], defaultMethod: "tfidf", betoStatus: "",
  activeExample: null, fileName: null,
  analyzedText: "", result: null, gold: null, storageKey: null,
  chosen: new Set(), reviewed: new Set(), undo: [], cursor: null, rows: [],
  controller: null, cancelled: false, returnTo: "load",
};

const numberFormat = new Intl.NumberFormat("es");
const decimal = (value) => new Intl.NumberFormat("es", { minimumFractionDigits: 3, maximumFractionDigits: 3 }).format(value);
const twoDecimals = (value) => new Intl.NumberFormat("es", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
const oneDecimal = (value) => new Intl.NumberFormat("es", { minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(value);
const percent = (value) => new Intl.NumberFormat("es", { style: "percent", maximumFractionDigits: 1 }).format(value);
const count = (n, one, many) => `${numberFormat.format(n)} ${n === 1 ? one : many}`;

function setText(selector, value) {
  $(selector).textContent = value;
}

function showError(selector, message) {
  const node = $(selector);
  node.textContent = message || "";
  node.hidden = !message;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

let toastTimer;
function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 3000);
}

function validationMessage(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    const first = detail[0];
    if (first.type === "string_too_short") return `El texto es demasiado corto: se necesitan al menos ${numberFormat.format(state.limits.min_characters)} caracteres (una resolución completa).`;
    if (first.type === "string_too_long") return `El texto supera el máximo de ${numberFormat.format(state.limits.max_characters)} caracteres.`;
    if (first.type === "extra_forbidden") return "La solicitud incluye campos no permitidos. Solo se acepta el texto y el método.";
    if (first.type === "value_error") return "El texto debe ser mayormente letras: pega la resolución completa en HTML o texto plano.";
  }
  return "La entrada no es válida. Revisa el texto y vuelve a intentarlo.";
}

async function requestJson(url, options = {}, timeout = REQUEST_TIMEOUT) {
  let response;
  try {
    response = await fetch(url, { ...options, signal: options.signal || AbortSignal.timeout(timeout) });
  } catch (error) {
    if (error.name === "AbortError" && state.cancelled) throw error;
    throw new Error(error.name === "TimeoutError" || error.name === "AbortError"
      ? "El análisis tardó demasiado. Prueba con TF-IDF o con un texto más corto."
      : "No hay conexión con el servidor. Revisa que python run.py siga en ejecución.");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 422) throw new Error(validationMessage(data.detail));
    if (response.status === 503) throw new Error(data.detail || "El modelo no está disponible. Consulta el README.");
    throw new Error("No se pudo completar el análisis. Revisa la terminal del servidor.");
  }
  return data;
}

/* ---------- Theme (dark by default; the choice is remembered) ---------- */

function applyTheme(theme) {
  const light = theme === "light";
  if (light) document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
  const toggle = $("#theme-toggle");
  toggle.setAttribute("aria-pressed", String(!light));
  setText("#theme-label", light ? "Modo claro" : "Modo oscuro");
  toggle.title = light ? "Cambiar a modo oscuro" : "Cambiar a modo claro";
}

$("#theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  applyTheme(next);
  try { localStorage.setItem("theme", next); } catch { /* the choice lasts for this visit */ }
});
applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");

/* ---------- Tabs ---------- */

const tabs = [...document.querySelectorAll('[role="tab"]')];

function selectTab(tab, focus = false) {
  tabs.forEach((item) => {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
  });
  document.body.classList.toggle("reviewing", tab.id === "tab-extract" && !$("#view-review").hidden);
  try { history.replaceState(null, "", `#${tab.dataset.hash}`); } catch { /* the hash is a convenience */ }
  if (focus) tab.focus();
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    const target = { ArrowRight: tabs[(index + 1) % tabs.length], ArrowLeft: tabs[(index - 1 + tabs.length) % tabs.length], Home: tabs[0], End: tabs[tabs.length - 1] }[event.key];
    if (target) {
      event.preventDefault();
      selectTab(target, true);
    }
  });
});

/* ---------- Views and steps ---------- */

function showView(name) {
  $("#view-load").hidden = name !== "load";
  $("#view-loading").hidden = name !== "loading";
  $("#view-review").hidden = name !== "review";
  document.body.classList.toggle("reviewing", name === "review");
  $("#step-load").className = name === "review" ? "done" : "current";
  $("#step-review").className = name === "review" ? "current" : "";
  const label = $("#step-load-label");
  label.replaceChildren();
  if (name === "review") {
    const back = element("button", "", "Cargar la resolución");
    back.type = "button";
    back.addEventListener("click", askToLoadAnother);
    label.append(back);
  } else {
    label.textContent = "Cargar la resolución";
  }
}

/* ---------- Step 1: load ---------- */

function inspectText(text) {
  const html = /<\s*(p|li|h[1-4]|div|br|span|b|table)\b/i.test(text);
  const paragraphs = html
    ? (text.match(/<\s*(p|li|h[1-4])\b/gi) || []).length
    : text.split(/\r?\n/).filter((line) => line.trim().length >= 3).length;
  const letters = (text.match(/\p{L}/gu) || []).length;
  return { html, paragraphs, letters, characters: text.length };
}

function validateText() {
  const { min_characters: min, max_characters: max } = state.limits;
  const minParagraphs = state.limits.min_paragraphs || 5;
  const info = inspectText(textInput.value);
  $("#drop-hint").hidden = info.characters > 0;
  setText("#character-count", count(info.characters, "carácter", "caracteres"));
  $("#character-fill").style.width = `${Math.min(100, (info.characters / min) * 100)}%`;

  let kind = "", message, valid = false;
  if (!info.characters) message = "Pega el texto, arrastra un archivo o elige un ejemplo.";
  else if (info.characters > max) { kind = "bad"; message = `El texto supera el máximo de ${numberFormat.format(max)} caracteres. Carga solo una resolución.`; }
  else if (info.characters < min) { kind = "bad"; message = `Faltan ${count(min - info.characters, "carácter", "caracteres")}. Pega la resolución completa, no un extracto.`; }
  else if (info.letters < min / 2) { kind = "bad"; message = "El texto debe ser mayormente letras: pega la resolución en HTML o texto plano."; }
  else if (info.paragraphs < minParagraphs) { kind = "bad"; message = `Se detectaron ${count(info.paragraphs, "párrafo", "párrafos")}; se necesitan al menos ${minParagraphs}. Si es texto plano, deja un párrafo por línea.`; }
  else if (info.html) { kind = "ok"; valid = true; message = `Listo: HTML de la base, unos ${count(info.paragraphs, "párrafo", "párrafos")}.`; }
  else { kind = "info"; valid = true; message = `Texto plano: ${count(info.paragraphs, "párrafo", "párrafos")}, uno por línea. El modelo se entrenó con el HTML de la base, así que la división puede variar.`; }

  const check = $("#text-check");
  check.className = kind ? `check ${kind}` : "check";
  check.textContent = message;
  analyzeButton.disabled = !(valid && state.ready);
  return valid;
}

function loadText(text, fileName = null) {
  textInput.value = text;
  state.fileName = fileName;
  $("#file-chip").hidden = !fileName;
  if (fileName) setText("#file-name", fileName);
  showError("#form-error", "");
  validateText();
}

textInput.addEventListener("input", () => {
  setActiveExample(null);
  showError("#form-error", "");
  validateText();
});
textInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    if (!analyzeButton.disabled) analyze(selectedMethod());
  }
});

function readFile(file) {
  if (!file) return;
  if (file.size > 2_000_000) {
    showError("#form-error", "El archivo supera 2 MB. Carga solo el texto de la resolución.");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    setActiveExample(null);
    loadText(String(reader.result), `${file.name} · ${numberFormat.format(Math.ceil(file.size / 1024))} KB`);
  };
  reader.onerror = () => showError("#form-error", "No se pudo leer el archivo. Usa un .txt o .html con codificación UTF-8.");
  reader.readAsText(file, "utf-8");
}

$("#pick-file").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (event) => {
  readFile(event.target.files[0]);
  event.target.value = "";
});
$("#file-clear").addEventListener("click", () => {
  loadText("");
  textInput.focus();
});
const dropzone = $("#dropzone");
["dragenter", "dragover"].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault();
  dropzone.classList.add("over");
}));
["dragleave", "drop"].forEach((type) => dropzone.addEventListener(type, () => dropzone.classList.remove("over")));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  readFile(event.dataTransfer.files[0]);
});

function selectedMethod() {
  const checked = document.querySelector('input[name="method"]:checked');
  return checked ? checked.value : state.defaultMethod;
}

function renderMethods() {
  const fieldset = $("#method-options");
  fieldset.querySelectorAll(".method-option").forEach((node) => node.remove());
  const select = $("#method-review");
  select.replaceChildren();
  METHOD_ORDER.forEach((method) => {
    const available = state.methods.includes(method);
    let help = METHOD_HELP[method];
    if (!available) help = method === "beto" ? `No está instalado: ${state.betoStatus}` : "Falta el modelo entrenado: ejecuta python -m app.train.";
    else if (method === state.defaultMethod) help += " Recomendado: es el mejor método instalado.";

    const option = element("label", "method-option");
    const input = element("input");
    Object.assign(input, { type: "radio", name: "method", value: method, disabled: !available, checked: method === state.defaultMethod });
    option.append(input, element("strong", "", METHOD_LABELS[method]), element("span", "", help));
    fieldset.append(option);

    const item = element("option", "", available ? METHOD_LABELS[method] : `${METHOD_LABELS[method]} (no disponible)`);
    item.value = method;
    item.disabled = !available;
    select.append(item);
  });
}

function setActiveExample(example) {
  state.activeExample = example;
  document.querySelectorAll(".example").forEach((node) => node.classList.toggle("active", Boolean(example) && node.dataset.id === example.id));
}

function renderExamples(examples) {
  const groups = { resolutions: $("#examples-resolutions"), demo: $("#examples-demo") };
  Object.values(groups).forEach((group) => group.querySelectorAll(".example").forEach((node) => node.remove()));
  examples.forEach((example) => {
    const isResolution = Boolean(example.resolucion);
    const control = element("button", "example");
    control.type = "button";
    control.dataset.id = example.id;
    control.append(element("strong", "", example.title), element("span", "", example.description));
    control.addEventListener("click", () => {
      loadText(example.text);
      setActiveExample(example);
      // Only load the text: the analyst reads it and runs the analysis with «Proponer ratio».
      textInput.scrollTop = 0;
      (analyzeButton.disabled ? textInput : analyzeButton).focus();
    });
    (isResolution ? groups.resolutions : groups.demo).append(control);
  });
  groups.resolutions.hidden = !groups.resolutions.querySelector(".example");
  groups.demo.hidden = !groups.demo.querySelector(".example");
  $("#examples-empty").hidden = examples.length > 0;
  if (!examples.length) setText("#examples-empty", "No hay ejemplos. Ejecuta python -m app.examples y reinicia.");
}

/* ---------- Analysis ---------- */

let elapsedTimer;

async function analyze(method, text = textInput.value) {
  showError("#form-error", "");
  state.returnTo = $("#view-review").hidden ? "load" : "review";
  const info = inspectText(text);
  setText("#loading-text", `Analizando ${info.html ? "unos " : ""}${count(info.paragraphs, "párrafo", "párrafos")} con ${METHOD_LABELS[method]}…`);
  showView("loading");
  const started = performance.now();
  setText("#elapsed", "0,0 s");
  elapsedTimer = setInterval(() => setText("#elapsed", `${oneDecimal((performance.now() - started) / 1000)} s`), 100);

  state.cancelled = false;
  state.controller = new AbortController();
  const timeout = setTimeout(() => state.controller.abort(), REQUEST_TIMEOUT);
  try {
    const result = await requestJson("/api/extract-ratio", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, method }), signal: state.controller.signal,
    });
    openResult(result, text);
  } catch (error) {
    if (state.cancelled) return;
    showView(state.returnTo);
    if (state.returnTo === "load") {
      showError("#form-error", error.message);
    } else {
      toast(error.message);
      $("#method-review").value = state.result.method;
    }
  } finally {
    clearTimeout(timeout);
    clearInterval(elapsedTimer);
  }
}

$("#extract-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!analyzeButton.disabled) analyze(selectedMethod());
});

$("#cancel-button").addEventListener("click", () => {
  state.cancelled = true;
  if (state.controller) state.controller.abort();
  clearInterval(elapsedTimer);
  showView(state.returnTo);
  if (state.returnTo === "review") $("#method-review").value = state.result.method;
  toast(state.returnTo === "load" ? "Análisis cancelado. El texto sigue cargado." : "Análisis cancelado. Sigues con el resultado anterior.");
});

/* ---------- Step 2: review ---------- */

function hashText(text) {
  let hash = 5381;
  for (let index = 0; index < text.length; index += 1) hash = ((hash << 5) + hash + text.charCodeAt(index)) | 0;
  return (hash >>> 0).toString(36);
}

function saveDecisions() {
  try {
    localStorage.setItem(state.storageKey, JSON.stringify({ chosen: [...state.chosen], reviewed: [...state.reviewed] }));
  } catch { /* storage is optional */ }
}

function restoreDecisions() {
  try {
    const saved = JSON.parse(localStorage.getItem(state.storageKey) || "null");
    if (!saved) return false;
    const valid = new Set(state.result.paragraphs.map((item) => item.idx));
    state.chosen = new Set(saved.chosen.filter((idx) => valid.has(idx)));
    state.reviewed = new Set(saved.reviewed.filter((idx) => valid.has(idx)));
    return true;
  } catch {
    return false;
  }
}

function proposal() {
  return new Set(state.result.paragraphs.filter((item) => item.selected).map((item) => item.idx));
}

function openResult(result, text) {
  state.result = result;
  state.analyzedText = text;
  const example = state.activeExample && state.activeExample.text === text ? state.activeExample : null;
  state.gold = example && example.gold_idx ? new Set(example.gold_idx) : null;
  state.storageKey = `ratio-decisions:${hashText(text)}:${result.method}`;
  state.chosen = proposal();
  state.reviewed = new Set();
  state.undo = [];
  state.cursor = null;
  const restored = restoreDecisions();

  const title = example && example.resolucion ? `Resolución ${example.resolucion}` : state.fileName ? "Resolución cargada" : "Resolución pegada";
  setText("#document-title", title);
  const facts = [];
  if (example && example.tipo) facts.push(example.tipo);
  if (state.fileName && !example) facts.push(state.fileName);
  facts.push(count(result.summary.paragraphs, "párrafo", "párrafos"));
  facts.push(result.input_format === "html" ? "HTML de la base" : "Texto plano");
  facts.push(state.gold ? "Ejemplo con ficha" : "Sin ficha");
  $("#case-facts").replaceChildren(...facts.map((fact) => element("span", "fact", fact)));

  $("#method-review").value = result.method;
  const isRule = result.method === "rule";
  const low = Math.max(0, result.threshold - result.review_margin), high = Math.min(1, result.threshold + result.review_margin);
  setText("#method-line", isRule
    ? `${result.method_name}: elige todos los párrafos de «Fundamentos jurídicos». No marca párrafos para revisar.`
    : `${result.method_name} · umbral ${twoDecimals(result.threshold)} elegido en validación · se marcan para revisar los puntajes entre ${twoDecimals(low)} y ${twoDecimals(high)}.`);
  const ruleCount = result.paragraphs.filter((item) => item.rule_selected).length;
  setText("#rule-line", isRule ? "" : `Como referencia, la regla de sección habría elegido ${count(ruleCount, "párrafo", "párrafos")}.`);
  $("#scores-switch").hidden = isRule;
  if (isRule) {
    $("#show-scores").checked = false;
    $("#view-review").classList.remove("show-scores");
  }
  $("#legend-gold").hidden = !state.gold;
  $("#legend-review").hidden = isRule;
  $("#gold-score").hidden = !state.gold;
  $("#mobile-gold").hidden = !state.gold;
  setText("#result-summary", `Análisis listo: ${count(result.summary.selected, "párrafo propuesto", "párrafos propuestos")}, ${count(result.summary.review, "por revisar", "por revisar")}.`);

  buildDocument();
  refresh();
  showView("review");
  window.scrollTo({ top: 0 });
  if (restored) toast("Se recuperaron tus decisiones anteriores sobre este texto. «Volver a la propuesta» empieza de cero.");
}

function buildDocument() {
  const { result } = state;
  const host = $("#paragraphs");
  host.replaceChildren();
  state.rows = [];
  let list = null, section = null;
  result.paragraphs.forEach((item) => {
    if (item.section !== section) {
      section = item.section;
      const head = element("div", "sec");
      head.dataset.section = section;
      head.append(element("h3", "", SECTION_NAMES[section] || section), element("span"));
      list = element("ol", "rows");
      host.append(head, list);
    }
    const row = element("li", "paragraph");
    row.dataset.idx = item.idx;
    if (item.words <= 12) row.classList.add("heading");
    if (state.gold && state.gold.has(item.idx)) row.classList.add("gold");
    if (item.selected) row.classList.add("model");

    const goldbar = element("span", "goldbar");
    if (row.classList.contains("gold")) goldbar.title = "Ratio según la ficha";
    const gutter = element("div", "gutter");
    const top = element("div", "gutter-top");
    const pick = element("button", "pick");
    pick.type = "button";
    pick.innerHTML = '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M2.5 7.5l3 3 6-7" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    pick.addEventListener("click", () => toggle(item.idx));
    pick.addEventListener("focus", () => setCursor(item.idx, false));
    top.append(element("span", "num", `§${item.idx + 1}`), pick);
    gutter.append(top);
    if (result.method !== "rule") {
      const score = element("span", "score");
      score.setAttribute("aria-hidden", "true");
      const track = element("span", "track");
      const fill = element("span", "fill");
      fill.style.width = `${Math.round(item.probability * 100)}%`;
      const tick = element("span", "tick");
      tick.style.left = `calc(${result.threshold * 100}% - 1px)`;
      track.append(fill, tick);
      score.append(element("span", "val", twoDecimals(item.probability)), track);
      gutter.append(score);
    }
    const body = element("div", "paragraph-body");
    body.append(element("p", "paragraph-text", item.text), element("div", "paragraph-meta"));
    row.append(goldbar, gutter, body);
    list.append(row);
    state.rows[item.idx] = row;
  });

  const map = $("#minimap");
  map.replaceChildren();
  map.classList.toggle("dense", result.paragraphs.length > 150);
  result.paragraphs.forEach((item) => {
    const tick = element("button", "mm");
    tick.type = "button";
    tick.tabIndex = -1;
    tick.style.flexGrow = item.words <= 12 ? 0.5 : Math.max(1, item.words / 25);
    tick.setAttribute("aria-label", `Ir al párrafo ${item.idx + 1}`);
    tick.addEventListener("click", () => goTo(item.idx));
    map.append(tick);
  });
}

function pendingReview() {
  return state.result.paragraphs.filter((item) => item.review && !state.reviewed.has(item.idx)).map((item) => item.idx);
}

function editCount() {
  return state.result.paragraphs.filter((item) => state.chosen.has(item.idx) !== item.selected).length;
}

function commit(change) {
  state.undo.push({ chosen: new Set(state.chosen), reviewed: new Set(state.reviewed) });
  if (state.undo.length > 200) state.undo.shift();
  change();
  saveDecisions();
  refresh();
}

function toggle(idx) {
  commit(() => {
    if (state.chosen.has(idx)) state.chosen.delete(idx); else state.chosen.add(idx);
    if (state.result.paragraphs[idx].review) state.reviewed.add(idx);
  });
}

function keepDecision(idx) {
  commit(() => state.reviewed.add(idx));
}

function undo() {
  const previous = state.undo.pop();
  if (!previous) return;
  state.chosen = previous.chosen;
  state.reviewed = previous.reviewed;
  saveDecisions();
  refresh();
  toast("Se deshizo el último cambio.");
}

function setCursor(idx, focus = true) {
  state.cursor = idx;
  state.rows.forEach((row, index) => row.classList.toggle("cursor", index === idx));
  document.querySelectorAll(".mm").forEach((tick, index) => tick.classList.toggle("cur", index === idx));
  if (focus) state.rows[idx].querySelector(".pick").focus({ preventScroll: true });
}

function goTo(idx) {
  const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  state.rows[idx].scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "center" });
  setCursor(idx);
}

function jumpToReview(direction) {
  const pending = pendingReview();
  const list = pending.length ? pending : state.result.paragraphs.filter((item) => item.review).map((item) => item.idx);
  if (!list.length) {
    toast("Este resultado no tiene párrafos por revisar.");
    return;
  }
  const from = state.cursor ?? (direction > 0 ? -1 : state.result.paragraphs.length);
  const target = direction > 0
    ? (list.find((idx) => idx > from) ?? list[0])
    : ([...list].reverse().find((idx) => idx < from) ?? list[list.length - 1]);
  goTo(target);
}

function tag(text, className) {
  return element("span", `tag ${className}`, text);
}

function refresh() {
  const { result, chosen } = state;
  const pending = new Set(pendingReview());
  result.paragraphs.forEach((item) => {
    const row = state.rows[item.idx];
    const isChosen = chosen.has(item.idx);
    const added = isChosen && !item.selected, removed = !isChosen && item.selected;
    row.classList.toggle("selected", isChosen);
    row.classList.toggle("removed", removed);
    row.classList.toggle("pending", pending.has(item.idx));
    const pick = row.querySelector(".pick");
    pick.setAttribute("aria-pressed", String(isChosen));
    pick.setAttribute("aria-label", isChosen ? `Quitar el párrafo ${item.idx + 1} de la ratio` : `Incluir el párrafo ${item.idx + 1} en la ratio`);

    const meta = row.querySelector(".paragraph-meta");
    meta.replaceChildren();
    if (pending.has(item.idx)) {
      const why = element("span", "why", item.selected
        ? `Puntaje ${twoDecimals(item.probability)}, apenas sobre el umbral: el modelo lo incluyó.`
        : `Puntaje ${twoDecimals(item.probability)}, apenas bajo el umbral: el modelo lo dejó fuera.`);
      const keep = element("button", "linkish", item.selected ? "Mantener incluido" : "Mantener fuera");
      keep.type = "button";
      keep.addEventListener("click", () => keepDecision(item.idx));
      meta.append(tag("Por revisar", "pending"), why, keep);
    } else {
      if (added) meta.append(tag("Agregado por ti", "added"));
      if (removed) meta.append(tag("Quitado por ti", "removed"));
      if (item.review && !added && !removed) meta.append(tag("Revisado", "done"));
    }
  });

  document.querySelectorAll(".mm").forEach((tick, index) => {
    tick.classList.toggle("chosen", chosen.has(index));
    tick.classList.toggle("pending", pending.has(index));
  });
  document.querySelectorAll(".sec").forEach((head) => {
    const items = result.paragraphs.filter((item) => item.section === head.dataset.section && item.words > 12);
    const inRatio = items.filter((item) => chosen.has(item.idx)).length;
    head.querySelector("span").textContent = `${count(items.length, "párrafo", "párrafos")} · ${numberFormat.format(inRatio)} en la ratio`;
  });

  const total = result.paragraphs.filter((item) => item.review).length;
  const done = total - pending.size;
  $("#progress").hidden = !total;
  $("#prev-review").disabled = $("#next-review").disabled = !total;
  $("#progress").replaceChildren(element("span", "", "Revisados "), element("strong", "", `${done} de ${total}`));
  const track = element("span", "progress-track");
  track.setAttribute("aria-hidden", "true");
  const fill = element("span", "progress-fill");
  fill.style.width = total ? `${(done / total) * 100}%` : "0";
  track.append(fill);
  $("#progress").append(track);
  $("#progress").classList.toggle("done", done === total);
  renderNotices(total, done);

  const selected = result.paragraphs.filter((item) => chosen.has(item.idx));
  const words = selected.reduce((sum, item) => sum + item.words, 0);
  setText("#count-selected", numberFormat.format(selected.length));
  setText("#count-words", numberFormat.format(words));
  setText("#count-review", total ? `${done}/${total}` : "—");
  setText("#mobile-count", numberFormat.format(selected.length));
  if (state.gold) {
    const f1 = goldF1(selected, words);
    setText("#gold-f1", percent(f1));
    setText("#mobile-f1", percent(f1));
    $("#gold-meter").style.width = `${f1 * 100}%`;
  }

  const preview = $("#ratio-preview");
  preview.replaceChildren();
  if (!selected.length) preview.append(element("li", "empty-ratio", "Ningún párrafo elegido. Marca la casilla junto a un párrafo para agregarlo."));
  selected.forEach((item) => {
    const control = element("button");
    control.type = "button";
    control.append(element("span", "pn", `§${item.idx + 1}`), element("span", "pt", item.text));
    control.addEventListener("click", () => goTo(item.idx));
    const entry = element("li");
    entry.append(control);
    preview.append(entry);
  });
  $("#copy-button").disabled = $("#mobile-copy").disabled = !selected.length;
  setText("#copy-button", selected.length ? `Copiar ratio (${count(selected.length, "párrafo", "párrafos")})` : "Copiar ratio");
  $("#undo-button").disabled = !state.undo.length;
}

function goldF1(selected, selectedWords) {
  const goldWords = state.result.paragraphs.filter((item) => state.gold.has(item.idx)).reduce((sum, item) => sum + item.words, 0);
  const overlap = selected.filter((item) => state.gold.has(item.idx)).reduce((sum, item) => sum + item.words, 0);
  return overlap === 0 ? 0 : (2 * overlap) / (selectedWords + goldWords);
}

function warningKind(text) {
  if (/Fundamentos|Ningún párrafo/.test(text)) return "alert";
  if (/Texto plano|tokens/.test(text)) return "info";
  return "warn";
}

function renderNotices(total, done) {
  const items = state.result.warnings.map((text) => ({ kind: warningKind(text), text }));
  if (total && done < total) items.push({ kind: "warn", strong: `${count(total - done, "párrafo", "párrafos")} cerca del umbral.`, text: " Revísalos antes de copiar la ratio: el modelo duda en ellos.", jump: true });
  if (total && done === total) items.push({ kind: "done", strong: "Revisión completa.", text: " Decidiste todos los párrafos dudosos; la ratio está lista para copiar." });
  const order = { alert: 0, warn: 1, info: 2, done: 3 };
  items.sort((a, b) => order[a.kind] - order[b.kind]);
  const labels = { alert: "Atención", warn: "Por revisar", info: "Información", done: "Listo" };
  $("#notices").replaceChildren(...items.map((item) => {
    const banner = element("div", `banner ${item.kind === "warn" ? "" : item.kind}`);
    const text = element("span");
    if (item.strong) text.append(element("strong", "", item.strong));
    text.append(document.createTextNode(item.text));
    banner.append(element("span", "kind", labels[item.kind]), text);
    if (item.jump) {
      const go = element("button", "btn", "Ir al primero");
      go.type = "button";
      go.addEventListener("click", () => jumpToReview(1));
      banner.append(go);
    }
    return banner;
  }));
}

/* ---------- Copy, reset and confirmations ---------- */

async function copyRatio() {
  const selected = state.result.paragraphs.filter((item) => state.chosen.has(item.idx));
  const text = selected.map((item) => item.text).join("\n\n");
  const words = selected.reduce((sum, item) => sum + item.words, 0);
  let copied = false;
  try {
    await navigator.clipboard.writeText(text);
    copied = true;
  } catch {
    const scratch = element("textarea");
    scratch.value = text;
    scratch.setAttribute("readonly", "");
    scratch.style.position = "fixed";
    scratch.style.opacity = "0";
    document.body.append(scratch);
    scratch.select();
    try { copied = document.execCommand("copy"); } catch { copied = false; }
    scratch.remove();
  }
  toast(copied
    ? `Ratio copiada: ${count(selected.length, "párrafo", "párrafos")}, ${count(words, "palabra", "palabras")}.`
    : "No se pudo copiar. Selecciona los párrafos en el documento y copia con Ctrl+C.");
}

function confirmInline(container, message, confirmLabel, onConfirm) {
  const original = [...container.childNodes];
  const box = element("div", "inline-confirm");
  box.setAttribute("role", "alert");
  const yes = element("button", "btn primary", confirmLabel);
  const no = element("button", "btn", "Cancelar");
  yes.type = no.type = "button";
  const restore = () => container.replaceChildren(...original);
  yes.addEventListener("click", () => { restore(); onConfirm(); });
  no.addEventListener("click", restore);
  box.append(element("span", "", message), yes, no);
  container.replaceChildren(box);
  no.focus();
}

function workSummary() {
  const edits = editCount(), reviews = state.reviewed.size;
  return { any: edits + reviews > 0, text: `${count(edits, "cambio", "cambios")} y ${count(reviews, "revisión", "revisiones")}` };
}

function askToLoadAnother() {
  const back = () => {
    showView("load");
    window.scrollTo({ top: 0 });
    textInput.focus();
  };
  const work = workSummary();
  if (!work.any) {
    back();
    return;
  }
  confirmInline($("#case-actions"), `Tus decisiones (${work.text}) quedan guardadas en este navegador.`, "Cargar otra", back);
}

$("#change-text").addEventListener("click", askToLoadAnother);

$("#method-review").addEventListener("change", (event) => {
  const next = event.target.value;
  event.target.value = state.result.method;
  const run = () => {
    const radio = document.querySelector(`input[name="method"][value="${next}"]`);
    if (radio) radio.checked = true;
    analyze(next, state.analyzedText);
  };
  const work = workSummary();
  if (!work.any) {
    run();
    return;
  }
  confirmInline($("#case-actions"), `Analizar con otro método deja de lado tus ${work.text} (quedan guardados para este método).`, "Analizar de nuevo", run);
});

$("#reset-button").addEventListener("click", () => {
  const work = workSummary();
  if (!work.any) {
    toast("Ya estás viendo la propuesta del modelo.");
    return;
  }
  confirmInline($("#panel-actions"), `Se descartan ${work.text}.`, "Descartar", () => {
    commit(() => {
      state.chosen = proposal();
      state.reviewed = new Set();
    });
    toast("Volviste a la propuesta. Puedes deshacerlo.");
  });
});

$("#copy-button").addEventListener("click", copyRatio);
$("#mobile-copy").addEventListener("click", copyRatio);
$("#mobile-see").addEventListener("click", () => $("#ratio-panel").scrollIntoView({ behavior: "smooth" }));
$("#undo-button").addEventListener("click", undo);
$("#prev-review").addEventListener("click", () => jumpToReview(-1));
$("#next-review").addEventListener("click", () => jumpToReview(1));
$("#show-scores").addEventListener("change", (event) => $("#view-review").classList.toggle("show-scores", event.target.checked));

document.addEventListener("keydown", (event) => {
  if ($("#view-review").hidden || $("#extract").hidden) return;
  const target = event.target;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName) || target.getAttribute("role") === "tab") return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undo();
    return;
  }
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key.toLowerCase();
  const paragraphs = state.result.paragraphs;
  if (key === "j" || key === "k") {
    event.preventDefault();
    goTo(Math.min(paragraphs.length - 1, Math.max(0, (state.cursor ?? -1) + (key === "j" ? 1 : -1))));
  } else if (key === "n" || key === "p") {
    event.preventDefault();
    jumpToReview(key === "n" ? 1 : -1);
  } else if (key === "m" && state.cursor !== null && paragraphs[state.cursor].review && !state.reviewed.has(state.cursor)) {
    event.preventDefault();
    keepDecision(state.cursor);
  } else if (key === " " && state.cursor !== null && target.tagName !== "BUTTON") {
    event.preventDefault();
    toggle(state.cursor);
  }
});

/* ---------- Results tab ---------- */

function cell(tag, text, className) {
  return element(tag, className, text);
}

function table(headers, rows, numericFrom = 1) {
  const node = element("table");
  const head = element("thead");
  const headRow = element("tr");
  headers.forEach((title, index) => headRow.append(cell("th", title, index >= numericFrom ? "num" : "")));
  head.append(headRow);
  const body = element("tbody");
  rows.forEach((values) => {
    const row = element("tr");
    values.forEach((value, index) => row.append(cell("td", value, index >= numericFrom ? "num" : "")));
    body.append(row);
  });
  node.append(head, body);
  return node;
}

function renderMetrics(report) {
  const container = $("#results-content");
  container.replaceChildren();
  const names = report.method_names;
  const methods = Object.keys(report.test);
  const best = methods[methods.length - 1];
  setText("#method-resolutions", numberFormat.format(report.resolutions_total));

  const headline = element("div", "headline-numbers");
  methods.forEach((name) => {
    const result = report.test[name];
    const box = element("div", name === best ? "best" : "");
    box.append(element("span", "", names[name]), element("strong", "", decimal(result.resolution_f1)),
      element("span", "", `F1 por resolución · ${percent(result.share_good)} bien extraídas (F1 ≥ 0,8)`));
    headline.append(box);
  });
  container.append(headline);

  const grid = element("div", "results-grid");
  const main = element("section", "result-block");
  main.append(element("h2", "", "Comparación en la prueba"),
    table(["Método", "F1 por resolución", "IC 95 %", "Bien extraídas", "F1 por párrafo"],
      methods.map((name) => {
        const result = report.test[name];
        return [names[name], decimal(result.resolution_f1), `${decimal(result.resolution_f1_ci95[0])}–${decimal(result.resolution_f1_ci95[1])}`, percent(result.share_good), decimal(result.paragraph.f1)];
      })),
    element("p", "small", `Prueba: ${numberFormat.format(report.partitions.test.resolutions)} resoluciones de ${report.partitions.test.years.join("–")}. F1 ponderado por palabras frente a la ratio de la ficha, promediado por resolución; intervalo por bootstrap.`));
  const comparisons = Object.entries(report.comparisons);
  if (comparisons.length) {
    main.append(element("h2", "", "Comparación pareada (McNemar)"),
      table(["Comparación", "Solo el primero", "Solo el segundo", "p"], comparisons.map(([key, value]) => {
        const [first, second] = key.split("_vs_");
        return [`${names[first]} vs ${names[second]}`, String(value.only_first), String(value.only_second), value.p_value < 0.001 ? "< 0,001" : decimal(value.p_value)];
      })),
      element("p", "small", "Resoluciones bien extraídas (F1 ≥ 0,8) por un método y no por el otro."));
  }
  grid.append(main);

  [["by_type", "Por tipo de resolución", "Tipo"], ["by_analyst", "Por analista de la ficha", "Analista (idusuario)"]].forEach(([key, title, label]) => {
    const block = element("section", "result-block");
    block.append(element("h2", "", `${title} · ${names[best]}`),
      table([label, "Resoluciones", "F1"], report.test[best][key].map((row) => [row.group, String(row.resolutions), decimal(row.resolution_f1)])));
    if (key === "by_analyst") block.append(element("p", "small", "Las diferencias entre analistas reflejan, en parte, criterios distintos al seleccionar la ratio."));
    grid.append(block);
  });

  const selection = report.tfidf_selection;
  const selectionBlock = element("section", "result-block");
  selectionBlock.append(element("h2", "", "Selección de TF-IDF en validación"),
    table(["Contexto vecino", "C", "Umbral", "F1 validación"], selection.candidates.map((row) => [row.use_context ? "sí" : "no", String(row.c), decimal(row.threshold), decimal(row.validation_resolution_f1)])));
  selectionBlock.append(element("p", "small", report.beto
    ? `BETO: ${report.beto.base_model}, ${report.beto.epochs} épocas, umbral ${decimal(report.beto.threshold)} elegido en validación.`
    : "BETO aún no se ha evaluado en este repositorio (ver README)."));
  grid.append(selectionBlock);
  container.append(grid);
}

/* ---------- Start ---------- */

async function init() {
  const initialTab = tabs.find((tab) => `#${tab.dataset.hash}` === location.hash);
  if (initialTab) selectTab(initialTab);
  showView("load");
  validateText();
  const status = $("#connection-status");
  try {
    const health = await requestJson("/api/health", {}, 15000);
    state.limits = { ...state.limits, ...health.limits };
    state.methods = health.methods;
    state.defaultMethod = health.default_method;
    state.betoStatus = health.beto_status;
    setText("#character-limits", `mín. ${numberFormat.format(state.limits.min_characters)} · máx. ${numberFormat.format(state.limits.max_characters)}`);
    renderMethods();
    status.textContent = health.status === "ok" ? `Servidor listo · ${METHOD_LABELS[health.default_method]}` : "Falta el modelo entrenado: ejecuta python -m app.train";
    status.className = `status-chip ${health.status === "ok" ? "ok" : "problem"}`;
    state.ready = true;
    validateText();
    renderExamples(await requestJson("/api/examples", {}, 15000));
    renderMetrics(await requestJson("/api/metrics", {}, 15000));
  } catch (error) {
    showError("#global-error", error.message);
    if (!state.ready) {
      status.textContent = "Sin conexión con el servidor";
      status.className = "status-chip problem";
    }
  }
}

init();
