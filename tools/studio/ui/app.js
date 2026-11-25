/* Omega Helix Studio — Zero-Dependency Offline Client */
(() => {
  "use strict";

  const STRAND_COLORS = ["#22d3ee", "#a78bfa", "#34d399", "#fbbf24"];
  const REDUCED_MOTION = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const PRESETS = [
    {
      id: "fast-nano",
      label: "Fast Starter Nano",
      patch: {
        name: "starter-nano",
        mode: "dna",
        arch: "custom",
        tokenizer: "byte",
        V: 256, T: 16, D: 32, H: 2, H_KV: 2, F: 64, L: 2, B: 2,
        LP: 0, LC: 0, NP: 1, adapt_thresh_micro: 0, msg_gain_micro: 0,
        base_steps: 12, lr_micro: 3000, spclip_mode: 1,
        export_hf: false, export_gguf: false,
      },
    },
    {
      id: "qwen3-rope",
      label: "Qwen3 RoPE+GQA",
      patch: {
        name: "qwen3-waldo",
        mode: "universal",
        arch: "qwen3",
        tokenizer: "bpe",
        V: 512, T: 32, D: 64, H: 4, H_KV: 2, F: 128, L: 4, B: 2,
        LP: 0, LC: 0, NP: 1, adapt_thresh_micro: 0, msg_gain_micro: 0,
        base_steps: 20, lr_micro: 2500, spclip_mode: 1,
        export_hf: true, hf_dtype: "bf16", export_gguf: true,
      },
    },
    {
      id: "spark-15m",
      label: "Spark 15.8M DNA",
      patch: {
        name: "spark-15m-dna",
        mode: "hybrid",
        arch: "custom",
        tokenizer: "bpe",
        V: 2048, T: 128, D: 384, H: 6, H_KV: 6, F: 1024, L: 8, B: 2,
        LP: 0, LC: 0, NP: 1, adapt_thresh_micro: 0, msg_gain_micro: 0,
        base_steps: 25, lr_micro: 2000, spclip_mode: 1,
        export_hf: true, hf_dtype: "f32", export_gguf: true,
      },
    },
    {
      id: "loop-adapt",
      label: "Looped Core + Adaptive Exit",
      patch: {
        name: "looped-adapt",
        mode: "dna",
        arch: "custom",
        tokenizer: "byte",
        V: 256, T: 32, D: 48, H: 4, H_KV: 2, F: 96, L: 4, B: 2,
        LP: 1, LC: 2, NP: 3, adapt_thresh_micro: 25000, msg_gain_micro: 150000,
        base_steps: 16, lr_micro: 2500, spclip_mode: 1,
        export_hf: false, export_gguf: false,
      },
    },
  ];

  const STAGE_DEFS = [
    { id: "openwaldo", label: "OpenWALDO Verify" },
    { id: "tokenize", label: "Tokenize" },
    { id: "base", label: "Base Trunk" },
    { id: "strands", label: "Merkle Strands" },
    { id: "spext", label: "SPEXT Merkle Lock" },
    { id: "export", label: "HF / GGUF / BOM" },
  ];

  const state = {
    system: null,
    catalog: { total: 0, count: 0, corpora: [], facets: { categories: [], licenses: [], prog_languages: [] } },
    filters: { q: "", cat: "", lic: "", prog: "" },
    spec: {
      name: "my-helix",
      mode: "dna",
      arch: "custom",
      tokenizer: "byte",
      V: 256, T: 16, D: 32, H: 2, H_KV: 2, F: 64, L: 2, B: 2,
      LP: 0, LC: 0, NP: 1,
      adapt_thresh_micro: 0,
      msg_gain_micro: 0,
      use_gpu: false,
      base_steps: 15,
      lr_micro: 3000,
      base_dataset: { kind: "preset", value: "omega-starter", label: "omega-starter (repo corpus)" },
      strands: [
        {
          id: "strand-a",
          name: "Code",
          color: STRAND_COLORS[0],
          add_layers: 1,
          growth: "growpro",
          use_fcachem: true,
          steps: 12,
          lr_micro: 2000,
          weight: 500,
          dataset: { kind: "preset", value: "omega-starter", label: "omega-starter" },
        },
        {
          id: "strand-b",
          name: "Reasoning",
          color: STRAND_COLORS[1],
          add_layers: 1,
          growth: "growpro",
          use_fcachem: true,
          steps: 12,
          lr_micro: 2000,
          weight: 500,
          dataset: { kind: "preset", value: "omega-starter", label: "omega-starter" },
        },
      ],
      spclip_mode: 1,
      export_hf: false,
      hf_dtype: "f32",
      export_gguf: false,
    },
    estimate: null,
    runs: [],
    activeRunId: null,
    activeRun: null,
    es: null,
    logLines: [],
    soundOn: false,
    audioCtx: null,
    lastStageAudio: "",
    playWeights: {},
    playOpts: { n_tokens: 48, temp_milli: 700, seed: 42, adapt_thresh_micro: 0, msg_gain_micro: 0 },
  };

  /* ---------------- Utilities ---------------- */
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  };
  const svgEl = (tag, attrs = {}) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    return n;
  };
  const stripAnsi = (s) => String(s || "").replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
  const fmtNum = (n) => (typeof n === "number" && Number.isFinite(n) ? n.toLocaleString("en-US") : "—");
  const fmtBytes = (b) => {
    if (typeof b !== "number" || !Number.isFinite(b)) return "—";
    if (b < 1024) return `${b} B`;
    if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
    return `${(b / 1048576).toFixed(2)} MB`;
  };
  const announce = (msg) => { $("sr-status").textContent = msg; };

  function toast(msg, kind = "info") {
    const box = $("toasts");
    const t = el("div", `toast ${kind}`, msg);
    box.appendChild(t);
    announce(msg);
    setTimeout(() => t.remove(), 4200);
  }

  async function api(url, opts) {
    const res = await fetch(url, opts);
    const data = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
    if (!res.ok && data.ok === undefined) data.ok = false;
    return data;
  }

  /* ---------------- Web Audio Synth ---------------- */
  function ensureAudio() {
    if (!state.soundOn) return null;
    if (!state.audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) state.audioCtx = new AC();
    }
    if (state.audioCtx && state.audioCtx.state === "suspended") state.audioCtx.resume();
    return state.audioCtx;
  }

  function tone(freq, dur = 0.07, type = "sine", vol = 0.05, delay = 0) {
    const ctx = ensureAudio();
    if (!ctx) return;
    const t0 = ctx.currentTime + delay;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t0);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(vol, t0 + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.connect(g).connect(ctx.destination);
    osc.start(t0);
    osc.stop(t0 + dur + 0.02);
  }

  const sfx = {
    tap: () => tone(580, 0.045, "triangle", 0.035),
    assign: () => { tone(523.25, 0.07, "sine", 0.045, 0); tone(659.25, 0.09, "sine", 0.045, 0.055); },
    milestone: () => { tone(440, 0.09, "triangle", 0.05, 0); tone(554.37, 0.11, "triangle", 0.05, 0.07); },
    celebrate: () => {
      [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => tone(f, 0.24, "sine", 0.06, i * 0.08));
    },
    err: () => { tone(220, 0.14, "sawtooth", 0.04, 0); tone(185, 0.18, "sawtooth", 0.035, 0.09); },
  };

  function celebrateBurst() {
    if (REDUCED_MOTION) return;
    const svg = $("burst");
    const w = window.innerWidth;
    const h = window.innerHeight;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.replaceChildren();
    const cx = w * 0.5;
    const cy = h * 0.34;
    for (let i = 0; i < 28; i++) {
      const a = (Math.PI * 2 * i) / 28;
      const dist = 80 + (i % 4) * 55;
      const c = svgEl("circle", {
        cx, cy, r: 3.5 + (i % 3),
        fill: STRAND_COLORS[i % STRAND_COLORS.length],
        opacity: "0.95",
      });
      svg.appendChild(c);
      c.animate(
        [
          { transform: "translate(0px, 0px) scale(1)", opacity: 0.95 },
          { transform: `translate(${Math.cos(a) * dist}px, ${Math.sin(a) * dist}px) scale(0.2)`, opacity: 0 },
        ],
        { duration: 820, easing: "cubic-bezier(0.16, 1, 0.3, 1)", fill: "forwards" }
      );
    }
    setTimeout(() => svg.replaceChildren(), 900);
  }

  /* ---------------- Animated 4-Strand DNA Header ---------------- */
  function initHelixAnimation() {
    const svg = $("helix-svg");
    if (!svg) return;
    const W = 640, H = 200;
    const layers = [];
    for (let i = 0; i < 4; i++) {
      layers.push(svgEl("path", { fill: "none", stroke: STRAND_COLORS[i], "stroke-width": 2.8, "stroke-linecap": "round", opacity: 0.9, filter: "url(#f-glow)" }));
    }
    const rungs = [];
    for (let x = 28; x <= W - 28; x += 28) {
      const r = svgEl("line", { x1: x, x2: x, y1: 100, y2: 100, stroke: "rgba(148,163,184,0.28)", "stroke-width": 1.6 });
      rungs.push({ x, el: r });
      svg.appendChild(r);
    }
    layers.forEach((p) => svg.appendChild(p));
    const dots = STRAND_COLORS.map((c) => {
      const d = svgEl("circle", { r: 4.5, fill: c, filter: "url(#f-glow)" });
      svg.appendChild(d);
      return d;
    });

    function draw(phase) {
      const pts = [[], [], [], []];
      for (let x = 16; x <= W - 16; x += 12) {
        for (let s = 0; s < 4; s++) {
          const y = H / 2 + Math.sin(x * 0.022 + phase + (s * Math.PI) / 2) * 58;
          pts[s].push(`${x === 16 ? "M" : "L"}${x} ${y.toFixed(1)}`);
        }
      }
      layers.forEach((p, s) => p.setAttribute("d", pts[s].join(" ")));
      rungs.forEach(({ x, el: line }) => {
        const y1 = H / 2 + Math.sin(x * 0.022 + phase) * 58;
        const y2 = H / 2 + Math.sin(x * 0.022 + phase + Math.PI) * 58;
        line.setAttribute("y1", y1.toFixed(1));
        line.setAttribute("y2", y2.toFixed(1));
      });
      const mx = W * 0.72;
      dots.forEach((d, s) => {
        const y = H / 2 + Math.sin(mx * 0.022 + phase + (s * Math.PI) / 2) * 58;
        d.setAttribute("cx", mx);
        d.setAttribute("cy", y.toFixed(1));
      });
    }

    draw(0.4);
    if (REDUCED_MOTION) return;
    let phase = 0.4;
    const tick = () => {
      phase += 0.024;
      draw(phase);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /* ---------------- Accessible Tabs ---------------- */
  function activateTab(tabId, focusTab = false) {
    const tabs = Array.from(document.querySelectorAll('.tabs [role="tab"]'));
    tabs.forEach((t) => {
      const on = t.id === tabId;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      const p = $(t.getAttribute("aria-controls"));
      if (p) p.hidden = !on;
      if (on && focusTab) t.focus();
    });
    if (tabId === "tab-design") refreshEstimate();
    if (tabId === "tab-forge") refreshRuns();
    if (tabId === "tab-play") refreshRuns(true);
    if (tabId === "tab-distill") refreshDistillRuns();
  }

  function bindTabs() {
    const tablist = document.querySelector('.tabs[role="tablist"]');
    const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
    tabs.forEach((t, idx) => {
      t.addEventListener("click", () => { sfx.tap(); activateTab(t.id); });
      t.addEventListener("keydown", (e) => {
        let next = null;
        if (e.key === "ArrowRight") next = tabs[(idx + 1) % tabs.length];
        else if (e.key === "ArrowLeft") next = tabs[(idx - 1 + tabs.length) % tabs.length];
        else if (e.key === "Home") next = tabs[0];
        else if (e.key === "End") next = tabs[tabs.length - 1];
        if (next) {
          e.preventDefault();
          sfx.tap();
          activateTab(next.id, true);
        }
      });
    });
  }

  /* ---------------- Tab 1: Discover Data ---------------- */
  async function loadSystem() {
    const sys = await api("/api/system");
    state.system = sys;
    if (sys.ok) {
      const gpu = sys.has_gb10 ? "GB10 GPU" : "CPU";
      const waldo = sys.has_waldo_cli ? "waldo CLI" : "index cache";
      $("sys-text").textContent = `${sys.os}/${sys.arch} · ${gpu} · ${waldo}`;
      const bins = Object.entries(sys.binaries || {}).filter(([, v]) => v).map(([k]) => k);
      $("foot-bin").textContent = bins.length ? `Binaries: ${bins.join(", ")}` : "";
    } else {
      $("sys-text").textContent = "host ready";
    }
  }

  async function loadCatalog() {
    const p = new URLSearchParams();
    if (state.filters.q) p.set("q", state.filters.q);
    if (state.filters.cat) p.set("cat", state.filters.cat);
    if (state.filters.lic) p.set("lic", state.filters.lic);
    if (state.filters.prog) p.set("prog", state.filters.prog);
    const data = await api(`/api/catalog?${p.toString()}`);
    if (!data.ok) {
      $("cat-meta").textContent = data.error || "Failed to load catalog.";
      return;
    }
    state.catalog = data;
    renderCatalogFacets();
    renderCatalogCards();
  }

  function renderChipGroup(containerId, items, activeVal, onSelect, maxItems = 18) {
    const box = $(containerId);
    box.replaceChildren();
    const allBtn = el("button", "chip", "All");
    allBtn.type = "button";
    allBtn.setAttribute("aria-pressed", activeVal === "" ? "true" : "false");
    allBtn.addEventListener("click", () => { sfx.tap(); onSelect(""); });
    box.appendChild(allBtn);

    (items || []).slice(0, maxItems).forEach((item) => {
      const name = typeof item === "string" ? item : item.name;
      const count = typeof item === "object" && item.count !== undefined ? ` (${item.count})` : "";
      const b = el("button", "chip", `${name}${count}`);
      b.type = "button";
      b.setAttribute("aria-pressed", activeVal === name ? "true" : "false");
      b.addEventListener("click", () => { sfx.tap(); onSelect(activeVal === name ? "" : name); });
      box.appendChild(b);
    });
  }

  function renderCatalogFacets() {
    const f = state.catalog.facets || {};
    renderChipGroup("cat-pills", f.categories, state.filters.cat, (v) => { state.filters.cat = v; loadCatalog(); }, 16);
    renderChipGroup("lic-chips", f.licenses, state.filters.lic, (v) => { state.filters.lic = v; loadCatalog(); }, 16);
    renderChipGroup("prog-chips", f.prog_languages, state.filters.prog, (v) => { state.filters.prog = v; loadCatalog(); }, 16);
    const active = [state.filters.cat, state.filters.lic, state.filters.prog].filter(Boolean);
    $("facet-summary").textContent = active.length ? `active: ${active.join(" · ")}` : "";
  }

  function isAssignedBase(path) {
    return state.spec.base_dataset && state.spec.base_dataset.kind === "waldo" && state.spec.base_dataset.value === path;
  }

  function isAssignedStrand(path) {
    return state.spec.strands.some((s) => s.dataset && s.dataset.kind === "waldo" && s.dataset.value === path);
  }

  function assignBaseDataset(kind, value, label) {
    state.spec.base_dataset = { kind, value, label };
    renderDataSlots();
    renderCatalogCards();
    refreshEstimate();
    sfx.assign();
    toast(`Base trunk set to ${label}`, "ok");
  }

  function addDatasetAsStrand(kind, value, label, suggestedName) {
    if (state.spec.strands.length >= 4) {
      toast("spclip fuses up to 4 strands — replace or remove one in Design.", "err");
      sfx.err();
      return;
    }
    const idx = state.spec.strands.length;
    const cleanName = (suggestedName || value.split("/").pop() || `Strand ${idx + 1}`).slice(0, 18);
    state.spec.strands.push({
      id: `strand-${Date.now().toString(36).slice(-4)}`,
      name: cleanName,
      color: STRAND_COLORS[idx % STRAND_COLORS.length],
      add_layers: 1,
      growth: "growpro",
      use_fcachem: true,
      steps: 12,
      lr_micro: 2000,
      weight: 500,
      dataset: { kind, value, label },
    });
    renderDataSlots();
    renderStrands();
    renderCatalogCards();
    refreshEstimate();
    sfx.assign();
    toast(`Added strand "${cleanName}" (${label})`, "ok");
  }

  function renderDataSlots() {
    $("slot-base-v").textContent = state.spec.base_dataset.label || state.spec.base_dataset.value;
    const ul = $("slot-strands");
    ul.replaceChildren();
    if (!state.spec.strands.length) {
      ul.appendChild(el("li", "muted small", "No strands assigned (Universal mode)."));
      return;
    }
    state.spec.strands.forEach((st, idx) => {
      const li = el("li", "slot-item");
      const sw = el("span", "sw");
      sw.style.background = st.color;
      const meta = el("span", "meta", `${st.name}: ${st.dataset.label || st.dataset.value}`);
      const rm = el("button", "btn btn-sm btn-ghost", "×");
      rm.type = "button";
      rm.title = `Remove ${st.name}`;
      rm.setAttribute("aria-label", `Remove ${st.name}`);
      rm.addEventListener("click", () => {
        sfx.tap();
        state.spec.strands.splice(idx, 1);
        renderDataSlots();
        renderStrands();
        renderCatalogCards();
        refreshEstimate();
      });
      li.append(sw, meta, rm);
      ul.appendChild(li);
    });
  }

  function renderCatalogCards() {
    const list = state.catalog.corpora || [];
    $("cat-meta").textContent = `Showing ${list.length} of ${state.catalog.total || list.length} verified OpenWALDO corpora`;
    const grid = $("cat-cards");
    grid.replaceChildren();
    if (!list.length) {
      grid.appendChild(el("p", "muted", "No matching corpora. Clear filters or sync the live index."));
      return;
    }
    list.forEach((c) => {
      const asBase = isAssignedBase(c.path);
      const asStrand = isAssignedStrand(c.path);
      const card = el("article", `cat-card${asBase ? " is-base" : ""}${asStrand ? " is-strand" : ""}`);

      const top = el("div", "card-top");
      const head = el("div");
      head.appendChild(el("h3", "card-title", c.title || c.name));
      head.appendChild(el("div", "card-path", c.path));
      top.appendChild(head);
      if (asBase || asStrand) {
        top.appendChild(el("span", "badge assigned", asBase && asStrand ? "Base + Strand" : asBase ? "Base" : "Strand"));
      }
      card.appendChild(top);

      card.appendChild(el("p", "card-desc", c.description || ""));

      const badges = el("div", "badges");
      if (c.category) badges.appendChild(el("span", "badge cat", c.category));
      const licText = c.license_summary || (c.licenses && c.licenses.slice(0, 2).join(", ")) || "";
      if (licText) badges.appendChild(el("span", "badge lic", licText));
      (c.programming_languages || []).slice(0, 3).forEach((pl) => badges.appendChild(el("span", "badge lang", pl)));
      if ((c.programming_languages || []).length > 3) {
        badges.appendChild(el("span", "badge lang", `+${c.programming_languages.length - 3}`));
      }
      if (c.sources_count) badges.appendChild(el("span", "badge", `${c.sources_count} src`));
      card.appendChild(badges);

      const actions = el("div", "card-actions");
      const bBase = el("button", "btn btn-sm btn-ghost", asBase ? "✓ Base Trunk" : "Assign to Base");
      bBase.type = "button";
      bBase.addEventListener("click", () => assignBaseDataset("waldo", c.path, c.title || c.path));

      const bStrand = el("button", "btn btn-sm btn-primary", "+ Add as Strand Dataset");
      bStrand.type = "button";
      bStrand.addEventListener("click", () => addDatasetAsStrand("waldo", c.path, c.title || c.path, c.name));
      actions.append(bBase, bStrand);
      card.appendChild(actions);
      grid.appendChild(card);
    });
  }

  async function inspectDataset(payload) {
    const out = $("inspect-out");
    out.replaceChildren(el("div", "inspect-card", "Inspecting…"));
    const res = await api("/api/dataset/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    out.replaceChildren();
    if (!res.ok) {
      out.appendChild(el("div", "inspect-card err", res.error || "Could not inspect dataset."));
      sfx.err();
      return;
    }
    const card = el("div", "inspect-card ok");
    const kind = res.kind || (payload.preset ? "preset" : "local");
    const value = payload.preset || res.path || payload.path;
    const label = res.name || value;
    card.appendChild(el("div", "mono", `${label} (${res.kind})`));
    card.appendChild(el("div", "muted small", `${fmtNum(res.total_files)} files · ${fmtBytes(res.total_bytes)}${res.total_docs ? ` · ${fmtNum(res.total_docs)} docs` : ""}${res.export_verified ? " · OpenWALDO verified" : ""}`));
    const row = el("div", "card-actions");
    const b1 = el("button", "btn btn-sm btn-ghost", "Assign to Base");
    b1.type = "button";
    b1.addEventListener("click", () => assignBaseDataset(kind, value, label));
    const b2 = el("button", "btn btn-sm btn-primary", "+ Add as Strand");
    b2.type = "button";
    b2.addEventListener("click", () => addDatasetAsStrand(kind, value, label, label));
    row.append(b1, b2);
    card.appendChild(row);
    out.appendChild(card);
    sfx.assign();
  }

  /* ---------------- Tab 2: Design Helix Chain ---------------- */
  const GEO_SLIDERS = [
    { key: "V", label: "Vocab V", min: 256, max: 65536, step: 256 },
    { key: "T", label: "Context T", min: 16, max: 1024, step: 16 },
    { key: "D", label: "Model Dim D", min: 16, max: 1024, step: 16 },
    { key: "H", label: "Query Heads H", min: 2, max: 16, step: 1 },
    { key: "H_KV", label: "KV Heads H_KV", min: 2, max: 16, step: 1 },
    { key: "F", label: "MLP F", min: 32, max: 4096, step: 32 },
    { key: "L", label: "Base Layers L", min: 2, max: 24, step: 1 },
    { key: "B", label: "Batch B", min: 1, max: 16, step: 1 },
  ];

  const LOOP_SLIDERS = [
    { key: "LP", label: "Loop Prologue LP", min: 0, max: 8, step: 1 },
    { key: "LC", label: "Loop Core LC", min: 0, max: 8, step: 1 },
    { key: "NP", label: "Loop Passes NP", min: 1, max: 8, step: 1 },
    { key: "adapt_thresh_micro", label: "Adaptive Exit Threshold (μ)", min: 0, max: 200000, step: 5000 },
    { key: "msg_gain_micro", label: "LTP0 Inter-Pass Gain (μ)", min: 0, max: 500000, step: 10000 },
  ];

  const TRAIN_SLIDERS = [
    { key: "base_steps", label: "Base Steps", min: 2, max: 200, step: 1 },
    { key: "lr_micro", label: "Base Learning Rate (μ)", min: 100, max: 20000, step: 100 },
  ];

  const sliderInputs = new Map();

  function buildSliderGroup(containerId, defs) {
    const box = $(containerId);
    box.replaceChildren();
    defs.forEach((d) => {
      const row = el("div", "slider-row");
      const id = `sl-${d.key}`;
      const lbl = el("label", "", d.label);
      lbl.setAttribute("for", id);
      const out = el("output", "", state.spec[d.key]);
      out.setAttribute("for", id);
      const inp = document.createElement("input");
      inp.type = "range";
      inp.id = id;
      inp.min = String(d.min);
      inp.max = String(d.max);
      inp.step = String(d.step);
      inp.value = String(state.spec[d.key]);
      inp.addEventListener("input", () => {
        const v = Number(inp.value);
        state.spec[d.key] = v;
        out.textContent = fmtNum(v);
        scheduleEstimate();
      });
      sliderInputs.set(d.key, { inp, out });
      row.append(lbl, out, inp);
      box.appendChild(row);
    });
  }

  function syncDesignControlsFromSpec() {
    const s = state.spec;
    const modeRadio = document.querySelector(`input[name="mode"][value="${s.mode}"]`);
    if (modeRadio) modeRadio.checked = true;
    $("f-name").value = s.name;
    $("f-arch").value = s.arch;
    $("f-tokenizer").value = s.tokenizer;
    $("f-hfdtype").value = s.hf_dtype;
    $("f-gpu").checked = Boolean(s.use_gpu);
    $("f-hf").checked = Boolean(s.export_hf);
    $("f-gguf").checked = Boolean(s.export_gguf);
    sliderInputs.forEach(({ inp, out }, k) => {
      inp.value = String(s[k]);
      out.textContent = fmtNum(s[k]);
    });
    $("strand-block").style.opacity = s.mode === "universal" ? "0.55" : "1";
    $("strand-note").textContent = s.mode === "universal"
      ? "Universal mode uses 0 DNA strands and trains a single OLLMCK04 checkpoint. Switch to DNA or Hybrid to attach strands."
      : "Each strand forks from the frozen base, grows layers and trains only its own weights. spclip fuses 2–4 strands at sample time.";
    renderStrands();
  }

  function renderPresets() {
    const box = $("preset-chips");
    box.replaceChildren();
    PRESETS.forEach((p) => {
      const b = el("button", "chip", p.label);
      b.type = "button";
      b.addEventListener("click", () => {
        sfx.assign();
        Object.assign(state.spec, p.patch);
        if (p.patch.mode === "universal") state.spec._savedStrands = state.spec.strands.slice();
        if (p.patch.mode !== "universal" && !state.spec.strands.length && state.spec._savedStrands) {
          state.spec.strands = state.spec._savedStrands;
        }
        syncDesignControlsFromSpec();
        refreshEstimate();
        toast(`Applied preset: ${p.label}`, "ok");
      });
      box.appendChild(b);
    });
  }

  function renderStrands() {
    const box = $("strands");
    box.replaceChildren();
    if (!state.spec.strands.length) {
      box.appendChild(el("p", "muted small", "No custom strands yet. Click “Add Custom Strand” to branch off the frozen trunk."));
      return;
    }
    state.spec.strands.forEach((st, idx) => {
      const card = el("div", "strand-card");
      card.style.setProperty("--strand-c", st.color || STRAND_COLORS[idx % STRAND_COLORS.length]);

      const head = el("div", "strand-head");
      head.appendChild(el("span", "strand-dot"));
      const nameIn = document.createElement("input");
      nameIn.type = "text";
      nameIn.value = st.name;
      nameIn.setAttribute("aria-label", `Strand ${idx + 1} name`);
      nameIn.addEventListener("input", () => { st.name = nameIn.value; renderDataSlots(); scheduleEstimate(); });
      const rm = el("button", "btn btn-sm btn-danger", "");
      rm.type = "button";
      rm.setAttribute("aria-label", `Remove ${st.name}`);
      const ic = svgEl("svg", { class: "ic" });
      ic.appendChild(svgEl("use", { href: "#i-trash" }));
      rm.appendChild(ic);
      rm.addEventListener("click", () => {
        sfx.tap();
        state.spec.strands.splice(idx, 1);
        renderDataSlots();
        renderStrands();
        renderCatalogCards();
        refreshEstimate();
      });
      head.append(nameIn, rm);
      card.appendChild(head);

      const dsLine = el("div", "mono small muted", `data: ${st.dataset.label || st.dataset.value}`);
      card.appendChild(dsLine);

      const grid = el("div", "strand-grid");
      const mkNum = (lblText, key, min, max, step) => {
        const l = el("label", "", lblText);
        const i = document.createElement("input");
        i.type = "number";
        i.min = String(min);
        i.max = String(max);
        i.step = String(step);
        i.value = String(st[key]);
        i.addEventListener("change", () => { st[key] = Number(i.value); scheduleEstimate(); });
        l.appendChild(i);
        return l;
      };
      const mkSel = (lblText, key, opts) => {
        const l = el("label", "", lblText);
        const s = document.createElement("select");
        opts.forEach(([v, txt]) => {
          const o = el("option", "", txt);
          o.value = v;
          if (String(st[key]) === String(v)) o.selected = true;
          s.appendChild(o);
        });
        s.addEventListener("change", () => {
          st[key] = key === "use_fcachem" ? s.value === "true" : s.value;
          scheduleEstimate();
        });
        l.appendChild(s);
        return l;
      };
      grid.append(
        mkNum("Added layers", "add_layers", 1, 8, 1),
        mkSel("Growth", "growth", [["growpro", "growpro (projector)"], ["grow", "grow (zero-init)"]]),
        mkNum("Steps", "steps", 1, 200, 1),
        mkNum("LR (micro)", "lr_micro", 100, 20000, 100),
        mkSel("Frozen cache", "use_fcachem", [["true", "fcachem + trainfc"], ["false", "trainf direct"]]),
        mkNum("Prior weight", "weight", 0, 1000, 50)
      );
      card.appendChild(grid);
      box.appendChild(card);
    });
  }

  /* ---------------- Merkle Chain SVG DAG ---------------- */
  function buildFallbackDag() {
    const nodes = [
      { id: "data", label: state.spec.base_dataset.label || "Dataset", kind: "data", color: "#22d3ee", meta: "OpenWALDO" },
      { id: "base", label: `Base Trunk (L=${state.spec.L})`, kind: "base", color: "#22d3ee", meta: `D=${state.spec.D}` },
      { id: "lock", label: "Frozen Merkle Lock", kind: "lock", color: "#34d399", meta: "FNV-1a / SHA" },
    ];
    const edges = [
      { from: "data", to: "base", label: "train" },
      { from: "base", to: "lock", label: "freeze" },
    ];
    const strands = state.spec.mode === "universal" ? [] : state.spec.strands;
    strands.forEach((st, i) => {
      const sid = st.id || `strand-${i}`;
      nodes.push({ id: sid, label: st.name, kind: "strand", color: st.color || STRAND_COLORS[i % 4], meta: `+${st.add_layers}L · OLLMSP01` });
      edges.push({ from: "lock", to: sid, label: st.growth });
    });
    if (strands.length) {
      nodes.push({ id: "spclip", label: "spclip Fusion", kind: "export", color: "#fbbf24", meta: `${strands.length}-helix` });
      strands.forEach((st, i) => edges.push({ from: st.id || `strand-${i}`, to: "spclip", label: "merkle" }));
    }
    if (state.spec.export_hf || state.spec.export_gguf || state.spec.mode !== "dna") {
      nodes.push({ id: "export", label: "OLLMCK04 / Export", kind: "export", color: "#a78bfa", meta: "HF · GGUF" });
      edges.push({ from: strands.length ? "spclip" : "lock", to: "export", label: "interchange" });
    }
    return { nodes, edges };
  }

  function renderDag(svgId, dag, activeStage = "", run = null) {
    const svg = $(svgId);
    const titleEl = svg.querySelector("title");
    svg.replaceChildren();
    if (titleEl) svg.appendChild(titleEl);
    const g = dag && dag.nodes && dag.nodes.length ? dag : buildFallbackDag();
    const W = 900, H = 320;

    // Topological column layout
    const depth = {};
    g.nodes.forEach((n) => { depth[n.id] = 0; });
    for (let iter = 0; iter < g.nodes.length; iter++) {
      (g.edges || []).forEach((e) => {
        if (depth[e.to] !== undefined && depth[e.from] !== undefined) {
          depth[e.to] = Math.max(depth[e.to], depth[e.from] + 1);
        }
      });
    }
    const cols = {};
    let maxCol = 0;
    g.nodes.forEach((n) => {
      const c = depth[n.id] || 0;
      maxCol = Math.max(maxCol, c);
      (cols[c] = cols[c] || []).push(n);
    });

    const pos = {};
    const padX = 96;
    const stepX = maxCol > 0 ? (W - padX * 2) / maxCol : 0;
    Object.entries(cols).forEach(([c, list]) => {
      const x = padX + Number(c) * stepX;
      const stepY = H / (list.length + 1);
      list.forEach((n, idx) => {
        pos[n.id] = { x, y: stepY * (idx + 1) };
      });
    });

    // Edges
    (g.edges || []).forEach((e) => {
      const a = pos[e.from], b = pos[e.to];
      if (!a || !b) return;
      const mx = (a.x + b.x) / 2;
      const d = `M${a.x} ${a.y} C${mx} ${a.y}, ${mx} ${b.y}, ${b.x} ${b.y}`;
      const path = svgEl("path", { d, class: `dag-edge${activeStage ? " lit" : ""}` });
      svg.appendChild(path);
      if (e.label) {
        const t = svgEl("text", { x: mx, y: (a.y + b.y) / 2 - 6, "text-anchor": "middle", fill: "#93a4bf", "font-size": 10 });
        t.textContent = e.label;
        svg.appendChild(t);
      }
    });

    // Nodes
    const strandMap = {};
    if (run && run.strands) run.strands.forEach((s) => { strandMap[s.id] = s; });

    g.nodes.forEach((n) => {
      const p = pos[n.id];
      if (!p) return;
      const stInfo = strandMap[n.id];
      let cls = "dag-node";
      if (run) {
        if (run.status === "done") cls += " done";
        else if (activeStage && n.kind === activeStage) cls += " active";
        else if (stInfo && stInfo.status === "done") cls += " done";
      }
      const boxW = maxCol >= 5 ? 124 : 148;
      const maxChars = maxCol >= 5 ? 15 : 18;
      const grp = svgEl("g", { class: cls, transform: `translate(${p.x - boxW / 2}, ${p.y - 25})` });
      const rect = svgEl("rect", { width: boxW, height: 50, rx: 10 });
      if (n.color) rect.setAttribute("stroke", n.color);
      grp.appendChild(rect);

      // Lock shield badge on lock node
      if (n.kind === "lock") {
        const badge = svgEl("circle", { cx: 14, cy: 16, r: 5, fill: "#34d399" });
        grp.appendChild(badge);
      }

      const title = svgEl("text", { x: n.kind === "lock" ? 26 : 12, y: 21 });
      title.textContent = String(n.label || n.id).slice(0, maxChars);
      grp.appendChild(title);

      let subText = typeof n.meta === "string"
        ? n.meta
        : (n.meta && typeof n.meta === "object"
          ? Object.entries(n.meta).slice(0, 2).map(([k, v]) => `${k}:${v}`).join(" ")
          : (n.kind || ""));
      if (stInfo && stInfo.pack_sha256) subText = `sha:${stInfo.pack_sha256.slice(0, 10)}`;
      else if (stInfo && stInfo.eval_loss) subText = `loss ${stInfo.eval_loss}`;
      const sub = svgEl("text", { x: 12, y: 39, class: "sub" });
      sub.textContent = String(subText).slice(0, maxChars + 2);
      grp.appendChild(sub);
      svg.appendChild(grp);
    });
  }

  let estTimer = null;
  function scheduleEstimate() {
    clearTimeout(estTimer);
    estTimer = setTimeout(refreshEstimate, 140);
  }

  function buildPayloadSpec() {
    const s = structuredClone(state.spec);
    if (s.mode === "universal") s.strands = [];
    return s;
  }

  async function refreshEstimate() {
    const payload = buildPayloadSpec();
    const res = await api("/api/chains/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.estimate = res;
    if (res.ok) {
      $("est-params").textContent = fmtNum(res.params);
      $("est-mb").textContent = res.est_mb !== undefined ? `${Number(res.est_mb).toFixed(2)} MB` : "—";
      $("est-tok").textContent = res.tok_bytes ? `${res.tok_bytes} B/tok` : "—";
      const v = $("est-valid");
      v.textContent = res.valid ? "Valid" : (res.error || "Invalid");
      v.className = `est-v ${res.valid ? "ok" : "err"}`;
      const cmd = Array.isArray(res.resolved_cmd_preview) ? res.resolved_cmd_preview.join("\n") : (res.resolved_cmd_preview || "—");
      $("est-cmd").textContent = cmd;
      renderDag("dag-svg", res.dag);
    } else {
      $("est-valid").textContent = res.error || "Check geometry";
      $("est-valid").className = "est-v err";
      renderDag("dag-svg", null);
    }
  }

  /* ---------------- Tab 3: Forge & Telemetry ---------------- */
  function renderStepper(currentStage, status) {
    const ol = $("stepper");
    ol.replaceChildren();
    const idx = STAGE_DEFS.findIndex((s) => currentStage && currentStage.toLowerCase().includes(s.id));
    STAGE_DEFS.forEach((st, i) => {
      const li = el("li", "step-item");
      let stState = "pending";
      if (status === "done") stState = "done";
      else if (status === "error" && i === Math.max(0, idx)) stState = "error";
      else if (idx > i) stState = "done";
      else if (idx === i && status === "running") stState = "active";
      li.dataset.state = stState;
      li.appendChild(el("span", "st-dot"));
      li.appendChild(el("span", "", st.label));
      ol.appendChild(li);
    });
  }

  function renderChart(metrics) {
    const svg = $("chart");
    const titleEl = svg.querySelector("title");
    svg.replaceChildren();
    if (titleEl) svg.appendChild(titleEl);
    const W = 640, H = 260, P = { l: 46, r: 18, t: 18, b: 28 };
    const iw = W - P.l - P.r;
    const ih = H - P.t - P.b;

    // Grid
    for (let i = 0; i <= 4; i++) {
      const y = P.t + (ih * i) / 4;
      svg.appendChild(svgEl("line", { x1: P.l, x2: W - P.r, y1: y, y2: y, stroke: "rgba(148,163,184,0.14)" }));
    }

    const pts = (metrics || []).filter((m) => typeof m.loss === "number" && Number.isFinite(m.loss));
    if (!pts.length) {
      const t = svgEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: "#64748b", "font-size": 13 });
      t.textContent = "Waiting for training telemetry…";
      svg.appendChild(t);
      return;
    }

    const losses = pts.map((m) => m.loss);
    const bpbs = pts.map((m) => (typeof m.bpb === "number" ? m.bpb : m.loss / Math.LN2));
    const allY = losses.concat(bpbs);
    const minY = Math.min(...allY) * 0.94;
    const maxY = Math.max(...allY) * 1.04 || 1;
    const spanY = Math.max(0.001, maxY - minY);

    const sx = (i) => P.l + (pts.length === 1 ? iw / 2 : (iw * i) / (pts.length - 1));
    const sy = (v) => P.t + ih - ((v - minY) / spanY) * ih;

    // Y labels
    [maxY, (maxY + minY) / 2, minY].forEach((val, idx) => {
      const t = svgEl("text", { x: P.l - 6, y: P.t + (ih * idx) / 2 + 4, "text-anchor": "end", fill: "#93a4bf", "font-size": 10.5, class: "mono" });
      t.textContent = val.toFixed(2);
      svg.appendChild(t);
    });

    const smoothPath = (vals) => {
      if (vals.length === 1) return `M${sx(0)} ${sy(vals[0])}`;
      let d = `M${sx(0).toFixed(1)} ${sy(vals[0]).toFixed(1)}`;
      for (let i = 1; i < vals.length; i++) {
        const x0 = sx(i - 1), y0 = sy(vals[i - 1]);
        const x1 = sx(i), y1 = sy(vals[i]);
        const cx = (x0 + x1) / 2;
        d += ` C${cx.toFixed(1)} ${y0.toFixed(1)}, ${cx.toFixed(1)} ${y1.toFixed(1)}, ${x1.toFixed(1)} ${y1.toFixed(1)}`;
      }
      return d;
    };

    // BPB dashed amber curve
    svg.appendChild(svgEl("path", { d: smoothPath(bpbs), fill: "none", stroke: "#fbbf24", "stroke-width": 2, "stroke-dasharray": "5 4", opacity: 0.85 }));
    // Loss glowing cyan curve
    svg.appendChild(svgEl("path", { d: smoothPath(losses), fill: "none", stroke: "#22d3ee", "stroke-width": 2.6, filter: "url(#f-glow)" }));

    // Latest glowing dot
    const last = pts.length - 1;
    svg.appendChild(svgEl("circle", { cx: sx(last), cy: sy(losses[last]), r: 5, fill: "#22d3ee", filter: "url(#f-glow)" }));
    svg.appendChild(svgEl("circle", { cx: sx(last), cy: sy(bpbs[last]), r: 4, fill: "#fbbf24" }));
  }

  function appendLog(line) {
    const clean = stripAnsi(line);
    if (!clean) return;
    state.logLines.push(clean);
    if (state.logLines.length > 600) state.logLines.shift();
    const term = $("term");
    term.textContent = state.logLines.join("\n");
    if ($("log-follow").checked) term.scrollTop = term.scrollHeight;
  }

  function applyRunState(run) {
    if (!run) return;
    state.activeRun = run;
    const pill = $("run-pill");
    pill.dataset.state = run.status || "idle";
    $("run-pill-text").textContent = `${run.run_id}: ${run.status}${run.stage ? ` (${run.stage})` : ""}`;
    $("btn-stop").disabled = run.status !== "running";
    $("forge-dag-cap").textContent = `${run.status} · ${run.stage || "init"}`;

    renderStepper(run.stage || "", run.status || "idle");
    const m = run.metrics || [];
    const last = m[m.length - 1] || {};
    $("kpi-step").textContent = last.step !== undefined ? fmtNum(last.step) : "0";
    $("kpi-loss").textContent = typeof last.loss === "number" ? last.loss.toFixed(4) : "—";
    $("kpi-bpb").textContent = typeof last.bpb === "number" ? last.bpb.toFixed(3) : "—";
    $("kpi-tps").textContent = typeof last.tok_s === "number" ? fmtNum(Math.round(last.tok_s)) : "—";
    const packs = (run.artifacts || []).filter((a) => a.kind === "pack" || String(a.name).endsWith(".sp")).length;
    $("kpi-packs").textContent = fmtNum(packs || (run.strands || []).filter((s) => s.pack_sha256).length);

    renderChart(m);
    renderDag("forge-dag", run.dag, run.stage, run);
    renderArtifacts(run.artifacts || []);
    renderMixerForRun(run);
    const doneBanner = $("forge-done-banner");
    if (doneBanner) {
      if (run.status === "done") {
        doneBanner.hidden = false;
        const lossTxt = typeof last.loss === "number" ? `final loss ${last.loss.toFixed(4)} nats` : "ready";
        const pkTxt = packs ? ` · ${packs} verified .sp pack${packs === 1 ? "" : "s"}` : "";
        $("forge-done-summary").textContent = `${run.run_id} finished (${lossTxt}${pkTxt}). Click to test or export it.`;
      } else {
        doneBanner.hidden = true;
      }
    }
  }

  function subscribeEvents(runId) {
    if (state.es) { state.es.close(); state.es = null; }
    state.logLines = [];
    $("term").textContent = "";
    const es = new EventSource(`/api/chains/${encodeURIComponent(runId)}/events`);
    state.es = es;
    es.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.kind === "log" && msg.message) appendLog(msg.message);
      if (msg.kind === "stage") {
        if (state.activeRun) {
          state.activeRun.stage = msg.stage || msg.label;
          applyRunState(state.activeRun);
        }
        if (msg.stage && msg.stage !== state.lastStageAudio) {
          state.lastStageAudio = msg.stage;
          sfx.milestone();
        }
        appendLog(`[stage] ${msg.stage || msg.label || ""}`);
      } else if (msg.kind === "metric") {
        if (state.activeRun) {
          state.activeRun.metrics = state.activeRun.metrics || [];
          state.activeRun.metrics.push(msg);
          applyRunState(state.activeRun);
        }
      } else if (msg.kind === "artifact" || msg.kind === "helix") {
        loadRunDetail(runId);
      } else if (msg.kind === "done") {
        es.close();
        state.es = null;
        loadRunDetail(runId);
        refreshRuns();
        sfx.celebrate();
        celebrateBurst();
        toast(`Forge complete: ${runId}`, "ok");
      } else if (msg.kind === "error") {
        es.close();
        state.es = null;
        appendLog(`[error] ${msg.error || msg.message || "Forge failed"}`);
        loadRunDetail(runId);
        sfx.err();
        toast(msg.error || "Forge failed", "err");
      }
    };
    es.onerror = () => {
      if (state.activeRun && state.activeRun.status !== "running") {
        es.close();
        state.es = null;
      }
    };
  }

  async function loadRunDetail(runId) {
    if (!runId) return;
    const res = await api(`/api/chains/${encodeURIComponent(runId)}`);
    if (res.ok && res.run) {
      if (state.activeRunId !== runId) state.playWeights = {};
      state.activeRunId = runId;
      applyRunState(res.run);
      if (res.run.error) appendLog(`[error] ${res.run.error}`);
    }
  }

  async function refreshRuns(selectForPlay = false) {
    const res = await api("/api/chains");
    if (!res.ok) return;
    state.runs = res.runs || [];
    const fill = (sel) => {
      const prev = sel.value || state.activeRunId || "";
      sel.replaceChildren();
      if (!state.runs.length) {
        const empty = el("option", "", "— no runs yet —");
        empty.value = "";
        sel.appendChild(empty);
        return;
      }
      state.runs.forEach((r) => {
        const opt = el("option", "", `${r.run_id} · ${r.name || r.mode} (${r.status})`);
        opt.value = r.run_id;
        if (r.run_id === prev) opt.selected = true;
        sel.appendChild(opt);
      });
    };
    fill($("run-select"));
    fill($("play-run"));
    const chosen = (selectForPlay ? $("play-run").value : $("run-select").value) || (state.runs[0] && state.runs[0].run_id);
    if (chosen && (!state.activeRun || state.activeRun.run_id !== chosen)) {
      await loadRunDetail(chosen);
    }
  }

  async function quickTrain(goal, customPath = "") {
    const baseDs = customPath
      ? { kind: "local", value: customPath, label: customPath }
      : (goal === "code"
        ? { kind: "preset", value: "omega-native", label: "omega-native (.om sources)" }
        : goal === "docs"
          ? { kind: "preset", value: "omega-docs", label: "omega-docs (architecture docs)" }
          : { kind: "preset", value: "omega-starter", label: "omega-starter (repo corpus)" });

    state.spec.base_dataset = baseDs;
    state.spec.use_gpu = false;
    if (goal === "qwen3") {
      state.spec.name = "quick-qwen3";
      state.spec.mode = "universal";
      state.spec.arch = "qwen3";
      state.spec.tokenizer = "byte";
      state.spec.V = 256; state.spec.T = 16; state.spec.D = 32; state.spec.H = 2; state.spec.H_KV = 2; state.spec.F = 64; state.spec.L = 2; state.spec.B = 2;
      state.spec.base_steps = 12;
      state.spec.export_hf = true;
      state.spec.export_gguf = true;
    } else {
      state.spec.name = customPath ? "my-custom-helix" : `quick-${goal}`;
      state.spec.mode = "dna";
      state.spec.arch = "custom";
      state.spec.tokenizer = "byte";
      state.spec.V = 256; state.spec.T = 16; state.spec.D = 32; state.spec.H = 2; state.spec.H_KV = 2; state.spec.F = 64; state.spec.L = 2; state.spec.B = 2;
      state.spec.base_steps = 12;
      state.spec.export_hf = false;
      state.spec.export_gguf = false;
      const strandName = goal === "code" ? "OMG-Code" : goal === "docs" ? "Docs-QA" : "Code";
      state.spec.strands = [
        {
          id: "strand-a",
          name: strandName,
          color: STRAND_COLORS[0],
          add_layers: 1,
          growth: "growpro",
          use_fcachem: true,
          steps: 10,
          lr_micro: 2000,
          weight: 500,
          dataset: { ...baseDs },
        },
      ];
    }
    syncDesignControlsFromSpec();
    renderDataSlots();
    renderStrands();
    activateTab("tab-forge", false);
    await startForge();
  }

  async function startForge() {
    const btn = $("btn-forge");
    btn.disabled = true;
    try {
      const res = await api("/api/chains/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayloadSpec()),
      });
      if (!res.ok) {
        toast(res.error || "Failed to start forge", "err");
        sfx.err();
        return;
      }
      state.activeRunId = res.run_id;
      state.lastStageAudio = "";
      if (res.run) applyRunState(res.run);
      await refreshRuns();
      $("run-select").value = res.run_id;
      subscribeEvents(res.run_id);
      sfx.milestone();
      toast(`Forging ${res.run_id}…`, "ok");
    } finally {
      btn.disabled = false;
    }
  }

  /* ---------------- Tab 4: Playground & Exports ---------------- */
  const PLAY_SLIDERS = [
    { key: "n_tokens", label: "Generate Tokens", min: 8, max: 256, step: 8 },
    { key: "temp_milli", label: "Temperature (milli)", min: 0, max: 1500, step: 50 },
    { key: "seed", label: "Seed", min: 1, max: 9999, step: 1 },
    { key: "adapt_thresh_micro", label: "Adaptive Exit Threshold (μ)", min: 0, max: 200000, step: 5000 },
    { key: "msg_gain_micro", label: "Inter-Pass Gain (μ)", min: 0, max: 500000, step: 10000 },
  ];

  function buildPlaySliders() {
    const box = $("play-sliders");
    box.replaceChildren();
    PLAY_SLIDERS.forEach((d) => {
      const row = el("div", "slider-row");
      const id = `pl-${d.key}`;
      const lbl = el("label", "", d.label);
      lbl.setAttribute("for", id);
      const out = el("output", "", state.playOpts[d.key]);
      out.setAttribute("for", id);
      const inp = document.createElement("input");
      inp.type = "range";
      inp.id = id;
      inp.min = String(d.min);
      inp.max = String(d.max);
      inp.step = String(d.step);
      inp.value = String(state.playOpts[d.key]);
      inp.addEventListener("input", () => {
        state.playOpts[d.key] = Number(inp.value);
        out.textContent = fmtNum(state.playOpts[d.key]);
      });
      row.append(lbl, out, inp);
      box.appendChild(row);
    });
  }

  function renderMixerForRun(run) {
    const box = $("mixer");
    box.replaceChildren();
    const strands = (run && run.strands && run.strands.length) ? run.strands : (run && run.spec && run.spec.strands) || [];
    if (!strands.length) {
      box.appendChild(el("p", "muted small", "Single checkpoint chain — strand mixing activates when 2+ DNA strands are present."));
      return;
    }
    strands.forEach((st, idx) => {
      const id = st.id || `strand-${idx}`;
      if (state.playWeights[id] === undefined) state.playWeights[id] = st.weight !== undefined ? st.weight : 500;
      const row = el("div", "mixer-row slider-row");
      row.style.setProperty("--strand-c", st.color || STRAND_COLORS[idx % 4]);
      const sid = `mix-${id}`;
      const lbl = el("label", "", st.name || id);
      lbl.setAttribute("for", sid);
      const out = el("output", "", state.playWeights[id]);
      out.setAttribute("for", sid);
      const inp = document.createElement("input");
      inp.type = "range";
      inp.id = sid;
      inp.min = "0";
      inp.max = "1000";
      inp.step = "25";
      inp.value = String(state.playWeights[id]);
      inp.addEventListener("input", () => {
        state.playWeights[id] = Number(inp.value);
        out.textContent = state.playWeights[id];
      });
      row.append(lbl, out, inp);
      box.appendChild(row);
    });
  }

  function renderArtifacts(artifacts) {
    const tbody = $("art-body");
    tbody.replaceChildren();
    if (!artifacts || !artifacts.length) {
      const tr = el("tr");
      const td = el("td", "muted", "No artifacts yet.");
      td.colSpan = 5;
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }
    artifacts.forEach((a) => {
      const tr = el("tr");
      tr.appendChild(el("td", "mono", a.name || a.path));
      tr.appendChild(el("td", "", a.kind || "file"));
      tr.appendChild(el("td", "mono", fmtBytes(a.bytes)));
      const shaTd = el("td", "sha", a.sha256 ? `${a.sha256.slice(0, 16)}…` : "—");
      shaTd.title = a.sha256 || "";
      tr.appendChild(shaTd);
      const act = el("td");
      if (a.sha256) {
        const cp = el("button", "btn btn-sm btn-ghost", "");
        cp.type = "button";
        cp.title = "Copy full SHA-256";
        cp.setAttribute("aria-label", `Copy SHA-256 for ${a.name}`);
        const ic = svgEl("svg", { class: "ic" });
        ic.appendChild(svgEl("use", { href: "#i-copy" }));
        cp.appendChild(ic);
        cp.addEventListener("click", async () => {
          await navigator.clipboard.writeText(a.sha256).catch(() => {});
          sfx.tap();
          toast(`Copied SHA-256 for ${a.name}`, "ok");
        });
        act.appendChild(cp);
      }
      tr.appendChild(act);
      tbody.appendChild(tr);
    });
  }

  async function runVerifyLocks() {
    const btn = $("btn-verify-locks");
    const st = $("locks-status");
    if (!btn || !st) return;
    btn.disabled = true;
    st.textContent = "Verifying 8-pillar locks…";
    try {
      const res = await api("/api/runtime/locks", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ steps: 6, lr_milli: 10 }) });
      if (res.ok) {
        const revTxt = res.rev_supported ? `Rev Q32 (${res.q32_inv_bits}-bit, 0B act) ✓` : "Rev (ARM64 only)";
        st.textContent = `OMG ✓ · Arboris ✓ · ${revTxt}`;
        toast("8-pillar pure-Omega locks verified.", "ok");
        sfx.assign();
      } else {
        st.textContent = res.error || "Lock check failed";
        toast("8-pillar lock check failed.", "err");
        sfx.err();
      }
    } finally {
      btn.disabled = false;
    }
  }

  async function runSample() {
    const playModeEl = document.querySelector('input[name="playmode"]:checked');
    const playMode = playModeEl ? playModeEl.value : "std";
    const runId = $("play-run").value || state.activeRunId;
    if (playMode === "omg" || playMode === "arbsamp") {
      const btn = $("btn-sample");
      btn.disabled = true;
      $("sample-out").textContent = playMode === "omg" ? "OMG-masked decoding…" : "Arboris 3-layer guided decoding…";
      try {
        const endpoint = playMode === "omg" ? "/api/runtime/omg" : "/api/runtime/arbsamp";
        const res = await api(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "samp",
            run_id: runId || undefined,
            seed: state.playOpts.seed,
            budget: Math.min(128, Math.max(16, state.playOpts.n_tokens || 64)),
            temp_milli: state.playOpts.temp_milli,
          }),
        });
        if (!res.ok) {
          $("sample-out").textContent = res.error || "8-pillar decode failed.";
          $("sample-meta").textContent = "";
          sfx.err();
          return;
        }
        $("sample-out").textContent = res.source_om || "(empty .om)";
        const compBadge = res.omc_compiled ? "omc ✓" : `omc rc=${res.omc_rc}`;
        if (playMode === "omg") {
          $("sample-meta").textContent = `OMG-Masked · ${res.ntok} toks · sem=${res.sem_id} · ${compBadge}`;
        } else {
          const pct = Math.round((res.bypass_rate || 0) * 100);
          $("sample-meta").textContent = `Arboris-Guided · ${res.bypass_hits}/${res.ntok} bypass (${pct}%) · LTP0=${res.ltp0_calls} · ${compBadge}`;
        }
        $("exit-pills").replaceChildren();
        sfx.assign();
      } finally {
        btn.disabled = false;
      }
      return;
    }
    if (!runId) {
      toast("Select or forge a chain first.", "err");
      sfx.err();
      return;
    }
    const spmode = Number(document.querySelector('input[name="spmode"]:checked').value);
    const btn = $("btn-sample");
    btn.disabled = true;
    $("sample-out").textContent = "Sampling…";
    try {
      const res = await api(`/api/chains/${encodeURIComponent(runId)}/sample`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: $("play-prompt").value,
          n_tokens: state.playOpts.n_tokens,
          temp_milli: state.playOpts.temp_milli,
          seed: state.playOpts.seed,
          spclip_mode: spmode,
          strand_weights: state.playWeights,
          adapt_thresh_micro: state.playOpts.adapt_thresh_micro,
          msg_gain_micro: state.playOpts.msg_gain_micro,
        }),
      });
      if (!res.ok) {
        $("sample-out").textContent = res.error || "Sampling failed.";
        sfx.err();
        return;
      }
      $("sample-out").textContent = res.text || res.stdout || "(empty output)";
      $("sample-meta").textContent = res.mode_used ? `mode: ${res.mode_used}` : "";
      const pills = $("exit-pills");
      pills.replaceChildren();
      const tel = res.telemetry || {};
      const passes = tel.exit_passes || [];
      const toks = tel.tokens || [];
      passes.slice(0, 64).forEach((p, idx) => {
        const pill = el("span", "exit-pill");
        const tokLbl = toks[idx] !== undefined ? String(toks[idx]) : `#${idx}`;
        pill.appendChild(document.createTextNode(`${tokLbl} `));
        pill.appendChild(el("b", "", `p${p}`));
        pills.appendChild(pill);
      });
      sfx.assign();
    } finally {
      btn.disabled = false;
    }
  }

  /* ---------------- Wiring & Boot ---------------- */
  function bindEvents() {
    $("sound-toggle").addEventListener("click", () => {
      state.soundOn = !state.soundOn;
      const btn = $("sound-toggle");
      btn.setAttribute("aria-pressed", String(state.soundOn));
      btn.querySelector("use").setAttribute("href", state.soundOn ? "#i-sound" : "#i-mute");
      btn.querySelector("span").textContent = state.soundOn ? "Sound: ON" : "Sound: OFF";
      if (state.soundOn) sfx.assign();
    });

    let qTimer = null;
    $("cat-q").addEventListener("input", () => {
      clearTimeout(qTimer);
      qTimer = setTimeout(() => { state.filters.q = $("cat-q").value.trim(); loadCatalog(); }, 160);
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "/" && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const tag = document.activeElement && document.activeElement.tagName;
        if (tag !== "INPUT" && tag !== "TEXTAREA" && tag !== "SELECT") {
          e.preventDefault();
          activateTab("tab-discover");
          $("cat-q").focus();
        }
      }
    });

    $("btn-sync").addEventListener("click", async () => {
      sfx.tap();
      $("btn-sync").disabled = true;
      try {
        const res = await api("/api/catalog/sync", { method: "POST" });
        if (res.ok) {
          toast(`Synced ${res.total} corpora (${res.synced_from || "live"})`, "ok");
          sfx.milestone();
          loadCatalog();
        } else {
          toast(res.error || "Sync unavailable offline; using cached index", "err");
          sfx.err();
        }
      } finally {
        $("btn-sync").disabled = false;
      }
    });

    $("btn-starter").addEventListener("click", () => inspectDataset({ preset: "omega-starter" }));
    $("btn-inspect").addEventListener("click", () => {
      const p = $("local-path").value.trim();
      if (!p) { toast("Enter a local folder or file path.", "err"); return; }
      inspectDataset({ path: p });
    });

    $("btn-to-design").addEventListener("click", () => { sfx.tap(); activateTab("tab-design", true); });
    $("btn-to-forge").addEventListener("click", () => { sfx.tap(); activateTab("tab-forge", true); });

    document.querySelectorAll('input[name="mode"]').forEach((r) => {
      r.addEventListener("change", () => {
        sfx.tap();
        state.spec.mode = r.value;
        syncDesignControlsFromSpec();
        refreshEstimate();
      });
    });

    $("f-name").addEventListener("input", () => { state.spec.name = $("f-name").value; scheduleEstimate(); });
    $("f-arch").addEventListener("change", () => { state.spec.arch = $("f-arch").value; scheduleEstimate(); });
    $("f-tokenizer").addEventListener("change", () => { state.spec.tokenizer = $("f-tokenizer").value; scheduleEstimate(); });
    $("f-hfdtype").addEventListener("change", () => { state.spec.hf_dtype = $("f-hfdtype").value; scheduleEstimate(); });
    $("f-gpu").addEventListener("change", () => { state.spec.use_gpu = $("f-gpu").checked; scheduleEstimate(); });
    $("f-hf").addEventListener("change", () => { state.spec.export_hf = $("f-hf").checked; scheduleEstimate(); });
    $("f-gguf").addEventListener("change", () => { state.spec.export_gguf = $("f-gguf").checked; scheduleEstimate(); });

    $("btn-add-strand").addEventListener("click", () => {
      addDatasetAsStrand(
        state.spec.base_dataset.kind,
        state.spec.base_dataset.value,
        state.spec.base_dataset.label,
        `Strand ${state.spec.strands.length + 1}`
      );
    });

    $("btn-forge").addEventListener("click", startForge);
    $("btn-stop").addEventListener("click", async () => {
      if (!state.activeRunId) return;
      await api(`/api/chains/${encodeURIComponent(state.activeRunId)}/stop`, { method: "POST" });
      toast(`Stop requested for ${state.activeRunId}`);
    });
    $("btn-refresh-runs").addEventListener("click", () => { sfx.tap(); refreshRuns(); });
    $("run-select").addEventListener("change", () => {
      const id = $("run-select").value;
      if (id) { loadRunDetail(id); subscribeEvents(id); }
    });
    $("play-run").addEventListener("change", () => {
      const id = $("play-run").value;
      if (id) loadRunDetail(id);
    });
    $("btn-sample").addEventListener("click", runSample);
    if ($("btn-verify-locks")) $("btn-verify-locks").addEventListener("click", runVerifyLocks);
    document.querySelectorAll(".qt-card[data-qt]").forEach((btn) => {
      btn.addEventListener("click", () => {
        sfx.milestone();
        quickTrain(btn.dataset.qt);
      });
    });
    if ($("btn-qt-local")) {
      $("btn-qt-local").addEventListener("click", () => {
        const p = ($("qt-local-path").value || "").trim();
        if (!p) {
          toast("Paste a local folder or file path first.", "err");
          sfx.err();
          return;
        }
        sfx.milestone();
        quickTrain("custom", p);
      });
    }
    if ($("btn-goto-play")) {
      $("btn-goto-play").addEventListener("click", async () => {
        sfx.tap();
        await refreshRuns(true);
        if (state.activeRunId) $("play-run").value = state.activeRunId;
        activateTab("tab-play", true);
      });
    }

    /* ---------------- Tab 5: Distillation Pipeline Network ---------------- */
    $("btn-distill-probe").addEventListener("click", probeDistillTeachers);
    $("btn-distill-start").addEventListener("click", startDistillRun);
    $("btn-distill-stop").addEventListener("click", async () => {
      if (!state.activeDistillId) return;
      await api(`/api/distill/runs/${encodeURIComponent(state.activeDistillId)}/stop`, { method: "POST" });
      toast(`Stop requested for ${state.activeDistillId}`);
    });
    $("distill-run-select").addEventListener("change", () => {
      const id = $("distill-run-select").value;
      if (id) { loadDistillDetail(id); subscribeDistillEvents(id); }
    });

    bindThemeAndRemote();
  }

  /* ---------------- Theme Toggle (dark-mode-toggle) & Remote / SSH Dialog ---------------- */
  function bindThemeAndRemote() {
    const meta = document.querySelector('meta[name="color-scheme"]');
    const storageKey = "omega-studio-color-scheme";
    const sysDark = () => window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const syncMeta = () => {
      let saved = null;
      try { saved = localStorage.getItem(storageKey); } catch (_) {}
      if (meta) meta.setAttribute("content", saved === "light" || saved === "dark" ? saved : "light dark");
      const effDark = saved === "dark" || (saved !== "light" && sysDark());
      const btn = $("theme-toggle");
      if (btn) btn.setAttribute("aria-label", effDark ? "Switch to light theme" : "Switch to starry dark theme");
    };
    syncMeta();
    const tBtn = $("theme-toggle");
    if (tBtn) {
      tBtn.addEventListener("click", () => {
        sfx.tap();
        let saved = null;
        try { saved = localStorage.getItem(storageKey); } catch (_) {}
        const currentDark = saved === "dark" || (saved !== "light" && sysDark());
        const target = currentDark ? "light" : "dark";
        const systemTarget = sysDark() ? "dark" : "light";
        try {
          if (target === systemTarget) localStorage.removeItem(storageKey);
          else localStorage.setItem(storageKey, target);
        } catch (_) {}
        syncMeta();
      });
    }
    window.addEventListener("storage", (ev) => { if (ev.key === storageKey) syncMeta(); });

    const dlg = $("remote-ssh-dialog");
    if (!dlg) return;

    dlg.addEventListener("close", () => { dlg.returnValue = ""; });

    // Feature-detected progressive enhancement fallbacks for commandfor and closedby="any"
    if (!("commandForElement" in HTMLButtonElement.prototype)) {
      $("btn-remote-ssh")?.addEventListener("click", () => { dlg.showModal(); refreshRemotes(); });
      $("btn-ssh-close")?.addEventListener("click", () => { dlg.close("cancel"); });
    } else {
      $("btn-remote-ssh")?.addEventListener("click", () => { refreshRemotes(); });
    }
    if (!("closedBy" in HTMLDialogElement.prototype)) {
      let pressedOut = false;
      const outside = (e) => {
        const r = dlg.getBoundingClientRect();
        return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
      };
      dlg.addEventListener("pointerdown", (e) => { pressedOut = outside(e); });
      dlg.addEventListener("pointerup", (e) => { if (pressedOut && outside(e)) dlg.close(); });
    }

    $("btn-ssh-refresh")?.addEventListener("click", () => { sfx.tap(); refreshRemotes(); });
    $("btn-ssh-probe")?.addEventListener("click", () => {
      const host = $("ssh-target").value.trim();
      const rport = Number($("ssh-rport").value || 18420);
      if (!host) { toast("Enter an SSH target first.", "err"); return; }
      probeSshHost(host, rport);
    });
    $("ssh-connect-form")?.addEventListener("submit", (e) => {
      e.preventDefault();
      const host = $("ssh-target").value.trim();
      const rport = Number($("ssh-rport").value || 18420);
      const bootstrap = Boolean($("ssh-bootstrap")?.checked);
      if (!host) { toast("Enter an SSH target.", "err"); return; }
      connectSshHost(host, rport, bootstrap);
    });
  }

  async function refreshRemotes() {
    const res = await api("/api/remotes");
    if (!res.ok) return;
    const hostsBox = $("ssh-hosts-list");
    const tunBox = $("ssh-tunnels-list");
    const lbl = $("remote-ssh-label");
    const pill = $("btn-remote-ssh");
    const tunnels = res.active_tunnels || [];
    const aliveCount = tunnels.filter((t) => t.alive).length;
    if (lbl) lbl.textContent = aliveCount > 0 ? `Remote SSH (${aliveCount})` : "Remote / SSH";
    if (pill) pill.classList.toggle("pill-remote-on", aliveCount > 0);
    if ($("ssh-snippet-box") && res.server_pairing) {
      $("ssh-snippet-box").textContent = `${res.server_pairing.headless_cmd}\n${res.server_pairing.ssh_forward_hint}`;
    }

    if (hostsBox) {
      hostsBox.replaceChildren();
      const hosts = res.ssh_hosts || [];
      if (!hosts.length) {
        hostsBox.appendChild(el("p", "muted small", "No hosts found in ~/.ssh/config yet. Enter user@host above."));
      } else {
        hosts.forEach((h) => {
          const row = el("div", "ssh-row");
          const meta = el("div", "ssh-row-meta");
          meta.appendChild(el("strong", "", h.name));
          meta.appendChild(el("span", "muted small mono", `${h.user ? h.user + "@" : ""}${h.hostname}:${h.port || 22}`));
          const acts = el("div", "ssh-row-actions");
          const bProbe = el("button", "btn btn-ghost btn-sm", "Probe");
          bProbe.type = "button";
          bProbe.addEventListener("click", () => {
            $("ssh-target").value = h.target;
            probeSshHost(h.target, Number($("ssh-rport").value || 18420));
          });
          const bConn = el("button", "btn btn-primary btn-sm", "Connect");
          bConn.type = "button";
          bConn.addEventListener("click", () => {
            $("ssh-target").value = h.target;
            connectSshHost(h.target, Number($("ssh-rport").value || 18420), Boolean($("ssh-bootstrap")?.checked));
          });
          acts.append(bProbe, bConn);
          row.append(meta, acts);
          hostsBox.appendChild(row);
        });
      }
    }

    if (tunBox) {
      tunBox.replaceChildren();
      if (!tunnels.length) {
        tunBox.appendChild(el("p", "muted small", "No active SSH tunnels."));
      } else {
        tunnels.forEach((t) => {
          const row = el("div", "ssh-row");
          const meta = el("div", "ssh-row-meta");
          const rs = t.remote_system || {};
          const tag = rs.has_gb10 ? "GB10 GPU" : (rs.os || "CPU");
          meta.appendChild(el("strong", "", `${t.host} (${tag})`));
          meta.appendChild(el("span", "muted small mono", `${t.local_url} -> remote:${t.remote_port}`));
          const acts = el("div", "ssh-row-actions");
          const aOpen = el("a", "btn btn-primary btn-sm", "Open Studio ↗");
          aOpen.href = t.open_url;
          aOpen.target = "_blank";
          aOpen.rel = "noopener";
          const bDisc = el("button", "btn btn-danger btn-sm", "Disconnect");
          bDisc.type = "button";
          bDisc.addEventListener("click", () => disconnectSshHost(t.id));
          acts.append(aOpen, bDisc);
          row.append(meta, acts);
          tunBox.appendChild(row);
        });
      }
    }
  }

  async function probeSshHost(host, rport) {
    const st = $("ssh-probe-status");
    if (st) st.textContent = `Probing ${host} over SSH…`;
    const res = await api("/api/remotes/probe", { method: "POST", body: { host, remote_port: rport } });
    if (!res.ok) {
      if (st) st.textContent = `Probe failed: ${res.error || "unreachable"}`;
      toast(`SSH probe failed: ${res.error || host}`, "err");
      sfx.err();
      return;
    }
    const gpu = res.has_gb10 ? "GB10 GPU detected" : "CPU only";
    const srv = res.studio_running ? `Studio running on :${rport}` : `Studio stopped (will auto-start)`;
    if (st) st.textContent = `${res.hostname} · ${res.os} ${res.arch} · ${gpu} · ${srv}`;
    sfx.assign();
  }

  async function connectSshHost(host, rport, bootstrap) {
    const btn = $("btn-ssh-connect");
    const st = $("ssh-probe-status");
    if (btn) btn.disabled = true;
    if (st) st.textContent = `Establishing SSH tunnel to ${host}…`;
    try {
      const res = await api("/api/remotes/connect", {
        method: "POST",
        body: { host, remote_port: rport, bootstrap },
      });
      if (!res.ok) {
        if (st) st.textContent = `Connect failed: ${res.error || "error"}`;
        toast(res.error || "SSH connect failed", "err");
        sfx.err();
        return;
      }
      const tun = res.tunnel;
      if (st) st.textContent = `Connected: ${tun.open_url}`;
      toast(`SSH tunnel ready -> ${tun.host} (${tun.local_url})`, "ok");
      sfx.milestone();
      await refreshRemotes();
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function disconnectSshHost(id) {
    const res = await api("/api/remotes/disconnect", { method: "POST", body: { id } });
    if (res.ok) {
      toast("SSH tunnel disconnected.");
      await refreshRemotes();
    }
  }

  let distillEs = null;
  function appendDistillLog(line) {
    const pre = $("distill-term");
    if (!pre) return;
    const prev = pre.textContent ? pre.textContent.split("\n") : [];
    prev.push(stripAnsi(line));
    if (prev.length > 400) prev.shift();
    pre.textContent = prev.join("\n");
    pre.scrollTop = pre.scrollHeight;
  }

  async function probeDistillTeachers() {
    sfx.tap();
    const res = await api("/api/distill/teachers/probe");
    const box = $("distill-probes");
    if (!box) return;
    box.replaceChildren();
    if (!res.ok) {
      box.textContent = "Probe failed.";
      return;
    }
    const eps = res.endpoints || [];
    eps.forEach((ep) => {
      const chip = el("button", { type: "button", class: `probe-chip${ep.reachable ? " on" : ""}` }, `${ep.name} (${ep.reachable ? (ep.models[0] || "online") : "offline"})`);
      chip.addEventListener("click", () => {
        $("d-teacher-url").value = ep.base_url;
        if (ep.models && ep.models.length) $("d-teacher-model").value = ep.models[0];
        sfx.assign();
      });
      box.appendChild(chip);
    });
  }

  function renderDistillChart(pts) {
    const svg = $("chart-distill");
    if (!svg) return;
    svg.replaceChildren();
    const W = 640, H = 220, pad = { l: 46, r: 16, t: 16, b: 26 };
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + ((H - pad.t - pad.b) * i) / 4;
      svg.appendChild(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: "rgba(148,163,184,0.12)" }));
    }
    if (!pts || !pts.length) {
      svg.appendChild(svgEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: "#64748b", "font-size": 12 }, "Distillation CE / KL telemetry appears here"));
      return;
    }
    const tot = pts.map((p) => Number(p.loss ?? 0));
    const ce = pts.map((p) => Number(p.ce_loss ?? p.loss ?? 0));
    const kl = pts.map((p) => Number(p.kl_loss ?? 0));
    const mx = Math.max(1e-6, ...tot, ...ce, ...kl);
    const sx = (i) => pad.l + ((W - pad.l - pad.r) * (pts.length === 1 ? 0.5 : i / (pts.length - 1)));
    const sy = (v) => pad.t + (1 - v / mx) * (H - pad.t - pad.b);
    const linePath = (arr) => arr.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)} ${sy(v).toFixed(1)}`).join(" ");
    svg.appendChild(svgEl("path", { d: linePath(ce), fill: "none", stroke: "#fbbf24", "stroke-width": 1.8, "stroke-dasharray": "4 3" }));
    svg.appendChild(svgEl("path", { d: linePath(kl), fill: "none", stroke: "#a78bfa", "stroke-width": 1.8 }));
    svg.appendChild(svgEl("path", { d: linePath(tot), fill: "none", stroke: "#22d3ee", "stroke-width": 2.4 }));
  }

  function applyDistillRun(run) {
    if (!run) return;
    $("btn-distill-stop").disabled = run.status !== "running";
    const m = run.metrics || [];
    const last = m[m.length - 1] || {};
    $("dk-step").textContent = last.step !== undefined ? fmtNum(last.step) : "0";
    $("dk-loss").textContent = typeof last.loss === "number" ? last.loss.toFixed(4) : "—";
    $("dk-ce").textContent = typeof last.ce_loss === "number" ? last.ce_loss.toFixed(4) : "—";
    $("dk-kl").textContent = typeof last.kl_loss === "number" ? last.kl_loss.toFixed(4) : "—";
    const vs = run.verify_stats || {};
    $("dk-pass").textContent = typeof vs.pass_rate === "number" ? `${(vs.pass_rate * 100).toFixed(0)}%` : "—";
    const dc = run.dcheck_receipt;
    if (dc) {
      $("distill-dcheck-meta").textContent = `dcheck PASS · ${dc.samples} params · mxrel=${Number(dc.mxrel_milli || 0).toFixed(6)} (${dc.retries || 0} retries)`;
    }
    const stageMap = { harvest: "dnode-src", verify: "dnode-ver", distill: "dnode-stu", dcheck: "dnode-stu" };
    ["dnode-src", "dnode-ver", "dnode-fmt", "dnode-stu"].forEach((id) => {
      const n = $(id);
      if (n) n.classList.remove("active", "done");
    });
    if (run.status === "done") {
      ["dnode-src", "dnode-ver", "dnode-fmt", "dnode-stu"].forEach((id) => $(id)?.classList.add("done"));
    } else if (stageMap[run.stage]) {
      $(stageMap[run.stage])?.classList.add("active");
    }
    renderDistillChart(m);
  }

  async function refreshDistillRuns() {
    const res = await api("/api/distill/runs");
    if (!res.ok) return;
    const sel = $("distill-run-select");
    if (!sel) return;
    const prev = state.activeDistillId || sel.value;
    sel.replaceChildren(el("option", { value: "" }, "— live / recent runs —"));
    (res.runs || []).forEach((r) => {
      sel.appendChild(el("option", { value: r.run_id }, `${r.run_id} (${r.mode} · ${r.status})`));
    });
    if (prev) sel.value = prev;
  }

  async function loadDistillDetail(runId) {
    const res = await api(`/api/distill/runs/${encodeURIComponent(runId)}`);
    if (res.ok && res.run) applyDistillRun(res.run);
  }

  function subscribeDistillEvents(runId) {
    if (distillEs) { distillEs.close(); distillEs = null; }
    if ($("distill-term")) $("distill-term").textContent = "";
    distillEs = new EventSource(`/api/distill/runs/${encodeURIComponent(runId)}/events`);
    distillEs.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.kind === "log" && msg.text) appendDistillLog(msg.text);
      else if (msg.kind === "stage") appendDistillLog(`[stage] ${msg.label || msg.stage || ""}`);
      else if (msg.kind === "dcheck") { appendDistillLog(`[dcheck] PASS samples=${msg.samples} mxrel=${msg.mxrel_milli}`); loadDistillDetail(runId); }
      else if (msg.kind === "metric") loadDistillDetail(runId);
      else if (msg.kind === "done") {
        appendDistillLog(`[done] bom=${msg.bom_sha256 || ""}`);
        sfx.celebrate();
        celebrateBurst();
        loadDistillDetail(runId);
        refreshDistillRuns();
        distillEs.close();
      } else if (msg.kind === "error") {
        appendDistillLog(`[error] ${msg.error || "failed"}`);
        sfx.err();
        toast(msg.error || "Distillation failed", "err");
        distillEs.close();
      }
    };
  }

  async function startDistillRun() {
    sfx.tap();
    const vers = Array.from(document.querySelectorAll('input[name="d-ver"]:checked')).map((i) => i.value);
    const traces = $("d-traces").value.split("\n").map((s) => s.trim()).filter(Boolean);
    const prompts = $("d-prompts").value.split("\n").map((s) => s.trim()).filter(Boolean);
    const tUrl = $("d-teacher-url").value.trim();
    const tMod = $("d-teacher-model").value.trim();
    const spec = {
      mode: $("d-mode").value,
      arch: $("d-arch").value,
      K: Number($("d-k").value || 8),
      alpha: Number($("d-alpha").value || 0.7),
      temperature: Number($("d-tau").value || 1.5),
      steps: Number($("d-steps").value || 10),
      verifiers: vers.length ? vers : ["nonempty"],
      trace_paths: traces,
      prompts: prompts,
      teachers: tUrl ? [{ kind: "openai_compat", base_url: tUrl, model: tMod || "q36-35b" }] : [],
    };
    const res = await api("/api/distill/runs", { method: "POST", body: spec });
    if (!res.ok) {
      toast(res.error || "Invalid distillation spec", "err");
      sfx.err();
      return;
    }
    state.activeDistillId = res.run_id;
    toast(`Distill run started: ${res.run_id}`, "ok");
    await refreshDistillRuns();
    subscribeDistillEvents(res.run_id);
  }

  async function boot() {
    initHelixAnimation();
    bindTabs();
    renderPresets();
    buildSliderGroup("geo-sliders", GEO_SLIDERS);
    buildSliderGroup("loop-sliders", LOOP_SLIDERS);
    buildSliderGroup("train-sliders", TRAIN_SLIDERS);
    buildPlaySliders();
    renderDataSlots();
    syncDesignControlsFromSpec();
    renderStepper("", "idle");
    renderChart([]);
    renderDistillChart([]);
    renderDag("dag-svg", null);
    renderDag("forge-dag", null);
    bindEvents();
    await Promise.all([loadSystem(), loadCatalog(), refreshEstimate(), refreshRuns(), refreshRemotes()]);
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
