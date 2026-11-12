/* EC Racing viewer: the training view's interface (spec 3.18 §2, Figure 2) — the panel with every statistic of §10.2
   and the best shown car, the progress card, the Training setup drawer, the top bar's Pause and the training keys.
   Classic script, after main.js; it owns #panel, #drawer, #progress and #hint while the panel's mode is "train" and
   hands them back otherwise. */
(function () {
  "use strict";
  const ECR = window.ECR, ui = ECR.ui, el = ui.el;
  const {chip, setChip, swatch, clock} = ui;

  const MPH_PER_MPS = 2.23694;
  const READOUT_MS = 100;                                  // the best car's numbers change at most this often
  const SOURCES = [["notebook", "Notebook run"], ["app", "App training"]];
  const SHOWS = [["all", "All"], ["best", "Best"], ["worst", "Worst"]];
  const UNITS = {Generation: "generations", Trial: "trials", Epoch: "epochs"};     // the stats row heading the panel
  const HINT = "Drag to look around · scroll to zoom · C camera · R rays · Space play or pause";
  const PLOT = {
    share: {title: "Lowest map success share",
            foot: "Dashed: the share needed to pass. " +
                  "Ticks: a level cleared (tall), or a target cut or hardening (short)."},
    accuracy: {title: "Best and mean validation accuracy",
               foot: "Filled: the best validation accuracy so far. Line: the mean."},
  };
  const PLOT_H = 96, TICK = {tall: 12, short: 6};          // px
  const SVG = "http://www.w3.org/2000/svg";

  const sim = document.getElementById("sim");
  const panelEl = document.getElementById("panel");
  const drawerEl = document.getElementById("drawer");
  const progressEl = document.getElementById("progress");
  const hintEl = document.getElementById("hint");
  const pauseBtn = document.getElementById("pauseBtn");

  const dash = "–";
  const count = n => Number(n).toLocaleString("en-US");
  const child = (parent, tag, className, text) => parent.appendChild(el(tag, className, text));
  function button(parent, className, text, label) {
    const b = child(parent, "button", className, text);
    b.type = "button";
    if (label) b.setAttribute("aria-label", label);
    return b;
  }
  function svg(tag, attrs, parent) {
    const node = document.createElementNS(SVG, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  let active = false, lastReadout = 0, sliding = false, plotKey = "";

  // ---------- the panel: the generation, every statistic, the best shown car ----------
  const root = el("div", "train-panel");
  const title = child(root, "div", "title");
  const titleUnit = child(title, "span", "", "Generation");
  title.appendChild(document.createTextNode(" "));
  const titleValue = child(title, "span", "", "");
  const cap = child(root, "div", "cap");
  const who = child(cap, "span", "who");
  const whoSwatch = who.appendChild(swatch("transparent"));
  const whoName = child(who, "span", "", "");
  const sourceEl = child(cap, "span", "", "");
  const statsEl = child(root, "div", "stats");
  const noteEl = child(root, "div", "empty", "");
  const noteStatusEl = child(root, "div", "empty", "");
  child(root, "div", "sep");
  function readout(label) {
    const row = child(root, "div", "row");
    child(row, "span", "k", label);
    return child(row, "b", "", dash);
  }
  const bestEl = readout("Best car");
  const liveEl = readout("Cars still driving");
  const clockEl = readout("Replay time");

  // the unit of the panel's heading: the report's own row, else what the approach reports
  function unitOf(p) {
    const row = p.stats.find(([label]) => label in UNITS);
    if (row) return row;
    const unit = p.approach.startsWith("kt") ? "Trial" : p.approach === "baseline" ? "Epoch" : "Generation";
    return [unit, null];
  }

  // a wrapped value breaks only after a comma: "seed 395988678, level 1, | road 12.4-16.0 m"
  const keep = text => text.replace(/(\d)-(\d)/g, "$1\u2011$2").replace(/(\d) (m|mph)\b/g, "$1\u00a0$2")
    .replace(/([A-Za-z]) (\d)/g, "$1\u00a0$2");

  function makeStat() {
    const row = el("div", "row");
    child(row, "span", "k");
    const value = child(row, "b");
    value.appendChild(document.createTextNode(""));
    child(value, "span", "need");
    return row;
  }

  function updateStat(row, {label, value}) {
    const [k, b] = row.children;
    if (k.textContent !== label) k.textContent = label;
    const m = /^(.*?)\s*\(need (.+)\)$/.exec(value);        // "3% 1% 3%  (need 50%)": the requirement muted
    const main = keep(m ? m[1] : value), need = m ? `need ${m[2]}` : "";
    if (b.firstChild.nodeValue !== main) b.firstChild.nodeValue = main;
    if (b.lastChild.textContent !== need) b.lastChild.textContent = need;
    b.lastChild.hidden = !need;
    b.title = m ? `${main} (${need})` : value;
  }

  function renderPanel(p) {
    const [unit, value] = unitOf(p);
    if (titleUnit.textContent !== unit) titleUnit.textContent = unit;
    titleValue.textContent = value === null ? dash : value;
    titleValue.className = value === null ? "none" : "";
    const approach = p.approaches.find(a => a.active) || {label: p.approach};
    whoSwatch.style.background = p.color;
    whoName.textContent = `${approach.label} · run ${p.run.label}`;
    const folder = p.folder.label.replace(/^Runs:\s*/, "").replace(/\s*▸$/, "");
    sourceEl.textContent = p.source !== "notebook" ? "App training"
      : folder === "current" ? "Notebook run" : `Notebook run · ${folder}`;
    ui.sync(statsEl, p.stats.filter(([label]) => !(label in UNITS)).map(([label, text]) => ({key: label, label,
                                                                                              value: text})),
            makeStat, updateStat);
    noteEl.textContent = p.note || "";
    noteStatusEl.textContent = p.note && p.status && p.status !== p.note ? p.status : "";
  }

  function readouts(now) {
    const t = performance.now();
    if (!now && t - lastReadout < READOUT_MS) return;
    lastReadout = t;
    const f = ECR.frame, s = ECR.sample();
    clockEl.textContent = clock(f ? f.t : 0);
    if (!f || !s || !s.n) {
      bestEl.textContent = liveEl.textContent = dash;
      bestEl.classList.remove("done");
      return;
    }
    const i = Math.min(s.focus, s.n - 1), crashed = !!(f.crashed && f.crashed[i]), done = !!(f.done && f.done[i]);
    bestEl.textContent = crashed ? "crashed" : done ? "finished" : `${Math.round(s.v(i) * MPH_PER_MPS)} mph`;
    bestEl.classList.toggle("done", crashed || done);
    let live = 0;
    for (let k = 0; k < f.done.length; k++) if (!f.done[k]) live++;
    liveEl.textContent = `${count(live)} of ${count(s.n)}`;
  }

  // ---------- the progress card: the lowest map share (or the accuracy) per generation ----------
  const card = el("div", "train-progress");
  const cardCap = child(card, "div", "cap");
  const cardTitle = child(cardCap, "span", "", "");
  const cardCount = child(cardCap, "span", "", "");
  const plot = svg("svg", {class: "plot", "aria-hidden": "true", preserveAspectRatio: "none"}, card);
  const foot = child(card, "div", "foot", "");

  /* One point pair per pixel column at most: each bucket of points adds its highest value at its first generation and
     its lowest at its last, so a dense run keeps its spikes and dips (the mockup's envelope). */
  function trace(points, k, px, py, width) {
    const out = [], usable = points.filter(q => q[k] !== null && q[k] !== undefined);
    const bucket = Math.max(1, Math.ceil(usable.length / width));
    for (let i = 0; i < usable.length; i += bucket) {
      const part = usable.slice(i, i + bucket);
      let lo = Infinity, hi = -Infinity;
      for (const q of part) { lo = Math.min(lo, q[k]); hi = Math.max(hi, q[k]); }
      out.push([px(part[0][0]), py(hi)]);
      if (part.length > 1) out.push([px(part[part.length - 1][0]), py(lo)]);
    }
    return out;
  }
  const pathOf = pts => pts.map((q, i) => `${i ? "L" : "M"}${q[0].toFixed(1)},${q[1].toFixed(1)}`).join("");

  function drawPlot(history, color) {
    const points = history.points, width = Math.round(plot.getBoundingClientRect().width);
    if (!points.length || width < 10) return;
    const last = points[points.length - 1];
    const key = [history.kind, points.length, points[0][0], last[0], last[1], last[2], history.need, color, width]
      .join("|");
    if (key === plotKey) return;
    plotKey = key;
    plot.replaceChildren();
    plot.setAttribute("viewBox", `0 0 ${width} ${PLOT_H}`);
    const share = history.kind === "share";
    let lo = 0, hi = 1;
    const values = [];
    for (const q of points) for (const v of [q[1], q[2]]) if (v !== null && v !== undefined) values.push(v);
    if (!share && values.length) {           // accuracy: the range of both series, a little room above and below
      lo = Math.min(...values); hi = Math.max(...values);
      const room = Math.max((hi - lo) * 0.08, 0.005);
      lo -= room; hi += room;
    }
    const top = 6, bottom = PLOT_H - 4, g0 = points[0][0], spanG = Math.max(last[0] - g0, 1);
    const single = last[0] === g0;           // one generation so far (a resumed run may report it twice)
    const px = g => (single ? width : (g - g0) / spanG * width);
    const py = v => bottom - (v - lo) / (hi - lo) * (bottom - top);
    const crisp = y => Math.round(y) + 0.5;
    for (const u of [0.25, 0.5, 0.75]) {
      const y = crisp(bottom - u * (bottom - top));
      svg("line", {x1: 0, x2: width, y1: y, y2: y, stroke: "rgba(255,255,255,.06)"}, plot);
    }
    if (share && history.need !== null && history.need !== undefined) {
      const y = crisp(py(history.need));
      svg("line", {x1: 0, x2: width, y1: y, y2: y, stroke: "rgba(255,255,255,.55)", "stroke-dasharray": "4 4"}, plot);
    }
    if (!share && !single) {
      const mean = trace(points, 2, px, py, width);
      if (mean.length > 1) {
        svg("path", {d: pathOf(mean), fill: "none", stroke: "rgba(255,255,255,.42)", "stroke-width": 1.2,
                     "stroke-linejoin": "round"}, plot);
      }
    }
    const main = trace(points, 1, px, py, width);
    if (main.length > 1 && !single) {
      const line = pathOf(main);
      svg("path", {d: `${line}L${main[main.length - 1][0].toFixed(1)},${PLOT_H}L${main[0][0].toFixed(1)},${PLOT_H}Z`,
                   fill: color, opacity: 0.16, stroke: "none"}, plot);
      svg("path", {d: line, fill: "none", stroke: color, "stroke-width": 1.4, "stroke-linejoin": "round"}, plot);
    } else if (main.length) {                // one generation so far: a dot at the right
      const y = last[1] !== null && last[1] !== undefined ? py(last[1]) : main[main.length - 1][1];
      svg("circle", {cx: width - 2.5, cy: y, r: 2.5, fill: color}, plot);
    }
    for (const q of points) {                // the ticks over the series, so a busy run never hides them
      if (!TICK[q[3]]) continue;
      const x = crisp(Math.min(px(q[0]), width - 1));
      svg("line", {x1: x, x2: x, y1: PLOT_H - TICK[q[3]], y2: PLOT_H,
                   stroke: q[3] === "tall" ? "rgba(255,255,255,.78)" : "rgba(255,255,255,.55)"}, plot);
    }
  }

  function renderProgress(p, unit) {
    const h = p.history, kind = PLOT[h.kind] || PLOT.share;
    const n = h.points.length, units = new Set(h.points.map(q => q[0])).size;    // a generation can repeat
    if (cardTitle.textContent !== kind.title) cardTitle.textContent = kind.title;
    cardCount.textContent = `${count(units)} ${units === 1 ? unit.toLowerCase() : UNITS[unit]}`;
    if (foot.dataset.text !== kind.foot) {     // one sentence per line
      foot.dataset.text = kind.foot;
      foot.replaceChildren(...kind.foot.split(/(?<=\.) /).map(text => el("span", "", text)));
    }
    placeBottom(n > 0);
    if (!progressEl.hidden) drawPlot(h, p.color);
  }

  // bottom left: the progress card once there is a generation, else the hint; the card gives way to a tall panel
  function placeBottom(hasPoints) {
    progressEl.hidden = !hasPoints;
    if (hasPoints && progressEl.offsetTop < panelEl.offsetTop + panelEl.offsetHeight + 8) progressEl.hidden = true;
    hintEl.textContent = HINT;
    hintEl.hidden = !progressEl.hidden;
  }

  // ---------- the drawer: Training setup ----------
  const setup = el("div", "train-setup");
  const heading = text => child(setup, "div", "h", text);
  function setting(label) {
    const row = child(setup, "div", "drow");
    child(row, "span", "k", label);
    return row;
  }
  function stepper(row, down, up, what) {
    const box = child(row, "div", "stepper");
    const less = button(box, "chip icon", undefined, `${what}: ${down === "minus" ? "smaller" : "previous"}`);
    less.appendChild(ui.icon(down));
    const value = child(box, "span", "val", "");
    const more = button(box, "chip icon", undefined, `${what}: ${up === "plus" ? "larger" : "next"}`);
    more.appendChild(ui.icon(up));
    return {box, less, value, more};
  }
  heading("Approach");
  const approachBox = child(setup, "div", "chips");
  heading("Source");
  const sourceBox = child(setup, "div", "chips");
  const folderRow = setting("Results folder");
  const folderBtn = button(folderRow, "chip cycle");
  folderBtn.appendChild(document.createTextNode(""));
  folderBtn.appendChild(ui.icon("next"));
  folderBtn.title = "Next results folder";
  const run = stepper(setting("Run"), "prev", "next", "Run");
  heading("Difficulty");
  const levelBox = child(setup, "div", "chips");
  const popRow = setting("Population");
  const pop = stepper(popRow, "minus", "plus", "Population");
  const shownHead = child(setup, "div", "h shown-head");
  child(shownHead, "span", "", "Cars shown");
  const shownValue = child(shownHead, "b");
  const shownCount = shownValue.appendChild(document.createTextNode(""));
  const shownOf = child(shownValue, "small", "", "");
  const showBox = child(setup, "div", "chips show");
  const range = child(setup, "input", "range");
  Object.assign(range, {type: "range", min: 1, max: 1, step: 1, value: 1});
  range.setAttribute("aria-label", "Cars shown");
  const gripRow = setting("Grip");
  const gripBox = child(gripRow, "div", "chips");
  const netRow = setting("Network window");
  const netBtn = button(netRow, "switch", undefined, "Network window");
  netBtn.setAttribute("role", "switch");
  const go = child(setup, "div", "go");
  const playBtn = button(go, "primary", "Play");
  const stopBtn = button(go, "", "Stop");
  const newBtn = button(go, "", "New run");
  newBtn.title = "Delete the saved run of this approach and run, then train afresh";
  const statusEl = child(setup, "p", "status", "");

  function showShown(value, size) {
    shownCount.nodeValue = count(value);
    shownOf.textContent = ` of ${count(size)}`;
    const u = size > 1 ? (value - 1) / (size - 1) : 1;      // the fill ends under the knob's centre
    range.style.setProperty("--fill", `calc(6px + (100% - 12px) * ${u.toFixed(4)})`);
  }

  function renderSetup(p) {
    ui.sync(approachBox, p.approaches.map(a => ({key: a.id, a})),
            item => chip(item.a.label, {dot: true}),
            (b, {a}) => setChip(b, {on: a.active, enabled: a.enabled, dot: a.color}));
    const idle = p.run.enabled;                               // source, run and approach lock while training runs
    ui.sync(sourceBox, SOURCES.map(([id, label]) => ({key: id, label})),
            item => chip(item.label),
            (b, item) => setChip(b, {on: item.key === p.source, enabled: idle}));
    const folder = p.folder.label.replace(/^Runs:\s*/, "").replace(/\s*▸$/, "");
    if (folderBtn.firstChild.nodeValue !== folder) folderBtn.firstChild.nodeValue = folder;
    folderBtn.disabled = !p.folder.enabled;
    run.value.textContent = p.run.label;
    run.less.disabled = run.more.disabled = !p.run.enabled;
    run.box.classList.toggle("off", !p.run.enabled);
    const level = p.difficulty, here = p.source === "app";    // the difficulty of training here, not of a notebook run
    ui.sync(levelBox, level.choices.map(([id, label]) => ({key: String(id), id, label})),
            item => chip(item.label),
            (b, item) => setChip(b, {on: here && item.id === level.active, enabled: level.enabled}));
    popRow.hidden = p.population === null;
    if (p.population) {
      pop.value.textContent = count(p.population.value);
      pop.less.disabled = pop.more.disabled = !p.population.enabled;
      pop.box.classList.toggle("off", !p.population.enabled);
    }
    ui.sync(showBox, SHOWS.map(([id, label]) => ({key: id, label})),
            item => chip(item.label),
            (b, item) => setChip(b, {on: item.key === p.show}));
    if (!sliding) {
      range.max = Math.max(p.size, 1);
      range.value = p.shown;
      range.disabled = p.size <= 1;
      showShown(p.shown, p.size);
    }
    ui.sync(gripBox, ui.GRIPS.map(([id, label]) => ({key: id, label})),
            item => chip(item.label),
            (b, item) => setChip(b, {on: item.key === p.grip.name, enabled: p.grip.enabled}));
    netBtn.classList.toggle("on", p.network_open);
    netBtn.setAttribute("aria-checked", String(p.network_open));
    playBtn.textContent = p.play.label;
    stopBtn.disabled = !p.stop.enabled;
    newBtn.disabled = !p.new_run.enabled;
    statusEl.textContent = p.status;
  }

  // the drawer's buttons: one listener, the button tells what it is
  setup.addEventListener("click", e => {
    const b = e.target.closest("button");
    const p = ECR.panel;
    if (!b || b.disabled || !p || !active) return;
    const box = b.parentNode, key = b.dataset.key, on = b.classList.contains("on");
    if (box === approachBox) { if (!on) ui.command("approach", key); }
    else if (box === sourceBox) { if (!on) ui.command("source"); }
    else if (b === folderBtn) ui.command("folder");
    else if (b === run.less) ui.command("run", -1);
    else if (b === run.more) ui.command("run", 1);
    else if (box === levelBox) { if (!on) ui.command("difficulty", Number(key)); }
    else if (b === pop.less) ui.command("population", -1);
    else if (b === pop.more) ui.command("population", 1);
    else if (box === showBox) { if (!on) ui.command("show", key); }
    else if (box === gripBox) { if (!on) ui.command("grip", key); }
    else if (b === netBtn) ui.command("network");
    else if (b === playBtn) ui.command("play");
    else if (b === stopBtn) ui.command("stop");
    else if (b === newBtn) ui.command("new_run");
  });

  // the slider shows its value while it moves and sends it when let go
  range.addEventListener("pointerdown", () => { sliding = true; });
  range.addEventListener("input", () => {
    sliding = true;
    showShown(Number(range.value), Number(range.max));
  });
  range.addEventListener("change", () => {
    sliding = false;
    if (active && ECR.panel && Number(range.value) !== ECR.panel.shown) ui.command("shown", Number(range.value));
  });
  const letGo = () => { if (sliding) setTimeout(() => { sliding = false; }, 0); };
  range.addEventListener("pointerup", letGo);
  range.addEventListener("pointercancel", letGo);

  // ---------- the top bar's Pause: holds the replay or the training; a speed lets it go on ----------
  const canPause = p => p.source === "notebook" || p.stop.enabled;     // training here pauses once it runs
  const isPaused = p => canPause(p) && !p.play.active;
  pauseBtn.addEventListener("click", () => {
    const p = ECR.panel;
    if (p && p.mode === "train" && !pauseBtn.disabled) ui.command("play");
  });
  document.getElementById("speeds").addEventListener("click", e => {
    const b = e.target.closest("[data-speed]"), p = ECR.panel;
    if (b && p && p.mode === "train" && isPaused(p)) ui.command("play");
  });

  // ---------- keys: Space plays or pauses ----------
  ui.onKey((e, key) => {
    if (!active || key !== " ") return false;
    if (!e.repeat) ui.command("play");
    return true;
  });

  // ---------- updates ----------
  function activate(on) {
    if (on === active) return;
    active = on;
    if (on) {
      panelEl.replaceChildren(root);
      panelEl.hidden = false;
      drawerEl.replaceChildren(setup);
      progressEl.replaceChildren(card);
      plotKey = "";
    } else {
      if (root.parentNode === panelEl) {
        root.remove();
        if (!panelEl.firstElementChild) panelEl.hidden = true;
      }
      if (setup.parentNode === drawerEl) setup.remove();
      if (card.parentNode === progressEl) card.remove();
      progressEl.hidden = true;
      if (hintEl.textContent === HINT) hintEl.hidden = true;
      sliding = false;
    }
  }

  function render(p) {
    activate(p.mode === "train");
    if (!active) return;
    const paused = isPaused(p);
    pauseBtn.disabled = !canPause(p);
    pauseBtn.classList.toggle("on", paused);
    pauseBtn.setAttribute("aria-pressed", String(paused));
    if (paused) for (const b of document.querySelectorAll("#speeds [data-speed]")) b.classList.remove("on");
    renderPanel(p);
    renderSetup(p);
    renderProgress(p, unitOf(p)[0]);
    readouts(true);
  }

  ECR.onPanel(render);
  ECR.onFrame(() => { if (active) readouts(false); });
  ECR.onScene(() => { if (active) readouts(true); });
  new ResizeObserver(() => {
    if (active && ECR.panel) renderProgress(ECR.panel, unitOf(ECR.panel)[0]);
  }).observe(sim);
})();
