/* EC Racing viewer: the race view's interface (spec 3.18 §2, Figure 1) — the panel with the followed car and the
   leaderboard of §10.3, the Pause button, the Race setup drawer and the race keys. Classic script, after main.js;
   it owns #panel, #drawer and #hint while the panel's mode is "race" and hands them back otherwise. */
(function () {
  "use strict";
  const ECR = window.ECR, ui = ECR.ui, el = ui.el;
  const {chip, setChip, swatch, clock} = ui;

  const MPH_PER_MPS = 2.23694;
  const RACE_LAPS = 3;                                     // config.RACE_LAPS
  const READOUT_MS = 100;                                  // the followed car's numbers change at most this often
  const RUNS = [["best", "Best runs"], ["all", "All runs"], ["seeded", "Seeded"], ["unseeded", "Unseeded"]];
  const GRIPS = ui.GRIPS;
  const GROUPS = [["final", "Reported"], ["app", "App cars"]];
  const HINT = "Click a leaderboard row to follow that car · drag to look around · scroll to zoom · " +
               "C camera · R rays · ← → track · Space start or stop";

  const sim = document.getElementById("sim");
  const panelEl = document.getElementById("panel");
  const drawerEl = document.getElementById("drawer");
  const hintEl = document.getElementById("hint");
  const pauseBtn = document.getElementById("pauseBtn");

  const dash = "–";
  const seconds = value => (value === null || value === undefined ? dash : value.toFixed(1));
  const child = (parent, tag, className, text) => parent.appendChild(el(tag, className, text));

  let active = false, focus = -1, lastReadout = 0;

  // ---------- the panel: the followed car, the leaderboard, the race time ----------
  const root = el("div", "race-panel");
  const speedEl = child(child(root, "div", "speed"), "span", "", "0");
  speedEl.parentNode.appendChild(el("small", "", "mph"));
  const cap = child(root, "div", "cap");
  const who = child(cap, "span", "who");
  const whoSwatch = who.appendChild(swatch("transparent"));
  const whoName = child(who, "span", "nm", "");
  const lapEl = child(cap, "span", "", "");
  function readout(label) {
    const row = child(root, "div", "row");
    child(row, "span", "k", label);
    return child(row, "b", "", dash);
  }
  const actionEl = readout("Action");
  const sensorEl = readout("Sensor length");
  const edgeEl = readout("Nearest edge");
  child(root, "div", "sep");
  const boardHead = child(root, "div", "board-head");
  child(boardHead, "span", "", `Leaderboard · ${RACE_LAPS} laps`);
  const leaderBtn = child(boardHead, "button", "lead", "Follow leader");
  leaderBtn.type = "button";
  const board = child(root, "div", "board");
  const head = child(board, "div", "lb th");
  for (const [text, cls] of [["", ""], ["Car", ""], ["Laps", "n"], ["Last", "n"], ["Best", "n"], ["Gap", "n gap"],
                             ["Status", "st"]]) child(head, "span", cls, text);
  const rowsEl = child(board, "div", "rows");
  const moreEl = child(board, "div", "more", "");
  child(root, "div", "sep");
  const clockEl = readout("Race time");
  const emptyEl = el("div", "empty", "");

  // "KT Hyperband (seeded)" → ["KT Hyperband", "(seeded)"]: a narrow row shortens the method, never the run
  const RUN_TAG = /^(.+?) ((?:\(app\) )?s\d+|r\d+|\(seeded\)|\(pole\))$/;
  function splitName(name) {
    const m = RUN_TAG.exec(name);
    return m ? [m[1], m[2]] : [name, ""];
  }

  function makeRow(item) {
    const row = el("div", "lb");
    child(row, "span", "p");
    const name = child(row, "span", "nm");
    name.appendChild(swatch("transparent"));
    child(name, "span", "lbl");
    child(name, "span", "run");
    for (const cls of ["n", "n", "n", "n gap", "st"]) child(row, "span", cls);
    if (item.row.car !== "pole") {
      row.setAttribute("role", "button");
      row.tabIndex = 0;
    } else {
      row.classList.add("pole");
    }
    return row;
  }

  function updateRow(row, {row: r, status}) {
    const cells = row.children;
    const set = (k, text) => { if (cells[k].textContent !== text) cells[k].textContent = text; };
    set(0, String(r.position));
    const name = cells[1], [label, run] = splitName(r.name);
    name.firstChild.style.background = r.color;
    if (name.children[1].textContent !== label) name.children[1].textContent = label;
    if (name.children[2].textContent !== run) name.children[2].textContent = run;
    name.title = r.name;
    set(2, String(r.laps));
    set(3, seconds(r.last));
    set(4, seconds(r.best));
    set(5, r.gap === null ? "" : r.gap);
    cells[5].classList.toggle("ahead", /^[-−]/.test(r.gap || ""));          // a flying lap faster than the pole
    set(6, status);
    cells[6].dataset.status = status;
    row.classList.toggle("out", status === "crashed");
    if (r.car !== "pole") row.setAttribute("aria-label", `Follow ${r.name}`);
  }

  // as many rows as fit above the hint, "… N more" for the rest
  function fitRows() {
    const rows = [...rowsEl.children];
    for (const row of rows) row.hidden = false;
    moreEl.hidden = true;
    const bottom = hintEl.hidden ? sim.clientHeight - 14 : hintEl.offsetTop - 8;
    const room = bottom - panelEl.offsetTop;
    const height = panelEl.scrollHeight + panelEl.offsetHeight - panelEl.clientHeight;    // uncut, with borders
    if (height <= room || !rows.length) return;
    const rowH = rows[0].offsetHeight || 21;
    const hide = Math.min(rows.length - 1, Math.ceil((height - room) / rowH) + 1);
    for (let k = rows.length - hide; k < rows.length; k++) rows[k].hidden = true;
    moreEl.hidden = false;
    moreEl.textContent = `… ${hide} more`;
  }

  // ---------- the drawer: Race setup ----------
  const setup = el("div", "race-setup");
  const heading = text => child(setup, "div", "h", text);
  heading("Track");
  const trackRow = child(setup, "div", "track");
  const prevBtn = child(trackRow, "button", "chip icon");
  prevBtn.type = "button";
  prevBtn.setAttribute("aria-label", "Previous track");
  prevBtn.appendChild(ui.icon("prev"));
  const trackName = child(trackRow, "div", "name");
  const trackTitle = child(trackName, "b", "", "");
  const trackNote = child(trackName, "small", "", "");
  const nextBtn = child(trackRow, "button", "chip icon");
  nextBtn.type = "button";
  nextBtn.setAttribute("aria-label", "Next track");
  nextBtn.appendChild(ui.icon("next"));
  heading("Difficulty");
  const levelBox = child(setup, "div", "chips");
  heading("Who races");
  const runsBox = child(setup, "div", "chips");
  const groupHead = heading("Controllers");
  const groupBox = child(setup, "div", "chips");
  const methodsHead = heading("Methods");
  const methodsBox = child(setup, "div", "chips");
  const appsBox = child(setup, "div", "apps");
  function setting(label) {
    const row = child(setup, "div", "drow");
    child(row, "span", "k", label);
    return row;
  }
  const gripRow = setting("Grip");
  const gripBox = child(gripRow, "div", "chips");
  const folderRow = setting("Results folder");
  folderRow.classList.add("keep");                         // the only control left when a folder has no cars
  const folderBtn = child(folderRow, "button", "chip cycle");
  folderBtn.type = "button";
  folderBtn.appendChild(document.createTextNode(""));
  folderBtn.appendChild(ui.icon("next"));
  const netRow = setting("Network window");
  const netBtn = child(netRow, "button", "switch");
  netBtn.type = "button";
  netBtn.setAttribute("role", "switch");
  netBtn.setAttribute("aria-label", "Network window");
  const go = child(setup, "div", "go");
  const startBtn = child(go, "button", "primary", "Start race");
  startBtn.type = "button";
  const stopBtn = child(go, "button", "", "Stop");
  stopBtn.type = "button";
  const statusEl = child(setup, "p", "status keep", "");

  function renderSetup(p) {
    const name = p.track.name, cut = name.indexOf(" · ");
    trackTitle.textContent = cut < 0 ? name : name.slice(0, cut);
    trackNote.textContent = cut < 0 ? "" : name.slice(cut + 3);
    prevBtn.disabled = nextBtn.disabled = !p.track.enabled;

    const level = p.race_level;
    ui.sync(levelBox, level.choices.map(([id, label]) => ({key: String(id), id, label})),
            item => chip(item.label),
            (b, item) => setChip(b, {on: item.id === level.active, enabled: level.enabled}));
    ui.sync(runsBox, RUNS.map(([id, label]) => ({key: id, label})),
            item => chip(item.label),
            (b, item) => setChip(b, {on: item.key === p.race_runs.value, enabled: p.race_runs.enabled}));

    groupHead.hidden = groupBox.hidden = !p.group.available;
    ui.sync(groupBox, p.group.available ? GROUPS.map(([id, label]) => ({key: id, label})) : [],
            item => chip(item.label),
            (b, item) => setChip(b, {on: item.key === p.group.value}));

    const apps = p.group.available && p.group.value === "app";
    methodsHead.textContent = apps ? "App cars" : "Methods";
    methodsBox.hidden = apps;
    appsBox.hidden = !apps;
    ui.sync(methodsBox, apps ? [] : p.methods.map(m => ({key: m.id, m})),
            item => chip(item.m.label, {dot: true}),
            (b, {m}) => setChip(b, {on: m.on, enabled: m.enabled, dot: m.color}));
    ui.sync(appsBox, apps ? p.app_choices.map(a => ({key: a.id, a})) : [],
            ({a}) => {
              const row = el("div", "app-row");
              const label = child(row, "span", "who");
              label.appendChild(swatch(a.color));
              label.appendChild(document.createTextNode(a.label));
              const b = child(row, "button", "chip cycle");
              b.type = "button";
              b.appendChild(document.createTextNode(""));
              b.appendChild(ui.icon("next"));
              return row;
            },
            (row, {a}) => {
              const b = row.lastChild;
              b.disabled = !a.enabled;
              b.classList.toggle("on", a.choice !== "off");
              if (b.firstChild.nodeValue !== a.choice) b.firstChild.nodeValue = a.choice;
              b.setAttribute("aria-label", `${a.label}: ${a.choice}`);
            });

    ui.sync(gripBox, GRIPS.map(([id, label]) => ({key: id, label})),
            item => chip(item.label),
            (b, item) => setChip(b, {on: item.key === p.grip.name, enabled: p.grip.enabled}));
    const folder = p.folder.label.replace(/^Runs:\s*/, "").replace(/\s*▸$/, "");
    if (folderBtn.firstChild.nodeValue !== folder) folderBtn.firstChild.nodeValue = folder;
    folderBtn.disabled = !p.folder.enabled;
    folderBtn.title = "Next results folder";
    netBtn.classList.toggle("on", p.network_open);
    netBtn.setAttribute("aria-checked", String(p.network_open));
    startBtn.disabled = !p.start.enabled;
    stopBtn.disabled = !p.stop.enabled;

    setup.classList.toggle("none", !p.methods.length && !p.app_choices.length);     // no controllers here
    statusEl.textContent = p.status;
  }

  // the drawer's buttons: one listener, the button tells what it is
  setup.addEventListener("click", e => {
    const b = e.target.closest("button");
    const p = ECR.panel;
    if (!b || b.disabled || !p) return;
    const box = b.parentNode, key = b.dataset.key;
    if (b === prevBtn) ui.command("track", -1);
    else if (b === nextBtn) ui.command("track", 1);
    else if (box === levelBox) ui.command("race_level", Number(key));
    else if (box === runsBox) ui.command("race_runs", key);
    else if (box === groupBox) { if (key !== p.group.value) ui.command("group"); }
    else if (box === methodsBox) ui.command("method", key);
    else if (box.classList.contains("app-row")) ui.command("app_choice", box.dataset.key);
    else if (box === gripBox) ui.command("grip", key);
    else if (b === folderBtn) ui.command("folder");
    else if (b === netBtn) ui.command("network");
    else if (b === startBtn) ui.command("start");
    else if (b === stopBtn) ui.command("stop");
  });

  // ---------- the leaderboard: follow a car, or the leader ----------
  function follow(row) {
    if (row && row.getAttribute("role") === "button") ui.command("follow", Number(row.dataset.key));
  }
  rowsEl.addEventListener("click", e => follow(e.target.closest(".lb")));
  rowsEl.addEventListener("keydown", e => {
    if (e.key === "Enter") follow(e.target.closest(".lb"));
  });
  leaderBtn.addEventListener("click", () => { if (!leaderBtn.disabled) ui.command("follow", null); });

  // ---------- Pause ----------
  pauseBtn.addEventListener("click", () => {
    const p = ECR.panel;
    if (p && p.mode === "race" && !pauseBtn.disabled) ui.command("pause", !p.paused);
  });
  document.getElementById("speeds").addEventListener("click", e => {
    const b = e.target.closest("[data-speed]");            // a speed while paused also lets the race go on
    if (b && ECR.panel && ECR.panel.mode === "race" && ECR.panel.paused) ui.command("pause", false);
  });

  // ---------- keys: ← → track, Space starts or stops the race ----------
  ui.onKey((e, key) => {
    const p = ECR.panel;
    if (!active || !p) return false;
    if (key === "ArrowLeft" || key === "ArrowRight") {
      if (p.track.enabled && !e.repeat) ui.command("track", key === "ArrowLeft" ? -1 : 1);
      return true;
    }
    if (key === " ") {
      if (e.repeat) return true;
      if (p.racing) ui.command("stop");
      else if (p.start.enabled) ui.command("start");
      return true;
    }
    return false;
  });

  // ---------- updates ----------
  function activate(on) {
    if (on === active) return;
    active = on;
    if (on) {
      panelEl.replaceChildren(root);
      panelEl.classList.add("race");
      panelEl.hidden = false;
      drawerEl.replaceChildren(setup);
      hintEl.textContent = HINT;
      hintEl.hidden = false;
    } else {
      if (root.parentNode === panelEl) {
        root.remove();
        panelEl.classList.remove("race");
        if (!panelEl.firstElementChild) panelEl.hidden = true;
      }
      if (setup.parentNode === drawerEl) setup.remove();
      if (hintEl.textContent === HINT) hintEl.hidden = true;
    }
  }

  function render(p) {
    activate(p.mode === "race");
    if (!active) return;
    pauseBtn.disabled = !p.racing;
    pauseBtn.classList.toggle("on", !!p.paused);
    pauseBtn.setAttribute("aria-pressed", String(!!p.paused));
    if (p.paused) for (const b of document.querySelectorAll("#speeds [data-speed]")) b.classList.remove("on");

    board.classList.toggle("nogap", p.pole === null);
    const rows = p.leaderboard.map(r => ({key: String(r.car), row: r, status: p.racing ? r.status : "ready"}));
    ui.sync(rowsEl, rows, makeRow, updateRow);
    const none = !rows.length;
    board.hidden = boardHead.hidden = none;
    if (none) {
      emptyEl.textContent = p.status;
      if (emptyEl.parentNode !== root) root.insertBefore(emptyEl, board);
    } else {
      emptyEl.remove();
    }
    leaderBtn.disabled = p.follow === null;
    leaderBtn.textContent = p.follow === null ? "Following the leader" : "Follow leader";
    markFocus();
    renderSetup(p);
    readouts(true);
    fitRows();
  }

  function markFocus() {
    const f = ECR.frame ? ECR.frame.focus : -1;
    for (const row of rowsEl.children) row.classList.toggle("sel", row.dataset.key === String(f));
    focus = f;
  }

  function nearestEdge(rays) {
    let k = 0;
    for (let i = 1; i < rays.length; i++) if (rays[i] < rays[k]) k = i;
    const middle = (rays.length - 1) / 2;                    // rays run from −90° (right) to +90° (left)
    const side = k < middle ? "right" : k > middle ? "left" : "ahead";
    return `${rays[k].toFixed(1)} m ${side}`;
  }

  function readouts(now) {
    const p = ECR.panel, f = ECR.frame, s = ECR.sample();
    const t = performance.now();
    if (!now && t - lastReadout < READOUT_MS) return;
    lastReadout = t;
    if (!p || !f || !s || !s.n || !p.cars.length) {
      speedEl.textContent = "0";
      whoSwatch.style.background = "transparent";
      whoName.textContent = p && !p.cars.length ? "No cars" : "";
      lapEl.textContent = "";
      actionEl.textContent = sensorEl.textContent = edgeEl.textContent = dash;
      clockEl.textContent = clock(f ? f.t : 0);
      return;
    }
    const i = s.focus, car = p.cars[i] || {name: "", color: "transparent"};
    const entry = p.leaderboard.find(r => r.car === i);
    const status = entry ? (p.racing ? entry.status : "ready") : "ready";
    const crashed = !!(f.crashed && f.crashed[i]);
    speedEl.textContent = crashed ? dash : String(Math.round(s.v(i) * MPH_PER_MPS));
    whoSwatch.style.background = car.color;
    if (whoName.textContent !== car.name) whoName.textContent = car.name;
    lapEl.textContent = status === "crashed" || status === "finished" || status === "stopped" ? status
      : `lap ${Math.min((entry ? entry.laps : 0) + 1, RACE_LAPS)} of ${RACE_LAPS}`;
    actionEl.textContent = f.action || dash;
    const sensor = f.sensor && f.sensor[i];
    sensorEl.textContent = typeof sensor === "number" ? `${Math.round(sensor)} m` : dash;
    const rays = s.rays(i);
    edgeEl.textContent = rays && rays.length ? nearestEdge(rays) : dash;
    clockEl.textContent = clock(f.t);
  }

  ECR.onPanel(render);
  ECR.onFrame(f => {
    if (!active) return;
    if (f.focus !== focus) markFocus();
    readouts(false);
  });
  ECR.onScene(() => { if (active) readouts(true); });
  new ResizeObserver(() => { if (active) fitRows(); }).observe(sim);
})();
