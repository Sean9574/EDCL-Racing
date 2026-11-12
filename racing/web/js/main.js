/* EC Racing viewer: the page's core — the event stream from the viewer, commands back to it, the frame clock that
   interpolates the cars between frames, and the display loop (spec 3.18 §3). Classic script; the ECR namespace:
     ECR.send(name, arg)            POST /command; resolves to null when done, else the error text
     ECR.onPanel / onFrame / onScene / onView(fn)   listeners (onPanel also fires at once with the latest panel)
     ECR.panel, ECR.frame, ECR.sceneData             the latest panel, accepted frame and loaded scene (or null)
     ECR.view                       {camera: "chase"|"wheels"|"top"|"map", rays: "car"|"all"|"off"}, page-only
     ECR.sample()                   the cars as drawn now (interpolated), or null
     ECR.close(text)                end the page: stop listening, hide the scene, show `text` */
(function () {
  "use strict";
  const ECR = window.ECR = window.ECR || {};

  const angDiff = (a, b) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const CONNECT_GRACE = 1500;                // ms before "Connecting…" shows

  // ---------- page-only view state; ?camera=…&rays=… set the first one ----------
  const params = new URLSearchParams(location.search);
  const pick = (value, allowed, fallback) => (allowed.includes(value) ? value : fallback);
  ECR.view = {camera: pick(params.get("camera"), ["chase", "wheels", "top", "map"], "top"),
              rays: pick(params.get("rays"), ["car", "all", "off"], "car")};
  ECR.panel = null;
  ECR.frame = null;
  ECR.sceneData = null;

  // ---------- listeners ----------
  const listeners = {panel: [], frame: [], scene: [], view: []};
  function emit(kind, value) {
    for (const fn of listeners[kind]) {
      try { fn(value); } catch (error) { console.error(error); }
    }
  }
  ECR.onPanel = fn => {
    listeners.panel.push(fn);
    if (ECR.panel) {
      try { fn(ECR.panel); } catch (error) { console.error(error); }
    }
  };
  ECR.onFrame = fn => { listeners.frame.push(fn); };
  ECR.onScene = fn => { listeners.scene.push(fn); };
  ECR.onView = fn => { listeners.view.push(fn); };
  ECR.viewChanged = () => emit("view", ECR.view);       // called by ECR.scene.setCamera / setRays

  // ---------- commands ----------
  ECR.send = (name, arg) => fetch("/command", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name, arg: arg === undefined ? null : arg}),
  }).then(response => (response.ok ? null : response.text().then(text => text || response.statusText)))
    .catch(error => String(error));

  // ---------- messages over the scene ----------
  const sim = document.getElementById("sim");
  const noticeEl = document.getElementById("notice");
  const notices = {final: null, link: null, base: null};
  function showNotice() {
    const text = notices.final || notices.link || notices.base;
    noticeEl.hidden = !text;
    noticeEl.textContent = text || "";
  }

  // ---------- the frame clock: the last two states, interpolated at display rate ----------
  /* `a` is the state drawn when frame `b` arrived (typed arrays), `b` the newest frame; between arrivals the cars move
     from a to b over one frame interval, so motion stays smooth whatever the frame rate. A new race or generation
     (fewer or more cars, time going back, another mode) snaps instead of sliding the cars across the map. */
  const clock = {a: null, b: null, at: 0, interval: 1 / 30, snapped: true, kappa: new Float64Array(0), ghostKappa: 0};

  function stateOf(f) {
    return {t: f.t, x: Float64Array.from(f.x), y: Float64Array.from(f.y), h: Float64Array.from(f.h),
            v: Float64Array.from(f.v), rays: f.rays || {}, ghost: f.ghost};
  }

  function continues(prev, f) {
    return prev && prev.mode === f.mode && prev.x.length === f.x.length && f.t >= prev.t && f.t - prev.t < 3;
  }

  function lerpState(a, b, u) {
    const n = b.x.length, out = {t: a.t + (b.t - a.t) * u, x: new Float64Array(n), y: new Float64Array(n),
                                 h: new Float64Array(n), v: new Float64Array(n), rays: {}, ghost: null};
    for (let i = 0; i < n; i++) {
      out.x[i] = a.x[i] + (b.x[i] - a.x[i]) * u;
      out.y[i] = a.y[i] + (b.y[i] - a.y[i]) * u;
      out.h[i] = a.h[i] + angDiff(b.h[i], a.h[i]) * u;
      out.v[i] = a.v[i] + (b.v[i] - a.v[i]) * u;
    }
    for (const k in b.rays) out.rays[k] = lerpRays(a.rays[k], b.rays[k], u);
    out.ghost = lerpGhost(a.ghost, b.ghost, u);
    return out;
  }

  function lerpRays(ra, rb, u) {
    if (!ra || ra.length !== rb.length) return rb;
    return rb.map((r, k) => ra[k] + (r - ra[k]) * u);
  }

  function lerpGhost(ga, gb, u) {
    if (!gb) return null;
    if (!ga) return gb;
    return [ga[0] + (gb[0] - ga[0]) * u, ga[1] + (gb[1] - ga[1]) * u, ga[2] + angDiff(gb[2], ga[2]) * u];
  }

  // the path's curvature (1/m) over the last frame step, for the front wheels of cars without a steering command
  function curvature(prev, f) {
    const n = f.x.length, k = clock.kappa.length === n ? clock.kappa : new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const ds = Math.hypot(f.x[i] - prev.x[i], f.y[i] - prev.y[i]);
      if (ds > 0.05) k[i] = angDiff(f.h[i], prev.h[i]) / ds;
    }
    clock.kappa = k;
    if (prev.ghost && f.ghost) {
      const ds = Math.hypot(f.ghost[0] - prev.ghost[0], f.ghost[1] - prev.ghost[1]);
      if (ds > 0.05) clock.ghostKappa = angDiff(f.ghost[2], prev.ghost[2]) / ds;
    }
  }

  function progress(now) {
    return clamp((now - clock.at) / clock.interval, 0, 1);
  }

  function receiveFrame(f) {
    if (!ECR.sceneData || f.scene !== ECR.sceneData.key) return;         // a frame of another track
    const now = performance.now() / 1000, prev = clock.b;
    if (continues(prev, f)) {
      const gap = now - clock.at;
      let a = lerpState(clock.a, prev, progress(now));
      // more than two frame steps behind: frames came in a burst while the page was busy, so catch up rather than
      // slide the cars across the corners
      if (f.t - a.t > 2.5 * Math.max(f.t - prev.t, 1e-3)) a = stateOf(prev);
      clock.a = a;
      if (gap > 0.004) clock.interval = clamp(clock.interval * 0.8 + Math.min(gap, 0.25) * 0.2, 1 / 90, 0.25);
      curvature(prev, f);
      clock.snapped = false;
    } else {
      clock.a = stateOf(f);
      clock.kappa = new Float64Array(f.x.length);
      clock.ghostKappa = 0;
      clock.snapped = true;
    }
    clock.b = f;
    clock.at = now;
    ECR.frame = f;
    emit("frame", f);
  }

  function makeSample(now) {
    const a = clock.a, b = clock.b;
    if (!a || !b) return null;
    const u = progress(now), kappa = clock.kappa;
    const g = lerpGhost(a.ghost, b.ghost, u);
    return {
      n: b.x.length, mode: b.mode, t: b.t, focus: b.focus, action: b.action, done: b.done, crashed: b.crashed,
      snapped: clock.snapped,
      x: i => a.x[i] + (b.x[i] - a.x[i]) * u,
      y: i => a.y[i] + (b.y[i] - a.y[i]) * u,
      h: i => a.h[i] + angDiff(b.h[i], a.h[i]) * u,
      v: i => a.v[i] + (b.v[i] - a.v[i]) * u,
      kappa: i => kappa[i] || 0,
      rays: i => (b.rays && b.rays[i] ? lerpRays(a.rays[i], b.rays[i], u) : null),
      sensor: (i, lengths) => {
        const s = b.sensor && b.sensor[i];
        return typeof s === "number" ? s : Math.max(...lengths);
      },
      ghost: g ? {x: g[0], y: g[1], h: g[2], kappa: clock.ghostKappa} : null,
    };
  }
  ECR.sample = () => makeSample(performance.now() / 1000);

  // ---------- the event stream ----------
  function receiveScene(sc) {
    if (sc && ECR.sceneData && sc.key === ECR.sceneData.key) return;   // a reconnect re-sends the same track
    ECR.sceneData = sc;
    ECR.frame = null;
    clock.a = clock.b = null;
    if (ECR.scene.ok) {
      if (sc) ECR.scene.load(sc); else ECR.scene.clear();
    }
    emit("scene", sc);
  }

  let source = null, linkTimer = 0;
  function connect() {
    source = new EventSource("/events");
    const parse = (handler) => event => {
      let data;
      try { data = JSON.parse(event.data); } catch (error) { console.error("bad event", event.type, error); return; }
      handler(data);
    };
    source.addEventListener("scene", parse(receiveScene));
    source.addEventListener("panel", parse(panel => { ECR.panel = panel; emit("panel", panel); }));
    source.addEventListener("frame", parse(receiveFrame));
    source.onopen = () => {
      clearTimeout(linkTimer);
      linkTimer = 0;
      notices.link = null;
      showNotice();
    };
    source.onerror = () => {
      if (notices.final) return;
      if (source.readyState === EventSource.CLOSED) {
        notices.link = "The viewer is not reachable.";
        showNotice();
      } else if (!linkTimer && !notices.link) {
        linkTimer = setTimeout(() => {
          linkTimer = 0;
          if (source.readyState !== EventSource.OPEN && !notices.final) {
            notices.link = "Connecting to the viewer…";
            showNotice();
          }
        }, CONNECT_GRACE);
      }
    };
  }

  ECR.close = text => {
    notices.final = text || "The viewer has closed.";
    if (source) source.close();
    clearTimeout(linkTimer);
    sim.classList.add("closed");
    showNotice();
  };

  // ---------- the display loop ----------
  let last = performance.now(), failed = null;
  function tick(ms) {
    requestAnimationFrame(tick);                 // first, so one bad frame never stops the display
    const dt = clamp((ms - last) / 1000, 0, 0.25), now = ms / 1000;     // a slow machine still settles the camera
    last = ms;
    if (!ECR.scene.ok || notices.final) return;
    try {
      const s = ECR.scene.track ? makeSample(now) : null;
      ECR.scene.place(s, dt, now);
      ECR.cars.update(s, dt, ECR.panel);
      ECR.scene.render();
      if (ECR.scene.track) ECR.scene.drawMini(s, s ? ECR.cars.miniDots(s, ECR.panel) : null);
      failed = null;
    } catch (error) {
      if (String(error) !== failed) console.error("display frame failed:", error);   // each new problem once
      failed = String(error);
    }
  }

  if (!ECR.scene.init(sim)) {
    notices.base = "The 3D view needs WebGL.";
    showNotice();
  }
  connect();
  requestAnimationFrame(tick);
})();
