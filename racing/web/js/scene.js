/* EC Racing viewer: the 3D scene — renderer, lights, the track, the cameras and the mini-map (spec 3.18 §2).
   World coordinates are the simulator's (x, y) in metres; three.js gets (x, height, −y). Classic script: ECR.scene. */
(function () {
  "use strict";
  const ECR = window.ECR = window.ECR || {};
  const T = window.THREE;

  const BG = 0x1b1f25;                       // scene background
  const GREEN = 0x3fbf78;                    // racing line
  const WALL_HEIGHT = 0.9;                   // m
  const EDGE_LINE = [0.45, 0.7];             // m inside each road edge
  const LINE_HALF = 0.16;                    // m, half the racing-line ribbon's width
  const PRESETS = {                          // the car frame: forward, left, up (m)
    chase: {off: [-13, 0, 4.3], look: [14, 0, 0.8]},
    wheels: {off: [3.3, 3.0, 0.95], look: [1.45, 0.8, 0.3]},
    top: {off: [-10, 0, 95], look: [30, 0, 0]},
  };
  const CAMERAS = ["chase", "wheels", "top", "map"];
  const RAY_MODES = ["car", "all", "off"];
  const FOLLOW_GLIDE = 0.8;                  // s the camera takes to glide to a newly followed car
  const MINI_PAD = 40;                       // m around the track in the mini-map
  const MINI_MAX_HEIGHT = 240;               // px

  const angDiff = (a, b) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const ease = u => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

  const api = {ok: false, three: null, track: null, textures: null, CAMERAS, RAY_MODES};
  let host, R, scene, cam, sun, fog;
  const orbit = {yaw: 0, pitch: 0, zoom: 1, drag: null};
  let off, look, camHd = null;
  const follow = {index: -1, from: -1, since: 0};

  function hasWebGL() {
    try {
      const c = document.createElement("canvas");
      const gl = window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl"));
      const lose = gl && gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();                      // only a probe: free the context at once
      return !!gl;
    } catch (e) {
      return false;
    }
  }

  function canvasTexture(w, h, draw) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    draw(c.getContext("2d"), w, h);
    const t = new T.CanvasTexture(c);
    t.wrapS = t.wrapT = T.RepeatWrapping;
    t.anisotropy = 8;
    return t;
  }

  function makeTextures() {
    const asphalt = canvasTexture(256, 256, (g, w, h) => {
      g.fillStyle = "#3a3e44"; g.fillRect(0, 0, w, h);
      for (let i = 0; i < 9000; i++) {
        const v = 48 + Math.random() * 30 | 0;
        g.fillStyle = `rgba(${v},${v + 2},${v + 5},.55)`;
        g.fillRect(Math.random() * w, Math.random() * h, 1.4, 1.4);
      }
    });
    const wall = canvasTexture(8, 128, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#8e959e"); gr.addColorStop(0.07, "#8e959e"); gr.addColorStop(0.08, "#e2e5ea");
      gr.addColorStop(0.72, "#d7dbe0"); gr.addColorStop(1, "#6c727a");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
    });
    const glow = canvasTexture(64, 64, g => {
      const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
      r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.25, "rgba(255,255,255,.55)");
      r.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = r; g.fillRect(0, 0, 64, 64);
    });
    return {asphalt, wall, glow};
  }

  // ---------- set-up ----------
  function init(el) {
    host = el;
    if (!T || !hasWebGL()) return false;
    try {
      R = new T.WebGLRenderer({antialias: true});
    } catch (e) {
      return false;
    }
    R.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    R.setClearColor(BG, 1);                      // without a track the canvas shows the plain background
    R.shadowMap.enabled = true;
    R.shadowMap.type = T.PCFSoftShadowMap;
    R.domElement.className = "view";
    host.prepend(R.domElement);

    scene = new T.Scene();
    scene.background = new T.Color(BG);
    fog = new T.Fog(BG, 220, 900);
    scene.fog = fog;
    cam = new T.PerspectiveCamera(48, 1, 0.1, 8000);
    scene.add(new T.HemisphereLight(0xe4ebff, 0x2a2d33, 0.62));
    sun = new T.DirectionalLight(0xfff6ea, 0.95);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, {left: -60, right: 60, top: 60, bottom: -60, near: 1, far: 300});
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.04;                // no acne bands on surfaces lit at a grazing angle
    scene.add(sun, sun.target);

    const first = PRESETS[ECR.view.camera] || PRESETS.top;          // start at the chosen preset, no glide in
    off = new T.Vector3(...first.off);
    look = new T.Vector3(...first.look);
    api.three = scene;
    api.camera = cam;
    api.textures = makeTextures();
    controls(R.domElement);
    const resize = () => {
      const w = Math.max(host.clientWidth, 1), h = Math.max(host.clientHeight, 1);
      R.setSize(w, h, false);
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(host);
    resize();
    setCamera(ECR.view.camera, true);
    api.ok = true;
    return true;
  }

  // drag to orbit, wheel to zoom, double-click to reset
  function controls(cv) {
    cv.addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      orbit.drag = [e.clientX, e.clientY];
      cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener("pointermove", e => {
      if (!orbit.drag) return;
      orbit.yaw -= (e.clientX - orbit.drag[0]) * 0.006;
      orbit.pitch = clamp(orbit.pitch + (e.clientY - orbit.drag[1]) * 0.006, -0.9, 1.2);
      orbit.drag = [e.clientX, e.clientY];
    });
    const release = () => { orbit.drag = null; };
    cv.addEventListener("pointerup", release);
    cv.addEventListener("pointercancel", release);
    cv.addEventListener("wheel", e => {
      e.preventDefault();
      orbit.zoom = clamp(orbit.zoom * Math.exp(e.deltaY * 0.0012), 0.3, 4);
    }, {passive: false});
    cv.addEventListener("dblclick", () => { orbit.yaw = 0; orbit.pitch = 0; orbit.zoom = 1; });
  }

  // ---------- page-only view state ----------
  function setCamera(name, quiet) {
    if (!CAMERAS.includes(name)) throw new Error(`unknown camera ${name}`);
    const changed = ECR.view.camera !== name;
    ECR.view.camera = name;
    orbit.yaw = 0; orbit.pitch = 0; orbit.zoom = 1;
    if (host) host.classList.toggle("mapview", name === "map");
    if (changed && !quiet && ECR.viewChanged) ECR.viewChanged();
  }

  function setRays(mode) {
    if (!RAY_MODES.includes(mode)) throw new Error(`unknown ray mode ${mode}`);
    const changed = ECR.view.rays !== mode;
    ECR.view.rays = mode;
    if (changed && ECR.viewChanged) ECR.viewChanged();
  }

  // ---------- the track ----------
  function arcLengths(A, B) {
    const n = A.length, along = [0];
    for (let i = 1; i <= n; i++) {
      const a = A[i - 1], b = A[i % n], c = B[i - 1], d = B[i % n];
      along.push(along[i - 1] + (Math.hypot(b[0] - a[0], b[1] - a[1]) + Math.hypot(d[0] - c[0], d[1] - c[1])) / 2);
    }
    return along;
  }

  // a closed strip between two outlines at height y
  function band(A, B, y, uvScale) {
    const n = A.length, along = arcLengths(A, B), pos = [], uv = [], idx = [];
    for (let i = 0; i <= n; i++) {
      const a = A[i % n], b = B[i % n], wd = Math.hypot(b[0] - a[0], b[1] - a[1]);
      pos.push(a[0], y, -a[1], b[0], y, -b[1]);
      uv.push(along[i] / uvScale, 0, along[i] / uvScale, wd / uvScale);
    }
    for (let i = 0; i < n; i++) { const k = 2 * i; idx.push(k, k + 2, k + 1, k + 1, k + 2, k + 3); }
    const g = new T.BufferGeometry();
    g.setAttribute("position", new T.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new T.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }

  /* A closed vertical wall along outline A, its faces toward the road (the side the other outline `inside` lies
     on). The wall is single-sided, so a camera outside it (the Wheels camera beside a car at the edge) sees through. */
  function wall(A, inside, h0, h1) {
    const n = A.length, along = arcLengths(A, A), pos = [], uv = [], idx = [];
    for (let i = 0; i <= n; i++) {
      const p = A[i % n];
      pos.push(p[0], h0, -p[1], p[0], h1, -p[1]);
      uv.push(along[i] / 5, 0, along[i] / 5, 1);
    }
    let left = 0;                                // > 0: the road lies to the left of A's direction
    for (let i = 0; i < n; i++) {
      const a = A[i], b = A[(i + 1) % n], c = inside[i];
      left += (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    }
    for (let i = 0; i < n; i++) {                // (k, k+2, k+1) faces the right of A's direction
      const k = 2 * i;
      if (left > 0) idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
      else idx.push(k, k + 2, k + 1, k + 1, k + 2, k + 3);
    }
    const g = new T.BufferGeometry();
    g.setAttribute("position", new T.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new T.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }

  // points `offset` m from the left edge towards the right one (fromLeft), or from the right edge towards the left
  function inset(L, Rt, fromLeft, offset) {
    return L.map((a, i) => {
      const b = Rt[i], w = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, ux = (b[0] - a[0]) / w, uy = (b[1] - a[1]) / w;
      return fromLeft ? [a[0] + ux * offset, a[1] + uy * offset] : [b[0] - ux * offset, b[1] - uy * offset];
    });
  }

  function load(sc) {
    clear();
    const L = sc.left, Rt = sc.right, line = sc.line;
    if (!L || L.length < 3 || !Rt || Rt.length !== L.length) return;
    let mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
    for (const p of L.concat(Rt)) {
      mnx = Math.min(mnx, p[0]); mxx = Math.max(mxx, p[0]); mny = Math.min(mny, p[1]); mxy = Math.max(mxy, p[1]);
    }
    const cx = (mnx + mxx) / 2, cy = (mny + mxy) / 2, span = Math.max(mxx - mnx, mxy - mny, 50);
    const tex = api.textures, group = new T.Group();
    const add = (geometry, material, opts) => {
      const m = new T.Mesh(geometry, material);
      Object.assign(m, opts || {});
      group.add(m);
      return m;
    };

    // ground and a faint 10 m grid. The ground is drawn first and writes no depth, so the grid can lie well below
    // the road (where the depth test always hides it) and still show over the ground
    const ground = add(new T.PlaneGeometry(span * 4, span * 4),
                       new T.MeshStandardMaterial({color: 0x21262b, roughness: 1, depthWrite: false}),
                       {receiveShadow: true, renderOrder: -1});
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(cx, -0.03, -cy);
    const gs = Math.ceil(span * 4 / 10) * 10, grid = new T.GridHelper(gs, gs / 10, 0x323941, 0x2a3036);
    grid.position.set(cx, -0.25, -cy);
    grid.material.transparent = true;
    grid.material.opacity = 0.55;
    group.add(grid);

    // asphalt, white edge lines, the racing line, walls
    add(band(L, Rt, 0, 7), new T.MeshStandardMaterial({map: tex.asphalt, roughness: 0.93, side: T.DoubleSide}),
        {receiveShadow: true});
    const lineMat = new T.MeshBasicMaterial({color: 0xe9ebee, side: T.DoubleSide, polygonOffset: true,
                                             polygonOffsetFactor: -2, polygonOffsetUnits: -4});
    add(band(inset(L, Rt, true, EDGE_LINE[0]), inset(L, Rt, true, EDGE_LINE[1]), 0.012, 7), lineMat);
    add(band(inset(L, Rt, false, EDGE_LINE[1]), inset(L, Rt, false, EDGE_LINE[0]), 0.012, 7), lineMat);
    if (line && line.length >= 3) {
      const nl = line.length, normal = line.map((p, i) => {
        const q = line[(i + 1) % nl], dx = q[0] - p[0], dy = q[1] - p[1], m = Math.hypot(dx, dy) || 1;
        return [-dy / m, dx / m];
      });
      const side = s => line.map((p, i) => [p[0] + s * normal[i][0] * LINE_HALF, p[1] + s * normal[i][1] * LINE_HALF]);
      add(band(side(1), side(-1), 0.02, 7),
          new T.MeshBasicMaterial({color: GREEN, transparent: true, opacity: 0.55, side: T.DoubleSide, depthWrite: false,
                                   polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -6}));
    }
    const wallMat = new T.MeshStandardMaterial({map: tex.wall, roughness: 0.75, shadowSide: T.DoubleSide});
    add(wall(L, Rt, 0, WALL_HEIGHT), wallMat, {castShadow: true});           // walls cast on the road but take no
    add(wall(Rt, L, 0, WALL_HEIGHT), wallMat, {castShadow: true});           // shadow themselves: no acne bands

    // the whole-map road: wider and solid, above the walls, so the lap reads as one clean line from far above
    const pad = span / 450;
    const mapRoad = add(band(inset(L, Rt, true, -pad), inset(L, Rt, false, -pad), 1.2, 7),
                        new T.MeshBasicMaterial({color: 0x5a616b, side: T.DoubleSide}), {visible: false});

    // the start line
    const a = L[0], b = Rt[0], w = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const start = add(new T.PlaneGeometry(1, 1), new T.MeshBasicMaterial({color: 0xf3f5f8, side: T.DoubleSide,
                                                                          polygonOffset: true, polygonOffsetFactor: -2,
                                                                          polygonOffsetUnits: -4}));
    start.scale.set(1.2, w, 1);
    start.rotation.x = -Math.PI / 2;
    start.rotation.z = Math.atan2(b[1] - a[1], b[0] - a[0]) - Math.PI / 2;
    start.position.set((a[0] + b[0]) / 2, 0.014, -(a[1] + b[1]) / 2);

    scene.add(group);
    const ahead = line && line.length > 1 ? Math.atan2(line[1][1] - line[0][1], line[1][0] - line[0][0])
                                          : Math.atan2(b[1] - a[1], b[0] - a[0]) + Math.PI / 2;
    api.track = {key: sc.key, name: sc.name, left: L, right: Rt, group, grid, mapRoad, span, cx, cy,
                 box: {mnx, mxx, mny, mxy}, home: {x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2, h: ahead}};
    camHd = null;
    follow.index = -1;
    buildMini(api.track);
  }

  function dispose(object) {
    object.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) [].concat(o.material).forEach(m => m.dispose());
    });
  }

  function clear() {
    if (!api.track) return;
    scene.remove(api.track.group);
    dispose(api.track.group);
    api.track = null;
    mini.el.hidden = true;
  }

  // ---------- the mini-map: the track drawn once, the cars' dots on top every frame ----------
  const mini = {el: null, canvas: null, ctx: null, base: null, scale: 1, x0: 0, y0: 0, dpr: 1};

  function buildMini(track) {
    mini.el = mini.el || document.getElementById("mini");
    mini.canvas = mini.canvas || document.getElementById("minimap");
    mini.el.hidden = false;
    const {mnx, mxx, mny, mxy} = track.box;
    const bw = mxx - mnx + 2 * MINI_PAD, bh = mxy - mny + 2 * MINI_PAD;
    const card = mini.el.clientWidth;                    // 0 while hidden (Whole map, drawer open)
    const inner = card > 40 ? card - 20 : 188;           // the card's padding is 10 px a side
    const scale = Math.min(inner / bw, MINI_MAX_HEIGHT / bh);
    const w = Math.round(bw * scale), h = Math.round(bh * scale), dpr = Math.min(window.devicePixelRatio || 1, 2);
    Object.assign(mini, {scale, x0: mnx - MINI_PAD, y0: mxy + MINI_PAD, dpr, w, h});
    mini.canvas.style.width = `${w}px`;
    mini.canvas.style.height = `${h}px`;
    mini.canvas.width = Math.round(w * dpr);
    mini.canvas.height = Math.round(h * dpr);
    mini.ctx = mini.canvas.getContext("2d");
    const base = document.createElement("canvas");
    base.width = mini.canvas.width; base.height = mini.canvas.height;
    const g = base.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const path = new Path2D();
    for (const edge of [track.left, track.right]) {
      edge.forEach((p, i) => (i ? path.lineTo : path.moveTo).call(path, ...miniPoint(p[0], p[1])));
      path.closePath();
    }
    g.fillStyle = "#3a3f47";
    g.fill(path, "evenodd");
    g.strokeStyle = "#7d8590";
    g.lineWidth = 1;
    g.lineJoin = "round";
    g.stroke(path);
    mini.base = base;
  }

  function miniPoint(x, y) {
    return [(x - mini.x0) * mini.scale, (mini.y0 - y) * mini.scale];
  }

  /* Draw the mini-map: `dot(i, out)` sets out.group (a small whole number, one per look), out.color (CSS), out.r (m),
     out.alpha and out.ring (a dark outline) for car i. Dots of one group are filled as one path with no allocation
     per car, so thousands of cars stay cheap; the focus car is drawn last, larger, always outlined. */
  const miniGroups = [];
  const miniDot = {group: 0, color: "#fff", r: 1, alpha: 1, ring: false};
  function drawMini(s, dot) {
    if (!api.track || !mini.ctx || mini.el.hidden || host.classList.contains("mapview") ||
        host.classList.contains("setup")) return;
    const g = mini.ctx;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, mini.canvas.width, mini.canvas.height);
    g.drawImage(mini.base, 0, 0);
    if (!s || !s.n) return;
    g.setTransform(mini.dpr, 0, 0, mini.dpr, 0, 0);
    g.strokeStyle = "#101215";
    g.lineWidth = 1.2;
    const out = miniDot, focus = s.focus, x0 = mini.x0, y0 = mini.y0, scale = mini.scale;
    miniGroups.length = 0;
    for (let i = 0; i < s.n; i++) {
      if (i === focus) continue;
      dot(i, out);
      let group = miniGroups[out.group];
      if (!group) {
        miniGroups[out.group] = group = {path: new Path2D(), color: out.color, alpha: out.alpha, ring: out.ring};
      }
      const px = (s.x(i) - x0) * scale, py = (y0 - s.y(i)) * scale, r = Math.max(out.r * scale, 1.2);
      group.path.moveTo(px + r, py);
      group.path.arc(px, py, r, 0, 2 * Math.PI);
    }
    for (const group of miniGroups) {
      if (!group) continue;
      g.globalAlpha = group.alpha;
      g.fillStyle = group.color;
      g.fill(group.path);
      if (group.ring) g.stroke(group.path);
    }
    g.globalAlpha = 1;
    if (focus >= 0 && focus < s.n) {
      dot(focus, out);
      const px = (s.x(focus) - x0) * scale, py = (y0 - s.y(focus)) * scale, r = Math.max(out.r * scale, 3);
      g.fillStyle = out.color;
      g.beginPath();
      g.arc(px, py, r, 0, 2 * Math.PI);
      g.fill();
      g.stroke();
    }
  }

  // ---------- cameras ----------
  function poseOf(s, i) {
    return {x: s.x(i), y: s.y(i), h: s.h(i)};
  }

  // the point the camera follows: the focus car, gliding over from the previous one after a change
  function followPose(s, now) {
    const track = api.track;
    if (!s || !s.n) { follow.index = -1; return track.home; }
    const f = clamp(s.focus | 0, 0, s.n - 1);
    if (follow.index < 0 || s.snapped) camHd = null;     // the first car, or a restart: face the car at once
    if (f !== follow.index) {
      follow.from = follow.index >= 0 && follow.index < s.n ? follow.index : -1;
      follow.index = f;
      follow.since = now;
    }
    const target = poseOf(s, f);
    if (follow.from < 0 || s.snapped) { follow.from = -1; return target; }
    const u = (now - follow.since) / FOLLOW_GLIDE;
    if (u >= 1) { follow.from = -1; return target; }
    const k = ease(clamp(u, 0, 1)), from = poseOf(s, follow.from);
    return {x: from.x + (target.x - from.x) * k, y: from.y + (target.y - from.y) * k,
            h: from.h + angDiff(target.h, from.h) * k};
  }

  // near and far planes: kilometres up in the Whole map, the depth buffer needs a far-off near plane
  function setDepthRange(near, far) {
    if (cam.near === near && cam.far === far) return;
    cam.near = near;
    cam.far = far;
    cam.updateProjectionMatrix();
  }

  function placeCamera(dt, pose) {
    const track = api.track, isMap = ECR.view.camera === "map";
    scene.fog = isMap ? null : fog;
    track.mapRoad.visible = isMap;
    track.grid.visible = !isMap;                 // 10 m cells shimmer from kilometres up
    if (isMap) {
      const {mnx, mxx, mny, mxy} = track.box, tf = Math.tan(cam.fov * Math.PI / 360);
      const h = 1.12 * orbit.zoom * Math.max((mxy - mny) / (2 * tf), (mxx - mnx) / (2 * tf * cam.aspect));
      setDepthRange(Math.round(h * 0.5), Math.round(h * 2));
      cam.position.set(track.cx, h, -track.cy + h * 0.02);
      cam.lookAt(track.cx, 0, -track.cy);
    } else {
      setDepthRange(0.1, 8000);
      const P = PRESETS[ECR.view.camera], k = 1 - Math.exp(-dt * 5);
      off.lerp(new T.Vector3(...P.off), k);
      look.lerp(new T.Vector3(...P.look), k);
      camHd = camHd === null ? pose.h : camHd + angDiff(pose.h, camHd) * (1 - Math.exp(-dt * 4));
      const toWorld = (f, l, u, h) => new T.Vector3(pose.x + f * Math.cos(h) - l * Math.sin(h), u,
                                                    -(pose.y + f * Math.sin(h) + l * Math.cos(h)));
      const target = toWorld(look.x, look.y, look.z, camHd), rel = off.clone().sub(look).multiplyScalar(orbit.zoom);
      const cy = Math.cos(orbit.yaw), sy = Math.sin(orbit.yaw);
      const f = rel.x * cy - rel.y * sy, l = rel.x * sy + rel.y * cy;
      const u = Math.max(0.25, rel.z + look.z + orbit.pitch * Math.max(2, Math.hypot(f, l)) * 0.5);
      cam.position.copy(toWorld(look.x + f, look.y + l, u, camHd));
      cam.lookAt(target);
    }
    sun.position.set(pose.x + 38, 95, -pose.y + 26);
    sun.target.position.set(pose.x, 0, -pose.y);
    sun.target.updateMatrixWorld();
  }

  /* Place the camera (and the sun) for one display frame, before the cars are drawn so they can use its position.
     `s` is the interpolated car state from main.js (or null before the first frame). */
  function place(s, dt, now) {
    if (api.ok && api.track) placeCamera(dt, followPose(s, now));
  }

  // draw the frame (the plain background without a track)
  function render() {
    if (!api.ok) return;
    if (!api.track) { R.clear(); return; }
    R.render(scene, cam);
  }

  Object.assign(api, {init, load, clear, setCamera, setRays, place, render, drawMini, PRESETS});
  ECR.scene = api;
})();
