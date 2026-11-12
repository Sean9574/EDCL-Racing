/* EC Racing viewer: the cars — the open-wheel model, the light instanced model, the sensor rays and the whole-map
   discs (spec 3.18 §2). Car frame: forward +x, left −z, up +y, metres. Classic script: ECR.cars. */
(function () {
  "use strict";
  const ECR = window.ECR = window.ECR || {};
  const T = window.THREE;

  const RED = 0xff4d5e;                      // sensor rays
  const BG = 0x1b1f25;                       // scene background
  const WHITE = 0xf3f5f8;
  const POLE = 0x2a2f38;                     // the pole lap: a see-through dark car
  const FALLBACK = "#7F7F7F";                // a car whose livery has not arrived yet
  const CRASHED_OPACITY = 0.35, POLE_OPACITY = 0.5, LIGHT_OPACITY = 0.3;
  const DONE_FADE = 0.65;                    // a done light car's color moves this far toward the background
  const FAN = 0.09, RAYS = 0.75, RIM = 0.4;  // sensor opacities; other cars' rays under "all cars" are about half
  const GLOW_MARGIN = 0.5;                   // m: a ray this much shorter than its sensor met the road edge
  const RING = [3.4, 3.9], RING_OPACITY = 0.35;
  const WHEELBASE = 3.0;                     // m, for the steering angle shown from the turn rate
  const STEER_SHOWN = 0.45;                  // rad, the front wheels' angle at a full steering command
  const SPIN = 0.06;                         // shown wheel spin per metre as a share of the true spin (no strobing)
  const STEER_LEVELS = {S: 0, "¼": 0.25, "½": 0.5, "¾": 0.75, 1: 1};
  /* The scene's lights put about 1.45× light on a surface facing up, which washed the paint out (KT Random's orange
     read as NEAT's yellow). The paint is matte-ish with a thin clear coat; its albedo is the method's color times
     PAINT_GAIN plus a faint self-light (PAINT_GLOW) that keeps the shaded sides readable, so the lit paint lands
     within about 20 of each channel of the hex from above and behind (spec §10.1, Okabe–Ito). The light model has
     no self-light and uses LIGHT_GAIN. */
  const PAINT_GAIN = 0.58, PAINT_GLOW = 0.15, LIGHT_GAIN = 0.68;
  const GLOW_BACK = 1.1;                     // m a glow sits back along its ray from the road edge, so walls never cut it
  const GLOW_HEIGHT = 0.8;                   // m, high enough that the road does not cut its lower half either
  const GLOW_NEAR = [2, 4];                  // m from the camera: glows fade out below this range (Wheels camera)
  const GLOW_SIZE = 2.2;                     // m across, from 25 m away and further (Top, Chase); closer ones shrink
  const GLOW_FULL = 25;                      // to 30 % so a Wheels view is not washed pink

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const EMPTY = new T.BufferGeometry();

  // ---------- the open-wheel model, built once per livery variant and shared by every car ----------
  function merge(geometries) {
    if (!geometries.length) return null;
    const flat = geometries.map(g => (g.index ? g.toNonIndexed() : g));
    const count = flat.reduce((sum, g) => sum + g.attributes.position.array.length, 0);
    const pos = new Float32Array(count), nor = new Float32Array(count);
    let o = 0;
    for (const g of flat) {
      pos.set(g.attributes.position.array, o);
      nor.set(g.attributes.normal.array, o);
      o += g.attributes.position.array.length;
    }
    const out = new T.BufferGeometry();
    out.setAttribute("position", new T.BufferAttribute(pos, 3));
    out.setAttribute("normal", new T.BufferAttribute(nor, 3));
    out.computeBoundingSphere();
    new Set(geometries.concat(flat)).forEach(g => g.dispose());
    return out;
  }

  const sym = half => half.concat(half.slice().reverse().map(([x, y]) => [x, -y]));
  const bodies = {};

  /* The body, merged per material: paint, white, carbon, black. Runs 1–3 (`stripe`) get a white stripe from the nose
     over the airbox and engine cover to the rear wing; app cars (`app`) get white wings. */
  function body(stripe, app) {
    const key = `${stripe ? 1 : 0}${app ? 1 : 0}`;
    if (bodies[key]) return bodies[key];
    const parts = {paint: [], white: [], carbon: [], black: []};
    const wing = app ? "white" : null;
    const slab = (pts, h, y0, mat, bv = 0.05) => {
      const s = new T.Shape();
      s.moveTo(pts[0][0], pts[0][1]);
      pts.slice(1).forEach(p => s.lineTo(p[0], p[1]));
      const g = new T.ExtrudeGeometry(s, {depth: Math.max(0.01, h - 2 * bv), bevelEnabled: true, bevelThickness: bv,
                                          bevelSize: bv * 0.8, bevelSegments: 3});
      g.rotateX(-Math.PI / 2);
      g.translate(0, y0 + bv, 0);
      parts[mat].push(g);
    };
    const box = (l, h, w, x, y, z, mat, rz = 0) => {
      const g = new T.BoxGeometry(l, h, w);
      if (rz) g.rotateZ(rz);
      g.translate(x, y, z);
      parts[mat].push(g);
    };
    slab(sym([[2.42, 0.02], [2.25, 0.12], [1.55, 0.2], [0.9, 0.26], [0.5, 0.3]]), 0.3, 0.17, "paint", 0.06);   // nose
    slab(sym([[0.75, 0.31], [0.15, 0.4], [-0.65, 0.41], [-1.05, 0.34]]), 0.56, 0.12, "paint", 0.07);          // tub
    slab(sym([[0.32, 0.42], [0.18, 0.74], [-0.75, 0.76], [-1.5, 0.56], [-1.78, 0.36]]), 0.42, 0.09, "paint", 0.08);
    slab(sym([[-0.35, 0.3], [-1.2, 0.27], [-2.0, 0.13], [-2.12, 0.02]]), 0.52, 0.2, "paint", 0.07);           // cover
    box(0.95, 0.012, 0.06, -0.35, 0.515, 0.62, "carbon");                    // sidepod trim
    box(0.95, 0.012, 0.06, -0.35, 0.515, -0.62, "carbon");
    box(0.42, 0.34, 0.3, -0.62, 0.83, 0, "paint");                           // airbox
    box(1.25, 0.24, 0.02, -1.35, 0.82, 0, stripe ? "white" : "paint", -0.08); // engine fin
    box(3.4, 0.025, 1.42, -0.2, 0.055, 0, "carbon");                         // floor
    const pit = new T.CircleGeometry(1, 28);
    pit.scale(0.44, 0.26, 1); pit.rotateX(-Math.PI / 2); pit.translate(0.12, 0.687, 0);
    parts.black.push(pit);
    const halo = new T.TorusGeometry(0.3, 0.024, 8, 24, Math.PI);
    halo.rotateY(Math.PI / 2); halo.translate(0.05, 0.69, 0);
    parts.carbon.push(halo);
    box(0.34, 0.034, 1.96, 2.33, 0.1, 0, wing || "carbon");                  // front wing
    box(0.2, 0.03, 1.8, 2.17, 0.16, 0, wing || "paint", 0.25);
    box(0.42, 0.22, 0.02, 2.28, 0.17, 0.98, wing || "paint");
    box(0.42, 0.22, 0.02, 2.28, 0.17, -0.98, wing || "paint");
    box(0.3, 0.04, 1.02, -2.33, 0.95, 0, wing || "carbon");                  // rear wing
    box(0.18, 0.03, 1.02, -2.48, 1.04, 0, wing || "paint", 0.35);
    box(0.52, 0.46, 0.02, -2.36, 0.86, 0.52, wing || "paint");
    box(0.52, 0.46, 0.02, -2.36, 0.86, -0.52, wing || "paint");
    if (stripe) {
      const w = 0.16;
      box(1.35, 0.012, w, 1.425, 0.476, 0, "white");                         // nose
      box(0.19, 0.012, w, 0.655, 0.686, 0, "white");                         // in front of the cockpit
      box(0.42, 0.012, w, -0.62, 1.006, 0, "white");                         // airbox
      box(1.15, 0.012, w, -1.4, 0.726, 0, "white");                          // engine cover
      if (!app) {
        box(0.3, 0.012, w, -2.33, 0.976, 0, "white");                        // over the rear wing
        box(0.18, 0.012, w, -2.487, 1.06, 0, "white", 0.35);
      }
    }
    const merged = {};
    for (const k in parts) merged[k] = merge(parts[k]);
    return (bodies[key] = merged);
  }

  const wheelParts = {};
  function wheel(r, w, side) {
    const key = `${r}|${w}|${side}`;
    if (wheelParts[key]) return wheelParts[key];
    const tire = new T.CylinderGeometry(r, r, w, 32);
    tire.rotateX(Math.PI / 2);
    const rim = new T.CylinderGeometry(r * 0.64, r * 0.64, w + 0.008, 24);
    rim.rotateX(Math.PI / 2);
    const spokes = [];
    for (let k = 0; k < 5; k++) {
      const s = new T.BoxGeometry(r * 1.18, 0.05, 0.012);
      s.rotateZ(k * Math.PI * 2 / 5);
      s.translate(0, 0, side * (w / 2 + 0.006));
      spokes.push(s);
    }
    return (wheelParts[key] = {tire, rim, spokes: merge(spokes)});
  }

  const WHEELS = [[1.55, 0.84, 0.33, 0.34, true], [1.55, -0.84, 0.33, 0.34, true],
                  [-1.45, 0.82, 0.355, 0.42, false], [-1.45, -0.82, 0.355, 0.42, false]];

  /* One open-wheel car in its own materials: look = {color, stripe, app}. */
  function makeCar(look) {
    const G = new T.Group();
    const mats = {
      paint: new T.MeshPhysicalMaterial({color: look.color, metalness: 0, roughness: 0.6, clearcoat: 0.15,
                                         clearcoatRoughness: 0.35}),
      white: new T.MeshPhysicalMaterial({color: WHITE, metalness: 0.1, roughness: 0.4, clearcoat: 0.6}),
      carbon: new T.MeshStandardMaterial({color: 0x1d2025, roughness: 0.5, metalness: 0.35}),
      black: new T.MeshStandardMaterial({color: 0x0c0d0f, roughness: 0.8}),
      tire: new T.MeshStandardMaterial({color: 0x141517, roughness: 0.88}),
      rim: new T.MeshStandardMaterial({color: 0x2f343b, metalness: 0.85, roughness: 0.3}),
      spoke: new T.MeshStandardMaterial({color: 0xb7bec8, metalness: 0.8, roughness: 0.3}),
    };
    const shells = {};
    for (const k of ["paint", "white", "carbon", "black"]) {
      shells[k] = new T.Mesh(EMPTY, mats[k]);
      G.add(shells[k]);
    }
    const wheels = [], steerers = [];
    for (const [x, z, r, w, front] of WHEELS) {
      const geo = wheel(r, w, z < 0 ? -1 : 1);
      const pivot = new T.Group(), spin = new T.Group();
      pivot.position.set(x, r, z);
      pivot.add(spin);
      spin.add(new T.Mesh(geo.tire, mats.tire), new T.Mesh(geo.rim, mats.rim), new T.Mesh(geo.spokes, mats.spoke));
      G.add(pivot);
      wheels.push({spin, r});
      if (front) steerers.push(pivot);
    }
    const car = {G, mats, wheels, steerers, look: {}, opacity: 1, shadows: true, steer: 0, dist: 0, last: null};
    car.setLook = next => {
      if (next.color !== car.look.color) {
        mats.paint.color.set(next.color).multiplyScalar(PAINT_GAIN);
        mats.paint.emissive.set(next.color).multiplyScalar(PAINT_GLOW);
      }
      if (next.stripe !== car.look.stripe || next.app !== car.look.app) {
        const parts = body(next.stripe, next.app);
        for (const k in shells) {
          shells[k].geometry = parts[k] || EMPTY;
          shells[k].visible = !!parts[k];
        }
      }
      car.look = {color: next.color, stripe: next.stripe, app: next.app};
    };
    car.setOpacity = (opacity, shadows) => {
      if (opacity === car.opacity && shadows === car.shadows) return;
      for (const k in mats) {
        mats[k].transparent = opacity < 1;
        mats[k].opacity = opacity;
      }
      const cast = shadows && opacity >= 1;
      G.traverse(o => { if (o.isMesh) o.castShadow = cast; });
      car.opacity = opacity;
      car.shadows = shadows;
    };
    car.setLook({color: look.color, stripe: !!look.stripe, app: !!look.app});
    G.traverse(o => { if (o.isMesh) o.castShadow = true; });
    return car;
  }

  // the steering command of an action label such as "L½ gas" or "R1 brake", −1 (right) … +1 (left); null if none
  function steerOf(label) {
    if (!label) return null;
    const token = label.split(" ")[0], sign = token[0] === "L" ? 1 : token[0] === "R" ? -1 : 0;
    const level = STEER_LEVELS[sign ? token.slice(1) : token];
    return level === undefined ? null : sign * level;
  }

  /* Put a full car where the sample says, steer its front wheels and spin its wheels. `command` is the steering
     command (−1…1) when known, else the angle comes from the turn rate. */
  function drive(car, s, i, dt, command) {
    const x = s.x(i), y = s.y(i), h = s.h(i);
    car.G.position.set(x, 0, -y);
    car.G.rotation.y = h;
    if (car.last) {
      const step = Math.hypot(x - car.last[0], y - car.last[1]);
      if (step < 25) car.dist += step;                     // a longer step is a jump (a restart), not driving
    }
    car.last = [x, y];
    const target = command !== null && command !== undefined ? command * STEER_SHOWN
      : clamp(3 * Math.atan(WHEELBASE * s.kappa(i)), -STEER_SHOWN, STEER_SHOWN);
    car.steer += (target - car.steer) * (1 - Math.exp(-dt * 10));
    car.steerers.forEach(p => { p.rotation.y = car.steer; });
    car.wheels.forEach(w => { w.spin.rotation.z = -(car.dist * SPIN) / w.r; });
  }

  // ---------- the light model: two instanced meshes (body in the car's color, tyres) for up to thousands of cars ----
  function lightGeometry() {
    const boxes = (list) => merge(list.map(([l, h, w, x, y, z]) => {
      const g = new T.BoxGeometry(l, h, w);
      g.translate(x, y, z);
      return g;
    }));
    return {
      body: boxes([[4.3, 0.42, 0.62, 0, 0.38, 0], [2.0, 0.36, 1.5, -0.45, 0.32, 0], [0.34, 0.05, 1.9, 2.3, 0.12, 0],
                   [0.32, 0.05, 1.05, -2.35, 0.95, 0]]),
      tyres: boxes([[0.62, 0.62, 0.36, 1.55, 0.33, 0.84], [0.62, 0.62, 0.36, 1.55, 0.33, -0.84],
                    [0.7, 0.7, 0.42, -1.45, 0.355, 0.82], [0.7, 0.7, 0.42, -1.45, 0.355, -0.82]]),
    };
  }

  // ---------- sensor rays: a translucent fan through the ray ends, the rays, a rim and glows at the road edge ----
  function makeSensor(scene, nr, glowTexture) {
    const fanPos = new Float32Array((nr + 1) * 3), fanIdx = [];
    for (let k = 1; k < nr; k++) fanIdx.push(0, k, k + 1);
    const fanG = new T.BufferGeometry();
    fanG.setAttribute("position", new T.BufferAttribute(fanPos, 3));
    fanG.setIndex(fanIdx);
    const fan = new T.Mesh(fanG, new T.MeshBasicMaterial({color: RED, transparent: true, opacity: FAN, depthWrite: false,
                                                          side: T.DoubleSide, polygonOffset: true,
                                                          polygonOffsetFactor: -4, polygonOffsetUnits: -8}));
    const rayPos = new Float32Array(nr * 6), rayG = new T.BufferGeometry();
    rayG.setAttribute("position", new T.BufferAttribute(rayPos, 3));
    const rays = new T.LineSegments(rayG, new T.LineBasicMaterial({color: RED, transparent: true, opacity: RAYS}));
    const rimPos = new Float32Array(nr * 3), rimG = new T.BufferGeometry();
    rimG.setAttribute("position", new T.BufferAttribute(rimPos, 3));
    const rim = new T.Line(rimG, new T.LineBasicMaterial({color: RED, transparent: true, opacity: RIM}));
    const glows = [];
    for (let k = 0; k < nr; k++) {
      const sprite = new T.Sprite(new T.SpriteMaterial({map: glowTexture, color: RED, transparent: true,
                                                        blending: T.AdditiveBlending, depthWrite: false}));
      sprite.scale.set(2.2, 2.2, 1);
      glows.push(sprite);
    }
    const lines = [fan, rays, rim];
    lines.forEach(o => { o.frustumCulled = false; });
    scene.add(fan, rays, rim, ...glows);
    const angles = Array.from({length: nr}, (_, k) => (-90 + 180 * k / (nr - 1)) * Math.PI / 180);
    const S = {nr, shown: false};
    S.hide = () => {
      if (!S.shown) return;
      lines.forEach(o => { o.visible = false; });
      glows.forEach(g => { g.visible = false; });
      S.shown = false;
    };
    // the rays of a car at (x, y) heading h; lengths[k] at angles[k] (index 0 points right); dim for other cars
    S.draw = (x, y, h, lengths, sensor, dim) => {
      const camera = ECR.scene.camera ? ECR.scene.camera.position : null;
      lines.forEach(o => { o.visible = true; });
      S.shown = true;
      fanPos[0] = x; fanPos[1] = 0.04; fanPos[2] = -y;
      for (let k = 0; k < nr; k++) {
        const a = h + angles[k], r = lengths[k], ex = x + Math.cos(a) * r, ey = y + Math.sin(a) * r;
        fanPos.set([ex, 0.04, -ey], (k + 1) * 3);
        rimPos.set([ex, 0.06, -ey], k * 3);
        rayPos.set([x, 0.06, -y, ex, 0.06, -ey], k * 6);
        const back = Math.min(GLOW_BACK, r), gx = ex - Math.cos(a) * back, gy = ey - Math.sin(a) * back;
        const near = camera ? Math.hypot(camera.x - gx, camera.y - GLOW_HEIGHT, camera.z + gy) : Infinity;
        const fade = Math.min(1, Math.max(0, (near - GLOW_NEAR[0]) / (GLOW_NEAR[1] - GLOW_NEAR[0])));
        const hit = r < sensor - GLOW_MARGIN && fade > 0;
        glows[k].visible = hit;
        if (hit) {
          const size = GLOW_SIZE * Math.min(1, Math.max(0.3, near / GLOW_FULL));
          glows[k].position.set(gx, GLOW_HEIGHT, -gy);
          glows[k].scale.set(size, size, 1);
        }
        glows[k].material.opacity = (dim ? 0.5 : 1) * fade;
      }
      fanG.attributes.position.needsUpdate = true;
      rayG.attributes.position.needsUpdate = true;
      rimG.attributes.position.needsUpdate = true;
      fan.material.opacity = dim ? FAN * 0.45 : FAN;
      rays.material.opacity = dim ? RAYS * 0.47 : RAYS;
      rim.material.opacity = dim ? RIM * 0.45 : RIM;
    };
    return S;
  }

  // ---------- whole-map discs: a car is 5 m long on a track kilometres across ----------
  function makeDisc(scene, color) {
    const g = new T.Group();
    const ring = new T.Mesh(new T.CircleGeometry(1, 32), new T.MeshBasicMaterial({color: 0x101215}));
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 1.5;
    const disc = new T.Mesh(new T.CircleGeometry(0.78, 32), new T.MeshBasicMaterial({color}));
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = 1.6;
    g.add(ring, disc);
    g.visible = false;
    scene.add(g);
    const D = {g, disc, ring, faded: false};
    D.fade = faded => {                          // a crashed car's disc, like its model, is faded
      if (faded === D.faded) return;
      [ring.material, disc.material].forEach(m => { m.transparent = faded; m.opacity = faded ? CRASHED_OPACITY : 1; });
      D.faded = faded;
    };
    return D;
  }

  // ---------- the cars on show ----------
  let scene = null;                          // the three.js scene, once ECR.scene is up
  const race = [];                           // per race car: {car, sensor, disc}
  let pole = null;                           // the pole lap's car
  const train = {car: null, ring: null, disc: null, sensor: null, fleet: null};

  function setUp() {
    if (scene || !ECR.scene || !ECR.scene.ok) return !!scene;
    scene = ECR.scene.three;
    return true;
  }

  /* Each car's livery in frame order: {color, stripe, app}; the race view reads panel.cars (aligned with the frame's
     car lists), the training view uses the approach color. */
  function liveries(panel, n, mode) {
    const out = [];
    const cars = panel && panel.cars || [];
    const color = panel && panel.color || FALLBACK;
    for (let i = 0; i < n; i++) {
      const c = mode === "race" ? cars[i] : null;
      out.push(c ? {color: c.color, stripe: !!c.stripe, app: !!c.app}
                 : {color: mode === "race" ? FALLBACK : color, stripe: false, app: false});
    }
    return out;
  }

  /* The cars' colors as CSS hex strings, one per car in frame order: the race view's from panel.cars, the training
     view's all the approach color (`count` of them, default 1). */
  function colorsFor(panel, count) {
    if (!panel) return [];
    if (panel.mode === "race") return (panel.cars || []).map(c => c.color);
    return Array.from({length: count === undefined ? 1 : count}, () => panel.color);
  }

  function raysWanted(i, focus) {
    const mode = ECR.view.rays;
    return mode === "all" || (mode === "car" && i === focus);
  }

  function updateRace(s, dt, panel, isMap, span) {
    const looks = liveries(panel, s.n, "race");
    const glow = ECR.scene.textures.glow;
    while (race.length < s.n) race.push({car: null, sensor: null, disc: null});
    for (let i = 0; i < race.length; i++) {
      const rc = race[i];
      if (i >= s.n) {
        if (rc.car) rc.car.G.visible = false;
        if (rc.disc) rc.disc.g.visible = false;
        if (rc.sensor) rc.sensor.hide();
        continue;
      }
      if (!rc.car) {
        rc.car = makeCar(looks[i]);
        scene.add(rc.car.G);
        rc.disc = makeDisc(scene, looks[i].color);
      }
      const car = rc.car, crashed = !!s.crashed[i], focus = i === s.focus;
      car.setLook(looks[i]);
      car.setOpacity(crashed ? CRASHED_OPACITY : 1, true);
      car.G.visible = !isMap;
      drive(car, s, i, dt, focus ? steerOf(s.action) : null);
      rc.disc.disc.material.color.set(looks[i].color);
      rc.disc.fade(crashed);
      rc.disc.g.visible = isMap;
      rc.disc.g.position.set(s.x(i), focus ? 6 : 0.12 * i, -s.y(i));     // stacked: no two discs share a height
      rc.disc.g.scale.setScalar(span / (focus ? 100 : 150));
      const lengths = !isMap && !crashed && raysWanted(i, s.focus) ? s.rays(i) : null;
      if (lengths) {
        if (!rc.sensor || rc.sensor.nr !== lengths.length) {
          if (rc.sensor) rc.sensor.hide();
          rc.sensor = makeSensor(scene, lengths.length, glow);
        }
        rc.sensor.draw(s.x(i), s.y(i), s.h(i), lengths, s.sensor(i, lengths), ECR.view.rays === "all" && !focus);
      } else if (rc.sensor) {
        rc.sensor.hide();
      }
    }
    // the pole lap, see-through dark
    if (s.ghost && !isMap) {
      if (!pole) {
        pole = makeCar({color: POLE});
        pole.setOpacity(POLE_OPACITY, false);
        scene.add(pole.G);
      }
      pole.G.visible = true;
      drive(pole, {x: () => s.ghost.x, y: () => s.ghost.y, h: () => s.ghost.h, kappa: () => s.ghost.kappa}, 0, dt, null);
    } else if (pole) {
      pole.G.visible = false;
    }
  }

  function hideRace() {
    race.forEach(rc => {
      if (rc.car) rc.car.G.visible = false;
      if (rc.disc) rc.disc.g.visible = false;
      if (rc.sensor) rc.sensor.hide();
    });
    if (pole) pole.G.visible = false;
  }

  // the light fleet: capacity grows to the largest display count seen
  function makeFleet() {
    const geo = lightGeometry();
    const bodyMat = new T.MeshStandardMaterial({color: 0xffffff, roughness: 0.6, metalness: 0, transparent: true,
                                                opacity: LIGHT_OPACITY, depthWrite: false});
    const tyreMat = new T.MeshStandardMaterial({color: 0x141517, roughness: 0.9, transparent: true,
                                                opacity: LIGHT_OPACITY, depthWrite: false});
    const dotGeo = new T.CircleGeometry(1, 16);
    dotGeo.rotateX(-Math.PI / 2);
    const dotMat = new T.MeshBasicMaterial({color: 0xffffff});
    const F = {cap: 0, body: null, tyres: null, dots: null, color: null, base: new T.Color(), faded: new T.Color(),
               done: new Uint8Array(0)};
    const M = new T.Matrix4(), HIDDEN = new T.Matrix4().makeScale(0, 0, 0), white = new T.Color(1, 1, 1);
    F.ensure = n => {
      if (n <= F.cap) return;
      const cap = Math.max(n, F.cap * 2, 128);
      [F.body, F.tyres, F.dots].forEach(m => { if (m) { scene.remove(m); if (m.dispose) m.dispose(); } });
      F.body = new T.InstancedMesh(geo.body, bodyMat, cap);
      F.tyres = new T.InstancedMesh(geo.tyres, tyreMat, cap);
      F.dots = new T.InstancedMesh(dotGeo, dotMat, cap);
      for (let i = 0; i < cap; i++) { F.body.setColorAt(i, white); F.dots.setColorAt(i, white); }
      [F.body, F.tyres, F.dots].forEach(m => {
        m.castShadow = false;
        m.frustumCulled = false;
        m.instanceMatrix.setUsage(T.DynamicDrawUsage);
        scene.add(m);
      });
      F.cap = cap;
      F.done = new Uint8Array(cap).fill(255);
      F.color = null;
    };
    F.update = (s, color, isMap, span) => {
      F.ensure(s.n);
      const recolor = color !== F.color;
      if (recolor) {
        F.color = color;
        F.base.set(color);
        F.faded.copy(F.base).lerp(new T.Color(BG), DONE_FADE).multiplyScalar(LIGHT_GAIN);
        F.base.multiplyScalar(LIGHT_GAIN);
      }
      let colorsChanged = false;
      const ds = span / 260;
      for (let i = 0; i < s.n; i++) {
        const x = s.x(i), y = s.y(i);
        if (i === s.focus || isMap) {
          F.body.setMatrixAt(i, HIDDEN);
          F.tyres.setMatrixAt(i, HIDDEN);
        } else {
          M.makeRotationY(s.h(i));
          M.setPosition(x, 0, -y);
          F.body.setMatrixAt(i, M);
          F.tyres.setMatrixAt(i, M);
        }
        if (isMap && i !== s.focus) {
          M.makeScale(ds, 1, ds);
          M.setPosition(x, 1.3, -y);
          F.dots.setMatrixAt(i, M);
        } else {
          F.dots.setMatrixAt(i, HIDDEN);
        }
        const done = s.done[i] ? 1 : 0;
        if (recolor || F.done[i] !== done) {
          const c = done ? F.faded : F.base;
          F.body.setColorAt(i, c);
          F.dots.setColorAt(i, c);
          F.done[i] = done;
          colorsChanged = true;
        }
      }
      F.body.count = F.tyres.count = F.dots.count = s.n;
      F.body.visible = F.tyres.visible = !isMap && s.n > 0;
      F.dots.visible = isMap && s.n > 0;
      F.body.instanceMatrix.needsUpdate = F.tyres.instanceMatrix.needsUpdate = F.dots.instanceMatrix.needsUpdate = true;
      if (colorsChanged) F.body.instanceColor.needsUpdate = F.dots.instanceColor.needsUpdate = true;
    };
    F.hide = () => { [F.body, F.tyres, F.dots].forEach(m => { if (m) m.visible = false; }); };
    return F;
  }

  function updateTrain(s, dt, panel, isMap, span) {
    const color = panel && panel.color || FALLBACK;
    if (!train.fleet) {
      train.fleet = makeFleet();
      train.car = makeCar({color});
      scene.add(train.car.G);
      train.ring = new T.Mesh(new T.RingGeometry(RING[0], RING[1], 48),
                              new T.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: RING_OPACITY,
                                                       depthWrite: false, side: T.DoubleSide}));
      train.ring.rotation.x = -Math.PI / 2;
      scene.add(train.ring);
      train.disc = makeDisc(scene, 0xffffff);
    }
    train.fleet.update(s, color, isMap, span);
    const f = s.focus, has = f >= 0 && f < s.n;
    const car = train.car;
    car.G.visible = has && !isMap;
    train.ring.visible = has && !isMap;
    train.disc.g.visible = has && isMap;
    if (!has) {
      if (train.sensor) train.sensor.hide();
      return;
    }
    car.setLook({color, stripe: false, app: false});
    car.setOpacity(s.crashed[f] ? CRASHED_OPACITY : 1, true);
    drive(car, s, f, dt, steerOf(s.action));
    const x = s.x(f), y = s.y(f);
    train.ring.position.set(x, 0.06, -y);
    train.disc.g.position.set(x, 0, -y);
    train.disc.g.scale.setScalar(span / 110);
    const lengths = !isMap && !s.done[f] && ECR.view.rays !== "off" ? s.rays(f) : null;
    if (lengths) {
      if (!train.sensor || train.sensor.nr !== lengths.length) {
        if (train.sensor) train.sensor.hide();
        train.sensor = makeSensor(scene, lengths.length, ECR.scene.textures.glow);
      }
      train.sensor.draw(x, y, s.h(f), lengths, s.sensor(f, lengths), false);
    } else if (train.sensor) {
      train.sensor.hide();
    }
  }

  function hideTrain() {
    if (!train.fleet) return;
    train.fleet.hide();
    train.car.G.visible = false;
    train.ring.visible = false;
    train.disc.g.visible = false;
    if (train.sensor) train.sensor.hide();
  }

  /* Draw the cars of one display frame. `s` is the interpolated sample from main.js (null: no cars). */
  function update(s, dt, panel) {
    if (!setUp()) return;
    const track = ECR.scene.track, mode = s && s.n && track ? s.mode : null;
    const isMap = ECR.view.camera === "map", span = track ? track.span : 1000;
    if (mode === "race") updateRace(s, dt, panel, isMap, span); else hideRace();
    if (mode === "train") updateTrain(s, dt, panel, isMap, span); else hideTrain();
  }

  /* The mini-map's dot for car i of the sample (see ECR.scene.drawMini). */
  function miniDots(s, panel) {
    const span = ECR.scene.track ? ECR.scene.track.span : 1000;
    if (s.mode === "race") {
      const looks = liveries(panel, s.n, "race");
      return (i, out) => {
        out.color = looks[i].color;
        out.r = span / (i === s.focus ? 45 : 70);
        out.alpha = s.crashed[i] ? 0.4 : 1;
        out.ring = true;
        out.group = 2 * i + (s.crashed[i] ? 1 : 0);
      };
    }
    const color = panel && panel.color || FALLBACK;
    return (i, out) => {
      const focus = i === s.focus;
      out.color = focus ? "#ffffff" : color;
      out.r = span / (focus ? 55 : 110);
      out.alpha = s.done[i] ? 0.25 : 0.75;
      out.ring = false;
      out.group = focus ? 2 : s.done[i] ? 1 : 0;
    };
  }

  ECR.cars = {update, miniDots, colorsFor, steerOf, makeCar};
})();
