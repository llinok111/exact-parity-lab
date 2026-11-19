// Deterministic sample generator.  No randomness without a fixed seed.
// Writes data/*.json (meshes + query sets).  `--check` re-renders in memory and
// compares bytes with what is on disk, proving determinism.
import { writeFileSync, readFileSync, mkdirSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { meshStats, meshToJSON, len3, sub3, cross3 } from "../src/core.mjs";

const OUT = new URL("../data/", import.meta.url);
const f = (x) => (Object.is(x, -0) ? 0 : x);

function weld(positions, indices) {
  const map = new Map();
  const remap = new Int32Array(positions.length / 3);
  const out = [];
  for (let i = 0; i < positions.length / 3; i++) {
    const key = `${f(positions[i * 3])}|${f(positions[i * 3 + 1])}|${f(positions[i * 3 + 2])}`;
    let id = map.get(key);
    if (id === undefined) {
      id = out.length / 3;
      map.set(key, id);
      out.push(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    }
    remap[i] = id;
  }
  return { positions: out, indices: Array.from(indices, (i) => remap[i]) };
}

function orientOutward(positions, indices) {
  let vol = 0;
  for (let t = 0; t < indices.length / 3; t++) {
    const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
    const A = [positions[a], positions[a + 1], positions[a + 2]];
    const B = [positions[b], positions[b + 1], positions[b + 2]];
    const C = [positions[c], positions[c + 1], positions[c + 2]];
    vol += (A[0] * (B[1] * C[2] - B[2] * C[1]) - A[1] * (B[0] * C[2] - B[2] * C[0]) + A[2] * (B[0] * C[1] - B[1] * C[0])) / 6;
  }
  if (vol < 0) {
    for (let t = 0; t < indices.length / 3; t++) {
      const tmp = indices[t * 3 + 1];
      indices[t * 3 + 1] = indices[t * 3 + 2];
      indices[t * 3 + 2] = tmp;
    }
  }
  return vol;
}

function sphere(radius = 1.6, slices = 64, stacks = 32) {
  const positions = [], indices = [];
  // north pole
  positions.push(0, radius, 0);
  for (let i = 1; i < stacks; i++) {
    const th = (Math.PI * i) / stacks;
    for (let j = 0; j < slices; j++) {
      const ph = (2 * Math.PI * j) / slices;
      positions.push(radius * Math.sin(th) * Math.cos(ph), radius * Math.cos(th), radius * Math.sin(th) * Math.sin(ph));
    }
  }
  positions.push(0, -radius, 0);
  const south = positions.length / 3 - 1;
  const ring = (i, j) => 1 + (i - 1) * slices + (j % slices);
  for (let j = 0; j < slices; j++) indices.push(0, ring(1, j + 1), ring(1, j));
  for (let i = 1; i < stacks - 1; i++) {
    for (let j = 0; j < slices; j++) {
      const a = ring(i, j), b = ring(i, j + 1), c = ring(i + 1, j), d = ring(i + 1, j + 1);
      indices.push(a, b, c, b, d, c);
    }
  }
  for (let j = 0; j < slices; j++) indices.push(south, ring(stacks - 1, j), ring(stacks - 1, j + 1));
  return { positions, indices };
}

function box(half = 1.2, n = 10) {
  const positions = [], indices = [];
  const faces = [
    { o: [1, 0, 0], u: [0, 1, 0], v: [0, 0, 1] },
    { o: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
    { o: [0, 1, 0], u: [0, 0, 1], v: [1, 0, 0] },
    { o: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
    { o: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
    { o: [0, 0, -1], u: [0, 1, 0], v: [1, 0, 0] },
  ];
  for (const face of faces) {
    const base = positions.length / 3;
    for (let i = 0; i <= n; i++) {
      for (let j = 0; j <= n; j++) {
        const a = -1 + (2 * i) / n, b = -1 + (2 * j) / n;
        positions.push(
          half * (face.o[0] + face.u[0] * a + face.v[0] * b),
          half * (face.o[1] + face.u[1] * a + face.v[1] * b),
          half * (face.o[2] + face.u[2] * a + face.v[2] * b),
        );
      }
    }
    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= n; j++) {
        const a = base + (i - 1) * (n + 1) + (j - 1);
        const b = a + 1, c = a + (n + 1), d = c + 1;
        indices.push(a, b, c, b, d, c);
      }
    }
  }
  const w = weld(positions, indices);
  return w;
}

function torus(R = 1.3, r = 0.45, nu = 72, nv = 36) {
  const positions = [], indices = [];
  for (let i = 0; i < nu; i++) {
    const u = (2 * Math.PI * i) / nu;
    for (let j = 0; j < nv; j++) {
      const v = (2 * Math.PI * j) / nv;
      positions.push((R + r * Math.cos(v)) * Math.cos(u), r * Math.sin(v), (R + r * Math.cos(v)) * Math.sin(u));
    }
  }
  const id = (i, j) => (i % nu) * nv + (j % nv);
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      const a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
      indices.push(a, b, c, a, c, d);
    }
  }
  return { positions, indices };
}

const SHAPES = [
  { name: "sphere", ...sphere() },
  { name: "box", ...box() },
  { name: "torus", ...torus() },
];

// deterministic LCG
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

function implicit(shape, p) {
  if (shape === "sphere") return len3(p) - 1.6;
  if (shape === "box") return Math.max(Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2])) - 1.2;
  const q = Math.hypot(p[0], p[2]) - 1.3;
  return Math.hypot(q, p[1]) - 0.45;
}

function makeQueries(shapeName, positions, indices, diag) {
  const ur = lcg(0x5eed1234);
  const tris = indices.length / 3;
  const near = [];
  const deltas = [1e-3, 1e-6, 1e-9, 1e-11];
  const stride = Math.max(1, Math.floor(tris / 24));
  for (let k = 0; k < 24; k++) {
    const t = (k * stride) % tris;
    const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
    const A = [positions[a], positions[a + 1], positions[a + 2]];
    const B = [positions[b], positions[b + 1], positions[b + 2]];
    const C = [positions[c], positions[c + 1], positions[c + 2]];
    const cen = [(A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3, (A[2] + B[2] + C[2]) / 3];
    const n = cross3(sub3(B, A), sub3(C, A));
    const ln = len3(n) || 1;
    const un = [n[0] / ln, n[1] / ln, n[2] / ln];
    for (const d of deltas) {
      for (const sgn of [1, -1]) {
        near.push({
          p: [cen[0] + un[0] * sgn * d * diag, cen[1] + un[1] * sgn * d * diag, cen[2] + un[2] * sgn * d * diag],
          expectedInside: sgn < 0,
          kind: "near",
          delta: d,
        });
      }
    }
  }
  const far = [];
  let guard = 0;
  while (far.length < 208 && guard++ < 200000) {
    const p = [
      -2.0 + 4.0 * ur(), -2.0 + 4.0 * ur(), -2.0 + 4.0 * ur(),
    ].map((v) => v * (diag / 3.2));
    if (Math.abs(implicit(shapeName, p)) < 0.02 * diag) continue;
    far.push({ p, expectedInside: implicit(shapeName, p) < 0, kind: "far", delta: 0 });
  }
  const axis = [];
  const grid = [];
  if (shapeName === "box") {
    for (let x = -0.96; x <= 0.961; x += 0.24) for (let y = -0.96; y <= 0.961; y += 0.24) grid.push([x, y]);
  } else if (shapeName === "sphere") {
    for (let x = -1.4; x <= 1.41; x += 0.2) for (let y = -1.4; y <= 1.41; y += 0.2) {
      if (Math.hypot(x, y) < 1.55) grid.push([x, y]);
    }
  } else {
    for (let x = -1.6; x <= 1.61; x += 0.2) for (let y = -0.5; y <= 0.51; y += 0.1) grid.push([x, y]);
  }
  for (const [x, y] of grid) {
    for (const z of [0, 2.6]) {
      const p = [x, y, z];
      if (Math.abs(implicit(shapeName, p)) < 1e-3) continue;
      axis.push({ p, expectedInside: implicit(shapeName, p) < 0, kind: "axis", delta: 0, dir: [0, 0, -1] });
    }
  }
  return [...near, ...far, ...axis];
}

function buildAll() {
  const files = new Map();
  for (const s of SHAPES) {
    const positions = s.positions.map(f);
    const indices = [...s.indices];
    const vol = orientOutward(positions, indices);
    const stats = meshStats(positions, indices);
    if (!stats.watertight) throw new Error(`${s.name} not watertight`);
    const qs = makeQueries(s.name, positions, indices, stats.bbox.diag);
    files.set(`${s.name}.json`, JSON.stringify({
      ...meshToJSON(s.name, positions, indices),
      generator: "tools/gen-sample.mjs",
      volume: +vol.toFixed(12),
      stats,
    }));
    files.set(`queries-${s.name}.json`, JSON.stringify({
      mesh: s.name,
      queries: qs.map((q) => ({
        p: q.p.map((v) => +v.toPrecision(17)),
        expectedInside: q.expectedInside, kind: q.kind, delta: q.delta,
        ...(q.dir ? { dir: q.dir } : {}),
      })),
    }));
  }
  return files;
}

function sha256(buf) { return createHash("sha256").update(buf).digest("hex"); }

const files = buildAll();
if (process.argv.includes("--check")) {
  let ok = true;
  for (const [name, text] of files) {
    const path = new URL(name, OUT);
    const disk = existsSync(path) ? readFileSync(path, "utf8") : "";
    if (disk !== text) { ok = false; console.error(`MISMATCH ${name}`); }
  }
  const hash = sha256([...files.values()].join("\n"));
  console.log(JSON.stringify({ check: ok ? "identical" : "MISMATCH", files: files.size, corpusSha256: hash }));
  process.exit(ok ? 0 : 1);
} else {
  mkdirSync(OUT, { recursive: true });
  for (const [name, text] of files) writeFileSync(new URL(name, OUT), text);
  const hash = sha256([...files.values()].join("\n"));
  console.log(JSON.stringify({ wrote: files.size, corpusSha256: hash, bytes: [...files.values()].reduce((a, b) => a + b.length, 0) }));
}
