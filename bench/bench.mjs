// Timing benchmark: cold start + 5 warm samples, median/p95/stddev.
// All numbers come from this run; nothing is copied from a previous session.
import { readFileSync, writeFileSync } from "node:fs";
import {
  buildBVH, closestPointBVH, exactInside, makeExactMesh, meshBounds, meshFromJSON, RAY_DIRS,
} from "../src/core.mjs";
import { closestBrute, floatRayInside } from "../baseline/naive.mjs";

const DATA = new URL("../data/", import.meta.url);
const SHAPES = ["sphere", "box", "torus"];
const load = (f) => JSON.parse(readFileSync(new URL(f, DATA), "utf8"));
const now = () => performance.now();
const stats = (a) => {
  const s = [...a].sort((x, y) => x - y);
  const med = s[Math.floor(s.length / 2)];
  const mean = s.reduce((x, y) => x + y, 0) / s.length;
  const sd = Math.sqrt(s.reduce((x, y) => x + (y - mean) ** 2, 0) / s.length);
  return { min: +s[0].toFixed(3), median: +med.toFixed(3), p95: +s[Math.min(s.length - 1, Math.ceil(0.95 * s.length) - 1)].toFixed(3), stddev: +sd.toFixed(3), rsdPct: +((100 * sd) / mean).toFixed(2) };
};

const meshes = {};
for (const name of SHAPES) {
  const m = meshFromJSON(load(`${name}.json`));
  meshes[name] = { ...m, bb: meshBounds(m.positions), qs: load(`queries-${name}.json`).queries };
}

const allQueries = [];
for (const name of SHAPES) for (const q of meshes[name].qs) allQueries.push({ name, ...q });

// distinct query subsets: keep total runtime bounded
const distQueries = [];
for (const name of SHAPES) for (let i = 0; i < meshes[name].qs.length; i += 3) distQueries.push({ name, p: meshes[name].qs[i].p });
const signQueries = allQueries;

function runDistanceBVH() {
  for (const q of distQueries) {
    const m = meshes[q.name];
    closestPointBVH(m.bvh, m.positions, m.indices, q.p);
  }
  return distQueries.length;
}
function runDistanceBrute() {
  for (const q of distQueries) {
    const m = meshes[q.name];
    closestBrute(m.positions, m.indices, q.p);
  }
  return distQueries.length;
}
function runExactSign() {
  for (const q of signQueries) {
    const m = meshes[q.name];
    const dirs = q.dir ? [q.dir, ...RAY_DIRS] : RAY_DIRS;
    exactInside(m.positions, m.indices, m.bvh, q.p, m.bb.diag, m.getTri, dirs);
  }
  return signQueries.length;
}
function runFloatSign() {
  for (const q of signQueries) {
    const m = meshes[q.name];
    floatRayInside(m.positions, m.indices, q.p, q.dir || RAY_DIRS[0], m.bb.diag * 4);
  }
  return signQueries.length;
}

for (const name of SHAPES) {
  const m = meshes[name];
  m.bvh = buildBVH(m.positions, m.indices);
  m.getTri = makeExactMesh(m.positions, m.indices);
}

// correctness counts recomputed here (not copied from verify)
let exactWrong = 0, exactInconclusive = 0, floatWrong = 0, exactRetries = 0;
for (const q of signQueries) {
  const m = meshes[q.name];
  const dirs = q.dir ? [q.dir, ...RAY_DIRS] : RAY_DIRS;
  const r = exactInside(m.positions, m.indices, m.bvh, q.p, m.bb.diag, m.getTri, dirs);
  if (r.status !== "ok") exactInconclusive++;
  else if (r.inside !== q.expectedInside) exactWrong++;
  if (q.dir && r.status === "ok" && r.dir > 0) exactRetries++;
  const f = floatRayInside(m.positions, m.indices, q.p, q.dir || RAY_DIRS[0], m.bb.diag * 4);
  if (f.inside !== q.expectedInside) floatWrong++;
}

// cold start (very first invocation of each routine)
const tasks = {
  bvh_build_sphere: () => buildBVH(meshes.sphere.positions, meshes.sphere.indices),
  distance_bvh: runDistanceBVH,
  distance_brute: runDistanceBrute,
  exact_sign: runExactSign,
  float_sign_baseline: runFloatSign,
};
const cold = {};
for (const [k, fn] of Object.entries(tasks)) {
  const t = now(); const n = fn(); cold[k] = { ms: +(now() - t).toFixed(3), queries: typeof n === "number" ? n : 1 };
}
// warmup once, then 5 timed samples
const samples = {};
for (const [k, fn] of Object.entries(tasks)) {
  fn();
  samples[k] = [];
  for (let i = 0; i < 5; i++) { const t = now(); fn(); samples[k].push(now() - t); }
}

const perQuery = (key, q) => +(stats(samples[key]).median * 1000 / q).toFixed(3);
const report = {
  generatedAt: new Date().toISOString(),
  node: process.version,
  corpus: { meshes: 3, triangles: SHAPES.reduce((a, n) => a + meshes[n].indices.length / 3, 0), containmentQueries: signQueries.length, distanceQueries: distQueries.length },
  counts: { exactWrong, exactInconclusive, floatWrong, exactRetries, queries: signQueries.length },
  cold,
  samples,
  summary: {},
};
report.summary = {
  bvhBuildMs: stats(samples.bvh_build_sphere).median,
  distanceBvhMs: stats(samples.distance_bvh).median,
  distanceBruteMs: stats(samples.distance_brute).median,
  distanceSpeedup: +(stats(samples.distance_brute).median / stats(samples.distance_bvh).median).toFixed(1),
  distanceBvhUsPerQuery: perQuery("distance_bvh", distQueries.length),
  distanceBruteUsPerQuery: perQuery("distance_brute", distQueries.length),
  exactSignMs: stats(samples.exact_sign).median,
  exactSignUsPerQuery: perQuery("exact_sign", signQueries.length),
  floatSignUsPerQuery: perQuery("float_sign_baseline", signQueries.length),
  exactOverheadVsFloat: +(stats(samples.exact_sign).median / stats(samples.float_sign_baseline).median).toFixed(2),
  rsdPct: Object.fromEntries(Object.keys(samples).map((k) => [k, stats(samples[k]).rsdPct])),
};

console.log(JSON.stringify(report, null, 2));
writeFileSync(new URL("./result.json", import.meta.url), JSON.stringify(report, null, 2) + "\n");