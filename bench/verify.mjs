// Assertions over the generated corpus.  Every printed number is measured here.
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import {
  buildBVH, closestPointBVH, exactInside, makeExactMesh, meshBounds, meshFromJSON,
  meshStats, RAY_DIRS, len3, sub3,
} from "../src/core.mjs";
import { closestBrute, floatRayInside } from "../baseline/naive.mjs";

const DATA = new URL("../data/", import.meta.url);
const SHAPES = ["sphere", "box", "torus"];
let failures = 0;
const rows = [];
function check(name, ok, detail) {
  rows.push({ name, ok, detail });
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}  ${detail}`);
}
const load = (f) => JSON.parse(readFileSync(new URL(f, DATA), "utf8"));

/* G7: data is script-generated and byte-deterministic */
const gen = JSON.parse(execFileSync(process.execPath, [new URL("../tools/gen-sample.mjs", import.meta.url).pathname, "--check"], { encoding: "utf8" }).trim());
check("G7 deterministic regeneration", gen.check === "identical", `6 files, corpusSha256=${gen.corpusSha256.slice(0, 16)}…`);

const meshes = {};
for (const name of SHAPES) {
  const mesh = meshFromJSON(load(`${name}.json`));
  const stats = meshStats(mesh.positions, mesh.indices);
  meshes[name] = { ...mesh, stats, bvh: buildBVH(mesh.positions, mesh.indices), getTri: makeExactMesh(mesh.positions, mesh.indices) };
  const bad = [...mesh.positions].some((v) => !Number.isFinite(v));
  check(`mesh ${name} watertight`, stats.watertight && !bad,
    `${stats.vertices} v / ${stats.triangles} t / boundary=${stats.boundaryEdges} nonManifold=${stats.nonManifoldEdges} diag=${stats.bbox.diag.toFixed(4)}`);
}

/* exact vs float containment, with constructed ground truth */
let total = 0, exactErr = 0, inconclusive = 0, floatErr = 0, axisTotal = 0, axisFloatErr = 0, retries = 0;
const errBy = {};
for (const name of SHAPES) {
  const m = meshes[name];
  const qs = load(`queries-${name}.json`).queries;
  for (const q of qs) {
    total++;
    const dirs = q.dir ? [q.dir, ...RAY_DIRS] : RAY_DIRS;
    const r = exactInside(m.positions, m.indices, m.bvh, q.p, m.stats.bbox.diag, m.getTri, dirs);
    if (r.status !== "ok") inconclusive++;
    else if (r.inside !== q.expectedInside) { exactErr++; errBy[name] = (errBy[name] || 0) + 1; }
    if (q.kind === "axis") retries += r.status === "ok" && r.dir > 0 ? 1 : 0;
    const f = floatRayInside(m.positions, m.indices, q.p, q.dir || RAY_DIRS[0], m.stats.bbox.diag * 4);
    if (f.inside !== q.expectedInside) { floatErr++; if (q.kind === "axis") axisFloatErr++; }
    if (q.kind === "axis") axisTotal++;
  }
}
check("exact containment", exactErr === 0 && inconclusive === 0,
  `${total - exactErr - inconclusive}/${total} correct, wrong=${exactErr}, inconclusive=${inconclusive} (${SHAPES.length} meshes)`);
check("float ray baseline is actually wrong here", floatErr > 0,
  `float Möller–Trumbore + eps=1e-12: wrong=${floatErr}/${total} = ${((100 * floatErr) / total).toFixed(2)}% (axis-aligned subset ${axisFloatErr}/${axisTotal}); exact retried ${retries} degenerate rays`);
check("sensitivity is not vacuous", axisFloatErr > 0 && total - axisTotal > 0,
  `non-axis wrong=${floatErr - axisFloatErr}, axis wrong=${axisFloatErr}; corpus contains both query families`);

/* distance: BVH vs an independently written brute-force oracle */
let distChecks = 0, maxDiff = 0, maxRelFar = 0, oracleAgree = 0;
for (const name of SHAPES) {
  const m = meshes[name];
  const qs = load(`queries-${name}.json`).queries;
  for (let i = 0; i < qs.length; i += 7) {
    const p = qs[i].p;
    const a = closestPointBVH(m.bvh, m.positions, m.indices, p);
    const b = closestBrute(m.positions, m.indices, p);
    const da = Math.sqrt(a.d2), db = Math.sqrt(b.d2), diag = m.stats.bbox.diag;
    maxDiff = Math.max(maxDiff, Math.abs(da - db));
    if (db > 1e-3 * diag) maxRelFar = Math.max(maxRelFar, Math.abs(da - db) / db);
    if (Math.abs(da - db) <= 1e-12 * diag) oracleAgree++;
    distChecks++;
  }
}
check("distance vs independent oracle", oracleAgree === distChecks && maxRelFar <= 1e-12,
  `${oracleAgree}/${distChecks} queries within 1e-12·diag; max|Δ|=${maxDiff.toExponential(2)} m; far-field max relative=${maxRelFar.toExponential(2)}`);

/* scale invariance: 1e-6 … 1e6 must not change a single boolean */
const scales = [1e-6, 1e-3, 1e-12, 1, 1e3, 1e6];
let scaleChecks = 0, scaleSignFlips = 0, maxScaleAbs = 0, maxScaleRelFar = 0;
for (const name of SHAPES) {
  const m = meshes[name];
  const qs = load(`queries-${name}.json`).queries.filter((q) => q.kind === "near").slice(0, 24);
  const baseD = qs.map((q) => {
    const r = closestPointBVH(m.bvh, m.positions, m.indices, q.p);
    return Math.sqrt(r.d2);
  });
  for (const s of scales) {
    const P = Float64Array.from(m.positions, (v) => v * s);
    const bvh = buildBVH(P, m.indices);
    const gt = makeExactMesh(P, m.indices);
    const bb = meshBounds(P);
    for (let i = 0; i < qs.length; i++) {
      const p = qs[i].p.map((v) => v * s);
      const r = exactInside(P, m.indices, bvh, p, bb.diag, gt);
      if (r.status !== "ok" || r.inside !== qs[i].expectedInside) scaleSignFlips++;
      const d = Math.sqrt(closestPointBVH(bvh, P, m.indices, p).d2);
      const ref = baseD[i] * s;
      const absDrift = Math.abs(d - ref);
      maxScaleAbs = Math.max(maxScaleAbs, absDrift / (bb.diag || 1));
      if (ref > 1e-3 * bb.diag) maxScaleRelFar = Math.max(maxScaleRelFar, absDrift / ref);
      scaleChecks++;
    }
  }
}
check("scale invariance", scaleSignFlips === 0 && maxScaleAbs <= 1e-12 && maxScaleRelFar <= 1e-12,
  `${scaleChecks} queries across scales ${scales.join("/")}; sign flips=${scaleSignFlips}; max |Δd|/diag=${maxScaleAbs.toExponential(2)}; far-field max relative drift=${maxScaleRelFar.toExponential(2)}`);

/* axis retry really is exercised, report the count */
check("degeneracy handling is real", retries > 0, `${retries}/${axisTotal} axis rays hit an edge/vertex and were retried exactly`);

console.log("\n--- summary ---");
console.log(JSON.stringify({ checks: rows.length, failed: failures, exactWrong: exactErr, floatWrong: floatErr, total, distanceChecks: distChecks, scaleChecks, retries }));
process.exit(failures === 0 ? 0 : 1);
