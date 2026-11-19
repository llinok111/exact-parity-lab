// core.mjs — certified point/mesh queries.
// Pure ES module: no DOM, no three.js, no I/O, no random, no clock.
//
// Two independent facilities:
//   1. float BVH closest-point distance (fast path, used by every query)
//   2. exact dyadic-rational inside/outside test (certified sign, no epsilon)
//
// A double is a dyadic rational m * 2^e, so every orientation predicate on the
// *represented* doubles can be evaluated exactly with BigInt.  The ray-parity
// oracle below never uses an epsilon and never divides by a float.

/* ------------------------------------------------------------------ */
/* vec3 helpers (plain arrays)                                        */
/* ------------------------------------------------------------------ */
export const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const len3 = (a) => Math.sqrt(dot3(a, a));

/* ------------------------------------------------------------------ */
/* Closest point on a triangle (Ericson, Real-Time Collision Detection) */
/* ------------------------------------------------------------------ */
export function closestPointTriangle(p, a, b, c) {
  const ab = sub3(b, a), ac = sub3(c, a), ap = sub3(p, a);
  const d1 = dot3(ab, ap), d2 = dot3(ac, ap);
  if (d1 <= 0 && d2 <= 0) return { x: a[0], y: a[1], z: a[2] };
  const bp = sub3(p, b);
  const d3 = dot3(ab, bp), d4 = dot3(ac, bp);
  if (d3 >= 0 && d4 <= d3) return { x: b[0], y: b[1], z: b[2] };
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return { x: a[0] + ab[0] * v, y: a[1] + ab[1] * v, z: a[2] + ab[2] * v };
  }
  const cp = sub3(p, c);
  const d5 = dot3(ab, cp), d6 = dot3(ac, cp);
  if (d6 >= 0 && d5 <= d6) return { x: c[0], y: c[1], z: c[2] };
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return { x: a[0] + ac[0] * w, y: a[1] + ac[1] * w, z: a[2] + ac[2] * w };
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return { x: b[0] + (c[0] - b[0]) * w, y: b[1] + (c[1] - b[1]) * w, z: b[2] + (c[2] - b[2]) * w };
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom, w = vc * denom;
  return {
    x: a[0] + ab[0] * v + ac[0] * w,
    y: a[1] + ab[1] * v + ac[1] * w,
    z: a[2] + ab[2] * v + ac[2] * w,
  };
}

/* ------------------------------------------------------------------ */
/* BVH over triangles                                                  */
/* ------------------------------------------------------------------ */
export function meshBounds(positions) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      if (positions[i + k] < min[k]) min[k] = positions[i + k];
      if (positions[i + k] > max[k]) max[k] = positions[i + k];
    }
  }
  return { min, max, diag: len3(sub3(max, min)) };
}

export function buildBVH(positions, indices, leafSize = 8) {
  const triCount = indices.length / 3;
  const order = new Int32Array(triCount);
  const centroids = new Float64Array(triCount * 3);
  for (let t = 0; t < triCount; t++) {
    order[t] = t;
    let cx = 0, cy = 0, cz = 0;
    for (let k = 0; k < 3; k++) {
      const v = indices[t * 3 + k] * 3;
      cx += positions[v]; cy += positions[v + 1]; cz += positions[v + 2];
    }
    centroids[t * 3] = cx / 3; centroids[t * 3 + 1] = cy / 3; centroids[t * 3 + 2] = cz / 3;
  }
  const nodes = [];
  const triBox = (t) => {
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < 3; k++) {
      const v = indices[t * 3 + k] * 3;
      for (let q = 0; q < 3; q++) {
        if (positions[v + q] < mn[q]) mn[q] = positions[v + q];
        if (positions[v + q] > mx[q]) mx[q] = positions[v + q];
      }
    }
    return { mn, mx };
  };
  const build = (start, end) => {
    const id = nodes.length;
    nodes.push(null);
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; i++) {
      const t = order[i], b = triBox(t);
      for (let q = 0; q < 3; q++) {
        if (b.mn[q] < mn[q]) mn[q] = b.mn[q];
        if (b.mx[q] > mx[q]) mx[q] = b.mx[q];
      }
    }
    if (end - start <= leafSize) {
      nodes[id] = { mn, mx, left: -1, right: -1, start, count: end - start };
      return id;
    }
    let ext = [0, 0, 0];
    for (let i = start; i < end; i++) {
      const t = order[i];
      for (let q = 0; q < 3; q++) {
        const lo = mn[q], hi = mx[q];
        if (hi - lo > ext[q]) ext[q] = hi - lo;
      }
    }
    let axis = ext[0] >= ext[1] && ext[0] >= ext[2] ? 0 : ext[1] >= ext[2] ? 1 : 2;
    const mid = (start + end) >> 1;
    const slice = Array.from(order.subarray(start, end));
    slice.sort((i, j) => centroids[i * 3 + axis] - centroids[j * 3 + axis] || i - j);
    order.set(slice, start);
    const left = build(start, mid);
    const right = build(mid, end);
    nodes[id] = { mn, mx, left, right, start: -1, count: 0 };
    return id;
  };
  const root = triCount > 0 ? build(0, triCount) : -1;
  return { nodes, order, root, triCount, leafSize };
}

function boxDist2(node, px, py, pz) {
  const mn = node.mn, mx = node.mx;
  let dx = 0, dy = 0, dz = 0;
  if (px < mn[0]) dx = mn[0] - px; else if (px > mx[0]) dx = px - mx[0];
  if (py < mn[1]) dy = mn[1] - py; else if (py > mx[1]) dy = py - mx[1];
  if (pz < mn[2]) dz = mn[2] - pz; else if (pz > mx[2]) dz = pz - mx[2];
  return dx * dx + dy * dy + dz * dz;
}

export function closestPointBVH(bvh, positions, indices, p) {
  if (bvh.root < 0) return { d2: Infinity, tri: -1 };
  const stack = [bvh.root];
  let best = Infinity, bestTri = -1;
  while (stack.length) {
    const node = bvh.nodes[stack.pop()];
    if (boxDist2(node, p[0], p[1], p[2]) >= best) continue;
    if (node.left < 0) {
      for (let i = node.start; i < node.start + node.count; i++) {
        const t = bvh.order[i];
        const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
        const cp = closestPointTriangle(
          p,
          [positions[a], positions[a + 1], positions[a + 2]],
          [positions[b], positions[b + 1], positions[b + 2]],
          [positions[c], positions[c + 1], positions[c + 2]],
        );
        const dx = cp.x - p[0], dy = cp.y - p[1], dz = cp.z - p[2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best) { best = d2; bestTri = t; }
      }
    } else {
      const l = bvh.nodes[node.left], r = bvh.nodes[node.right];
      const dl = boxDist2(l, p[0], p[1], p[2]), dr = boxDist2(r, p[0], p[1], p[2]);
      if (dl < dr) { if (dr < best) stack.push(node.right); if (dl < best) stack.push(node.left); }
      else { if (dl < best) stack.push(node.left); if (dr < best) stack.push(node.right); }
    }
  }
  return { d2: best, tri: bestTri };
}

/* ------------------------------------------------------------------ */
/* Exact dyadic rationals: a double is exactly m * 2^e                */
/* ------------------------------------------------------------------ */
const _dv = new DataView(new ArrayBuffer(8));

export function toDyadic(x) {
  _dv.setFloat64(0, x);
  const hi = _dv.getUint32(0), lo = _dv.getUint32(4);
  const neg = (hi >>> 31) === 1;
  const exp = (hi >>> 20) & 0x7ff;
  let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e;
  if (exp === 0) {
    if (m === 0n) return { m: 0n, e: 0 };
    e = -1074;
  } else { m |= 1n << 52n; e = exp - 1075; }
  return { m: neg ? -m : m, e };
}

const dnorm = (a) => {
  let m = a.m, e = a.e;
  if (m === 0n) return { m: 0n, e: 0 };
  while ((m & 1n) === 0n) { m >>= 1n; e += 1; }
  return { m, e };
};
export const dadd = (a, b) => {
  if (a.m === 0n) return b;
  if (b.m === 0n) return a;
  const e = Math.min(a.e, b.e);
  return dnorm({ m: (a.m << BigInt(a.e - e)) + (b.m << BigInt(b.e - e)), e });
};
export const dsub = (a, b) => dadd(a, { m: -b.m, e: b.e });
export const dmul = (a, b) => (a.m === 0n || b.m === 0n ? { m: 0n, e: 0 } : dnorm({ m: a.m * b.m, e: a.e + b.e }));
export const dsign = (a) => (a.m > 0n ? 1 : a.m < 0n ? -1 : 0);

function orient3dD(A, B, C, D) {
  const ax = dsub(A[0], D[0]), ay = dsub(A[1], D[1]), az = dsub(A[2], D[2]);
  const bx = dsub(B[0], D[0]), by = dsub(B[1], D[1]), bz = dsub(B[2], D[2]);
  const cx = dsub(C[0], D[0]), cy = dsub(C[1], D[1]), cz = dsub(C[2], D[2]);
  const m1 = dsub(dmul(by, cz), dmul(bz, cy));
  const m2 = dsub(dmul(bx, cz), dmul(bz, cx));
  const m3 = dsub(dmul(bx, cy), dmul(by, cx));
  return dadd(dsub(dmul(ax, m1), dmul(ay, m2)), dmul(az, m3));
}

export function signOrient3d(a, b, c, d) {
  const A = a.map(toDyadic), B = b.map(toDyadic), C = c.map(toDyadic), D = d.map(toDyadic);
  return dsign(orient3dD(A, B, C, D));
}

// sign of the 2D orientation of (A,B,X) where X = Xnum / D
function orient2dSign(A, B, Xnum, D, u, v) {
  const left = dmul(dsub(B[u], A[u]), dsub(Xnum[v], dmul(A[v], D)));
  const right = dmul(dsub(B[v], A[v]), dsub(Xnum[u], dmul(A[u], D)));
  return dsign(dsub(left, right)) * dsign(D);
}

/**
 * Exact segment/triangle crossing on represented doubles.
 * status: "cross" | "miss" | "degenerate" (touches an edge/vertex or coplanar).
 */
export function segmentTriangleExact(p, q, A, B, C, fa, fb, fc) {
  const P = p.map(toDyadic), Q = q.map(toDyadic);
  const dp = orient3dD(A, B, C, P);
  if (dp.m === 0n) return { status: "degenerate" };
  const dq = orient3dD(A, B, C, Q);
  if (dq.m === 0n) return { status: "degenerate" };
  if ((dp.m > 0n) === (dq.m > 0n)) return { status: "miss" };

  const D = dsub(dp, dq);
  const X = [0, 1, 2].map((k) => dsub(dmul(dp, Q[k]), dmul(dq, P[k])));

  // projection axis from the float normal; only affects degeneracy detection
  const nx = Math.abs((fb[1] - fa[1]) * (fc[2] - fa[2]) - (fb[2] - fa[2]) * (fc[1] - fa[1]));
  const ny = Math.abs((fb[2] - fa[2]) * (fc[0] - fa[0]) - (fb[0] - fa[0]) * (fc[2] - fa[2]));
  const nz = Math.abs((fb[0] - fa[0]) * (fc[1] - fa[1]) - (fb[1] - fa[1]) * (fc[0] - fa[0]));
  let u = 0, v = 1;
  if (nx >= ny && nx >= nz) { u = 1; v = 2; }
  else if (ny >= nz) { u = 2; v = 0; }
  else { u = 0; v = 1; }

  const s1 = orient2dSign(A, B, X, D, u, v);
  const s2 = orient2dSign(B, C, X, D, u, v);
  const s3 = orient2dSign(C, A, X, D, u, v);
  if (s1 === 0 || s2 === 0 || s3 === 0) return { status: "degenerate" };
  const allPos = s1 > 0 && s2 > 0 && s3 > 0;
  const allNeg = s1 < 0 && s2 < 0 && s3 < 0;
  if (allPos || allNeg) return { status: "cross" };
  return { status: "miss" };
}

/* ------------------------------------------------------------------ */
/* BVH-assisted exact ray parity                                      */
/* ------------------------------------------------------------------ */
export const RAY_DIRS = [
  [0.2672612419124244, 0.5345224838248488, 0.8017837257372732],
  [-0.4082482904638631, 0.8164965809277261, 0.4082482904638631],
  [0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
];

function segmentHitsBox(mn, mx, p, q) {
  let tmin = 0, tmax = 1;
  for (let k = 0; k < 3; k++) {
    const d = q[k] - p[k];
    if (Math.abs(d) < 1e-300) {
      if (p[k] < mn[k] || p[k] > mx[k]) return false;
    } else {
      let t1 = (mn[k] - p[k]) / d, t2 = (mx[k] - p[k]) / d;
      if (t1 > t2) { const s = t1; t1 = t2; t2 = s; }
      if (t1 > tmin) tmin = t1;
      if (t2 < tmax) tmax = t2;
      if (tmin > tmax) return false;
    }
  }
  return true;
}

// conservative candidate triangles for the segment p->q
export function segmentCandidates(bvh, positions, indices, p, q) {
  if (bvh.root < 0) return [];
  const out = [];
  const stack = [bvh.root];
  while (stack.length) {
    const node = bvh.nodes[stack.pop()];
    if (!segmentHitsBox(node.mn, node.mx, p, q)) continue;
    if (node.left < 0) {
      for (let i = node.start; i < node.start + node.count; i++) out.push(bvh.order[i]);
    } else {
      stack.push(node.left, node.right);
    }
  }
  return out;
}

export function makeExactMesh(positions, indices) {
  const cache = new Array(indices.length / 3);
  return (t) => {
    if (!cache[t]) {
      const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
      cache[t] = [
        [toDyadic(positions[a]), toDyadic(positions[a + 1]), toDyadic(positions[a + 2])],
        [toDyadic(positions[b]), toDyadic(positions[b + 1]), toDyadic(positions[b + 2])],
        [toDyadic(positions[c]), toDyadic(positions[c + 1]), toDyadic(positions[c + 2])],
      ];
    }
    return cache[t];
  };
}

/**
 * Exact inside/outside by parity of ray crossings, computed in BigInt rationals.
 * Returns { status:"ok", inside, crossings, dir } or { status:"inconclusive", ... }.
 * Three generic directions are tried; a ray touching an edge/vertex is retried.
 */
export function exactInside(positions, indices, bvh, p, diag, getTri = makeExactMesh(positions, indices), dirs = RAY_DIRS) {
  const attempts = [];
  for (let di = 0; di < dirs.length; di++) {
    const d = dirs[di], L = diag * 4;
    const q = [p[0] + d[0] * L, p[1] + d[1] * L, p[2] + d[2] * L];
    const cand = segmentCandidates(bvh, positions, indices, p, q);
    let crossings = 0, degenerate = false;
    for (const t of cand) {
      const tri = getTri(t);
      const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
      const fa = [positions[a], positions[a + 1], positions[a + 2]];
      const fb = [positions[b], positions[b + 1], positions[b + 2]];
      const fc = [positions[c], positions[c + 1], positions[c + 2]];
      const r = segmentTriangleExact(p, q, tri[0], tri[1], tri[2], fa, fb, fc);
      if (r.status === "degenerate") { degenerate = true; break; }
      if (r.status === "cross") crossings++;
    }
    attempts.push({ dir: di, candidates: cand.length, crossings, degenerate });
    if (!degenerate) {
      return { status: "ok", inside: (crossings & 1) === 1, crossings, dir: di, attempts };
    }
  }
  return { status: "inconclusive", inside: null, crossings: null, dir: -1, attempts };
}

/* ------------------------------------------------------------------ */
/* float baseline: Möller–Trumbore with an epsilon (for comparison)   */
/* ------------------------------------------------------------------ */
export function floatInside(positions, indices, p, dir = RAY_DIRS[0], L = 1e6, eps = 1e-12) {
  const q = [p[0] + dir[0] * L, p[1] + dir[1] * L, p[2] + dir[2] * L];
  const dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
  let crossings = 0;
  for (let t = 0; t < indices.length / 3; t++) {
    const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
    const e1x = positions[b] - positions[a], e1y = positions[b + 1] - positions[a + 1], e1z = positions[b + 2] - positions[a + 2];
    const e2x = positions[c] - positions[a], e2y = positions[c + 1] - positions[a + 1], e2z = positions[c + 2] - positions[a + 2];
    const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (Math.abs(det) < eps) continue;
    const inv = 1 / det;
    const tx = p[0] - positions[a], ty = p[1] - positions[a + 1], tz = p[2] - positions[a + 2];
    const u = (tx * px + ty * py + tz * pz) * inv;
    if (u < -eps || u > 1 + eps) continue;
    const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
    const v = (dx * qx + dy * qy + dz * qz) * inv;
    if (v < -eps || u + v > 1 + eps) continue;
    const tt = (e2x * qx + e2y * qy + e2z * qz) * inv;
    if (tt > eps && tt < 1 - eps) crossings++;
  }
  return { inside: (crossings & 1) === 1, crossings };
}

/* ------------------------------------------------------------------ */
/* mesh stats                                                         */
/* ------------------------------------------------------------------ */
export function meshStats(positions, indices) {
  const { min, max, diag } = meshBounds(positions);
  const edge = new Map();
  const key = (i, j) => (i < j ? `${i}_${j}` : `${j}_${i}`);
  for (let t = 0; t < indices.length / 3; t++) {
    const a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
    for (const [i, j] of [[a, b], [b, c], [c, a]]) {
      const k = key(i, j);
      edge.set(k, (edge.get(k) || 0) + 1);
    }
  }
  let boundary = 0, nonManifold = 0, interior = 0;
  for (const n of edge.values()) {
    if (n === 1) boundary++; else if (n === 2) interior++; else nonManifold++;
  }
  return {
    vertices: positions.length / 3,
    triangles: indices.length / 3,
    edges: edge.size,
    boundaryEdges: boundary,
    nonManifoldEdges: nonManifold,
    interiorEdges: interior,
    watertight: boundary === 0 && nonManifold === 0,
    bbox: { min, max, diag },
  };
}

/* ------------------------------------------------------------------ */
/* mesh (de)serialisation helpers — pure, no filesystem               */
/* ------------------------------------------------------------------ */
export function meshFromJSON(obj) {
  return {
    name: obj.name,
    positions: Float64Array.from(obj.positions),
    indices: Uint32Array.from(obj.indices),
  };
}
export function meshToJSON(name, positions, indices) {
  return {
    name,
    positions: Array.from(positions, (v) => +v.toPrecision(17)),
    indices: Array.from(indices),
  };
}
