// Independent baseline oracles.  Deliberately different implementations from
// src/core.mjs: barycentric projection + edge clamp for distance, and a plain
// double-precision Möller–Trumbore parity test with an epsilon for the sign.
// No shared constants with the exact path.

function d2(a, b) {
  const x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
  return x * x + y * y + z * z;
}

function segDist2(p, a, b) {
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
  const den = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
  let t = den > 0 ? (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / den : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return d2(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t]);
}

// distance from p to triangle: project to the plane and test barycentrics via
// dot products; outside -> closest of the three edges.
export function triDist2(p, a, b, c) {
  const v0 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const v2 = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
  const d00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
  const d01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
  const d11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
  const d20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2];
  const d21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];
  const den = d00 * d11 - d01 * d01;
  if (den > 0) {
    const v = (d11 * d20 - d01 * d21) / den;
    const w = (d00 * d21 - d01 * d20) / den;
    if (v >= 0 && w >= 0 && v + w <= 1) {
      const n = [
        v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0],
      ];
      const nn = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
      if (nn > 0) {
        const t = (v2[0] * n[0] + v2[1] * n[1] + v2[2] * n[2]) / nn;
        return d2(p, [p[0] - n[0] * t, p[1] - n[1] * t, p[2] - n[2] * t]);
      }
    }
  }
  return Math.min(segDist2(p, a, b), segDist2(p, b, c), segDist2(p, c, a));
}

export function closestBrute(positions, indices, p) {
  let best = Infinity, tri = -1;
  for (let t = 0; t < indices.length / 3; t++) {
    const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
    const d = triDist2(p,
      [positions[a], positions[a + 1], positions[a + 2]],
      [positions[b], positions[b + 1], positions[b + 2]],
      [positions[c], positions[c + 1], positions[c + 2]]);
    if (d < best) { best = d; tri = t; }
  }
  return { d2: best, tri };
}

export function floatRayInside(positions, indices, p, dir, rayLen, eps = 1e-12) {
  const q = [p[0] + dir[0] * rayLen, p[1] + dir[1] * rayLen, p[2] + dir[2] * rayLen];
  const dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
  let crossings = 0;
  for (let t = 0; t < indices.length / 3; t++) {
    const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
    const ax = positions[a], ay = positions[a + 1], az = positions[a + 2];
    const e1x = positions[b] - ax, e1y = positions[b + 1] - ay, e1z = positions[b + 2] - az;
    const e2x = positions[c] - ax, e2y = positions[c + 1] - ay, e2z = positions[c + 2] - az;
    const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (Math.abs(det) < eps) continue;
    const inv = 1 / det;
    const tx = p[0] - ax, ty = p[1] - ay, tz = p[2] - az;
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
