// Render layer only. Every classification comes from src/core.mjs.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildBVH, closestPointBVH, exactInside, makeExactMesh, meshBounds, meshFromJSON } from "./src/core.mjs";

const err = document.getElementById("err");
const fail = (e) => { err.style.display = "block"; err.textContent = "Failed to start: " + (e?.message || e); console.error(e); };

async function main() {
  const [meshRaw, queryRaw] = await Promise.all([
    fetch("./data/sphere.json").then((r) => r.json()),
    fetch("./data/queries-sphere.json").then((r) => r.json()),
  ]);
  const mesh = meshFromJSON(meshRaw);
  const { positions, indices } = mesh;
  const bb = meshBounds(positions);
  const bvh = buildBVH(positions, indices);
  const getTri = makeExactMesh(positions, indices);

  // ---- classify every query with the exact predicate (same code as node) ----
  const t0 = performance.now();
  const classified = queryRaw.queries.map((q) => {
    const dirs = q.dir ? [q.dir, [0.2672612419124244, 0.5345224838248488, 0.8017837257372732], [-0.4082482904638631, 0.8164965809277261, 0.4082482904638631]] : undefined;
    const res = exactInside(positions, indices, bvh, q.p, bb.diag, getTri, dirs);
    const near = closestPointBVH(bvh, positions, indices, q.p);
    return { ...q, res, distance: Math.sqrt(near.d2) };
  });
  const ms = performance.now() - t0;

  const inside = classified.filter((c) => c.res.status === "ok" && c.res.inside).length;
  const outside = classified.filter((c) => c.res.status === "ok" && !c.res.inside).length;
  const retried = classified.filter((c) => c.res.status === "ok" && c.res.dir > 0).length;
  const bad = classified.filter((c) => c.res.status !== "ok").length;
  document.getElementById("n-in").textContent = inside;
  document.getElementById("n-out").textContent = outside;
  document.getElementById("n-retry").textContent = retried;
  document.getElementById("n-bad").textContent = bad;
  document.getElementById("n-ms").textContent = ms.toFixed(0) + " ms / " + classified.length + " q";

  // ---- scene ----
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  document.getElementById("app").appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e13);
  const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 200);
  camera.position.set(4.6, 2.8, 4.9);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(positions), 3));
  geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(indices), 1));
  geometry.computeVertexNormals();
  const solid = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
    color: 0x5a7ea8, metalness: 0.15, roughness: 0.55, flatShading: true,
    transparent: true, opacity: 0.55, side: THREE.DoubleSide,
  }));
  scene.add(solid);
  const wire = new THREE.LineSegments(
    new THREE.WireframeGeometry(geometry),
    new THREE.LineBasicMaterial({ color: 0x2d4a66, transparent: true, opacity: 0.35 }),
  );
  scene.add(wire);

  // query dots: green inside, amber outside, magenta = exact ray needed a retry
  const palette = { in: [0.36, 0.88, 0.63], out: [1.0, 0.71, 0.33], retry: [0.83, 0.55, 1.0] };
  const dotGeom = new THREE.SphereGeometry(bb.diag * 0.0045, 8, 6);
  for (const c of classified) {
    const isIn = c.res.inside;
    const col = c.res.dir > 0 && c.res.status === "ok" ? palette.retry : isIn ? palette.in : palette.out;
    const dot = new THREE.Mesh(dotGeom, new THREE.MeshBasicMaterial({ color: new THREE.Color(...col) }));
    dot.position.set(c.p[0], c.p[1], c.p[2]);
    scene.add(dot);
  }

  // a subtle normal-offset line from the surface centroid probes: no, keep one interaction.
  scene.add(new THREE.HemisphereLight(0xcfe4ff, 0x11151c, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(3, 6, 4);
  scene.add(key);

  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
}

main().catch(fail);