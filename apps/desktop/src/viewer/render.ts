// Three.js scene for point clouds: perspective or axis-locked orthographic
// cameras, per-point visibility masks, a slice outline and an optional
// side-by-side compare pane. Draws on demand only.
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {LineSegments2} from 'three/addons/lines/LineSegments2.js';
import {LineSegmentsGeometry} from 'three/addons/lines/LineSegmentsGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';
import type {Box, Cloud, Orientation} from '../types';
import {IDENTITY, quaternion} from './orient';

export type ViewMode = 'persp' | 'top' | 'front' | 'side';
export const DEPTH_AXIS: Record<Exclude<ViewMode, 'persp'>, 0 | 1 | 2> = {top: 2, front: 1, side: 0};

const VERT = `
uniform float pointSize; uniform vec2 zRange; attribute vec3 color; attribute float visible; varying vec3 vColor;
void main(){
  vColor = color;
  vec4 world = modelMatrix * vec4(position, 1.0);
  if (visible < 0.5 || world.z < zRange.x || world.z > zRange.y) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  gl_Position = projectionMatrix * viewMatrix * world;
  gl_PointSize = pointSize;
}`;
const FRAG = `
varying vec3 vColor;
void main(){ vec2 d = gl_PointCoord - vec2(0.5); if (dot(d, d) > 0.25) discard; gl_FragColor = vec4(vColor, 1.0); }`;

type Entry = {cloud: Cloud; points: THREE.Points; visible: THREE.BufferAttribute; material: THREE.ShaderMaterial; localBounds: THREE.Box3; bounds: THREE.Box3; visibleBounds: THREE.Box3; shown: boolean; mask: Float32Array | null; localVisible: THREE.Box3 | null};

export class CloudRenderer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private persp = new THREE.PerspectiveCamera(48, 1, 0.01, 10000);
  private ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -10000, 10000);
  private controls: OrbitControls;
  private entries = new Map<string, Entry>();
  private outline: THREE.LineSegments | null = null;
  private polygons: LineSegments2[] = [];
  private lineMaterials: LineMaterial[] = [];
  private fills: THREE.Mesh[] = [];
  private grid: THREE.GridHelper | null = null;
  private observer: ResizeObserver;
  worldOrigin: THREE.Vector3 | null = null;
  mode: ViewMode = 'persp';
  flipped = false;
  compare: string | null = null;
  pointSize = 1.6;
  /** Height window in scene z; points outside are hidden by the shader. */
  private zWindow: [number, number] = [-1e9, 1e9];
  onChange: (() => void) | null = null;

  /** Hide points below `lo` or above `hi` (world z); null lifts the limit. */
  setHeightWindow(lo: number | null, hi: number | null) {
    const oz = this.worldOrigin?.z ?? 0;
    this.zWindow = [lo == null ? -1e9 : lo - oz, hi == null ? 1e9 : hi - oz];
    for (const e of this.entries.values()) (e.material.uniforms.zRange.value as THREE.Vector2).set(this.zWindow[0], this.zWindow[1]);
    this.draw();
  }
  /** World z extent of the shown clouds, for the height slider. */
  worldZRange(): [number, number] | null {
    const b = this.shownBounds();
    if (!b || !this.worldOrigin) return null;
    return [b.min.z + this.worldOrigin.z, b.max.z + this.worldOrigin.z];
  }

  constructor(private canvas: HTMLCanvasElement, private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: false, powerPreference: 'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x1b1e21);
    this.renderer.autoClear = false;
    this.persp.up.set(0, 0, 1);
    this.ortho.up.set(0, 0, 1);
    this.controls = this.makeControls(this.persp);
    this.canvas.addEventListener('wheel', this.flyThrough, {capture: true, passive: false});
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();
  }

  private get camera(): THREE.Camera { return this.mode === 'persp' ? this.persp : this.ortho; }

  private makeControls(camera: THREE.Camera) {
    // Listen on the canvas only: overlays (slice handles, buttons) are
    // siblings and must never be captured by the controls.
    const c = new OrbitControls(camera, this.canvas);
    c.enableDamping = false;
    c.screenSpacePanning = true;
    c.zoomToCursor = true;
    // Never let the orbit centre become a wall: below this distance a wheel
    // tick flies the camera forward instead (see flyThrough).
    c.minDistance = 0.25;
    c.addEventListener('change', () => this.draw());
    return c;
  }

  /**
   * In perspective the orbit target limits how close the camera can get, and
   * once there every wheel tick shrinks an already tiny distance, which also
   * makes zooming back out slow. At the limit, move camera and target along
   * the cursor ray so the view keeps travelling into the scene.
   */
  private flyThrough = (event: WheelEvent) => {
    if (this.mode !== 'persp' || event.deltaY >= 0 || !this.controls.enabled) return;
    const distance = this.persp.position.distanceTo(this.controls.target);
    if (distance > this.controls.minDistance * 1.5) return;
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Vector3(ndc.x, ndc.y, 0.5).unproject(this.persp).sub(this.persp.position).normalize();
    const step = 0.4 + 0.2 * distance;
    this.persp.position.addScaledVector(ray, step);
    this.controls.target.addScaledVector(ray, step);
    this.controls.update();
    this.draw();
    event.preventDefault();
    event.stopPropagation();
  };

  setPointSize(n: number) {
    this.pointSize = n;
    for (const e of this.entries.values()) e.material.uniforms.pointSize.value = n * Math.min(devicePixelRatio, 2);
    this.draw();
  }

  /** Add or remove clouds so the scene matches `clouds`. Returns true when the set changed. */
  setClouds(clouds: Map<string, Cloud>): boolean {
    let changed = false;
    for (const [path, e] of [...this.entries]) {
      if (!clouds.has(path) || clouds.get(path) !== e.cloud) { this.remove(path); changed = true; }
    }
    for (const [path, cloud] of clouds) {
      if (this.entries.has(path)) continue;
      changed = true;
      if (!this.worldOrigin) this.worldOrigin = new THREE.Vector3(...(cloud.info.origin as [number, number, number]));
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(cloud.positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(cloud.colors, 3, true));
      const n = cloud.positions.length / 3;
      const visible = new THREE.BufferAttribute(new Float32Array(n).fill(1), 1);
      visible.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('visible', visible);
      const material = new THREE.ShaderMaterial({uniforms: {pointSize: {value: this.pointSize * Math.min(devicePixelRatio, 2)}, zRange: {value: new THREE.Vector2(this.zWindow[0], this.zWindow[1])}}, vertexShader: VERT, fragmentShader: FRAG});
      const points = new THREE.Points(geometry, material);
      points.frustumCulled = false;
      points.position.copy(new THREE.Vector3(...(cloud.info.origin as [number, number, number])).sub(this.worldOrigin));
      const localBounds = new THREE.Box3(new THREE.Vector3(...(cloud.info.bounds[0] as [number, number, number])), new THREE.Vector3(...(cloud.info.bounds[1] as [number, number, number])));
      this.scene.add(points);
      const entry: Entry = {cloud, points, visible, material, localBounds, bounds: new THREE.Box3(), visibleBounds: new THREE.Box3(), shown: true, mask: null, localVisible: null};
      this.entries.set(path, entry);
      this.setOrientation(path, IDENTITY);
    }
    if (changed) this.draw();
    return changed;
  }
  private remove(path: string) {
    const e = this.entries.get(path);
    if (!e) return;
    this.scene.remove(e.points);
    e.points.geometry.dispose();
    e.material.dispose();
    this.entries.delete(path);
  }

  /** Rotate about the cloud's own origin and shift it; bounds follow. */
  setOrientation(path: string, o: Orientation) {
    const e = this.entries.get(path);
    if (!e || !this.worldOrigin) return;
    e.points.quaternion.copy(quaternion(o));
    e.points.position.copy(new THREE.Vector3(...(e.cloud.info.origin as [number, number, number])).sub(this.worldOrigin).add(new THREE.Vector3(...o.translation)));
    e.points.updateMatrixWorld(true);
    e.bounds.copy(e.localBounds).applyMatrix4(e.points.matrixWorld);
    this.updateVisibleBounds(e);
    this.draw();
  }

  /**
   * null shows every point; an all-zero mask hides the cloud entirely.
   * `localBounds` is the box of the visible points in the cloud's own
   * coordinates, computed off-thread with the mask; without it the whole
   * cloud's box is used.
   */
  setMask(path: string, mask: Float32Array | null, localBounds?: [number, number, number, number, number, number] | null) {
    const e = this.entries.get(path);
    if (!e) return;
    e.mask = mask;
    if (!mask) { (e.visible.array as Float32Array).fill(1); e.localVisible = null; }
    else { (e.visible.array as Float32Array).set(mask); e.localVisible = localBounds === undefined ? e.localBounds.clone() : localBounds ? new THREE.Box3(new THREE.Vector3(localBounds[0], localBounds[1], localBounds[2]), new THREE.Vector3(localBounds[3], localBounds[4], localBounds[5])) : new THREE.Box3(); }
    this.updateVisibleBounds(e);
    e.points.visible = e.shown;
    e.visible.needsUpdate = true;
    this.draw();
  }
  private updateVisibleBounds(e: Entry) {
    if (!e.mask) { e.shown = true; e.visibleBounds.copy(e.bounds); return; }
    const local = e.localVisible;
    e.shown = !!local && !local.isEmpty();
    e.visibleBounds.copy(e.shown ? local!.clone().applyMatrix4(e.points.matrixWorld) : e.bounds);
    e.points.visible = e.shown;
  }

  /** Bounds of one cloud after orientation, in world coordinates. */
  worldBounds(path: string): Box | null {
    const e = this.entries.get(path);
    if (!e || !this.worldOrigin) return null;
    const lo = e.bounds.min.clone().add(this.worldOrigin), hi = e.bounds.max.clone().add(this.worldOrigin);
    return [[lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]];
  }

  /** Slice mode: left drag is for drawing, right drag still pans and the wheel zooms. */
  private sliceMode = false;
  setSliceMode(on: boolean) {
    this.sliceMode = on;
    this.applySliceMode();
  }
  private applySliceMode() {
    this.controls.mouseButtons.LEFT = this.sliceMode ? (-1 as unknown as THREE.MOUSE) : THREE.MOUSE.ROTATE;
  }

  /** Direction the camera looks along, unit vector in scene/world axes. */
  viewDirection(): THREE.Vector3 {
    const d = new THREE.Vector3();
    this.camera.getWorldDirection(d);
    return d;
  }
  /** Which world axes run horizontally and vertically across the screen in the current axis view. */
  screenAxes(): {h: 0 | 1 | 2; v: 0 | 1 | 2} {
    if (this.mode === 'top') return {h: 0, v: 1};
    if (this.mode === 'front') return {h: 0, v: 2};
    return {h: 1, v: 2};
  }
  /** CSS pixel position of a scene-space point. */
  project(point: THREE.Vector3): {x: number; y: number} {
    const {w, h} = this.size();
    const p = point.clone().project(this.camera);
    return {x: (p.x + 1) / 2 * w, y: (1 - p.y) / 2 * h};
  }

  /** Bounds of the currently shown points (union of shown cloud boxes), scene coordinates. */
  shownBounds(): THREE.Box3 | null {
    const box = new THREE.Box3();
    for (const e of this.entries.values()) if (e.shown) box.union(e.visibleBounds);
    return box.isEmpty() ? null : box;
  }

  setOutline(box: Box | null) {
    if (this.outline) { this.scene.remove(this.outline); this.outline.geometry.dispose(); (this.outline.material as THREE.Material).dispose(); this.outline = null; }
    if (box && this.worldOrigin) {
      const b = new THREE.Box3(new THREE.Vector3(...box[0]).sub(this.worldOrigin), new THREE.Vector3(...box[1]).sub(this.worldOrigin));
      const size = b.getSize(new THREE.Vector3()), center = b.getCenter(new THREE.Vector3());
      const geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(Math.max(size.x, 1e-6), Math.max(size.y, 1e-6), Math.max(size.z, 1e-6)));
      this.outline = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({color: 0xd8c27a, transparent: true, opacity: 0.9}));
      this.outline.position.copy(center);
      this.scene.add(this.outline);
    }
    this.draw();
  }

  private updateGrid() {
    if (this.grid) { this.scene.remove(this.grid); this.grid.geometry.dispose(); (this.grid.material as THREE.Material).dispose(); this.grid = null; }
    if (this.mode === 'persp' || !this.worldOrigin) return;
    const span = Math.ceil(this.sceneSpan() * 1.5 / 10) * 10;
    this.grid = new THREE.GridHelper(span, span, 0x3a4048, 0x2a2f35);
    this.grid.rotation.x = Math.PI / 2; // XY plane
    this.grid.position.z = -this.worldOrigin.z; // world z = 0
    (this.grid.material as THREE.Material).transparent = true;
    (this.grid.material as THREE.Material).opacity = 0.6;
    this.scene.add(this.grid);
  }

  /**
   * Outline polygons (world XY, a z range) drawn as a top and bottom loop with
   * corner posts; guides are one dashed loop at the lower height. Fat lines
   * with a dark backing so they read on any background, drawn over the points.
   */
  setPolygons(polys: {points: [number, number][]; z: [number, number]; strong: boolean; dashed?: boolean; fill?: {color: number; opacity: number; hole?: [number, number][]}}[]) {
    for (const line of this.polygons) { this.scene.remove(line); line.geometry.dispose(); }
    for (const m of this.lineMaterials) m.dispose();
    for (const mesh of this.fills) { this.scene.remove(mesh); mesh.geometry.dispose(); (mesh.material as THREE.Material).dispose(); }
    this.polygons = [];
    this.lineMaterials = [];
    this.fills = [];
    const o = this.worldOrigin;
    if (!o) { this.draw(); return; }
    // Translucent washes at ground level, under the lines.
    for (const poly of polys) {
      if (!poly.fill) continue;
      const shape = new THREE.Shape(poly.points.map(([x, y]) => new THREE.Vector2(x - o.x, y - o.y)));
      if (poly.fill.hole) shape.holes.push(new THREE.Path(poly.fill.hole.map(([x, y]) => new THREE.Vector2(x - o.x, y - o.y))));
      const geometry = new THREE.ShapeGeometry(shape);
      const material = new THREE.MeshBasicMaterial({color: poly.fill.color, transparent: true, opacity: poly.fill.opacity, depthTest: false, depthWrite: false, side: THREE.DoubleSide});
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.z = poly.z[0] - o.z;
      mesh.renderOrder = 5;
      this.scene.add(mesh);
      this.fills.push(mesh);
    }
    const {w, h} = this.size();
    const add = (pts: number[], options: {color: number; width: number; opacity: number; dashed?: boolean; dashSize?: number; gapSize?: number; dashOffset?: number; order: number}) => {
      const geometry = new LineSegmentsGeometry();
      geometry.setPositions(pts);
      const material = new LineMaterial({
        color: options.color, linewidth: options.width, transparent: true, opacity: options.opacity, depthTest: false, depthWrite: false,
        dashed: !!options.dashed, dashSize: options.dashSize ?? 1, gapSize: options.gapSize ?? 1, dashOffset: options.dashOffset ?? 0, worldUnits: false,
      });
      material.resolution.set(w, h);
      const line = new LineSegments2(geometry, material);
      line.computeLineDistances();
      line.renderOrder = options.order;
      this.scene.add(line);
      this.polygons.push(line);
      this.lineMaterials.push(material);
    };
    for (const poly of polys) {
      const pts: number[] = [];
      const n = poly.points.length;
      const levels = poly.dashed ? [poly.z[0]] : poly.z;
      for (let i = 0; i < n; i++) {
        const [ax, ay] = poly.points[i], [bx, by] = poly.points[(i + 1) % n];
        for (const z of levels) pts.push(ax - o.x, ay - o.y, z - o.z, bx - o.x, by - o.y, z - o.z);
        if (!poly.dashed) pts.push(ax - o.x, ay - o.y, poly.z[0] - o.z, ax - o.x, ay - o.y, poly.z[1] - o.z);
      }
      if (poly.dashed) {
        // Buildings and other guides: bright dashed over a dark backing.
        add(pts, {color: 0x10151a, width: 6, opacity: 0.8, order: 10});
        add(pts, {color: 0x7fd8ff, width: 3.5, opacity: 0.95, dashed: true, dashSize: 0.8, gapSize: 0.5, order: 11});
      } else if (poly.strong) {
        // The outline: dark backing, then amber and white dashes alternating.
        add(pts, {color: 0x10151a, width: 7, opacity: 0.85, order: 12});
        add(pts, {color: 0xffd35a, width: 4, opacity: 1, dashed: true, dashSize: 1, gapSize: 1, order: 13});
        add(pts, {color: 0xffffff, width: 4, opacity: 1, dashed: true, dashSize: 1, gapSize: 1, dashOffset: 1, order: 13});
      } else {
        // Perimeter band edge.
        add(pts, {color: 0x10151a, width: 5, opacity: 0.8, order: 10});
        add(pts, {color: 0x9fe6a0, width: 2.5, opacity: 0.95, order: 11});
      }
    }
    this.draw();
  }

  setView(mode: ViewMode, flipped = false) {
    const previousTarget = this.controls.target.clone();
    const previousDistance = this.controls.object.position.distanceTo(previousTarget);
    this.mode = mode;
    this.flipped = flipped;
    const wasEnabled = this.controls.enabled;
    this.controls.dispose();
    this.controls = this.makeControls(this.camera);
    this.controls.enabled = wasEnabled;
    this.applySliceMode();
    this.controls.target.copy(previousTarget);
    if (mode === 'persp') {
      this.controls.enableRotate = true;
      this.persp.position.copy(previousTarget).add(new THREE.Vector3(-0.6, -0.8, 0.6).normalize().multiplyScalar(previousDistance || 10));
    } else {
      this.controls.enableRotate = false;
      const dir = new THREE.Vector3();
      if (mode === 'top') { dir.set(0, 0, 1); this.ortho.up.set(0, 1, 0); }
      if (mode === 'front') { dir.set(0, -1, 0); this.ortho.up.set(0, 0, 1); }
      if (mode === 'side') { dir.set(1, 0, 0); this.ortho.up.set(0, 0, 1); }
      if (flipped) { dir.negate(); if (mode === 'top') this.ortho.up.set(0, 1, 0); }
      const span = this.sceneSpan();
      this.ortho.position.copy(previousTarget).add(dir.multiplyScalar(span * 4));
      this.ortho.lookAt(previousTarget);
    }
    this.controls.update();
    this.updateGrid();
    this.frameVisible(mode === 'persp' ? undefined : true);
  }

  private sceneSpan() {
    const b = this.shownBounds() ?? new THREE.Box3(new THREE.Vector3(-5, -5, -5), new THREE.Vector3(5, 5, 5));
    return Math.max(b.getSize(new THREE.Vector3()).length(), 1);
  }

  /** Fit the shown points into view. Keeps the current direction. */
  frameVisible(keepDirection?: boolean) {
    const b = this.shownBounds();
    if (!b) { this.draw(); return; }
    const center = b.getCenter(new THREE.Vector3()), size = b.getSize(new THREE.Vector3()), span = Math.max(size.length(), 1);
    if (this.mode === 'persp') {
      const dir = keepDirection ? this.persp.position.clone().sub(this.controls.target).normalize() : new THREE.Vector3(-0.6, -0.8, 0.6).normalize();
      this.persp.position.copy(center).add(dir.multiplyScalar(span * 0.9));
      this.persp.near = Math.max(0.001, span / 10000);
      this.persp.far = span * 20;
      this.persp.updateProjectionMatrix();
    } else {
      const dir = this.ortho.position.clone().sub(this.controls.target).normalize();
      this.ortho.position.copy(center).add(dir.multiplyScalar(span * 4));
      this.ortho.near = -span * 10;
      this.ortho.far = span * 10;
      const {w, h} = this.size();
      // Ortho frustum is in CSS pixels, so zoom = pixels per metre.
      const [horizontal, vertical] = this.mode === 'top' ? [size.x, size.y] : this.mode === 'front' ? [size.x, size.z] : [size.y, size.z];
      this.ortho.zoom = Math.min(w / Math.max(horizontal, 1e-3), h / Math.max(vertical, 1e-3)) / 1.1;
      this.ortho.updateProjectionMatrix();
    }
    this.controls.target.copy(center);
    this.controls.update();
    this.draw();
    this.onChange?.();
  }

  private size() {
    const r = this.host.getBoundingClientRect();
    return {w: Math.max(1, r.width), h: Math.max(1, r.height)};
  }
  private setAspect(w: number, h: number) {
    this.persp.aspect = w / h;
    this.persp.updateProjectionMatrix();
    this.ortho.left = -w / 2; this.ortho.right = w / 2; this.ortho.top = h / 2; this.ortho.bottom = -h / 2;
    this.ortho.updateProjectionMatrix();
  }
  private resize() {
    const {w, h} = this.size();
    this.renderer.setSize(w, h, false);
    this.setAspect(w, h);
    for (const m of this.lineMaterials) m.resolution.set(w, h);
    this.draw();
  }

  /** World metres per CSS pixel in the orthographic views. */
  metresPerPixel(): number | null {
    if (this.mode === 'persp') return null;
    return 1 / this.ortho.zoom;
  }

  /** Scene-space point under a CSS pixel on the orthographic view plane through the target. */
  unproject(px: number, py: number): THREE.Vector3 | null {
    if (this.mode === 'persp') return null;
    const {w, h} = this.size();
    const ndc = new THREE.Vector3((px / w) * 2 - 1, -(py / h) * 2 + 1, 0);
    const point = ndc.unproject(this.ortho);
    // Slide along the view direction onto the plane through the orbit target.
    const dir = new THREE.Vector3(); this.ortho.getWorldDirection(dir);
    const t = this.controls.target.clone().sub(point).dot(dir);
    return point.add(dir.multiplyScalar(t));
  }

  private raf = 0;
  draw() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => { this.raf = 0; this.render(); this.onChange?.(); });
  }
  private render() {
    const {w, h} = this.size();
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, w, h);
    this.renderer.clear();
    const compare = this.compare && this.entries.has(this.compare) ? this.compare : null;
    if (!compare) {
      this.renderer.render(this.scene, this.camera);
      return;
    }
    const half = Math.floor(w / 2);
    this.renderer.setScissorTest(true);
    const panes: [number, number, (path: string) => boolean][] = [[0, half - 1, p => p !== compare], [half + 1, w - half - 1, p => p === compare]];
    for (const [x, width, show] of panes) {
      for (const [path, e] of this.entries) e.points.visible = e.shown && show(path);
      this.renderer.setViewport(x, 0, width, h);
      this.renderer.setScissor(x, 0, width, h);
      this.setAspect(width, h);
      this.renderer.render(this.scene, this.camera);
    }
    this.setAspect(w, h);
    for (const e of this.entries.values()) e.points.visible = e.shown;
  }

  dispose() {
    this.canvas.removeEventListener('wheel', this.flyThrough, {capture: true} as EventListenerOptions);
    this.setPolygons([]);
    if (this.grid) { this.scene.remove(this.grid); this.grid.geometry.dispose(); (this.grid.material as THREE.Material).dispose(); }
    cancelAnimationFrame(this.raf);
    this.observer.disconnect();
    this.controls.dispose();
    for (const path of [...this.entries.keys()]) this.remove(path);
    this.setOutline(null);
    this.renderer.dispose();
  }
}
