// Three.js scene for point clouds: perspective or axis-locked orthographic
// cameras, per-point visibility masks, a slice outline and an optional
// side-by-side compare pane. Draws on demand only.
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Box, Cloud} from '../types';

export type ViewMode = 'persp' | 'top' | 'front' | 'side';
export const DEPTH_AXIS: Record<Exclude<ViewMode, 'persp'>, 0 | 1 | 2> = {top: 2, front: 1, side: 0};

const VERT = `
uniform float pointSize; attribute vec3 color; attribute float visible; varying vec3 vColor;
void main(){
  vColor = color;
  if (visible < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = pointSize;
}`;
const FRAG = `
varying vec3 vColor;
void main(){ vec2 d = gl_PointCoord - vec2(0.5); if (dot(d, d) > 0.25) discard; gl_FragColor = vec4(vColor, 1.0); }`;

type Entry = {cloud: Cloud; points: THREE.Points; visible: THREE.BufferAttribute; material: THREE.ShaderMaterial; bounds: THREE.Box3; visibleBounds: THREE.Box3; shown: boolean};

export class CloudRenderer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private persp = new THREE.PerspectiveCamera(48, 1, 0.01, 10000);
  private ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -10000, 10000);
  private controls: OrbitControls;
  private entries = new Map<string, Entry>();
  private outline: THREE.LineSegments | null = null;
  private observer: ResizeObserver;
  worldOrigin: THREE.Vector3 | null = null;
  mode: ViewMode = 'persp';
  flipped = false;
  compare: string | null = null;
  pointSize = 1.6;
  onChange: (() => void) | null = null;

  constructor(private canvas: HTMLCanvasElement, private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: false, powerPreference: 'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x1b1e21);
    this.renderer.autoClear = false;
    this.persp.up.set(0, 0, 1);
    this.ortho.up.set(0, 0, 1);
    this.controls = this.makeControls(this.persp);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();
  }

  private get camera(): THREE.Camera { return this.mode === 'persp' ? this.persp : this.ortho; }

  private makeControls(camera: THREE.Camera) {
    const c = new OrbitControls(camera, this.host);
    c.enableDamping = false;
    c.screenSpacePanning = true;
    c.zoomToCursor = true;
    c.addEventListener('change', () => this.draw());
    return c;
  }

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
      const buffer = new THREE.InterleavedBuffer(cloud.data, 6);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(buffer, 3, 0));
      geometry.setAttribute('color', new THREE.InterleavedBufferAttribute(buffer, 3, 3));
      const n = cloud.data.length / 6;
      const visible = new THREE.BufferAttribute(new Float32Array(n).fill(1), 1);
      visible.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('visible', visible);
      const material = new THREE.ShaderMaterial({uniforms: {pointSize: {value: this.pointSize * Math.min(devicePixelRatio, 2)}}, vertexShader: VERT, fragmentShader: FRAG});
      const points = new THREE.Points(geometry, material);
      points.frustumCulled = false;
      points.position.copy(new THREE.Vector3(...(cloud.info.origin as [number, number, number])).sub(this.worldOrigin));
      const bounds = new THREE.Box3(new THREE.Vector3(...(cloud.info.bounds[0] as [number, number, number])), new THREE.Vector3(...(cloud.info.bounds[1] as [number, number, number]))).translate(points.position);
      this.scene.add(points);
      this.entries.set(path, {cloud, points, visible, material, bounds, visibleBounds: bounds.clone(), shown: true});
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

  /** null shows every point; an all-zero mask hides the cloud entirely. */
  setMask(path: string, mask: Float32Array | null) {
    const e = this.entries.get(path);
    if (!e) return;
    if (!mask) { (e.visible.array as Float32Array).fill(1); e.shown = true; e.visibleBounds.copy(e.bounds); }
    else {
      (e.visible.array as Float32Array).set(mask);
      const d = e.cloud.data, box = new THREE.Box3();
      const v = new THREE.Vector3();
      for (let i = 0; i < mask.length; i++) if (mask[i]) box.expandByPoint(v.set(d[i * 6], d[i * 6 + 1], d[i * 6 + 2]));
      e.shown = !box.isEmpty();
      e.visibleBounds.copy(box.isEmpty() ? e.bounds : box.translate(e.points.position));
    }
    e.points.visible = e.shown;
    e.visible.needsUpdate = true;
    this.draw();
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

  setView(mode: ViewMode, flipped = false) {
    const previousTarget = this.controls.target.clone();
    const previousDistance = this.controls.object.position.distanceTo(previousTarget);
    this.mode = mode;
    this.flipped = flipped;
    this.controls.dispose();
    this.controls = this.makeControls(this.camera);
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
    cancelAnimationFrame(this.raf);
    this.observer.disconnect();
    this.controls.dispose();
    for (const path of [...this.entries.keys()]) this.remove(path);
    this.setOutline(null);
    this.renderer.dispose();
  }
}
