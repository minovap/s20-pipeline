// Full-window point cloud viewer with axis-locked views, box slices and export.
import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {ChevronLeft, Columns2, Compass, Crop, FlipHorizontal2, Maximize, PanelRightClose, PanelRightOpen, Pentagon} from 'lucide-react';
import * as THREE from 'three';
import {IDENTITY, isIdentity, levelGround, rotateAboutAxis, transformFor} from './orient';
import {api, errorText, listen} from '../api';
import {basename, bytes, count, when} from '../format';
import {ConfirmDialog, Modal, NameDialog, Segmented, Spinner, useContextMenu} from '../ui';
import type {Box, Cloud, ExportEvent, Orientation, Project, Shape, Slice} from '../types';
import {PREVIEW_MAX, defaultBudget} from '../types';
import {CloudRenderer, DEPTH_AXIS, type ViewMode} from './render';
import {children, countMask, descendants, effectiveBox, effectiveRegion, intersect, newId, nextSliceName, normalize, unionMask} from './slices';
import {LYCKAN_8, area as polygonArea, centred, formatVertices, offsetPolygon, parseVertices, rectangle, shapeBox, shapeOf, worldPolygon} from './shape';
import type {XY} from './shape';

type Source = {path: string; name: string; detail: string; kind: 'result' | 'export' | 'import'};
type V3 = [number, number, number];
/** A slice being drawn: two opposite corners in world coordinates on the view plane. */
type Draft = {p1: V3; p2: V3};
type Handle = 'tl' | 'tr' | 'bl' | 'br' | 'l' | 'r' | 't' | 'b' | 'move';

export function Viewer({path, focus, onBack, onError}: {path: string; focus?: string; onBack: () => void; onError: (m: string) => void}) {
  const [project, setProject] = useState<Project | null>(null);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [clouds, setClouds] = useState<Map<string, Cloud>>(new Map());
  const [loading, setLoading] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>(() => (new URLSearchParams(location.search).get('view') as ViewMode | null) ?? 'persp');
  const [flipped, setFlipped] = useState(false);
  const [slicing, setSlicing] = useState(false);
  const [compare, setCompare] = useState<string | null>(null);
  const [pointSize, setPointSize] = useState(1.6);
  /** Share of each cloud to show, in percent; null means the size-based default. */
  const [percent, setPercent] = useState<number | null>(() => { const v = localStorage.getItem('previewPercent'); return v ? +v : null; });
  const [draftPercent, setDraftPercent] = useState<number | null>(null);
  const [totals, setTotals] = useState<Record<string, number>>({});
  const [panel, setPanel] = useState(true);
  const [scale, setScale] = useState<number | null>(null);
  /** Height window (world z) that hides points above and below; null means no limit on that side. */
  const [heightWindow, setHeightWindow] = useState<{lo: number | null; hi: number | null}>({lo: null, hi: null});
  const [zRange, setZRange] = useState<[number, number] | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  /** An outline being placed before it becomes a slice: dragged and turned over the whole cloud, then cropped. */
  type Placing = {name: string; vertices: XY[]; band: number | null; source: string; parent: Slice | null; z: [number, number]; position: [number, number]; rotation: number};
  const [placing, setPlacing] = useState<Placing | null>(null);
  const placeDrag = useRef<{startPosition: [number, number]; origin: V3} | null>(null);
  /** An outline being dragged: its slice and the position it has right now. */
  const [dragShape, setDragShape] = useState<{id: string; position: [number, number]} | null>(null);
  const shapeDrag = useRef<{id: string; startPosition: [number, number]; origin: V3} | null>(null);
  /** Calibration mode edits one cloud's orientation locally until Save. */
  const [calibrating, setCalibrating] = useState<{source: string; before: Orientation; current: Orientation; checkedBefore: Set<string>; modeBefore: ViewMode} | null>(null);
  const drag = useRef<{handle: Handle; start: Draft; origin: V3} | null>(null);
  const [viewTick, setViewTick] = useState(0);
  const [dialog, setDialog] = useState<React.ReactNode>(null);
  const [exporting, setExporting] = useState<{name: string; done: number; total: number} | null>(null);
  type ExportJob = {name: string; sources: {path: string; regions?: ExportRegion[]; boxes?: Box[]; transform?: ReturnType<typeof transformFor>}[]};
  const [queue, setQueue] = useState<ExportJob[]>([]);
  const [toast, setToast] = useState('');
  const {openMenu, menu} = useContextMenu();

  const host = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null);
  const renderer = useRef<CloudRenderer | null>(null);

  // ---- project and sources
  const reload = useCallback(async () => {
    try { setProject(await api.openProject(path)); } catch (e) { onError(errorText(e)); onBack(); }
  }, [path, onError, onBack]);
  useEffect(() => { void reload(); }, [reload]);

  const sources = useMemo<Source[]>(() => {
    if (!project) return [];
    const list: Source[] = [];
    for (const r of project.runs) if (r.result) list.push({path: r.result, name: `Run ${when(r.started)}`, detail: r.result_points ? `${count(r.result_points)} points` : basename(r.result), kind: 'result'});
    for (const x of project.exports) list.push({path: x.path, name: x.name, detail: `Export, ${bytes(x.bytes)}`, kind: 'export'});
    for (const c of project.clouds) list.push({path: c.path, name: c.name, detail: 'Imported', kind: 'import'});
    return list;
  }, [project]);
  const storedSlices = project?.slices ?? [];
  const slices = useMemo(() => (dragShape ? withShapeBoxes(storedSlices.map(s => (s.id === dragShape.id && s.shape ? {...s, shape: {...s.shape, position: dragShape.position}} : s))) : storedSlices), [storedSlices, dragShape]);
  const orientations = useMemo<Record<string, Orientation>>(() => {
    const stored = project?.orientations ?? {};
    return calibrating ? {...stored, [calibrating.source]: calibrating.current} : stored;
  }, [project, calibrating]);

  const initialised = useRef(false);
  useEffect(() => {
    if (!project || initialised.current || !sources.length) return;
    initialised.current = true;
    const first = (focus && sources.find(s => s.path === focus)?.path) ?? sources[0].path;
    const params = new URLSearchParams(location.search);
    const slice = params.get('slice') ? project.slices.find(x => x.name === params.get('slice')) : null;
    setChecked(new Set(params.get('check') === 'all' ? sources.map(x => x.path) : [slice ? slice.id : first]));
    setSelected(slice ? slice.id : first);
    if (params.get('calibrate')) setTimeout(() => startCalibration(first), 800);
    if (params.get('outline')) setTimeout(() => openOutlineRef.current(), 2500);
    if (params.get('zhi')) setTimeout(() => setHeightWindow({lo: params.get('zlo') ? +params.get('zlo')! : null, hi: +params.get('zhi')!}), 2500);
    if (params.get('place')) setTimeout(() => setPlacing({name: 'Lyckan 8', vertices: centred(LYCKAN_8), band: 2, source: first, parent: null, z: [-1, 5], position: [0, 0], rotation: 20}), 2500);
  }, [project, sources, focus]);

  // ---- renderer lifecycle
  useEffect(() => {
    if (!canvas.current || !host.current) return;
    let r: CloudRenderer;
    try { r = new CloudRenderer(canvas.current, host.current); }
    catch (e) { onError(`The point cloud view could not start: ${errorText(e)}`); return; }
    r.onChange = () => { setScale(r.metresPerPixel()); setViewTick(t => t + 1); };
    renderer.current = r;
    return () => { r.dispose(); renderer.current = null; };
  }, []);
  useEffect(() => { renderer.current?.setPointSize(pointSize); }, [pointSize]);
  useEffect(() => { renderer.current?.setView(mode, flipped); }, [mode, flipped]);
  useEffect(() => { if (renderer.current) { renderer.current.compare = compare; renderer.current.draw(); } }, [compare]);

  // ---- load previews for checked clouds
  const budgetFor = useCallback((total: number) => {
    if (percent == null) return defaultBudget(total);
    return Math.max(10_000, Math.min(PREVIEW_MAX, total, Math.round((total * percent) / 100)));
  }, [percent]);
  const loadCloud = useCallback(async (source: string) => {
    setLoading(l => new Set(l).add(source));
    try {
      const total = totals[source] ?? (await api.cloudInfo(source)).source_points;
      setTotals(t => (t[source] === total ? t : {...t, [source]: total}));
      const cloud = await api.loadCloud(source, budgetFor(total));
      setClouds(c => new Map(c).set(source, cloud));
    } catch (e) { onError(errorText(e)); setChecked(c => { const n = new Set(c); n.delete(source); return n; }); }
    finally { setLoading(l => { const n = new Set(l); n.delete(source); return n; }); }
  }, [budgetFor, totals, onError]);

  useEffect(() => {
    const needed = new Set<string>();
    for (const id of checked) {
      const slice = slices.find(s => s.id === id);
      needed.add(slice ? slice.source : id);
    }
    if (selected) { const slice = slices.find(s => s.id === selected); needed.add(slice ? slice.source : selected); }
    for (const source of needed) if (sources.some(s => s.path === source) && !clouds.has(source) && !loading.has(source)) void loadCloud(source);
  }, [checked, selected, slices, sources, clouds, loading, loadCloud]);
  const firstPercent = useRef(true);
  useEffect(() => {
    if (firstPercent.current) { firstPercent.current = false; return; }
    if (percent == null) localStorage.removeItem('previewPercent'); else localStorage.setItem('previewPercent', String(percent));
    setClouds(new Map());
  }, [percent]);

  const firstFrame = useRef(true);
  useEffect(() => {
    const r = renderer.current;
    if (!r) return;
    r.setClouds(clouds);
    for (const path of clouds.keys()) r.setOrientation(path, orientations[path] ?? IDENTITY);
  }, [clouds, orientations]);
  useEffect(() => { renderer.current?.setSliceMode(slicing && mode !== 'persp'); }, [slicing, mode]);
  useEffect(() => { if (!slicing) setDraft(null); }, [slicing]);

  // ---- visibility masks: a checked cloud shows everything; otherwise the union of its checked slices
  useEffect(() => {
    const r = renderer.current;
    if (!r) return;
    for (const [source, cloud] of clouds) {
      if (checked.has(source)) { r.setMask(source, null); continue; }
      const regions = slices.filter(s => s.source === source && checked.has(s.id)).map(s => effectiveRegion(s, slices)).filter((b): b is NonNullable<typeof b> => !!b);
      if (!regions.length) { r.setMask(source, new Float32Array(cloud.positions.length / 3)); continue; }
      r.setMask(source, unionMask(cloud.positions, cloud.info.origin, regions, transformFor(orientations[source], cloud.info.origin)));
    }
    // Frame the first thing that becomes visible, after its mask is in place.
    if (firstFrame.current && clouds.size) { firstFrame.current = false; r.frameVisible(); }
    setZRange(r.worldZRange());
  }, [checked, slices, clouds, orientations]);
  useEffect(() => { renderer.current?.setHeightWindow(heightWindow.lo, heightWindow.hi); }, [heightWindow, clouds]);

  const selectedSlice = slices.find(s => s.id === selected) ?? null;
  const selectedShape = selectedSlice ? shapeOf(selectedSlice, slices) : null;
  /** The outline slice a drag or rotation acts on: the selected outline, or the parent of a selected band. */
  const movableOutline = selectedSlice ? (selectedSlice.shape ? selectedSlice : selectedSlice.ring ? slices.find(s => s.id === selectedSlice.parent) ?? null : null) : null;
  const canMoveOutline = !!movableOutline?.shape && mode === 'top' && !slicing && !calibrating && !placing;

  useEffect(() => {
    const r = renderer.current;
    if (!r) return;
    if (placing) {
      const base = worldPolygon({vertices: placing.vertices, position: placing.position, rotation: placing.rotation});
      const polys = [{points: base, z: placing.z, strong: true}];
      if (placing.band != null) polys.push({points: offsetPolygon(base, placing.band), z: placing.z, strong: false});
      r.setOutline(null);
      r.setPolygons(polys);
      return;
    }
    const box = selectedSlice ? effectiveBox(selectedSlice, slices) : null;
    if (selectedSlice && selectedShape && box) {
      const z: [number, number] = [box[0][2], box[1][2]];
      const base = worldPolygon(selectedShape);
      const polys = [{points: base, z, strong: !selectedSlice.ring}];
      if (selectedSlice.ring) polys.push({points: offsetPolygon(base, selectedSlice.ring.expand), z, strong: true});
      r.setOutline(null);
      r.setPolygons(polys);
    } else {
      r.setPolygons([]);
      r.setOutline(box);
    }
  }, [selectedSlice, selectedShape, slices, placing]);

  const sliceCounts = useMemo(() => {
    const out = new Map<string, number>();
    for (const s of slices) {
      const cloud = clouds.get(s.source);
      const region = effectiveRegion(s, slices);
      if (cloud && region) out.set(s.id, countMask(unionMask(cloud.positions, cloud.info.origin, [region], transformFor(orientations[s.source], cloud.info.origin))));
    }
    return out;
  }, [slices, clouds, orientations]);

  // ---- persistence
  async function saveSlices(next: Slice[]) {
    if (!project) return;
    setProject(p => (p ? {...p, slices: next} : p));
    try { await api.writeProject(project.path, {slices: next}); } catch (e) { onError(errorText(e)); }
  }

  // ---- slicing
  const sliceTarget = useMemo<{source: string; parent: Slice | null} | null>(() => {
    if (calibrating) return {source: calibrating.source, parent: null};
    if (selectedSlice) return {source: selectedSlice.source, parent: selectedSlice};
    if (selected && sources.some(s => s.path === selected)) return {source: selected, parent: null};
    const firstChecked = [...checked].find(id => sources.some(s => s.path === id));
    return firstChecked ? {source: firstChecked, parent: null} : null;
  }, [selectedSlice, selected, sources, checked, calibrating]);
  const targetName = sliceTarget ? (sliceTarget.parent?.name ?? sources.find(s => s.path === sliceTarget.source)?.name ?? '') : '';

  function worldBoxOf(source: string): Box | null {
    return renderer.current?.worldBounds(source) ?? null;
  }
  async function saveOrientation(source: string, o: Orientation) {
    if (!project) return;
    const next = {...orientations};
    if (isIdentity(o)) delete next[source]; else next[source] = o;
    setProject(p => (p ? {...p, orientations: next} : p));
    try { await api.writeProject(project.path, {orientations: next}); } catch (e) { onError(errorText(e)); }
  }

  function startSlicing() {
    if (calibrating || placing) return;
    if (!sliceTarget) { onError('Check a point cloud first, then slice it.'); return; }
    if (mode === 'persp') setMode('top');
    setSlicing(true);
  }
  /** World point under the pointer on the slicing plane, or null outside axis views. */
  function pointerWorld(e: React.PointerEvent): V3 | null {
    const rend = renderer.current;
    if (!rend || !host.current || !rend.worldOrigin) return null;
    const r = host.current.getBoundingClientRect();
    const p = rend.unproject(e.clientX - r.left, e.clientY - r.top);
    return p ? [p.x + rend.worldOrigin.x, p.y + rend.worldOrigin.y, p.z + rend.worldOrigin.z] : null;
  }
  function toScreen(p: V3): {x: number; y: number} | null {
    const rend = renderer.current;
    if (!rend?.worldOrigin) return null;
    return rend.project(new THREE.Vector3(p[0] - rend.worldOrigin.x, p[1] - rend.worldOrigin.y, p[2] - rend.worldOrigin.z));
  }
  function onPointerDown(e: React.PointerEvent) {
    const target = e.target as HTMLElement;
    if (placing && mode === 'top' && e.button === 0 && target.tagName === 'CANVAS') {
      const ptr = pointerWorld(e);
      if (!ptr) return;
      placeDrag.current = {startPosition: placing.position, origin: ptr};
      try { host.current?.setPointerCapture(e.pointerId); } catch { /* optional */ }
      return;
    }
    if (canMoveOutline && e.button === 0 && target.tagName === 'CANVAS' && movableOutline?.shape) {
      const ptr = pointerWorld(e);
      if (!ptr) return;
      shapeDrag.current = {id: movableOutline.id, startPosition: movableOutline.shape.position, origin: ptr};
      try { host.current?.setPointerCapture(e.pointerId); } catch { /* optional */ }
      return;
    }
    if (!drawing || e.button !== 0) return;
    if (target.tagName !== 'CANVAS' && !target.dataset.handle) return;
    const ptr = pointerWorld(e);
    if (!ptr) return;
    const handle = ((e.target as HTMLElement).dataset.handle as Handle | undefined) ?? null;
    if (handle && draft) drag.current = {handle, start: draft, origin: ptr};
    else { const d = {p1: ptr, p2: ptr}; setDraft(d); drag.current = {handle: 'br', start: d, origin: ptr}; }
    try { host.current?.setPointerCapture(e.pointerId); } catch { /* capture is a convenience only */ }
  }
  function onPointerMove(e: React.PointerEvent) {
    const placingDrag = placeDrag.current;
    if (placingDrag) {
      const ptr = pointerWorld(e);
      if (ptr) setPlacing(pl => (pl ? {...pl, position: [placingDrag.startPosition[0] + ptr[0] - placingDrag.origin[0], placingDrag.startPosition[1] + ptr[1] - placingDrag.origin[1]]} : pl));
      return;
    }
    const moving = shapeDrag.current;
    if (moving) {
      const ptr = pointerWorld(e);
      if (ptr) setDragShape({id: moving.id, position: [moving.startPosition[0] + ptr[0] - moving.origin[0], moving.startPosition[1] + ptr[1] - moving.origin[1]]});
      return;
    }
    const d = drag.current;
    if (!d || !renderer.current) return;
    const ptr = pointerWorld(e);
    if (!ptr) return;
    const {h, v} = renderer.current.screenAxes();
    const p1: V3 = [...d.start.p1], p2: V3 = [...d.start.p2];
    const k = d.handle;
    if (k === 'move') { for (const a of [h, v]) { const delta = ptr[a] - d.origin[a]; p1[a] += delta; p2[a] += delta; } }
    else {
      if (k === 'tl' || k === 'l' || k === 'bl') p1[h] = ptr[h];
      if (k === 'tr' || k === 'r' || k === 'br') p2[h] = ptr[h];
      if (k === 'tl' || k === 't' || k === 'tr') p1[v] = ptr[v];
      if (k === 'bl' || k === 'b' || k === 'br') p2[v] = ptr[v];
    }
    setDraft({p1, p2});
  }
  function onPointerUp(e: React.PointerEvent) {
    if (placeDrag.current) {
      placeDrag.current = null;
      try { if (host.current?.hasPointerCapture(e.pointerId)) host.current.releasePointerCapture(e.pointerId); } catch { /* released */ }
      return;
    }
    if (shapeDrag.current) {
      const moving = shapeDrag.current;
      shapeDrag.current = null;
      try { if (host.current?.hasPointerCapture(e.pointerId)) host.current.releasePointerCapture(e.pointerId); } catch { /* released */ }
      const slice = storedSlices.find(s => s.id === moving.id);
      const position = dragShape?.position;
      setDragShape(null);
      if (slice?.shape && position) updateShape(slice, {...slice.shape, position});
      return;
    }
    if (!drag.current) return;
    drag.current = null;
    try { if (host.current?.hasPointerCapture(e.pointerId)) host.current.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    setDraft(d => {
      if (!d) return d;
      const a = toScreen(d.p1), b = toScreen(d.p2);
      if (!a || !b) return d;
      if (Math.abs(a.x - b.x) < 4 && Math.abs(a.y - b.y) < 4) return null;
      // Keep p1 as the screen top-left corner so handles stay where they look.
      const {h, v} = renderer.current!.screenAxes();
      const p1: V3 = [...d.p1], p2: V3 = [...d.p2];
      if (a.x > b.x) { [p1[h], p2[h]] = [p2[h], p1[h]]; }
      if (a.y > b.y) { [p1[v], p2[v]] = [p2[v], p1[v]]; }
      return {p1, p2};
    });
  }
  function commitDraft() {
    if (!draft || !sliceTarget || mode === 'persp') return;
    const parentBox = sliceTarget.parent ? effectiveBox(sliceTarget.parent, slices) : worldBoxOf(sliceTarget.source);
    if (!parentBox) { onError('The point cloud is still loading.'); return; }
    const axis = DEPTH_AXIS[mode];
    const lo: V3 = [...draft.p1], hi: V3 = [...draft.p2];
    lo[axis] = parentBox[0][axis]; hi[axis] = parentBox[1][axis];
    const box = intersect(normalize([lo, hi]), parentBox);
    if (!box) { onError(sliceTarget.parent ? 'The rectangle lies outside the parent slice.' : 'The rectangle lies outside the point cloud.'); return; }
    const parent = sliceTarget.parent;
    setDialog(<NameDialog title={parent ? `Slice ${parent.name}` : 'New slice'} defaultValue={nextSliceName(slices, parent)} confirm="Create" onCancel={() => setDialog(null)}
      onSubmit={name => {
        setDialog(null);
        const slice: Slice = {id: newId(), name, source: sliceTarget.source, parent: parent?.id ?? null, box, created: Date.now() / 1000};
        void saveSlices([...slices, slice]);
        setChecked(c => { const n = new Set(c); n.delete(parent ? parent.id : sliceTarget.source); n.add(slice.id); return n; });
        setSelected(slice.id);
        setSlicing(false);
      }} />);
  }
  /** Turn the target cloud about the axis the camera looks along; clockwise on screen is positive. */
  function rotateView(degrees: number) {
    if (placing) { setPlacing({...placing, rotation: Math.round((placing.rotation - degrees) * 100) / 100}); return; }
    if (movableOutline?.shape && !slicing && !calibrating) {
      // Clockwise on screen in the top view is a negative turn of the outline.
      updateShape(movableOutline, {...movableOutline.shape, rotation: Math.round((movableOutline.shape.rotation - degrees) * 100) / 100});
      return;
    }
    const rend = renderer.current;
    const source = sliceTarget?.source;
    if (!rend || !source) return;
    const d = rend.viewDirection();
    applyOrientation(source, rotateAboutAxis(orientations[source] ?? IDENTITY, [d.x, d.y, d.z], degrees));
  }
  /** While calibrating, changes stay local; otherwise they are saved at once. */
  function applyOrientation(source: string, o: Orientation) {
    if (calibrating && calibrating.source === source) setCalibrating(c => (c ? {...c, current: o} : c));
    else void saveOrientation(source, o);
  }
  function startCalibration(source: string) {
    setSlicing(false);
    setCalibrating({source, before: orientations[source] ?? IDENTITY, current: orientations[source] ?? IDENTITY, checkedBefore: checked, modeBefore: mode});
    setChecked(new Set([source]));
    setSelected(source);
    if (mode === 'persp') setMode('front');
  }
  function endCalibration(save: boolean) {
    if (!calibrating) return;
    if (save) void saveOrientation(calibrating.source, calibrating.current);
    setChecked(calibrating.checkedBefore);
    setCalibrating(null);
  }
  function saveCalibrationAs() {
    if (!calibrating || !project) return;
    const box = worldBoxOf(calibrating.source);
    const cloud = clouds.get(calibrating.source);
    if (!box || !cloud) { onError('The point cloud is still loading.'); return; }
    const source = calibrating.source, current = calibrating.current;
    setDialog(<NameDialog title="Save leveled copy" label="File name" defaultValue={`${labelFor(source)} leveled`} confirm="Save"
      note="Writes a new LAS file with this orientation applied. The original keeps its previous orientation." onCancel={() => setDialog(null)}
      onSubmit={name => { setDialog(null); enqueue([{name, sources: [{path: source, boxes: [box], transform: transformFor(current, cloud.info.origin)}]}]); endCalibration(false); }} />);
  }
  const rotationLabel = (() => {
    if (placing) return {name: placing.name, value: placing.rotation};
    if (movableOutline?.shape && !slicing && !calibrating) return {name: movableOutline.name, value: movableOutline.shape.rotation};
    const o = sliceTarget ? orientations[sliceTarget.source] ?? IDENTITY : IDENTITY;
    if (mode === 'top') return {name: 'Yaw', value: o.rotation[2]};
    if (mode === 'front') return {name: 'Pitch', value: o.rotation[1]};
    return {name: 'Roll', value: o.rotation[0]};
  })();

  /** Recompute the boxes of outline slices and their perimeter bands from their shapes. */
  function withShapeBoxes(list: Slice[]): Slice[] {
    return list.map(s => {
      if (s.shape) return {...s, box: shapeBox(s.shape, 'inside', 0, [s.box[0][2], s.box[1][2]])};
      if (s.ring) { const shape = shapeOf(s, list); return shape ? {...s, box: shapeBox(shape, 'ring', s.ring.expand, [s.box[0][2], s.box[1][2]])} : s; }
      return s;
    });
  }
  const openOutlineRef = useRef<() => void>(() => {});
  function openOutlineDialog() {
    if (!sliceTarget) { onError('Check a point cloud first.'); return; }
    const parentBox = sliceTarget.parent ? effectiveBox(sliceTarget.parent, slices) : worldBoxOf(sliceTarget.source);
    if (!parentBox) { onError('The point cloud is still loading.'); return; }
    const target = sliceTarget;
    setDialog(<OutlineDialog defaultName={nextSliceName(slices, target.parent)} onCancel={() => setDialog(null)}
      onSubmit={(name, vertices, band) => {
        setDialog(null);
        const centre: [number, number] = [(parentBox[0][0] + parentBox[1][0]) / 2, (parentBox[0][1] + parentBox[1][1]) / 2];
        const z: [number, number] = [parentBox[0][2], parentBox[1][2]];
        setSlicing(false);
        setPlacing({name, vertices: centred(vertices), band, source: target.source, parent: target.parent, z, position: centre, rotation: 0});
        if (mode === 'persp') setMode('top');
      }} />);
  }
  /** Turn the placed outline into slices at its current position. */
  function crop() {
    if (!placing) return;
    const shape: Shape = {vertices: placing.vertices, position: placing.position, rotation: placing.rotation};
    const outline: Slice = {id: newId(), name: placing.name, source: placing.source, parent: placing.parent?.id ?? null, box: shapeBox(shape, 'inside', 0, placing.z), created: Date.now() / 1000, shape};
    const created = [outline];
    if (placing.band != null) created.push({id: newId(), name: `${placing.name} perimeter`, source: placing.source, parent: outline.id, box: shapeBox(shape, 'ring', placing.band, placing.z), created: Date.now() / 1000, ring: {expand: placing.band}});
    void saveSlices([...slices, ...created]);
    const parentId = placing.parent ? placing.parent.id : placing.source;
    setChecked(c => { const n = new Set(c); n.delete(parentId); for (const x of created) n.add(x.id); return n; });
    setSelected(outline.id);
    setPlacing(null);
  }
  openOutlineRef.current = openOutlineDialog;
  function updateShape(slice: Slice, shape: Shape) {
    void saveSlices(withShapeBoxes(slices.map(s => (s.id === slice.id ? {...s, shape} : s))));
  }
  function updateRing(slice: Slice, expand: number) {
    void saveSlices(withShapeBoxes(slices.map(s => (s.id === slice.id ? {...s, ring: {expand}} : s))));
  }
  function addBand(slice: Slice) {
    if (!slice.shape) return;
    const z: [number, number] = [slice.box[0][2], slice.box[1][2]];
    const band: Slice = {id: newId(), name: `${slice.name} perimeter`, source: slice.source, parent: slice.id, box: shapeBox(slice.shape, 'ring', 2, z), created: Date.now() / 1000, ring: {expand: 2}};
    void saveSlices([...slices, band]);
    setChecked(c => new Set(c).add(band.id));
    setSelected(band.id);
  }

  function deleteSlice(slice: Slice) {
    const kids = descendants(slice.id, slices);
    const remove = () => {
      const gone = new Set([slice.id, ...kids.map(k => k.id)]);
      void saveSlices(slices.filter(s => !gone.has(s.id)));
      setChecked(c => new Set([...c].filter(id => !gone.has(id))));
      if (selected && gone.has(selected)) setSelected(slice.parent ?? slice.source);
      setDialog(null);
    };
    if (!kids.length) { remove(); return; }
    setDialog(<ConfirmDialog title="Delete slice" body={<>Delete <b>{slice.name}</b> and the {kids.length === 1 ? 'slice' : `${kids.length} slices`} cut from it?</>} confirm="Delete" danger onCancel={() => setDialog(null)} onConfirm={remove} />);
  }
  function renameSlice(slice: Slice) {
    setDialog(<NameDialog title="Rename slice" defaultValue={slice.name} onCancel={() => setDialog(null)} onSubmit={name => { setDialog(null); void saveSlices(slices.map(s => (s.id === slice.id ? {...s, name} : s))); }} />);
  }
  function updateBox(slice: Slice, box: Box) {
    void saveSlices(slices.map(s => (s.id === slice.id ? {...s, box: normalize(box)} : s)));
  }

  // ---- export
  const exportNames = new Set((project?.exports ?? []).map(x => x.name.toLowerCase()));
  type ExportRegion = {box: Box; polygons: {vertices: XY[]; mode: 'inside' | 'ring'; expand: number}[]};
  function groupsFor(ids: string[]): {path: string; regions: ExportRegion[]}[] | null {
    const groups = new Map<string, ExportRegion[]>();
    for (const id of ids) {
      const slice = slices.find(s => s.id === id);
      const source = slice ? slice.source : id;
      const region = slice ? effectiveRegion(slice, slices) : (() => { const b = worldBoxOf(id); return b ? {box: b, tests: []} : null; })();
      if (!region) { onError(`${slice?.name ?? basename(id)} is not loaded yet.`); return null; }
      const box: Box = [[region.box[0][0], region.box[0][1], Math.max(region.box[0][2], heightWindow.lo ?? -Infinity)], [region.box[1][0], region.box[1][1], Math.min(region.box[1][2], heightWindow.hi ?? Infinity)]];
      groups.set(source, [...(groups.get(source) ?? []), {box, polygons: region.tests.map(t => ({vertices: t.vertices, mode: t.mode, expand: t.expand}))}]);
    }
    return [...groups].map(([p, regions]) => ({path: p, regions, transform: transformFor(orientations[p], clouds.get(p)?.info.origin ?? [0, 0, 0])}));
  }
  const labelFor = (id: string) => slices.find(s => s.id === id)?.name ?? sources.find(s => s.path === id)?.name ?? basename(id);
  function startExport(ids: string[]) {
    if (!project || !ids.length) return;
    if (ids.length === 1) {
      const groups = groupsFor(ids); if (!groups) return;
      setDialog(<NameDialog title="Export" label="File name" defaultValue={labelFor(ids[0])} confirm="Export" note={<>Saved as a LAS file in the project's exports folder.</>} onCancel={() => setDialog(null)}
        onSubmit={name => { setDialog(null); enqueue([{name, sources: groups}]); }} />);
      return;
    }
    setDialog(<ExportDialog names={ids.map(labelFor)} defaultName={`${project.name} composite`} existing={exportNames} onCancel={() => setDialog(null)}
      onSubmit={(kind, name) => {
        setDialog(null);
        if (kind === 'composite') { const groups = groupsFor(ids); if (groups) enqueue([{name, sources: groups}]); }
        else {
          const jobs = ids.map((id): ExportJob | null => { const groups = groupsFor([id]); return groups ? {name: labelFor(id), sources: groups} : null; }).filter((j): j is ExportJob => !!j);
          enqueue(jobs);
        }
      }} />);
  }
  function enqueue(jobs: ExportJob[]) {
    const clash = jobs.find(j => exportNames.has(j.name.toLowerCase()));
    if (clash) { onError(`An export named ${clash.name} already exists. Choose another name.`); return; }
    setQueue(q => [...q, ...jobs]);
  }
  useEffect(() => {
    if (exporting || !queue.length || !project) return;
    const [job, ...rest] = queue;
    setQueue(rest);
    setExporting({name: job.name, done: 0, total: 0});
    api.exportSlices({output: `${project.path}/exports/${job.name.replace(/[/:]/g, '-')}.las`, sources: job.sources}).catch(e => { onError(errorText(e)); setExporting(null); });
  }, [queue, exporting, project]);
  useEffect(() => {
    let off: (() => void) | undefined;
    listen<ExportEvent>('export-event', p => {
      if (p.event === 'progress') setExporting(x => (x ? {...x, done: p.done ?? 0, total: p.total ?? 0} : x));
      else {
        setExporting(null);
        if (p.event === 'completed') { setToast(`Exported ${p.name}, ${count(p.points)} points`); void reload(); }
        else onError(p.event === 'cancelled' ? `Export of ${p.name} cancelled` : `Export of ${p.name} failed: ${p.message ?? ''}`);
      }
    }).then(f => (off = f));
    return () => off?.();
  }, [reload, onError]);
  useEffect(() => { if (!toast) return; const t = setTimeout(() => setToast(''), 4000); return () => clearTimeout(t); }, [toast]);

  // ---- keyboard
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || dialog) return;
      if (e.key === '7') setMode('top'); else if (e.key === '1') setMode('front'); else if (e.key === '3') setMode('side'); else if (e.key === '5') setMode('persp');
      else if (e.key === 'f') renderer.current?.frameVisible(true);
      else if (e.key === 's') (slicing ? setSlicing(false) : startSlicing());
      else if (e.key === 'Escape') { if (placing) setPlacing(null); else if (draft) setDraft(null); else if (slicing) setSlicing(false); else if (calibrating) endCalibration(false); }
      else if (e.key === 'Enter' && placing) crop();
      else if (e.key === 'Enter' && draft && slicing) commitDraft();
      else if ((e.key === 'Backspace' || e.key === 'Delete') && selectedSlice) deleteSlice(selectedSlice);
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  });

  // ---- checked helpers
  const toggle = (id: string) => setChecked(c => { const n = new Set(c); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const checkedExportable = [...checked].filter(id => slices.some(s => s.id === id) || sources.some(s => s.path === id));

  // ---- panel rows
  function sliceRows(source: string, parent: string | null, depth: number): React.ReactNode {
    return slices.filter(s => s.source === source && s.parent === parent).map(s => (
      <React.Fragment key={s.id}>
        <Row id={s.id} depth={depth} name={s.name} detail={sliceCounts.has(s.id) ? `${count(sliceCounts.get(s.id))} shown` : ''} checked={checked.has(s.id)} selected={selected === s.id} onToggle={() => toggle(s.id)} onSelect={() => setSelected(s.id)}
          onMenu={e => openMenu(e, [
            {label: 'Slice from here', onClick: () => { setSelected(s.id); startSlicing(); }},
            ...(s.shape && !children(s.id, slices).some(c => c.ring) ? [{label: 'Add perimeter band', onClick: () => addBand(s)}] : []),
            {label: 'Export…', onClick: () => startExport([s.id])},
            {label: 'Rename', onClick: () => renameSlice(s)},
            {separator: true, label: ''},
            {label: 'Delete', danger: true, onClick: () => deleteSlice(s)},
          ])} />
        {sliceRows(source, s.id, depth + 1)}
      </React.Fragment>
    ));
  }

  const drawing = slicing && mode !== 'persp';
  return (
    <div className="viewer" onContextMenu={e => e.preventDefault()}>
      <div className={'canvas-host' + (drawing ? ' drawing' : '')} ref={host} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
        <canvas ref={canvas} />
        {drawing && draft && <Rubber a={toScreen(draft.p1)} b={toScreen(draft.p2)} key={viewTick} />}
        {mode !== 'persp' && (placing || (sliceTarget && (slicing || calibrating || canMoveOutline))) && (
          <div className="rotate" onPointerDown={e => e.stopPropagation()}>
            <span className="angle"><b>{rotationLabel.name}</b> {rotationLabel.value.toFixed(1)}°</span>
            <span className="buttons">
              {[-2, -0.5, -0.1].map(d => <button key={d} onClick={() => rotateView(d)} title={`Counterclockwise ${-d}°`}>↺ {-d}</button>)}
              <i />
              {[0.1, 0.5, 2].map(d => <button key={d} onClick={() => rotateView(d)} title={`Clockwise ${d}°`}>{d} ↻</button>)}
            </span>
          </div>
        )}
        {calibrating && (
          <div className="slicebar calibrate" onPointerDown={e => e.stopPropagation()}>
            <span>Calibrating {labelFor(calibrating.source)}. Level the ground, then turn it until edges line up.</span>
            <button onClick={() => endCalibration(false)}>Cancel</button>
            <button onClick={saveCalibrationAs}>Save as…</button>
            <button className="primary" onClick={() => endCalibration(true)}>Save</button>
          </div>
        )}
        {placing && (
          <div className="slicebar" onPointerDown={e => e.stopPropagation()}>
            <span>{mode === 'top' ? `Drag ${placing.name} into place and turn it with the corner buttons, then press Crop.` : 'Switch to Top to place the outline.'}</span>
            <button onClick={() => setPlacing(null)}>Cancel</button>
            <button className="primary" onClick={crop}>Crop</button>
          </div>
        )}
        {canMoveOutline && !drawing && (
          <div className="slicebar outline-hint" onPointerDown={e => e.stopPropagation()}>
            <span>Drag {movableOutline!.name} to move it; turn it with the buttons in the corner.</span>
          </div>
        )}
        {drawing && (
          <div className="slicebar" onPointerDown={e => e.stopPropagation()}>
            <span>{draft ? `Adjust the corners of the slice in ${targetName}, then press Done.` : `Drag a rectangle over ${targetName}.`}</span>
            <button onClick={() => setSlicing(false)}>Cancel</button>
            <button className="primary" disabled={!draft} onClick={commitDraft}>Done</button>
          </div>
        )}
        {compare && <div className="compare-labels"><span>{sources.filter(s => checked.has(s.path) && s.path !== compare).map(s => s.name).join(', ') || 'Slices'}</span><span>{labelFor(compare)}</span></div>}
        {!clouds.size && (
          <div className="viewer-empty">{loading.size ? <><Spinner /> Loading point cloud</> : sources.length ? 'Check a point cloud on the right to show it.' : 'No point clouds in this project yet. Process a scan or import a LAS or PLY file.'}</div>
        )}
        {scale != null && <ScaleBar metresPerPixel={scale} />}
        {zRange && clouds.size > 0 && <HeightSlider range={zRange} value={heightWindow} onChange={setHeightWindow} />}
      </div>

      <div className="toolbar">
        <button className="icon" onClick={onBack} aria-label="Back to project" title="Back to project"><ChevronLeft size={18} /></button>
        <span className="title">{project?.name ?? ''}</span>
        <Segmented small value={mode} onChange={m => { setMode(m); if (m === 'persp') setSlicing(false); }} options={[{value: 'persp', label: 'Perspective', title: 'Free orbit (5)'}, {value: 'top', label: 'Top', title: 'Look down (7)'}, {value: 'front', label: 'Front', title: 'Look north (1)'}, {value: 'side', label: 'Side', title: 'Look west (3)'}]} />
        <button className="icon" disabled={mode === 'persp'} onClick={() => setFlipped(f => !f)} title="Look from the opposite side" aria-label="Flip view"><FlipHorizontal2 size={16} /></button>
        <button className="icon" onClick={() => renderer.current?.frameVisible(true)} title="Fit to view (f)" aria-label="Fit to view"><Maximize size={16} /></button>
        <span className="sep" />
        <button className={'tool' + (slicing ? ' on' : '')} disabled={!!calibrating} onClick={() => (slicing ? setSlicing(false) : startSlicing())} title="Draw a rectangle to cut a slice (s)"><Crop size={15} />Slice</button>
        <button className="tool" disabled={!!calibrating} onClick={openOutlineDialog} title="Place a fixed outline, such as the plot boundary, as a slice"><Pentagon size={15} />Outline</button>
        {slicing && <span className="hint">{mode === 'persp' ? 'Choose Top, Front or Side' : `Drag to cut ${targetName}`}</span>}
        <span className="sep" />
        <label className="size" title="Point size"><input type="range" min={0.6} max={4} step={0.1} value={pointSize} onChange={e => setPointSize(+e.target.value)} aria-label="Point size" /></label>
        <PointShare percent={draftPercent ?? percent} clouds={clouds} totals={totals} onDraft={setDraftPercent} onCommit={v => { setDraftPercent(null); setPercent(v); }} />
        {loading.size > 0 && <Spinner />}
      </div>
      {!panel && <button className="icon panel-open" onClick={() => setPanel(true)} aria-label="Show point clouds" title="Show point clouds"><PanelRightOpen size={18} /></button>}

      {panel && (
        <aside className="cloud-panel">
          <header><h2>{calibrating ? 'Calibrate orientation' : 'Point clouds'}</h2><button className="icon" onClick={() => setPanel(false)} aria-label="Hide panel"><PanelRightClose size={17} /></button></header>
          {calibrating ? (
            <div className="tree">
              <p className="pad muted">{labelFor(calibrating.source)}</p>
              <OrientationEditor orientation={calibrating.current} onChange={o => applyOrientation(calibrating.source, o)}
                onLevel={() => { const c = clouds.get(calibrating.source); const o = c ? levelGround(c.positions, calibrating.current) : null; if (o) applyOrientation(calibrating.source, o); else onError('No dominant ground plane found. Adjust roll and pitch by hand.'); }} />
              <p className="pad muted small">Use Front or Side to level and Top to turn. The buttons in the lower left rotate about the axis you look along. Save keeps the orientation with this cloud, Save as writes a new leveled LAS file.</p>
            </div>
          ) : (
          <div className="tree">
            {sources.map(s => (
              <React.Fragment key={s.path}>
                <Row id={s.path} depth={0} name={s.name} detail={loading.has(s.path) ? 'Loading' : clouds.has(s.path) ? `${count(clouds.get(s.path)!.info.display_points)} of ${count(clouds.get(s.path)!.info.source_points)} shown` : s.detail}
                  checked={checked.has(s.path)} selected={selected === s.path} compare={compare === s.path} calibrated={!isIdentity(orientations[s.path])} onToggle={() => toggle(s.path)} onSelect={() => setSelected(s.path)}
                  onMenu={e => openMenu(e, [
                    {label: 'Slice from here', onClick: () => { setSelected(s.path); setChecked(c => new Set(c).add(s.path)); startSlicing(); }},
                    {label: 'Calibrate orientation…', onClick: () => startCalibration(s.path)},
                    {label: 'Export…', onClick: () => startExport([s.path])},
                    {label: compare === s.path ? 'Stop comparing' : 'Compare side by side', onClick: () => setCompare(compare === s.path ? null : s.path)},
                    {label: 'Show in Finder', onClick: () => api.reveal(s.path)},
                    ...(s.kind === 'result' ? [] : [{separator: true, label: ''}, {label: s.kind === 'export' ? 'Delete export' : 'Remove from project', danger: true, onClick: () => removeSource(s)}]),
                  ])} />
                {sliceRows(s.path, null, 1)}
              </React.Fragment>
            ))}
            {!sources.length && <p className="muted pad">Nothing to show yet.</p>}
          </div>
          )}
          {placing && (
            <OutlineEditor slice={{id: 'placing', name: placing.name, source: placing.source, parent: null, box: [[0, 0, 0], [0, 0, 0]], created: 0}} shape={{vertices: placing.vertices, position: placing.position, rotation: placing.rotation}}
              onShape={sh => setPlacing({...placing, position: sh.position, rotation: sh.rotation})} onExpand={() => {}} />
          )}
          {!placing && selectedSlice && !calibrating && !selectedShape && <BoundsEditor slice={selectedSlice} onChange={box => updateBox(selectedSlice, box)} />}
          {!placing && selectedSlice && !calibrating && selectedShape && movableOutline?.shape && (
            <OutlineEditor slice={selectedSlice} shape={movableOutline.shape} onShape={shape => updateShape(movableOutline, shape)} onExpand={v => updateRing(selectedSlice, v)} />
          )}
          {!calibrating && <footer>
            {exporting ? (
              <div className="export-progress">
                <span><Spinner size={12} /> Exporting {exporting.name}{exporting.total ? `, ${Math.round((exporting.done / exporting.total) * 100)}%` : ''}</span>
                <button onClick={() => api.cancelExport()}>Cancel</button>
              </div>
            ) : (
              <button className="primary" disabled={!checkedExportable.length} onClick={() => startExport(checkedExportable)}>
                {checkedExportable.length > 1 ? `Export ${checkedExportable.length} checked` : 'Export checked'}
              </button>
            )}
            {queue.length > 0 && <span className="muted">{queue.length} more waiting</span>}
          </footer>}
        </aside>
      )}
      {toast && <div className="toast">{toast}</div>}
      {dialog}
      {menu}
    </div>
  );

  function removeSource(s: Source) {
    if (!project) return;
    setDialog(<ConfirmDialog title={s.kind === 'export' ? 'Delete export' : 'Remove point cloud'} confirm={s.kind === 'export' ? 'Delete' : 'Remove'} danger onCancel={() => setDialog(null)}
      body={s.kind === 'export' ? <>Delete the file <b>{s.name}.las</b> and its slices? This cannot be undone.</> : <>Remove <b>{s.name}</b> and its slices from this project? The file itself stays where it is.</>}
      onConfirm={async () => {
        setDialog(null);
        try {
          if (s.kind === 'export') await api.deleteExport(s.path);
          else await api.writeProject(project.path, {clouds: project.clouds.filter(c => c.path !== s.path)});
          await api.writeProject(project.path, {slices: slices.filter(x => x.source !== s.path)});
          setChecked(c => { const n = new Set(c); n.delete(s.path); return n; });
          setClouds(c => { const n = new Map(c); n.delete(s.path); return n; });
          await reload();
        } catch (e) { onError(errorText(e)); }
      }} />);
  }
}

function Rubber({a, b}: {a: {x: number; y: number} | null; b: {x: number; y: number} | null}) {
  if (!a || !b) return null;
  const left = Math.min(a.x, b.x), top = Math.min(a.y, b.y), width = Math.abs(a.x - b.x), height = Math.abs(a.y - b.y);
  const handles: Handle[] = ['tl', 't', 'tr', 'l', 'r', 'bl', 'b', 'br'];
  return (
    <div className="rubber" style={{left, top, width, height}} data-handle="move">
      {handles.map(h => <span key={h} className={`handle ${h}`} data-handle={h} />)}
    </div>
  );
}

function Row({depth, name, detail, checked, selected, compare, calibrated, onToggle, onSelect, onMenu}:
  {id: string; depth: number; name: string; detail: string; checked: boolean; selected: boolean; compare?: boolean; calibrated?: boolean; onToggle: () => void; onSelect: () => void; onMenu: (e: React.MouseEvent) => void}) {
  return (
    <div className={'cloud-row' + (selected ? ' selected' : '')} style={{paddingLeft: 10 + depth * 18}} onClick={onSelect} onContextMenu={onMenu}>
      <input type="checkbox" checked={checked} onChange={onToggle} onClick={e => e.stopPropagation()} aria-label={`Show ${name}`} />
      <span className="name">{name}{calibrated && <Compass size={12} className="compare-mark" aria-label="Orientation calibrated" />}{compare && <Columns2 size={12} className="compare-mark" />}</span>
      <span className="detail">{detail}</span>
    </div>
  );
}

type Preset = 'lyckan' | 'rectangle' | 'custom';
/** Choose an outline: the plot from the cadastral extract, a rectangle of a given area, or typed corners. */
function OutlineDialog({defaultName, onSubmit, onCancel}: {defaultName: string; onSubmit: (name: string, vertices: XY[], band: number | null) => void; onCancel: () => void}) {
  const [preset, setPreset] = useState<Preset>('lyckan');
  const [name, setName] = useState('Lyckan 8');
  const [areaText, setAreaText] = useState('2358');
  const [aspect, setAspect] = useState('1.5');
  const [text, setText] = useState(formatVertices(LYCKAN_8));
  const [withBand, setWithBand] = useState(true);
  const [band, setBand] = useState('2');
  const vertices = preset === 'lyckan' ? LYCKAN_8 : preset === 'rectangle' ? rectangle(Math.max(1, +areaText || 1), Math.max(0.2, +aspect || 1)) : parseVertices(text);
  const size = vertices ? polygonArea(vertices) : 0;
  const ok = !!vertices && name.trim().length > 0 && (!withBand || (+band > 0));
  const choose = (p: Preset) => { setPreset(p); if (p === 'lyckan') setName('Lyckan 8'); else if (name === 'Lyckan 8') setName(defaultName); if (p === 'custom' && preset !== 'custom') setText(formatVertices(vertices ?? LYCKAN_8)); };
  return (
    <Modal title="Outline slice" onClose={onCancel} width={520}>
      <form onSubmit={e => { e.preventDefault(); if (ok && vertices) onSubmit(name.trim(), vertices, withBand ? +band : null); }}>
        <div className="field"><span>Shape</span>
          <Segmented value={preset} onChange={choose} options={[{value: 'lyckan', label: 'Lyckan 8 plot'}, {value: 'rectangle', label: 'Rectangle'}, {value: 'custom', label: 'Corners'}]} />
          <small>{preset === 'lyckan' ? 'Five corners from the certified cadastral extract, 1:400. Registered area 2358 m².' : preset === 'rectangle' ? 'A rectangle of the given area and width-to-height ratio.' : 'One corner per line as x y in metres, in order around the shape.'}</small>
        </div>
        {preset === 'rectangle' && (
          <div className="row">
            <label className="field inline"><span>Area</span><span className="unit"><input type="number" min={1} value={areaText} onChange={e => setAreaText(e.target.value)} /> m²</span></label>
            <label className="field inline"><span>Width to height</span><input type="number" min={0.2} step={0.1} value={aspect} onChange={e => setAspect(e.target.value)} /></label>
          </div>
        )}
        <div className="outline-preview">
          <ShapePreview vertices={vertices ?? []} />
          <label className="field"><span>Corners, x y in metres</span>
            <textarea rows={7} value={preset === 'custom' ? text : formatVertices(vertices ?? [])} readOnly={preset !== 'custom'} onChange={e => setText(e.target.value)} spellCheck={false} />
          </label>
        </div>
        <p className="note">{vertices ? `${vertices.length} corners, ${Math.round(size).toLocaleString()} m². It is placed at the centre of the view; drag it into place and turn it with the corner buttons.` : 'Enter at least three corners.'}</p>
        <label className="field"><span>Name</span><input value={name} onChange={e => setName(e.target.value)} /></label>
        <label className="choice"><input type="checkbox" checked={withBand} onChange={e => setWithBand(e.target.checked)} />
          <span><strong>Add a perimeter band</strong><small>A second slice covering the ground outside the outline, this far out.</small></span></label>
        {withBand && <label className="field inline indent"><span>Band width</span><span className="unit"><input type="number" min={0.1} step={0.5} value={band} onChange={e => setBand(e.target.value)} /> m</span></label>}
        <footer><button type="button" onClick={onCancel}>Cancel</button><button type="submit" className="primary" disabled={!ok}>Create</button></footer>
      </form>
    </Modal>
  );
}

/** Small drawing of an outline, north up, with the corners marked. */
function ShapePreview({vertices}: {vertices: XY[]}) {
  const size = 170;
  if (vertices.length < 3) return <svg className="shape-preview" width={size} height={size} />;
  const xs = vertices.map(v => v[0]), ys = vertices.map(v => v[1]);
  const w = Math.max(...xs) - Math.min(...xs), h = Math.max(...ys) - Math.min(...ys);
  const k = (size - 24) / Math.max(w, h, 1e-6);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const pts = vertices.map(([x, y]) => [size / 2 + (x - cx) * k, size / 2 - (y - cy) * k] as XY);
  return (
    <svg className="shape-preview" width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-label="Outline preview">
      <polygon points={pts.map(p => p.join(',')).join(' ')} />
      {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={2.5} />)}
      <text x={size - 4} y={12} textAnchor="end">N ↑</text>
    </svg>
  );
}

/** Position, turn and band width of an outline slice. Arrow keys nudge. */
function OutlineEditor({slice, shape, onShape, onExpand}: {slice: Slice; shape: Shape; onShape: (s: Shape) => void; onExpand: (v: number) => void}) {
  const fields: {label: string; value: number; step: number; unit: string; set: (v: number) => void}[] = [
    {label: 'X', value: shape.position[0], step: 0.1, unit: 'm', set: v => onShape({...shape, position: [v, shape.position[1]]})},
    {label: 'Y', value: shape.position[1], step: 0.1, unit: 'm', set: v => onShape({...shape, position: [shape.position[0], v]})},
    {label: 'Turn', value: shape.rotation, step: 0.5, unit: '°', set: v => onShape({...shape, rotation: v})},
  ];
  if (slice.ring) fields.push({label: 'Band', value: slice.ring.expand, step: 0.5, unit: 'm', set: v => onExpand(Math.max(0.1, v))});
  const [draft, setDraft] = useState<Record<string, string>>({});
  useEffect(() => { setDraft({}); }, [shape, slice.ring?.expand]);
  return (
    <div className="bounds orientation">
      <h3>{slice.name}, {Math.round(polygonArea(shape.vertices)).toLocaleString()} m²</h3>
      {fields.map(f => (
        <div className="axis" key={f.label}><span>{f.label}</span>
          <input value={draft[f.label] ?? f.value.toFixed(2)} onChange={e => setDraft(d => ({...d, [f.label]: e.target.value}))}
            onBlur={() => { const v = Number(draft[f.label]); if (draft[f.label] != null && isFinite(v)) f.set(v); else setDraft(d => ({...d, [f.label]: undefined as unknown as string})); }}
            onKeyDown={e => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); if (e.key === 'ArrowUp' || e.key === 'ArrowDown') { e.preventDefault(); f.set(Math.round((f.value + (e.key === 'ArrowUp' ? f.step : -f.step)) * 100) / 100); } }}
            aria-label={f.label} />
          <span className="unit">{f.unit}</span>
        </div>
      ))}
    </div>
  );
}

function BoundsEditor({slice, onChange}: {slice: Slice; onChange: (box: Box) => void}) {
  const [draft, setDraft] = useState<string[]>(() => flat(slice.box));
  useEffect(() => { setDraft(flat(slice.box)); }, [slice.id, slice.box]);
  function flat(b: Box) { return [b[0][0], b[1][0], b[0][1], b[1][1], b[0][2], b[1][2]].map(v => v.toFixed(2)); }
  function commit() {
    const v = draft.map(Number);
    if (v.some(x => !isFinite(x))) { setDraft(flat(slice.box)); return; }
    onChange([[v[0], v[2], v[4]], [v[1], v[3], v[5]]]);
  }
  const axes = ['X', 'Y', 'Z'];
  return (
    <div className="bounds">
      <h3>{slice.name}</h3>
      {axes.map((axis, i) => (
        <div className="axis" key={axis}><span>{axis}</span>
          <input value={draft[i * 2]} onChange={e => setDraft(d => d.map((x, j) => (j === i * 2 ? e.target.value : x)))} onBlur={commit} onKeyDown={e => e.key === 'Enter' && commit()} aria-label={`${axis} minimum`} />
          <span className="to">to</span>
          <input value={draft[i * 2 + 1]} onChange={e => setDraft(d => d.map((x, j) => (j === i * 2 + 1 ? e.target.value : x)))} onBlur={commit} onKeyDown={e => e.key === 'Enter' && commit()} aria-label={`${axis} maximum`} />
          <span className="unit">m</span>
        </div>
      ))}
    </div>
  );
}

function OrientationEditor({orientation, onChange, onLevel}: {orientation: Orientation; onChange: (o: Orientation) => void; onLevel: () => void}) {
  const fields: {label: string; get: () => number; set: (v: number) => Orientation; step: number; unit: string}[] = [
    {label: 'Roll', get: () => orientation.rotation[0], set: v => ({...orientation, rotation: [v, orientation.rotation[1], orientation.rotation[2]]}), step: 0.1, unit: '°'},
    {label: 'Pitch', get: () => orientation.rotation[1], set: v => ({...orientation, rotation: [orientation.rotation[0], v, orientation.rotation[2]]}), step: 0.1, unit: '°'},
    {label: 'Yaw', get: () => orientation.rotation[2], set: v => ({...orientation, rotation: [orientation.rotation[0], orientation.rotation[1], v]}), step: 0.5, unit: '°'},
    {label: 'Height', get: () => orientation.translation[2], set: v => ({...orientation, translation: [orientation.translation[0], orientation.translation[1], v]}), step: 0.01, unit: 'm'},
  ];
  const [draft, setDraft] = useState<Record<string, string>>({});
  useEffect(() => { setDraft({}); }, [orientation]);
  return (
    <div className="bounds orientation">
      <div className="row"><button onClick={onLevel}>Level ground</button><button onClick={() => onChange(IDENTITY)} disabled={isIdentity(orientation)}>Reset</button></div>
      {fields.map(f => (
        <div className="axis" key={f.label}><span>{f.label}</span>
          <input value={draft[f.label] ?? String(f.get())} onChange={e => setDraft(d => ({...d, [f.label]: e.target.value}))}
            onBlur={() => { const v = Number(draft[f.label]); if (draft[f.label] != null && isFinite(v)) onChange(f.set(v)); else setDraft(d => ({...d, [f.label]: undefined as unknown as string})); }}
            onKeyDown={e => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); if (e.key === 'ArrowUp' || e.key === 'ArrowDown') { e.preventDefault(); onChange(f.set(Math.round((f.get() + (e.key === 'ArrowUp' ? f.step : -f.step)) * 1000) / 1000)); } }}
            aria-label={f.label} />
          <span className="unit">{f.unit}</span>
        </div>
      ))}
    </div>
  );
}

function ExportDialog({names, defaultName, existing, onSubmit, onCancel}:
  {names: string[]; defaultName: string; existing: Set<string>; onSubmit: (kind: 'composite' | 'separate', name: string) => void; onCancel: () => void}) {
  const [kind, setKind] = useState<'composite' | 'separate'>('composite');
  const [name, setName] = useState(defaultName);
  const clash = kind === 'composite' ? existing.has(name.trim().toLowerCase()) : names.some(n => existing.has(n.toLowerCase()));
  const ok = kind === 'separate' ? !clash : name.trim().length > 0 && !clash;
  return (
    <Modal title={`Export ${names.length} items`} onClose={onCancel} width={460}>
      <form onSubmit={e => { e.preventDefault(); if (ok) onSubmit(kind, name.trim()); }}>
        <label className="choice"><input type="radio" checked={kind === 'composite'} onChange={() => setKind('composite')} />
          <span><strong>One point cloud</strong><small>{names.join(', ')} combined. A point inside more than one slice is written once.</small></span></label>
        {kind === 'composite' && <label className="field indent"><span>File name</span><input value={name} onChange={e => setName(e.target.value)} autoFocus /></label>}
        <label className="choice"><input type="radio" checked={kind === 'separate'} onChange={() => setKind('separate')} />
          <span><strong>Separate files</strong><small>One LAS file per item, named after it.</small></span></label>
        {clash && <p className="warn-text">{kind === 'composite' ? 'An export with this name already exists.' : 'An export with one of these names already exists. Rename it first.'}</p>}
        <p className="note">Files are saved in the project's exports folder and appear in the list.</p>
        <footer><button type="button" onClick={onCancel}>Cancel</button><button type="submit" className="primary" disabled={!ok}>Export</button></footer>
      </form>
    </Modal>
  );
}

/** Slider for the share of points shown. The label reports the loaded clouds' shown and total counts. */
function PointShare({percent, clouds, totals, onDraft, onCommit}:
  {percent: number | null; clouds: Map<string, Cloud>; totals: Record<string, number>; onDraft: (v: number) => void; onCommit: (v: number | null) => void}) {
  const shown = [...clouds.values()].reduce((n, c) => n + c.info.display_points, 0);
  const total = [...clouds.keys()].reduce((n, k) => n + (totals[k] ?? clouds.get(k)!.info.source_points), 0);
  // The slider position follows the actual share when the size-based default is in use.
  const value = percent ?? (total ? Math.max(1, Math.round((100 * shown) / total)) : 10);
  const label = clouds.size ? `${count(shown)} of ${count(total)} points` : 'Points shown';
  return (
    <label className="share" title="Share of each point cloud to show. Higher is slower to load and draw.">
      <span>{label}</span>
      <input type="range" min={1} max={100} step={1} value={value} aria-label="Share of points shown"
        onChange={e => onDraft(+e.target.value)} onPointerUp={e => onCommit(+(e.target as HTMLInputElement).value)} onKeyUp={e => onCommit(+(e.target as HTMLInputElement).value)} />
      <b>{value}%</b>
    </label>
  );
}

/** Two handles on a vertical track: hide points above the top handle and below the bottom one. Double-click resets. */
function HeightSlider({range, value, onChange}: {range: [number, number]; value: {lo: number | null; hi: number | null}; onChange: (v: {lo: number | null; hi: number | null}) => void}) {
  const track = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<'lo' | 'hi' | null>(null);
  const [lo, hi] = [value.lo ?? range[0], value.hi ?? range[1]];
  const span = Math.max(range[1] - range[0], 1e-6);
  const pct = (z: number) => (100 * (Math.min(Math.max(z, range[0]), range[1]) - range[0])) / span;
  const zAt = (clientY: number) => {
    const r = track.current!.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (r.bottom - clientY) / r.height));
    return range[0] + t * span;
  };
  const move = (which: 'lo' | 'hi', z: number) => {
    const snapped = Math.round(z * 100) / 100;
    if (which === 'lo') onChange({lo: snapped <= range[0] + span * 0.002 ? null : Math.min(snapped, hi - 0.01), hi: value.hi});
    else onChange({lo: value.lo, hi: snapped >= range[1] - span * 0.002 ? null : Math.max(snapped, lo + 0.01)});
  };
  const label = (z: number) => `${z.toFixed(2)} m`;
  return (
    <div className="height-slider" onPointerDown={e => e.stopPropagation()} onDoubleClick={() => onChange({lo: null, hi: null})} title="Hide points above and below. Double-click to reset.">
      <span className="cap">{label(range[1])}</span>
      <div className="track" ref={track}
        onPointerMove={e => { if (drag) move(drag, zAt(e.clientY)); }}
        onPointerUp={e => { setDrag(null); (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId); }}
        onPointerCancel={() => setDrag(null)}>
        <div className="band" style={{bottom: `${pct(lo)}%`, top: `${100 - pct(hi)}%`}} />
        {(['hi', 'lo'] as const).map(which => (
          <button key={which} className={`thumb ${which}`} style={{bottom: `${pct(which === 'lo' ? lo : hi)}%`}} aria-label={which === 'lo' ? 'Hide below' : 'Hide above'}
            onPointerDown={e => { e.stopPropagation(); setDrag(which); (e.currentTarget.parentElement as HTMLElement).setPointerCapture?.(e.pointerId); }}>
            <i>{label(which === 'lo' ? lo : hi)}</i>
          </button>
        ))}
      </div>
      <span className="cap">{label(range[0])}</span>
    </div>
  );
}

function ScaleBar({metresPerPixel}: {metresPerPixel: number}) {
  const steps = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500];
  const length = steps.find(s => s / metresPerPixel >= 70) ?? steps[steps.length - 1];
  const px = length / metresPerPixel;
  return <div className="scalebar" style={{width: Math.min(px, 400)}}><span>{length >= 1 ? `${length} m` : `${Math.round(length * 100)} cm`}</span></div>;
}
