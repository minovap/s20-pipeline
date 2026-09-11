// Full-window point cloud viewer with axis-locked views, box slices and export.
import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {ChevronLeft, Columns2, Compass, Crop, FlipHorizontal2, Maximize, PanelRightClose, PanelRightOpen} from 'lucide-react';
import * as THREE from 'three';
import {IDENTITY, isIdentity, levelGround, rotateAboutAxis, transformFor} from './orient';
import {api, errorText, listen} from '../api';
import {basename, bytes, count, when} from '../format';
import {ConfirmDialog, Modal, NameDialog, Segmented, Spinner, useContextMenu} from '../ui';
import type {Box, Cloud, ExportEvent, Orientation, Project, Slice} from '../types';
import {CloudRenderer, DEPTH_AXIS, type ViewMode} from './render';
import {children, countMask, descendants, effectiveBox, intersect, newId, nextSliceName, normalize, unionMask} from './slices';

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
  const [budget, setBudget] = useState(() => +(localStorage.getItem('budget') ?? 1000000));
  const [panel, setPanel] = useState(true);
  const [scale, setScale] = useState<number | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  /** Calibration mode edits one cloud's orientation locally until Save. */
  const [calibrating, setCalibrating] = useState<{source: string; before: Orientation; current: Orientation; checkedBefore: Set<string>; modeBefore: ViewMode} | null>(null);
  const drag = useRef<{handle: Handle; start: Draft; origin: V3} | null>(null);
  const [viewTick, setViewTick] = useState(0);
  const [dialog, setDialog] = useState<React.ReactNode>(null);
  const [exporting, setExporting] = useState<{name: string; done: number; total: number} | null>(null);
  type ExportJob = {name: string; sources: {path: string; boxes: Box[]; transform?: ReturnType<typeof transformFor>}[]};
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
  const slices = project?.slices ?? [];
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
  const loadCloud = useCallback(async (source: string) => {
    setLoading(l => new Set(l).add(source));
    try {
      const info = await api.loadPreview(source, budget);
      const raw = await api.readPreview(info.key);
      if (raw.byteLength !== info.bytes) throw new Error('Preview data was incomplete. Try again.');
      setClouds(c => new Map(c).set(source, {info, data: new Float32Array(raw)}));
    } catch (e) { onError(errorText(e)); setChecked(c => { const n = new Set(c); n.delete(source); return n; }); }
    finally { setLoading(l => { const n = new Set(l); n.delete(source); return n; }); }
  }, [budget, onError]);

  useEffect(() => {
    const needed = new Set<string>();
    for (const id of checked) {
      const slice = slices.find(s => s.id === id);
      needed.add(slice ? slice.source : id);
    }
    if (selected) { const slice = slices.find(s => s.id === selected); needed.add(slice ? slice.source : selected); }
    for (const source of needed) if (sources.some(s => s.path === source) && !clouds.has(source) && !loading.has(source)) void loadCloud(source);
  }, [checked, selected, slices, sources, clouds, loading, loadCloud]);
  useEffect(() => { localStorage.setItem('budget', String(budget)); setClouds(new Map()); }, [budget]);

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
      const boxes = slices.filter(s => s.source === source && checked.has(s.id)).map(s => effectiveBox(s, slices)).filter((b): b is Box => !!b);
      if (!boxes.length) { r.setMask(source, new Float32Array(cloud.data.length / 6)); continue; }
      r.setMask(source, unionMask(cloud.data, cloud.info.origin, boxes, transformFor(orientations[source], cloud.info.origin)));
    }
    // Frame the first thing that becomes visible, after its mask is in place.
    if (firstFrame.current && clouds.size) { firstFrame.current = false; r.frameVisible(); }
  }, [checked, slices, clouds, orientations]);

  const selectedSlice = slices.find(s => s.id === selected) ?? null;
  useEffect(() => { renderer.current?.setOutline(selectedSlice ? effectiveBox(selectedSlice, slices) : null); }, [selectedSlice, slices]);

  const sliceCounts = useMemo(() => {
    const out = new Map<string, number>();
    for (const s of slices) {
      const cloud = clouds.get(s.source);
      const box = effectiveBox(s, slices);
      if (cloud && box) out.set(s.id, countMask(unionMask(cloud.data, cloud.info.origin, [box], transformFor(orientations[s.source], cloud.info.origin))));
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
    if (calibrating) return;
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
    if (!drawing || e.button !== 0) return;
    const target = e.target as HTMLElement;
    if (target.tagName !== 'CANVAS' && !target.dataset.handle) return;
    const ptr = pointerWorld(e);
    if (!ptr) return;
    const handle = ((e.target as HTMLElement).dataset.handle as Handle | undefined) ?? null;
    if (handle && draft) drag.current = {handle, start: draft, origin: ptr};
    else { const d = {p1: ptr, p2: ptr}; setDraft(d); drag.current = {handle: 'br', start: d, origin: ptr}; }
    try { host.current?.setPointerCapture(e.pointerId); } catch { /* capture is a convenience only */ }
  }
  function onPointerMove(e: React.PointerEvent) {
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
    const o = sliceTarget ? orientations[sliceTarget.source] ?? IDENTITY : IDENTITY;
    if (mode === 'top') return {name: 'Yaw', value: o.rotation[2]};
    if (mode === 'front') return {name: 'Pitch', value: o.rotation[1]};
    return {name: 'Roll', value: o.rotation[0]};
  })();

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
  function groupsFor(ids: string[]): {path: string; boxes: Box[]}[] | null {
    const groups = new Map<string, Box[]>();
    for (const id of ids) {
      const slice = slices.find(s => s.id === id);
      const source = slice ? slice.source : id;
      const box = slice ? effectiveBox(slice, slices) : worldBoxOf(id);
      if (!box) { onError(`${slice?.name ?? basename(id)} is not loaded yet.`); return null; }
      groups.set(source, [...(groups.get(source) ?? []), box]);
    }
    return [...groups].map(([p, boxes]) => ({path: p, boxes, transform: transformFor(orientations[p], clouds.get(p)?.info.origin ?? [0, 0, 0])}));
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
          const jobs = ids.map(id => { const groups = groupsFor([id]); return groups ? {name: labelFor(id), sources: groups} : null; }).filter((j): j is ExportJob => !!j);
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
      else if (e.key === 'Escape') { if (draft) setDraft(null); else if (slicing) setSlicing(false); else if (calibrating) endCalibration(false); }
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
        {mode !== 'persp' && sliceTarget && (slicing || calibrating) && (
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
      </div>

      <div className="toolbar">
        <button className="icon" onClick={onBack} aria-label="Back to project" title="Back to project"><ChevronLeft size={18} /></button>
        <span className="title">{project?.name ?? ''}</span>
        <Segmented small value={mode} onChange={m => { setMode(m); if (m === 'persp') setSlicing(false); }} options={[{value: 'persp', label: 'Perspective', title: 'Free orbit (5)'}, {value: 'top', label: 'Top', title: 'Look down (7)'}, {value: 'front', label: 'Front', title: 'Look north (1)'}, {value: 'side', label: 'Side', title: 'Look west (3)'}]} />
        <button className="icon" disabled={mode === 'persp'} onClick={() => setFlipped(f => !f)} title="Look from the opposite side" aria-label="Flip view"><FlipHorizontal2 size={16} /></button>
        <button className="icon" onClick={() => renderer.current?.frameVisible(true)} title="Fit to view (f)" aria-label="Fit to view"><Maximize size={16} /></button>
        <span className="sep" />
        <button className={'tool' + (slicing ? ' on' : '')} disabled={!!calibrating} onClick={() => (slicing ? setSlicing(false) : startSlicing())} title="Draw a rectangle to cut a slice (s)"><Crop size={15} />Slice</button>
        {slicing && <span className="hint">{mode === 'persp' ? 'Choose Top, Front or Side' : `Drag to cut ${targetName}`}</span>}
        <span className="sep" />
        <label className="size" title="Point size"><input type="range" min={0.6} max={4} step={0.1} value={pointSize} onChange={e => setPointSize(+e.target.value)} aria-label="Point size" /></label>
        <select value={budget} onChange={e => setBudget(+e.target.value)} aria-label="Points shown" title="How many points to show">
          <option value={250000}>250 k points</option><option value={1000000}>1 M points</option><option value={2000000}>2 M points</option><option value={4000000}>4 M points</option><option value={8000000}>8 M points</option>
        </select>
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
                onLevel={() => { const c = clouds.get(calibrating.source); const o = c ? levelGround(c.data, calibrating.current) : null; if (o) applyOrientation(calibrating.source, o); else onError('No dominant ground plane found. Adjust roll and pitch by hand.'); }} />
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
          {selectedSlice && !calibrating && <BoundsEditor slice={selectedSlice} onChange={box => updateBox(selectedSlice, box)} />}
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

function ScaleBar({metresPerPixel}: {metresPerPixel: number}) {
  const steps = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500];
  const length = steps.find(s => s / metresPerPixel >= 70) ?? steps[steps.length - 1];
  const px = length / metresPerPixel;
  return <div className="scalebar" style={{width: Math.min(px, 400)}}><span>{length >= 1 ? `${length} m` : `${Math.round(length * 100)} cm`}</span></div>;
}
