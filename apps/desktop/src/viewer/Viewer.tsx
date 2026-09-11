// Full-window point cloud viewer with axis-locked views, box slices and export.
import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {ChevronLeft, Columns2, Crop, FlipHorizontal2, Maximize, PanelRightClose, PanelRightOpen} from 'lucide-react';
import {IDENTITY, isIdentity, levelGround, transformFor} from './orient';
import {api, errorText, listen} from '../api';
import {basename, bytes, count, when} from '../format';
import {ConfirmDialog, Modal, NameDialog, Segmented, Spinner, useContextMenu} from '../ui';
import type {Box, Cloud, ExportEvent, Orientation, Project, Slice} from '../types';
import {CloudRenderer, DEPTH_AXIS, type ViewMode} from './render';
import {children, countMask, descendants, effectiveBox, intersect, newId, nextSliceName, normalize, unionMask} from './slices';

type Source = {path: string; name: string; detail: string; kind: 'result' | 'export' | 'import'};
type Rect = {x0: number; y0: number; x1: number; y1: number};

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
  const [rect, setRect] = useState<Rect | null>(null);
  const [dialog, setDialog] = useState<React.ReactNode>(null);
  const [exporting, setExporting] = useState<{name: string; done: number; total: number} | null>(null);
  const [queue, setQueue] = useState<{name: string; sources: {path: string; boxes: Box[]}[]}[]>([]);
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
  const orientations = project?.orientations ?? {};

  const initialised = useRef(false);
  useEffect(() => {
    if (!project || initialised.current || !sources.length) return;
    initialised.current = true;
    const first = (focus && sources.find(s => s.path === focus)?.path) ?? sources[0].path;
    const params = new URLSearchParams(location.search);
    const slice = params.get('slice') ? project.slices.find(x => x.name === params.get('slice')) : null;
    setChecked(new Set([slice ? slice.id : first]));
    setSelected(slice ? slice.id : first);
  }, [project, sources, focus]);

  // ---- renderer lifecycle
  useEffect(() => {
    if (!canvas.current || !host.current) return;
    let r: CloudRenderer;
    try { r = new CloudRenderer(canvas.current, host.current); }
    catch (e) { onError(`The point cloud view could not start: ${errorText(e)}`); return; }
    r.onChange = () => setScale(r.metresPerPixel());
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
  useEffect(() => { renderer.current?.setInteractive(!(slicing && mode !== 'persp')); }, [slicing, mode]);

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
    if (selectedSlice) return {source: selectedSlice.source, parent: selectedSlice};
    if (selected && sources.some(s => s.path === selected)) return {source: selected, parent: null};
    const firstChecked = [...checked].find(id => sources.some(s => s.path === id));
    return firstChecked ? {source: firstChecked, parent: null} : null;
  }, [selectedSlice, selected, sources, checked]);
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
    if (!sliceTarget) { onError('Check a point cloud first, then slice it.'); return; }
    if (mode === 'persp') setMode('top');
    setSlicing(true);
  }
  function finishRect(r: Rect) {
    const rend = renderer.current;
    if (!rend || !sliceTarget || mode === 'persp' || !rend.worldOrigin) return;
    if (Math.abs(r.x1 - r.x0) < 4 || Math.abs(r.y1 - r.y0) < 4) return;
    const a = rend.unproject(r.x0, r.y0), b = rend.unproject(r.x1, r.y1);
    if (!a || !b) return;
    const o = rend.worldOrigin;
    const parentBox = sliceTarget.parent ? effectiveBox(sliceTarget.parent, slices) : worldBoxOf(sliceTarget.source);
    if (!parentBox) { onError('The point cloud is still loading.'); return; }
    const axis = DEPTH_AXIS[mode];
    const lo: [number, number, number] = [a.x + o.x, a.y + o.y, a.z + o.z], hi: [number, number, number] = [b.x + o.x, b.y + o.y, b.z + o.z];
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
      }} />);
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
          const jobs = ids.map(id => { const groups = groupsFor([id]); return groups ? {name: labelFor(id), sources: groups} : null; }).filter((j): j is {name: string; sources: {path: string; boxes: Box[]}[]} => !!j);
          enqueue(jobs);
        }
      }} />);
  }
  function enqueue(jobs: {name: string; sources: {path: string; boxes: Box[]}[]}[]) {
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
      else if (e.key === 'Escape') { if (rect) setRect(null); else if (slicing) setSlicing(false); }
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
      <div className={'canvas-host' + (drawing ? ' drawing' : '')} ref={host}
        onPointerDown={e => { if (!drawing || e.button !== 0) return; const r = host.current!.getBoundingClientRect(); setRect({x0: e.clientX - r.left, y0: e.clientY - r.top, x1: e.clientX - r.left, y1: e.clientY - r.top}); (e.target as HTMLElement).setPointerCapture?.(e.pointerId); }}
        onPointerMove={e => { if (!rect) return; const r = host.current!.getBoundingClientRect(); setRect({...rect, x1: e.clientX - r.left, y1: e.clientY - r.top}); }}
        onPointerUp={() => { if (!rect) return; const r = rect; setRect(null); finishRect(r); }}>
        <canvas ref={canvas} />
        {rect && <div className="rubber" style={{left: Math.min(rect.x0, rect.x1), top: Math.min(rect.y0, rect.y1), width: Math.abs(rect.x1 - rect.x0), height: Math.abs(rect.y1 - rect.y0)}} />}
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
        <button className={'tool' + (slicing ? ' on' : '')} onClick={() => (slicing ? setSlicing(false) : startSlicing())} title="Draw a rectangle to cut a slice (s)"><Crop size={15} />Slice</button>
        {slicing && <span className="hint">{mode === 'persp' ? 'Choose Top, Front or Side' : `Drag to cut ${targetName}`}</span>}
        <span className="sep" />
        <label className="size" title="Point size"><input type="range" min={0.6} max={4} step={0.1} value={pointSize} onChange={e => setPointSize(+e.target.value)} aria-label="Point size" /></label>
        <select value={budget} onChange={e => setBudget(+e.target.value)} aria-label="Points shown" title="How many points to show">
          <option value={250000}>250 k points</option><option value={1000000}>1 M points</option><option value={2000000}>2 M points</option>
        </select>
        {loading.size > 0 && <Spinner />}
      </div>
      {!panel && <button className="icon panel-open" onClick={() => setPanel(true)} aria-label="Show point clouds" title="Show point clouds"><PanelRightOpen size={18} /></button>}

      {panel && (
        <aside className="cloud-panel">
          <header><h2>Point clouds</h2><button className="icon" onClick={() => setPanel(false)} aria-label="Hide panel"><PanelRightClose size={17} /></button></header>
          <div className="tree">
            {sources.map(s => (
              <React.Fragment key={s.path}>
                <Row id={s.path} depth={0} name={s.name} detail={loading.has(s.path) ? 'Loading' : clouds.has(s.path) ? `${count(clouds.get(s.path)!.info.display_points)} of ${count(clouds.get(s.path)!.info.source_points)} shown` : s.detail}
                  checked={checked.has(s.path)} selected={selected === s.path} compare={compare === s.path} onToggle={() => toggle(s.path)} onSelect={() => setSelected(s.path)}
                  onMenu={e => openMenu(e, [
                    {label: 'Slice from here', onClick: () => { setSelected(s.path); setChecked(c => new Set(c).add(s.path)); startSlicing(); }},
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
          {selectedSlice && <BoundsEditor slice={selectedSlice} onChange={box => updateBox(selectedSlice, box)} />}
          {selected && !selectedSlice && clouds.has(selected) && (
            <OrientationEditor orientation={orientations[selected] ?? IDENTITY} onChange={o => saveOrientation(selected, o)}
              onLevel={() => { const o = levelGround(clouds.get(selected)!.data, orientations[selected] ?? IDENTITY); if (o) void saveOrientation(selected, o); else onError('No dominant ground plane found. Adjust roll and pitch by hand.'); }} />
          )}
          <footer>
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
          </footer>
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

function Row({depth, name, detail, checked, selected, compare, onToggle, onSelect, onMenu}:
  {id: string; depth: number; name: string; detail: string; checked: boolean; selected: boolean; compare?: boolean; onToggle: () => void; onSelect: () => void; onMenu: (e: React.MouseEvent) => void}) {
  return (
    <div className={'cloud-row' + (selected ? ' selected' : '')} style={{paddingLeft: 10 + depth * 18}} onClick={onSelect} onContextMenu={onMenu}>
      <input type="checkbox" checked={checked} onChange={onToggle} onClick={e => e.stopPropagation()} aria-label={`Show ${name}`} />
      <span className="name">{name}{compare && <Columns2 size={12} className="compare-mark" />}</span>
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
      <h3>Orientation</h3>
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
      <small className="muted">Level ground fits the largest flat area and sets it to height 0. Arrow keys nudge a value.</small>
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
