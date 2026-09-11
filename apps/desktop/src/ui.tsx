// Small presentational pieces shared by every screen.
import React, {useEffect, useRef, useState} from 'react';
import {X} from 'lucide-react';

export function Modal({title, children, onClose, width = 420}: {title: string; children: React.ReactNode; onClose: () => void; width?: number}) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [onClose]);
  return (
    <div className="scrim" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title} style={{width}}>
        <header><h2>{title}</h2><button className="icon" onClick={onClose} aria-label="Close"><X size={16} /></button></header>
        {children}
      </div>
    </div>
  );
}

/** Ask for a single name. Used for projects, slices and exports. */
export function NameDialog({title, label = 'Name', defaultValue, confirm = 'Save', note, onSubmit, onCancel}:
  {title: string; label?: string; defaultValue: string; confirm?: string; note?: React.ReactNode; onSubmit: (name: string) => void; onCancel: () => void}) {
  const [value, setValue] = useState(defaultValue);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { input.current?.focus(); input.current?.select(); }, []);
  const ok = value.trim().length > 0;
  return (
    <Modal title={title} onClose={onCancel}>
      <form onSubmit={e => { e.preventDefault(); if (ok) onSubmit(value.trim()); }}>
        <label className="field"><span>{label}</span><input ref={input} value={value} onChange={e => setValue(e.target.value)} /></label>
        {note && <p className="note">{note}</p>}
        <footer><button type="button" onClick={onCancel}>Cancel</button><button type="submit" className="primary" disabled={!ok}>{confirm}</button></footer>
      </form>
    </Modal>
  );
}

export function ConfirmDialog({title, body, confirm, danger, onConfirm, onCancel}:
  {title: string; body: React.ReactNode; confirm: string; danger?: boolean; onConfirm: () => void; onCancel: () => void}) {
  return (
    <Modal title={title} onClose={onCancel}>
      <p className="note">{body}</p>
      <footer><button onClick={onCancel}>Cancel</button><button className={danger ? 'danger' : 'primary'} onClick={onConfirm} autoFocus>{confirm}</button></footer>
    </Modal>
  );
}

export type MenuItem = {label: string; onClick?: () => void; danger?: boolean; disabled?: boolean; separator?: boolean};
export function ContextMenu({x, y, items, onClose}: {x: number; y: number; items: MenuItem[]; onClose: () => void}) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = () => onClose();
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('mousedown', close);
    window.addEventListener('keydown', key);
    window.addEventListener('blur', close);
    return () => { window.removeEventListener('mousedown', close); window.removeEventListener('keydown', key); window.removeEventListener('blur', close); };
  }, [onClose]);
  // Keep the menu inside the window.
  const [pos, setPos] = useState({x, y});
  useEffect(() => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return;
    setPos({x: Math.min(x, window.innerWidth - r.width - 8), y: Math.min(y, window.innerHeight - r.height - 8)});
  }, [x, y]);
  return (
    <div ref={box} className="menu" role="menu" style={{left: pos.x, top: pos.y}} onMouseDown={e => e.stopPropagation()}>
      {items.map((item, i) => item.separator
        ? <hr key={i} />
        : <button key={i} role="menuitem" className={item.danger ? 'danger' : ''} disabled={item.disabled} onClick={() => { item.onClick?.(); onClose(); }}>{item.label}</button>)}
    </div>
  );
}

export function useContextMenu() {
  const [menu, setMenu] = useState<{x: number; y: number; items: MenuItem[]} | null>(null);
  const openMenu = (e: React.MouseEvent, items: MenuItem[]) => { e.preventDefault(); e.stopPropagation(); setMenu({x: e.clientX, y: e.clientY, items}); };
  const element = menu ? <ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={() => setMenu(null)} /> : null;
  return {openMenu, menu: element};
}

export function Segmented<T extends string>({value, options, onChange, disabled, small}:
  {value: T; options: {value: T; label: string; title?: string}[]; onChange: (v: T) => void; disabled?: boolean; small?: boolean}) {
  return (
    <div className={'segmented' + (small ? ' small' : '')} role="radiogroup">
      {options.map(o => <button key={o.value} role="radio" aria-checked={o.value === value} title={o.title} className={o.value === value ? 'on' : ''} disabled={disabled} onClick={() => onChange(o.value)}>{o.label}</button>)}
    </div>
  );
}

export function Toggle({checked, onChange, label, hint, disabled}: {checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string; disabled?: boolean}) {
  return (
    <label className="toggle">
      <span>{label}{hint && <small>{hint}</small>}</span>
      <input type="checkbox" role="switch" checked={checked} disabled={disabled} onChange={e => onChange(e.target.checked)} />
      <i />
    </label>
  );
}

export function Spinner({size = 14}: {size?: number}) {
  return <span className="spinner" style={{width: size, height: size}} aria-hidden />;
}

/** Inline error strip with a dismiss button. */
export function ErrorBar({message, onClose}: {message: string; onClose: () => void}) {
  if (!message) return null;
  return <div className="error" role="alert"><span>{message}</span><button className="icon" onClick={onClose} aria-label="Dismiss"><X size={14} /></button></div>;
}
