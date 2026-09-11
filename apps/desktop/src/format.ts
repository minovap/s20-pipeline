// Formatting helpers. Durations never show fractions of a second.

export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return '';
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
  if (h) return `${h}h ${String(m).padStart(2, '0')}m ${String(s).padStart(2, '0')}s`;
  if (m) return `${m}m ${String(s).padStart(2, '0')}s`;
  return `${s}s`;
}

/** Rough range for estimates: "3 to 12 min", "about 40 s". */
export function roughRange(range: [number, number] | null | undefined): string {
  if (!range) return '';
  const [lo, hi] = range;
  if (hi < 90) return `about ${Math.round(hi / 10) * 10 || 10} seconds`;
  const unit = (v: number) => (v >= 3600 ? `${(v / 3600).toFixed(1)} h` : `${Math.max(1, Math.round(v / 60))} min`);
  return `${unit(lo)} to ${unit(hi)}`;
}

export function bytes(value: number | null | undefined): string {
  if (value == null) return '';
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GB`;
  if (value >= 1e6) return `${Math.round(value / 1e6)} MB`;
  return `${Math.round(value / 1e3)} kB`;
}

export function count(value: number | null | undefined): string {
  if (value == null) return '';
  if (value >= 1e6) return `${(value / 1e6).toFixed(value >= 1e7 ? 0 : 1)} M`;
  if (value >= 1e4) return `${Math.round(value / 1e3)} k`;
  return value.toLocaleString();
}

const dateFormat = new Intl.DateTimeFormat(undefined, {month: 'short', day: 'numeric'});
const timeFormat = new Intl.DateTimeFormat(undefined, {hour: '2-digit', minute: '2-digit'});
const fullFormat = new Intl.DateTimeFormat(undefined, {year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'});

/** "Today 14:05", "Yesterday 09:12", "Sep 3 16:40", "Sep 3 2025 16:40". */
export function when(unixSeconds: number | null | undefined): string {
  if (!unixSeconds) return '';
  const d = new Date(unixSeconds * 1000), now = new Date();
  const day = (x: Date) => Math.floor((x.getTime() - x.getTimezoneOffset() * 60000) / 86400000);
  const diff = day(now) - day(d);
  if (diff === 0) return `Today ${timeFormat.format(d)}`;
  if (diff === 1) return `Yesterday ${timeFormat.format(d)}`;
  if (d.getFullYear() !== now.getFullYear()) return fullFormat.format(d);
  return `${dateFormat.format(d)} ${timeFormat.format(d)}`;
}

/** Folder-safe timestamp for run directories: 2026-09-11 18-04-33 */
export function runFolderName(date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())} ${p(date.getHours())}-${p(date.getMinutes())}-${p(date.getSeconds())}`;
}

export const basename = (path: string) => path.replace(/\/$/, '').split('/').pop() ?? path;
