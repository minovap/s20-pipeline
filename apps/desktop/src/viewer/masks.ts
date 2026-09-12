// Main-thread side of the mask worker: promise-based requests, latest wins.
import type {Box} from '../types';
import type {Region} from './slices';
import type {Transform} from './orient';

export type MaskResult = {mask: Float32Array; count: number; bounds: [number, number, number, number, number, number] | null};

export class MaskService {
  private worker: Worker | null = null;
  private next = 1;
  private pending = new Map<number, {resolve: (v: unknown) => void; reject: (e: Error) => void}>();

  private get w(): Worker {
    if (!this.worker) {
      this.worker = new Worker(new URL('./mask.worker.ts', import.meta.url), {type: 'module'});
      this.worker.onmessage = (e: MessageEvent<{type: string; id?: number} & Record<string, unknown>>) => {
        const {id} = e.data;
        if (id == null) return;
        const p = this.pending.get(id);
        if (!p) return;
        this.pending.delete(id);
        if (e.data.type === 'missing') p.reject(new Error('Cloud is not loaded in the mask worker'));
        else p.resolve(e.data);
      };
      this.worker.onerror = e => { for (const p of this.pending.values()) p.reject(new Error(e.message)); this.pending.clear(); };
    }
    return this.worker;
  }
  load(source: string, positions: Float32Array) {
    // The worker gets its own copy; the renderer keeps the original for the GPU.
    this.w.postMessage({type: 'load', source, positions: positions.slice()});
  }
  unload(source: string) { this.w.postMessage({type: 'unload', source}); }
  private request<T>(message: Record<string, unknown>): Promise<T> {
    const id = this.next++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {resolve: resolve as (v: unknown) => void, reject});
      this.w.postMessage({...message, id});
    });
  }
  mask(source: string, origin: number[], regions: (Box | Region)[], transform: Transform | null): Promise<MaskResult> {
    return this.request<MaskResult>({type: 'mask', source, origin, regions, transform});
  }
  count(source: string, origin: number[], region: Box | Region, transform: Transform | null): Promise<number> {
    return this.request<{count: number}>({type: 'count', source, origin, region, transform}).then(r => r.count);
  }
  dispose() { this.worker?.terminate(); this.worker = null; this.pending.clear(); }
}
