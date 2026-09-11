import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {Maximize2,RotateCcw,Layers3} from 'lucide-react';
import type {Cloud} from './types';

// Adapts the existing comparison viewer's shared-camera/scissor renderer and
// OrbitControls, replacing scan-specific presets with bounds-derived framing.
export function CloudView({clouds,busy}:{clouds:Cloud[];busy:boolean}){
 const box=useRef<HTMLDivElement>(null),canvas=useRef<HTMLCanvasElement>(null),panes=useRef<(HTMLDivElement|null)[]>([]);
 const reset=useRef<()=>void>(()=>{}),setPointSize=useRef<(n:number)=>void>(()=>{});const [error,setError]=useState('');const [size,setSize]=useState(1.6);
 useEffect(()=>{
  if(!canvas.current||!box.current||!clouds.length)return;
  let renderer:THREE.WebGLRenderer;
  try{renderer=new THREE.WebGLRenderer({canvas:canvas.current,antialias:false,powerPreference:'high-performance'});}catch(e){setError(String(e));return;}
  setError('');renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.autoClear=false;
  renderer.setClearColor(0x101b20);const camera=new THREE.PerspectiveCamera(48,1,.01,10000);camera.up.set(0,0,1);
  const worldOrigin=new THREE.Vector3(...clouds[0].info.origin as [number,number,number]);
  const bounds=new THREE.Box3();const scenes:THREE.Scene[]=[],geometries:THREE.BufferGeometry[]=[],materials:THREE.Material[]=[],controls:OrbitControls[]=[];
  clouds.forEach((cloud,index)=>{
   const scene=new THREE.Scene();scenes.push(scene);
   const buffer=new THREE.InterleavedBuffer(cloud.data,6),geometry=new THREE.BufferGeometry();
   geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(buffer,3,0));geometry.setAttribute('color',new THREE.InterleavedBufferAttribute(buffer,3,3));
   const material=new THREE.ShaderMaterial({uniforms:{pointSize:{value:size*Math.min(devicePixelRatio,2)}},vertexShader:'uniform float pointSize; attribute vec3 color; varying vec3 photoColor; void main(){ photoColor=color; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); gl_PointSize=pointSize; }',fragmentShader:'varying vec3 photoColor; void main(){ vec2 d=gl_PointCoord-vec2(.5); if(dot(d,d)>.25)discard; gl_FragColor=vec4(photoColor,1.); }'});
   const points=new THREE.Points(geometry,material);points.position.copy(new THREE.Vector3(...cloud.info.origin as [number,number,number]).sub(worldOrigin));scene.add(points);
   const b=new THREE.Box3(new THREE.Vector3(...cloud.info.bounds[0] as [number,number,number]),new THREE.Vector3(...cloud.info.bounds[1] as [number,number,number])).translate(points.position);bounds.union(b);
   geometries.push(geometry);materials.push(material);
   const control=new OrbitControls(camera,panes.current[index]!);control.enableDamping=false;control.screenSpacePanning=true;controls.push(control);
  });
  setPointSize.current=(n:number)=>{materials.forEach(m=>{(m as THREE.ShaderMaterial).uniforms.pointSize.value=n*Math.min(devicePixelRatio,2)});draw();};
  const center=bounds.getCenter(new THREE.Vector3()),span=Math.max(bounds.getSize(new THREE.Vector3()).length(),1);
  const draw=()=>{if(!box.current)return;const r=box.current.getBoundingClientRect();renderer.setScissorTest(false);renderer.setViewport(0,0,r.width,r.height);renderer.clear();renderer.setScissorTest(true);scenes.forEach((scene,i)=>{const p=panes.current[i]!.getBoundingClientRect();renderer.setViewport(p.left-r.left,r.bottom-p.bottom,p.width,p.height);renderer.setScissor(p.left-r.left,r.bottom-p.bottom,p.width,p.height);camera.aspect=p.width/p.height;camera.updateProjectionMatrix();renderer.render(scene,camera);});};
  reset.current=()=>{camera.position.copy(center).add(new THREE.Vector3(-.6,-.8,.6).multiplyScalar(span));camera.near=Math.max(.001,span/10000);camera.far=span*20;controls.forEach(c=>{c.target.copy(center);c.update();});draw();};
  controls.forEach(control=>control.addEventListener('change',()=>{controls.forEach(c=>{if(c!==control)c.target.copy(control.target);});draw();}));
  const resize=()=>{if(box.current){renderer.setSize(box.current.clientWidth,box.current.clientHeight,false);draw();}};
  const observer=new ResizeObserver(resize);observer.observe(box.current);resize();reset.current();
  const lost=(e:Event)=>{e.preventDefault();setError('Graphics context lost. Reload this cloud to restore the view.');};canvas.current.addEventListener('webglcontextlost',lost);
  const cv=canvas.current;return()=>{observer.disconnect();controls.forEach(c=>c.dispose());geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());renderer.dispose();cv.removeEventListener('webglcontextlost',lost);};
 },[clouds]);
 useEffect(()=>{setPointSize.current(size)},[size]);
 return <section className="cloud-card"><div className="cloud-toolbar"><span><Layers3 size={15}/>Point cloud inspector</span><div><label>Point size <input aria-label="Point size" type="range" min=".7" max="3" step=".1" value={size} onChange={e=>setSize(+e.target.value)}/></label><button className="icon-button" title="Reset camera" onClick={()=>reset.current()}><RotateCcw size={15}/></button><button className="icon-button" title="Expand viewer" onClick={()=>box.current?.requestFullscreen?.().catch(()=>{})}><Maximize2 size={15}/></button></div></div><div className="cloud-canvas" ref={box}><canvas ref={canvas}/>{clouds.length?clouds.map((cloud,i)=><div className="cloud-pane" key={i} ref={el=>{panes.current[i]=el;}} tabIndex={0} role="application" aria-label={`${cloud.info.name}. Drag to orbit, shift-drag to pan, scroll to zoom.`}><div className="cloud-tag">{i===0?'OUTPUT':'REFERENCE'}<strong>{cloud.info.name}</strong><small>{(cloud.info.display_points/1e6).toFixed(2)}M displayed / {(cloud.info.source_points/1e6).toFixed(2)}M source</small></div></div>):<div className="empty-view"><div className="point-glyph">{Array.from({length:49},(_,i)=><i key={i} style={{opacity:.18+((i*7%11)/14)}}/>)}</div><h3>{busy?'Preparing your cloud':'A clear view of your capture'}</h3><p>{busy?'The preview opens automatically when processing finishes.':'Process a capture or open an existing LAS / PLY output.'}</p></div>}{error&&<div className="viewer-error" role="alert">{error}</div>}</div><div className="cloud-help">Drag to orbit · Shift-drag to pan · Scroll to zoom<span>Linked cameras · no alignment applied</span></div></section>;
}
