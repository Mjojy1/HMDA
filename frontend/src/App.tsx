import { useEffect, useRef, useState } from 'react'
import { MessageSquare, X, Menu, ArrowUpRight } from 'lucide-react'
import { Overview, Fairness } from './pages/Dashboard'
import { Classifier } from './pages/Classifier'
import { Analyst } from './pages/Analyst'
const pages=[
 {id:'overview',name:'Overview',title:'Understand the Lending Picture',description:'Filter the public sample and explore lending outcomes and loan characteristics.',component:Overview},
 {id:'model',name:'Model Classifier',title:'Explore the Model Classifier',description:'Try test scenarios and review saved model performance.',component:Classifier},
 {id:'fairness',name:'Fairness',title:'Review Group Outcomes',description:'Explore descriptive differences with the context needed to interpret them.',component:Fairness},
 {id:'analyst',name:'Analyst',title:'Ask the Analyst',description:'Explore aggregate evidence with a read-only research assistant.',component:Analyst}
]
export default function App(){
 const [active,setActive]=useState(location.hash.slice(1)||'overview'),[menu,setMenu]=useState(false)
 const [analystOpen,setAnalystOpen]=useState(false)
 const toggle=useRef<HTMLButtonElement>(null), drawer=useRef<HTMLElement>(null)
 const page=pages.find(p=>p.id===active)||pages[0],Content=page.component
 const showAnalyst=analystOpen&&page.id!=='analyst'
 function closeAnalyst(){setAnalystOpen(false);toggle.current?.focus()}
 useEffect(()=>{const fn=()=>{const id=location.hash.slice(1)||'overview';setActive(id);setMenu(false)};window.addEventListener('hashchange',fn);return ()=>window.removeEventListener('hashchange',fn)},[])
 useEffect(()=>{if(!showAnalyst)return;const key=(e:KeyboardEvent)=>{if(e.key==='Escape'){setAnalystOpen(false);toggle.current?.focus()}};window.addEventListener('keydown',key);return ()=>window.removeEventListener('keydown',key)},[showAnalyst])
 return <div className="app-shell"><a className="skip-link" href="#main-content" onClick={e=>{e.preventDefault();document.getElementById('main-content')?.focus()}}>Skip to content</a>
 <header className="app-header"><a className="brand" href="#overview"><span className="brand-mark">H</span><span>HMDA <span className="brand-description">Mortgage Intelligence</span></span></a><button className="mobile-menu" aria-label={menu?'Close navigation':'Open navigation'} aria-expanded={menu} aria-controls="primary-navigation" onClick={()=>setMenu(!menu)}>{menu?<X size={20}/>:<Menu size={20}/>}</button><nav id="primary-navigation" aria-label="Main navigation" className={menu?'primary-nav open':'primary-nav'}>{pages.map(p=><a href={`#${p.id}`} key={p.id} aria-current={p.id===page.id?'page':undefined} className={p.id===page.id?'active':''}>{p.name}</a>)}</nav></header>
 <div className={`workspace ${showAnalyst?'with-analyst':''}`}><main id="main-content" tabIndex={-1}><div className="page-heading"><div><p className="eyebrow">PUBLIC HMDA RESEARCH</p><h1>{page.title}</h1><p>{page.description}</p></div>{page.id!=='analyst'&&<button ref={toggle} className="ask-button" aria-expanded={showAnalyst} aria-controls="contextual-analyst" onClick={()=>{setAnalystOpen(!analystOpen);if(!analystOpen)setTimeout(()=>drawer.current?.querySelector<HTMLButtonElement>('button')?.focus(),0)}}><MessageSquare size={16}/>{showAnalyst?'Hide analyst':'Ask analyst'}</button>}</div><Content/><footer><span>Public HMDA · Research and monitoring only</span><a href="#overview">Sources & limitations <ArrowUpRight size={13}/></a><span>No individual lending decisions</span></footer></main>
 {showAnalyst&&<aside ref={drawer} className="analyst-drawer" id="contextual-analyst" aria-label="Contextual analyst"><div className="drawer-heading"><div><h2>Ask about this view</h2><p>{page.name}{['model','fairness'].includes(page.id)?' · Saved 2025 evaluation':''}</p></div><button onClick={closeAnalyst} aria-label="Close analyst" title="Close analyst"><X size={19}/></button></div><Analyst embedded context={page.id}/></aside>}</div></div>
}
