import { useState } from 'react'
import { Notice, Panel } from '../components/UI'
import { useApi } from '../lib/api'

type Turn = {question:string, result?:any, error?:string}
export function Analyst(){
 const [message,setMessage]=useState(''),[turns,setTurns]=useState<Turn[]>([]),[loading,setLoading]=useState(false)
 const status=useApi('analyst/status')
 async function ask(text:string){
  if(!text.trim()||loading)return
  setLoading(true);setMessage('')
  setTurns(t=>[...t,{question:text}])
  try{
   const response=await fetch(new URL('api/analyst',document.baseURI),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,history:turns.slice(-6).map(t=>t.question)}),signal:AbortSignal.timeout(150000)})
   const data=await response.json()
   if(!response.ok)throw new Error(data.error?.message||'Analyst unavailable. Try a supported aggregate question.')
   setTurns(t=>[...t.slice(0,-1),{question:text,result:data}])
  }catch(e){setTurns(t=>[...t.slice(0,-1),{question:text,error:e instanceof Error?e.message:'Request failed'}])}
  finally{setLoading(false)}
 }
 return <><Notice>Ask about aggregate historical outcomes, saved model results, fairness groups, or metric definitions. No individual lending decisions. The agent can gather and combine evidence across tools. Restate year filters when changing scope.</Notice>
 <Panel title="Conversational analyst" subtitle={status.loading?'Checking availability…':status.error?'Analyst status unavailable':status.data?.agent_available?`${status.data.provider_label} agent configured — live access is checked when you ask a question.`:status.data?.configuration_error||'Agent unavailable: configure OpenAI or Domino AI Gateway. Basic single-tool lookup remains available.'}>
 {status.data?.provider==='openai'&&<Notice>Questions, recent question history, and aggregate HMDA tool evidence are sent to OpenAI. Do not enter personal or confidential information. The API key stays on the server.</Notice>}
 <div className="analyst-examples">{['Review model performance and fairness, and explain the target','Show aggregate outcomes for 2025','Show trends over time','Show model performance','Explain denial recall','Show fairness groups'].map(q=><button key={q} disabled={loading} onClick={()=>ask(q)}>{q}</button>)}</div>
 <div aria-live="polite" aria-busy={loading}>{turns.map((t,i)=><article className="analyst-turn" key={i}><h3>{t.question}</h3>{t.error?<p role="alert">{t.error}</p>:t.result?<><p style={{whiteSpace:'pre-line'}}>{t.result.answer}</p>{t.result.mode&&<p>{t.result.mode}</p>}{t.result.trace&&<details open><summary>Agent activity ({t.result.steps} steps)</summary><ol>{t.result.trace.map((step:any,i:number)=><li key={i}>{step.action}: {step.tool?.name || step.evidence_ids?.join(', ') || 'Stopped'} — {step.status}{step.evidence_id&&` [${step.evidence_id}]`}{step.error&&<p role="alert">{step.error}</p>}</li>)}</ol></details>}{t.result.source&&<p><strong>Evidence:</strong> {t.result.source}</p>}{t.result.evidence&&<details open><summary>Tool evidence (exact API values; rates are proportions)</summary><pre>{JSON.stringify(t.result.evidence,null,2)}</pre></details>}<p>{t.result.target_note}</p><p className="muted">{t.result.limitations}</p></>:<p role="status">Reading aggregate evidence…</p>}</article>)}</div>
 <form onSubmit={e=>{e.preventDefault();ask(message)}}><label htmlFor="analyst-question">Your aggregate question</label><textarea id="analyst-question" value={message} maxLength={2000} onChange={e=>setMessage(e.target.value)} placeholder="How did sampled outcomes change over time?"/><button disabled={loading||!message.trim()} type="submit">{loading?'Reading evidence…':'Ask analyst'}</button></form>
 </Panel></>
}
