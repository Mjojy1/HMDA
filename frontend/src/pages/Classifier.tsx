import { FormEvent, useState } from 'react'
import { useApi, num, pct } from '../lib/api'
import { Notice, Panel, State, Stat } from '../components/UI'
import { Model } from './Dashboard'

export function Classifier(){
 const options=useApi('classifier-demo/options')
 const [values,setValues]=useState({loan_amount:'300000',income:'100',combined_loan_to_value_ratio:'80',debt_to_income_ratio:'30%-<36%',loan_type:'1',loan_term:'360'})
 const [ack,setAck]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState(''),[result,setResult]=useState<any>(null)
 function change(key:string,value:string){setValues(v=>({...v,[key]:value}));setResult(null);setError('')}
 async function submit(event:FormEvent){
  event.preventDefault();setBusy(true);setError('');setResult(null)
  try{
   const response=await fetch(new URL('api/classifier-demo',document.baseURI),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...values,loan_amount:Number(values.loan_amount),income:Number(values.income),combined_loan_to_value_ratio:Number(values.combined_loan_to_value_ratio),loan_term:Number(values.loan_term),fictional_scenario:ack})})
   const body=await response.json()
   if(!response.ok)throw new Error(body.error?.message||(typeof body.detail==='string'?body.detail:'Check all six inputs and confirm this is a fictional scenario.'))
   setResult(body)
  }catch(e){setError(e instanceof Error?e.message:'The classifier is unavailable. Try again.')}
  finally{setBusy(false)}
 }
 return <><Notice>Presentation simulator: use fictional loan scenarios only. No names or personal records. Results describe the saved model’s historical classification, not an actual approval decision.</Notice>
 <Panel title="Try the saved gradient boosting classifier" subtitle="Six model inputs · saved preprocessing · no retraining">
 <State request={options}><form onSubmit={submit} className="classifier-form"><fieldset disabled={busy}><legend>Fictional scenario inputs</legend><div className="grid-two">
 {[['loan_amount','Loan amount (USD)','1','100000000'],['income','Annual income (thousands of USD)','0','1000000'],['combined_loan_to_value_ratio','Combined loan-to-value (%)','0','1000'],['loan_term','Loan term (months)','1','600']].map(([key,title,min,max])=><label key={key}>{title}<input required type="number" min={min} max={max} step={key==='loan_term'?'1':'any'} value={values[key as keyof typeof values]} onChange={e=>change(key,e.target.value)}/></label>)}
 <label>Debt-to-income category<select value={values.debt_to_income_ratio} onChange={e=>change('debt_to_income_ratio',e.target.value)}>{options.data?.debt_to_income_ratio.map((v:string)=><option key={v}>{v}</option>)}</select></label>
 <label>Loan type<select value={values.loan_type} onChange={e=>change('loan_type',e.target.value)}>{[['1','Conventional'],['2','FHA-insured'],['3','VA-guaranteed'],['4','USDA RHS/FSA-guaranteed']].map(([v,label])=><option key={v} value={v}>{label}</option>)}</select></label></div>
 <p className="chart-note">Income 100 means $100,000 annually. Term 360 means 30 years. Scope: home purchase, first lien, principal residence, site-built, 1–4 units.</p>
 <Panel title="Optional demographic context" subtitle="For fairness context only—not used in prediction"><p className="chart-note">Choose fictional demographics to view existing group-level evidence. Selections stay in this browser, are not saved or sent to the classifier or OpenAI, and do not change the score. Each group is evaluated separately, not as an intersection.</p><div className="grid-two">{['race','sex','ethnicity'].map(dimension=><DemographicContext key={dimension} dimension={dimension}/>)}</div></Panel>
 <label className="scenario-confirm"><input type="checkbox" checked={ack} required onChange={e=>{setAck(e.target.checked);setResult(null)}}/>These are fictional inputs for a test demonstration.</label>
 <button type="submit" disabled={!ack||busy}>{busy?'Running saved model…':'Classify test scenario'}</button></fieldset></form></State>
 {error&&<p role="alert" className="error">{error}</p>}
 {result&&<div role="status" aria-live="polite"><h3>Predicted historical class: {result.label}</h3><div className="mini-stats"><Stat label="Positive-class model score" value={result.positive_class_score.toFixed(6)} note="Uncalibrated score on a 0–1 scale; not a chance of approval"/><Stat label="Classification threshold" value={String(result.threshold)} note="Score at or above threshold → positive class"/></div><p>Score ≥ {result.threshold}: positive (actions 1 + 2). Score &lt; {result.threshold}: denied (action 3). Classification uses the full-precision score.</p><p>{result.target_note}</p><p className="chart-note">{result.disclaimer}</p></div>}
 </Panel><h2>Saved model performance and evidence</h2><Model/></>
}

function DemographicContext({dimension}:{dimension:string}){
 const evidence=useApi(`fairness?dimension=${dimension}&min_samples=0`)
 const [selected,setSelected]=useState('')
 const rows=evidence.data?.rows||[]
 const row=rows.find((r:any)=>r.group===selected)
 const title=dimension.charAt(0).toUpperCase()+dimension.slice(1)
 return <div><State request={evidence}><label>{title} (optional)<select aria-label={`${title} (optional)`} value={selected} onChange={event=>setSelected(event.target.value)}><option value="">Not selected</option>{rows.map((r:any)=><option key={r.group} value={r.group}>{r.group}</option>)}</select></label>
 {row&&<div className="chart-note"><strong>{evidence.data.model.test_year} group evaluation · {row.group}</strong><p>{num(row.applications)} records · {num(row.actual_denials)} historical denials</p><p>Denial recall: {pct(row.denial_recall)} · Approval recall: {pct(row.approval_recall)}</p>{row.small_group&&<p>Limited sample support: interpret rates cautiously.</p>}{row.demographic_unavailable&&<p>Demographic information unavailable for this group.</p>}<p>Descriptive group evidence, not an assessment of this fictional scenario. Undefined rates appear as a dash.</p></div>}
 </State></div>
}
