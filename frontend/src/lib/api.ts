import { useEffect, useState } from 'react'
export function useApi<T=any>(path:string) {
  const [data,setData]=useState<T|null>(null),[error,setError]=useState(''),[loading,setLoading]=useState(true),[retry,setRetry]=useState(0)
  useEffect(()=>{const controller=new AbortController();setLoading(true);setError('');setData(null)
    fetch(new URL(`api/${path}`,document.baseURI),{signal:controller.signal}).then(async r=>{const body=await r.json();if(!r.ok)throw new Error(body.error?.message||'This view is unavailable. Check artifact readiness.');return body}).then(setData).catch(e=>{if(e.name!=='AbortError')setError(e.message)}).finally(()=>{if(!controller.signal.aborted)setLoading(false)})
    return ()=>controller.abort()
  },[path,retry])
  return {data,error,loading,reload:()=>setRetry(v=>v+1)}
}
export const pct=(v:number|null|undefined)=>v==null?'—':`${(v*100).toFixed(2)}%`
export const num=(v:number|null|undefined)=>v==null?'—':v.toLocaleString()
export const label=(s:string)=>s.replace(/_/g,' ')
