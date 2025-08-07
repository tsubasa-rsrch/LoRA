import React, { useState } from 'react'
import api from './api'

const Step = ({ n, label, active }) => (
  <div className={'step' + (active ? ' active' : '')}>{n}. {label}</div>
)

export default function App() {
  const [step, setStep] = useState(1)
  return (
    <div className="container">
      <div className="header">
        <h1>LoRA GUI (LLM)</h1>
        <div className="muted">ターミナル不要のLoRA/QLoRA微調整</div>
      </div>
      <div className="steps">
        <Step n={1} label="環境 & マシン" active={step===1} />
        <Step n={2} label="データ作成/取り込み" active={step===2} />
        <Step n={3} label="モデル & 設定" active={step===3} />
        <Step n={4} label="学習" active={step===4} />
        <Step n={5} label="評価 & 出力" active={step===5} />
      </div>
      {step===1 && <EnvPage onNext={()=>setStep(2)} />}
      {step===2 && <DataPage onNext={()=>setStep(3)} />}
      {step===3 && <TrainPage onNext={()=>setStep(4)} />}
      {step===4 && <RunPage onNext={()=>setStep(5)} />}
      {step===5 && <EvalPage />}
    </div>
  )
}

function Tip({text}){
  return <span className="tooltip" title={text}>?</span>
}

function EnvPage({onNext}){
  const [env, setEnv] = useState(null)
  const [loading, setLoading] = useState(false)

  const check = async()=>{
    setLoading(true)
    const res = await fetch('http://localhost:8000/env')
    const j = await res.json()
    setEnv(j); setLoading(false)
  }

  return (
    <div className="card">
      <h2>環境 & マシン選択</h2>
      <p className="muted">ローカルGPU / Apple Silicon / CPU / Colab / AWS のうち、まずはローカル推奨。<br/>
      <span className="small">ヒント: <span className="kbd">bitsandbytes</span> が使えるとQLoRA（4bit）が速い。</span></p>

      <div className="row" style={{marginTop: 12}}>
        <button className="btn" onClick={check} disabled={loading}>{loading? '検出中...' : '自動検出 (GPU/VRAM)'}</button>
        <span className="muted small">環境をチェックします</span>
      </div>

      {env && (
        <div style={{marginTop:12}}>
          <pre style={{whiteSpace:'pre-wrap', background:'#0f172a', padding:12, borderRadius:8}}>{JSON.stringify(env, null, 2)}</pre>
        </div>
      )}

      <div className="row" style={{marginTop: 12}}>
        <button className="btn" onClick={onNext}>次へ</button>
        <span className="muted small">続いてデータを準備します</span>
      </div>
    </div>
  )
}

function DataPage({onNext}){
  const [pairs, setPairs] = useState([{instruction:'', input:'', output:'', tags:[] }])
  const [files, setFiles] = useState(null)
  const [text, setText] = useState('')
  const [capPath, setCapPath] = useState(null)
  const [status, setStatus] = useState('')

  const addRow = ()=> setPairs(p=>[...p,{instruction:'',input:'',output:'',tags:[]}])
  const update = (i,k,v)=> setPairs(p=>p.map((x,ix)=> ix===i? {...x,[k]:v} : x))

  const ingest = async()=>{
    const fd = new FormData()
    fd.append('files', new Blob()) // keep form-data shape even if empty (FastAPI quirk)
    const res = await fetch('http://localhost:8000/ingest', {
      method:'POST',
      body: fd
    })
    const j = await res.json()
    setStatus(JSON.stringify(j))
  }

  const caption = async()=>{
    const res = await fetch('http://localhost:8000/caption', {
      method:'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ text, mode:'summary_qa' })
    })
    const j = await res.json()
    setCapPath(j.path)
    setStatus(JSON.stringify(j))
  }

  return (
    <div className="card">
      <h2>データ作成 / 取り込み</h2>

      <div className="grid grid-2">
        <div>
          <label>一問一答エディタ <Tip text="GUIで質問と回答を追加。タグは任意（例: qa|jp）" /></label>
          {pairs.map((p,i)=>(
            <div key={i} className="card" style={{padding:12}}>
              <input placeholder="instruction（質問や指示）" value={p.instruction} onChange={e=>update(i,'instruction',e.target.value)} />
              <textarea placeholder="input（追加入力、省略可）" rows={2} value={p.input} onChange={e=>update(i,'input',e.target.value)} />
              <textarea placeholder="output（回答）空でもOK" rows={2} value={p.output} onChange={e=>update(i,'output',e.target.value)} />
            </div>
          ))}
          <div className="row">
            <button className="btn secondary" onClick={addRow}>行を追加</button>
            <span className="muted small">CSV/JSONLは右側の「ファイル投入」でもOK</span>
          </div>
        </div>

        <div>
          <label>ファイル投入 <Tip text="CSV/JSONL/ShareGPT/MD/TXT を投入。自動解析します" /></label>
          <input type="file" multiple onChange={e=>setFiles(e.target.files)} />

          <label style={{marginTop:12, display:'block'}}>キャプション自動生成 <Tip text="生文書→ instruction/input/output ペア化。ヒューリスティック（要約/重要点）" /></label>
          <textarea rows={8} placeholder="ここに生の文書を貼ってください" value={text} onChange={e=>setText(e.target.value)} />

          <div className="row">
            <button className="btn" onClick={caption}>ペアを自動生成</button>
            {capPath && <span className="muted small">生成ファイル: {capPath}</span>}
          </div>
        </div>
      </div>

      <div style={{marginTop:12}}>
        <pre style={{whiteSpace:'pre-wrap', background:'#0f172a', padding:12, borderRadius:8}}>{status}</pre>
      </div>

      <div className="row" style={{marginTop: 12}}>
        <button className="btn" onClick={onNext}>次へ</button>
      </div>
    </div>
  )
}

function TrainPage({onNext}){
  const [baseModel, setBaseModel] = useState('meta-llama/Meta-Llama-3-8B')
  const [loraType, setLoraType] = useState('qlora')
  const [epochs, setEpochs] = useState(1)
  const [lr, setLr] = useState(2e-4)
  const [bs, setBs] = useState(1)
  const [ga, setGa] = useState(4)
  const [maxLen, setMaxLen] = useState(2048)
  const [status, setStatus] = useState('')

  const start = async()=>{
    const res = await api.post('train', { json: {
      base_model: baseModel, lora_type: loraType, num_epochs: Number(epochs),
      lr: Number(lr), batch_size: Number(bs), grad_accum_steps: Number(ga),
      max_seq_len: Number(maxLen),
    }}).json()
    setStatus(JSON.stringify(await res, null, 2))
  }

  return (
    <div className="card">
      <h2>モデル & 学習設定</h2>
      <div className="grid grid-2">
        <div>
          <label>ベースモデル <span className="muted small">(例: Llama-3 8B, Qwen2 7B)</span></label>
          <input value={baseModel} onChange={e=>setBaseModel(e.target.value)} />
        </div>
        <div>
          <label>方式プリセット <span className="muted small">QLoRA=4bitで低VRAM、LoRA=16bit/FP16想定</span></label>
          <select value={loraType} onChange={e=>setLoraType(e.target.value)}>
            <option value="qlora">QLoRA</option>
            <option value="lora">LoRA</option>
          </select>
        </div>
        <div>
          <label>エポック</label>
          <input type="number" value={epochs} onChange={e=>setEpochs(e.target.value)} />
        </div>
        <div>
          <label>学習率</label>
          <input type="number" value={lr} onChange={e=>setLr(e.target.value)} />
        </div>
        <div>
          <label>バッチサイズ</label>
          <input type="number" value={bs} onChange={e=>setBs(e.target.value)} />
        </div>
        <div>
          <label>勾配累積</label>
          <input type="number" value={ga} onChange={e=>setGa(e.target.value)} />
        </div>
        <div>
          <label>最大トークン長</label>
          <input type="number" value={maxLen} onChange={e=>setMaxLen(e.target.value)} />
        </div>
      </div>

      <div className="row" style={{marginTop:12}}>
        <button className="btn" onClick={start}>学習を開始</button>
        <span className="muted small">VRAM不足時はQLoRA推奨</span>
      </div>

      <div style={{marginTop:12}}>
        <pre style={{whiteSpace:'pre-wrap', background:'#0f172a', padding:12, borderRadius:8}}>{status}</pre>
      </div>

      <div className="row" style={{marginTop:12}}>
        <button className="btn" onClick={onNext}>次へ</button>
      </div>
    </div>
  )
}

function RunPage({onNext}){
  return (
    <div className="card">
      <h2>学習</h2>
      <p className="muted">学習ログはバックエンド標準出力 / <span className="kbd">outputs/logs/train.log</span>（MVPでは簡易）</p>
      <div className="row" style={{marginTop:12}}>
        <button className="btn" onClick={onNext}>評価へ</button>
      </div>
    </div>
  )
}

function EvalPage(){
  const [baseModel, setBaseModel] = useState('meta-llama/Meta-Llama-3-8B')
  const [adapterPath, setAdapterPath] = useState('outputs/adapters/adapter')
  const [prompts, setPrompts] = useState('あなたは誰？\n次の文章を2行で要約して: ...')
  const [res, setRes] = useState('')

  const run = async()=>{
    const resj = await api.post('eval', { json: {
      base_model: baseModel,
      adapter_path: adapterPath,
      prompts: prompts.split('\n').filter(Boolean)
    }}).json()
    setRes(JSON.stringify(await resj, null, 2))
  }

  const card = async()=>{
    const r = await fetch('http://localhost:8000/export/card')
    const j = await r.json()
    setRes(JSON.stringify(j, null, 2))
  }

  return (
    <div className="card">
      <h2>評価 & 出力</h2>
      <div className="grid">
        <div>
          <label>ベースモデル</label>
          <input value={baseModel} onChange={e=>setBaseModel(e.target.value)} />
        </div>
        <div>
          <label>Adapterパス</label>
          <input value={adapterPath} onChange={e=>setAdapterPath(e.target.value)} />
        </div>
        <div>
          <label>評価プロンプト（改行区切り）</label>
          <textarea rows={4} value={prompts} onChange={e=>setPrompts(e.target.value)} />
        </div>
        <div className="row">
          <button className="btn" onClick={run}>比較テストを走らせる</button>
          <button className="btn secondary" onClick={card}>モデルカードを保存</button>
        </div>
      </div>
      <div style={{marginTop:12}}>
        <pre style={{whiteSpace:'pre-wrap', background:'#0f172a', padding:12, borderRadius:8}}>{res}</pre>
      </div>
    </div>
  )
}
