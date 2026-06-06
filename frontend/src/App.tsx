import { useState, useEffect, useRef, useId } from 'react'
import { downloadResult } from './download'
import type { TaskResult, RunResult, RunState } from './types'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL

const MODELS: { group: string; provider: string; models: string[] }[] = [
  {
    group: 'Anthropic',
    provider: 'anthropic',
    models: ['claude-haiku-4-5', 'claude-haiku-4-5-20251001', 'claude-sonnet-4-6', 'claude-opus-4-8'],
  },
  {
    group: 'OpenAI',
    provider: 'openai',
    models: ['gpt-4o-mini', 'gpt-4o', 'gpt-5.4-mini', 'gpt-5.4', 'gpt-5.5'],
  },
]

function providerOf(model: string): string {
  return MODELS.find(g => g.models.includes(model))?.provider ?? ''
}

// ── Tooltip ───────────────────────────────────────────────────────────────────

function Tooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLSpanElement>(null)
  const id = useId()

  useEffect(() => {
    if (!open) return
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  return (
    <span className="tooltip-wrap" ref={ref}>
      <button
        type="button"
        className="tooltip-trigger"
        aria-describedby={id}
        aria-expanded={open}
        onClick={() => setOpen(o => !o)}
      >?</button>
      {open && (
        <span className="tooltip-box" id={id} role="tooltip">{text}</span>
      )}
    </span>
  )
}

// ── Accuracy helpers ──────────────────────────────────────────────────────────

type AccRow = { task_kind: string; row_correct: number; col_correct: number; total: number }

function computeAccuracy(results: TaskResult[]): AccRow[] {
  const byKind: Record<string, AccRow> = {}
  for (const r of results) {
    byKind[r.task_kind] ??= { task_kind: r.task_kind, row_correct: 0, col_correct: 0, total: 0 }
    byKind[r.task_kind].row_correct += r.row_correct ? 1 : 0
    byKind[r.task_kind].col_correct += r.col_correct ? 1 : 0
    byKind[r.task_kind].total += 1
  }
  return Object.values(byKind).sort((a, b) => a.task_kind.localeCompare(b.task_kind))
}

function pct(n: number, total: number) {
  return total === 0 ? '—' : `${((n / total) * 100).toFixed(1)}%`
}

function fmtDelta(row: AccRow) {
  if (row.total === 0) return '—'
  const d = ((row.col_correct - row.row_correct) / row.total) * 100
  return `${d >= 0 ? '+' : ''}${d.toFixed(1)}%`
}

function deltaClass(row: AccRow) {
  const d = row.col_correct - row.row_correct
  return d > 0 ? 'pos' : d < 0 ? 'neg' : 'neutral'
}

// ── Shared components ─────────────────────────────────────────────────────────

function AccuracyTable({ results }: { results: TaskResult[] }) {
  const rows = computeAccuracy(results)
  const totals: AccRow = {
    task_kind: 'TOTAL',
    row_correct: results.filter(r => r.row_correct).length,
    col_correct: results.filter(r => r.col_correct).length,
    total: results.length,
  }
  return (
    <table className="acc-table">
      <thead>
        <tr><th>Task type</th><th>Row acc</th><th>Col acc</th><th>Delta</th><th>n</th></tr>
      </thead>
      <tbody>
        {rows.map(row => (
          <tr key={row.task_kind}>
            <td>{row.task_kind}</td>
            <td>{pct(row.row_correct, row.total)}</td>
            <td>{pct(row.col_correct, row.total)}</td>
            <td className={`delta ${deltaClass(row)}`}>{fmtDelta(row)}</td>
            <td>{row.total}</td>
          </tr>
        ))}
        <tr className="totals-row">
          <td>TOTAL</td>
          <td>{pct(totals.row_correct, totals.total)}</td>
          <td>{pct(totals.col_correct, totals.total)}</td>
          <td className={`delta ${deltaClass(totals)}`}>{fmtDelta(totals)}</td>
          <td>{totals.total}</td>
        </tr>
      </tbody>
    </table>
  )
}

function TaskDetail({ results }: { results: TaskResult[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="task-detail">
      <button className="link-btn" onClick={() => setOpen(o => !o)}>
        {open ? 'Hide' : 'Show'} individual tasks ({results.length})
      </button>
      {open && (
        <div className="task-list">
          {results.map((r, i) => {
            const cls = r.row_correct && r.col_correct ? 'both-correct'
              : !r.row_correct && !r.col_correct ? 'both-wrong' : 'mixed'
            return (
              <div key={i} className={`task-item ${cls}`}>
                <div className="task-header">
                  <span className="task-kind">{r.task_kind}</span>
                  <span className="task-badges">
                    <span className={`badge ${r.row_correct ? 'ok' : 'fail'}`}>row {r.row_correct ? '✓' : '✗'}</span>
                    <span className={`badge ${r.col_correct ? 'ok' : 'fail'}`}>col {r.col_correct ? '✓' : '✗'}</span>
                  </span>
                </div>
                <div className="task-q"><strong>Q:</strong> {r.question}</div>
                <div className="task-truth"><strong>Truth:</strong> {r.ground_truth}</div>
                <div className="task-answers">
                  <span><strong>Row:</strong> {r.row_answer}</span>
                  <span><strong>Col:</strong> {r.col_answer}</span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function ResultCard({ result }: { result: RunResult }) {
  return (
    <div className="result-box">
      <div className="result-header">
        <h3>{result.dataset} — {result.model}</h3>
        <button className="download-btn" onClick={() => downloadResult(result)}>
          Download JSON
        </button>
      </div>
      <AccuracyTable results={result.results} />
      <TaskDetail results={result.results} />
    </div>
  )
}

// ── File input ───────────────────────────────────────────────────────────────

function FileInput({ onChange, accept = '.csv', required = true }: {
  onChange: (f: File | null) => void
  accept?: string
  required?: boolean
}) {
  const ref = useRef<HTMLInputElement>(null)
  const [name, setName] = useState<string | null>(null)

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null
    setName(f?.name ?? null)
    onChange(f)
  }

  return (
    <div className="file-input-row">
      <button type="button" className="file-btn" onClick={() => ref.current?.click()}>
        Choose file
      </button>
      <span className="file-name">{name ?? 'No file chosen'}</span>
      <input
        ref={ref}
        type="file"
        accept={accept}
        required={required}
        onChange={handleChange}
        style={{ display: 'none' }}
      />
    </div>
  )
}

// ── Confirm modal ────────────────────────────────────────────────────────────

function ConfirmModal({ message, onConfirm, onCancel }: {
  message: string
  onConfirm: () => void
  onCancel: () => void
}) {
  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <p>{message}</p>
        <div className="modal-actions">
          <button className="modal-btn modal-btn-danger" onClick={onConfirm}>Yes, start new run</button>
          <button className="modal-btn modal-btn-cancel" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  )
}

// ── Reload saved JSON ─────────────────────────────────────────────────────────

function ReloadJsonSection({ onLoaded }: { onLoaded: (result: RunResult) => void }) {
  const [error, setError] = useState<string | null>(null)

  function handleFile(f: File | null) {
    if (!f) return
    setError(null)
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const parsed = JSON.parse(reader.result as string)
        if (!parsed || !Array.isArray(parsed.results) || !parsed.dataset || !parsed.model) {
          setError('File does not look like a results JSON from this app.')
          return
        }
        onLoaded(parsed as RunResult)
      } catch {
        setError('Could not parse file — make sure it is a valid JSON file.')
      }
    }
    reader.readAsText(f)
  }

  return (
    <details className="recovery-section">
      <summary>Load a saved results file</summary>
      <div className="recovery-body">
        <p className="recovery-hint">Upload a JSON file previously downloaded from this app to view the results again.</p>
        <div className="field">
          <label>Results JSON</label>
          <FileInput onChange={handleFile} accept=".json" required={false} />
        </div>
        {error && <p className="status-msg error">{error}</p>}
      </div>
    </details>
  )
}

// ── Re-parse raw batch JSONL ──────────────────────────────────────────────────

function BatchReparseSection({ onLoaded }: { onLoaded: (result: RunResult) => void }) {
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [jsonlFile, setJsonlFile] = useState<File | null>(null)
  const [model, setModel] = useState(MODELS[0].models[0])
  const [idCol, setIdCol] = useState('content_id')
  const [n, setN] = useState(20)
  const [seed, setSeed] = useState(42)
  const [maxRows, setMaxRows] = useState(15)
  const [cols, setCols] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!csvFile || !jsonlFile) return
    setError(null)
    setSubmitting(true)
    const fd = new FormData()
    fd.append('csv', csvFile)
    fd.append('batch_results', jsonlFile)
    fd.append('model', model)
    fd.append('id_col', idCol)
    fd.append('n', String(n))
    fd.append('seed', String(seed))
    fd.append('max_rows', String(maxRows))
    fd.append('cols', cols)
    try {
      const res = await fetch(`${API_URL}/upload-batch`, { method: 'POST', body: fd })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError(body.detail ?? `Server error ${res.status}`)
        return
      }
      onLoaded(await res.json() as RunResult)
    } catch (err) {
      setError(String(err))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <details className="recovery-section">
      <summary>Re-parse raw batch results from provider</summary>
      <div className="recovery-body">
        <p className="recovery-hint">
          Upload the original CSV and the raw JSONL batch output from Anthropic or OpenAI to re-score the results.
          Use the same parameters as the original run.
        </p>
        <form onSubmit={handleSubmit} className="run-form">
          <div className="field">
            <label>Original CSV</label>
            <FileInput onChange={setCsvFile} accept=".csv" />
          </div>
          <div className="field">
            <label>Batch results JSONL</label>
            <FileInput onChange={setJsonlFile} accept=".jsonl,.txt" />
          </div>
          <div className="field">
            <label>Model</label>
            <select value={model} onChange={e => setModel(e.target.value)}>
              {MODELS.map(({ group, models }) => (
                <optgroup key={group} label={group}>
                  {models.map(m => <option key={m} value={m}>{m}</option>)}
                </optgroup>
              ))}
            </select>
          </div>
          <div className="field">
            <label>
              ID column
              <Tooltip text="Must match the ID column used in the original run." />
            </label>
            <input type="text" value={idCol} onChange={e => setIdCol(e.target.value)} required />
          </div>
          <div className="field-row">
            <div className="field">
              <label>n tasks</label>
              <input type="number" min={1} value={n} onChange={e => setN(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>Seed</label>
              <input type="number" value={seed} onChange={e => setSeed(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>Max rows</label>
              <input type="number" min={1} value={maxRows} onChange={e => setMaxRows(Number(e.target.value))} />
            </div>
          </div>
          <div className="field">
            <label>
              Columns
              <Tooltip text="Must match the columns filter used in the original run. Leave blank if you did not filter." />
            </label>
            <input type="text" value={cols} onChange={e => setCols(e.target.value)} placeholder="blank = all" />
          </div>
          <button type="submit" disabled={submitting || !csvFile || !jsonlFile}>
            {submitting ? 'Parsing…' : 'Parse results'}
          </button>
          {error && <p className="status-msg error">{error}</p>}
        </form>
      </div>
    </details>
  )
}

// ── Run panel ─────────────────────────────────────────────────────────────────

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [model, setModel] = useState(MODELS[0].models[0])
  const [idCol, setIdCol] = useState('content_id')
  const [n, setN] = useState(20)
  const [seed, setSeed] = useState(42)
  const [maxRows, setMaxRows] = useState(15)
  const [cols, setCols] = useState('')
  const [anthropicKey, setAnthropicKey] = useState('')
  const [openaiKey, setOpenaiKey] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)
  const [runState, setRunState] = useState<RunState | null>(null)
  const [showConfirm, setShowConfirm] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Resume polling for a run that survived a page refresh
  useEffect(() => {
    const saved = localStorage.getItem('activeRunId')
    if (saved && !runId) {
      setRunId(saved)
      setRunState({ status: 'running' })
    }
  }, [])

  useEffect(() => {
    if (!runId) return
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/run/${runId}`)
        if (res.status === 404) {
          // Run expired from backend before we could fetch it
          localStorage.removeItem('activeRunId')
          setRunState({ status: 'failed', error: 'Run expired — the server evicted the result before it could be retrieved.' })
          clearInterval(pollRef.current!)
          return
        }
        const data: RunState = await res.json()
        setRunState(data)
        if (data.status !== 'running') {
          localStorage.removeItem('activeRunId')
          clearInterval(pollRef.current!)
          if (data.status === 'done' && data.result) {
            downloadResult(data.result)
          }
        }
      } catch (err) {
        localStorage.removeItem('activeRunId')
        setRunState({ status: 'failed', error: String(err) })
        clearInterval(pollRef.current!)
      }
    }, 2000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [runId])

  async function startRun() {
    if (!file) return
    setShowConfirm(false)
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
    setSubmitting(true)
    setRunId(null)
    setRunState(null)

    const fd = new FormData()
    fd.append('csv', file)
    fd.append('model', model)
    fd.append('id_col', idCol)
    fd.append('n', String(n))
    fd.append('seed', String(seed))
    fd.append('max_rows', String(maxRows))
    fd.append('cols', cols)
    fd.append('api_key', providerOf(model) === 'openai' ? openaiKey : anthropicKey)

    try {
      const res = await fetch(`${API_URL}/run`, { method: 'POST', body: fd })
      const data: { id: string } = await res.json()
      localStorage.setItem('activeRunId', data.id)
      setRunId(data.id)
      setRunState({ status: 'running' })
    } catch (err) {
      setRunState({ status: 'failed', error: String(err) })
    } finally {
      setSubmitting(false)
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    if (runState !== null) { setShowConfirm(true); return }
    startRun()
  }

  const busy = submitting || runState?.status === 'running'

  function handleRecoveredResult(result: RunResult) {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
    setRunState({ status: 'done', result })
  }

  return (
    <>
    <div className="app">
      <header>
        <h1>CSV Orientation Experiment</h1>
        <p className="subtitle">Row-wise vs column-wise CSV accuracy benchmark</p>
      </header>

      <div className="panel">
          <form onSubmit={handleSubmit} className="run-form">
            <div className="field">
              <label>CSV file</label>
              <FileInput onChange={setFile} />
            </div>
            <div className="field">
              <label>Model</label>
              <select value={model} onChange={e => setModel(e.target.value)}>
                {MODELS.map(({ group, models }) => (
                  <optgroup key={group} label={group}>
                    {models.map(m => <option key={m} value={m}>{m}</option>)}
                  </optgroup>
                ))}
              </select>
            </div>
            <div className="field">
              <label>API key</label>
              <input
                type="password"
                value={providerOf(model) === 'openai' ? openaiKey : anthropicKey}
                onChange={e =>
                  providerOf(model) === 'openai'
                    ? setOpenaiKey(e.target.value)
                    : setAnthropicKey(e.target.value)
                }
                placeholder="Your API key"
                required
              />
            </div>
            <div className="field">
              <label>
                ID column
                <Tooltip text="The column that uniquely identifies each row — used as the entity name in questions (e.g. 'What is the genre of X?'). Must be present in the CSV." />
              </label>
              <input type="text" value={idCol} onChange={e => setIdCol(e.target.value)} required />
            </div>
            <div className="field-row">
              <div className="field">
                <label>n tasks</label>
                <input type="number" min={1} value={n} onChange={e => setN(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>Seed</label>
                <input type="number" value={seed} onChange={e => setSeed(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>Max rows</label>
                <input type="number" min={1} value={maxRows} onChange={e => setMaxRows(Number(e.target.value))} />
              </div>
            </div>
            <div className="field">
              <label>
                Columns
                <Tooltip text="Restrict which columns are included in the CSV context sent to the model. Comma-separated. Leave blank to use all columns. The ID column is always included automatically." />
              </label>
              <input
                type="text"
                value={cols}
                onChange={e => setCols(e.target.value)}
                placeholder="e.g. title, genre, rating  (blank = all)"
              />
            </div>
            <button type="submit" disabled={busy}>
              {submitting ? 'Starting…' : busy ? 'Running…' : 'Run experiment'}
            </button>
          </form>

          {runState && (
            <div className="run-status">
              {runState.status === 'running' && (
                <p className="status-msg"><span className="spinner" /> Running… polling every 2s</p>
              )}
              {runState.status === 'failed' && (
                <p className="status-msg error">Run failed: {runState.error}</p>
              )}
              {runState.status === 'done' && runState.result && (
                <>
                  <p className="status-msg download-note">
                    Download started automatically — use the button below if it was blocked.
                  </p>
                  <ResultCard result={runState.result} />
                </>
              )}
            </div>
          )}
        </div>
      <div className="panel recovery-panel">
        <h2 className="recovery-heading">Recover results</h2>
        <ReloadJsonSection onLoaded={handleRecoveredResult} />
        <BatchReparseSection onLoaded={handleRecoveredResult} />
      </div>
    </div>

    {showConfirm && (
      <ConfirmModal
        message={
          runState?.status === 'running'
            ? 'A run is currently in progress. Starting a new one will abandon it here, but the batch will keep running on Anthropic\'s side — check your Anthropic console to cancel it manually. Results will be lost. Continue?'
            : 'This will clear the current results. Are you sure you want to start a new run?'
        }
        onConfirm={startRun}
        onCancel={() => setShowConfirm(false)}
      />
    )}
    </>
  )
}
