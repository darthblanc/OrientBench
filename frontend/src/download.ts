import type { RunResult } from './types'

export function downloadResult(result: RunResult): void {
  const ts = result.timestamp.replace(/[:.]/g, '-').slice(0, 19)
  const filename = `${result.dataset}_${result.model}_${ts}.json`
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
