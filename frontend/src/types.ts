export type TaskResult = {
  task_kind: string
  question: string
  ground_truth: string
  row_answer: string
  col_answer: string
  row_correct: boolean
  col_correct: boolean
}

export type RunResult = {
  dataset: string
  model: string
  timestamp: string
  id_col: string
  mode?: string
  batch_id?: string
  results: TaskResult[]
}

export type RunState = {
  status: 'running' | 'done' | 'failed'
  error?: string
  result?: RunResult
}
