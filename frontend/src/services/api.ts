import { env } from '../config/environment'

export interface PipelineResponse {
  status: string
  audio_job_id?: string | null
  audio_url?: string | null
  transcript_telugu: string
  translated_query?: string
  answer_telugu: string
  answer_english?: string
  answer_grounded?: boolean
  llm_provider?: string
  sources?: Array<{
    source_filename?: string
    page_number?: number | string
    section?: string
    text?: string
    faiss_score?: number
    reranker_score?: number
    chunk_id?: string
  }>
}

export interface AudioJobResponse {
  status: 'queued' | 'running' | 'complete' | 'failed' | string
  audio_url?: string
  error?: string
}

export class BackendUnavailableError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'BackendUnavailableError'
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = data?.error?.message || data?.detail || `Backend returned ${response.status}`
    throw new BackendUnavailableError(message)
  }
  return data as T
}

export async function runAudioPipeline(file: Blob, filename: string): Promise<PipelineResponse> {
  const form = new FormData()
  form.append('file', file, filename)
  let response: Response
  try {
    response = await fetch(`${env.apiBaseUrl}/api/pipeline/run`, { method: 'POST', body: form })
  } catch {
    throw new BackendUnavailableError('Could not connect to the TeluguBridge backend.')
  }
  return parseResponse<PipelineResponse>(response)
}

export async function getAudioJob(jobId: string): Promise<AudioJobResponse> {
  let response: Response
  try {
    response = await fetch(`${env.apiBaseUrl}/api/audio-jobs/${encodeURIComponent(jobId)}`)
  } catch {
    throw new BackendUnavailableError('Could not check the audio generation status.')
  }
  return parseResponse<AudioJobResponse>(response)
}
