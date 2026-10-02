// Central type definitions for the TeluguBridge AI voice pipeline.
// Keep these in sync with the eventual backend contract (see services/api.ts).

export type PipelineStatus =
  | 'idle'
  | 'recording'
  | 'uploading'
  | 'transcribing'
  | 'translating'
  | 'retrieving'
  | 'generating'
  | 'synthesizing'
  | 'success'
  | 'error'
  | 'out_of_domain'

export type ErrorReason =
  | 'mic_permission_denied'
  | 'recording_failed'
  | 'backend_unavailable'
  | 'transcription_failed'
  | 'translation_failed'
  | 'retrieval_failed'
  | 'generation_failed'
  | 'tts_failed'
  | 'unknown'

export interface SourceReference {
  title: string
  page?: string | number
  section?: string
  text?: string
  relevance?: number
  rerankerScore?: number
  chunkId?: string
  date?: string
  /** True when this citation is demo/sample data, not a real retrieved document. */
  demo?: boolean
}

export interface VoiceQueryResult {
  transcript: string
  translation?: string
  answer?: string
  answerTranslation?: string
  audioUrl?: string
  sources?: SourceReference[]
  outOfDomain?: boolean
}

export interface TranscribeResponse {
  transcript: string
}

export interface TranslateResponse {
  translation: string
}

export interface QueryResponse {
  answer: string
  answerTranslation?: string
  sources: SourceReference[]
  outOfDomain: boolean
}

export interface SynthesizeResponse {
  audioUrl: string
}

/** Discriminated-union-friendly snapshot of the whole interaction, owned by useVoiceInteraction. */
export interface VoiceInteractionState {
  status: PipelineStatus
  transcript: string | null
  translation: string | null
  answer: string | null
  answerTranslation: string | null
  sources: SourceReference[]
  audioUrl: string | null
  error: ErrorReason | null
  outOfDomain: boolean
  answerGrounded: boolean | null
  llmProvider: string | null
}
