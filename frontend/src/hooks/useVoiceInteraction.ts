import { useCallback, useRef, useState } from 'react'
import { useAudioRecorder } from './useAudioRecorder'
import { BackendUnavailableError, getAudioJob, runAudioPipeline, type AudioJobResponse } from '../services/api'
import { env } from '../config/environment'
import type { VoiceInteractionState } from '../types/api'

const initialState: VoiceInteractionState = {
  status: 'idle', transcript: null, translation: null, answer: null,
  answerTranslation: null, sources: [], audioUrl: null, error: null, outOfDomain: false,
  answerGrounded: null, llmProvider: null,
}

const delay = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

function recordingFilename(blob: Blob): string {
  const mime = blob.type.toLowerCase()
  if (mime.includes('wav')) return 'recording.wav'
  if (mime.includes('mp4')) return 'recording.mp4'
  if (mime.includes('ogg')) return 'recording.ogg'
  if (mime.includes('mpeg') || mime.includes('mp3')) return 'recording.mp3'
  return 'recording.webm'
}

function apiAudioUrl(path: string): string {
  return new URL(path, `${env.apiBaseUrl}/`).toString()
}

export function useVoiceInteraction() {
  const [state, setState] = useState<VoiceInteractionState>(initialState)
  const recorder = useAudioRecorder()
  const cancelledRef = useRef(false)

  const processAudio = useCallback(async (audio: Blob, filename = recordingFilename(audio)) => {
    cancelledRef.current = false
    setState({ ...initialState, status: 'uploading' })
    try {
      setState((previous) => ({ ...previous, status: 'transcribing' }))
      const result = await runAudioPipeline(audio, filename)
      if (cancelledRef.current) return

      const isGrounded = result.answer_grounded === true
      const sources = isGrounded
        ? (result.sources ?? []).map((source) => ({
            title: source.source_filename || 'Agriculture knowledge base',
            page: source.page_number,
            section: source.section,
            text: source.text,
            relevance: source.faiss_score,
            rerankerScore: source.reranker_score,
            chunkId: source.chunk_id,
          }))
        : []  // Don't expose retrieval candidates for general LLM answers
      const baseState: VoiceInteractionState = {
        status: result.audio_job_id ? 'synthesizing' : 'success',
        transcript: result.transcript_telugu,
        translation: result.translated_query ?? null,
        answer: result.answer_telugu,
        answerTranslation: result.answer_english ?? null,
        sources,
        audioUrl: result.audio_url ? apiAudioUrl(result.audio_url) : null,
        error: null,
        outOfDomain: false,
        answerGrounded: result.answer_grounded ?? null,
        llmProvider: result.llm_provider ?? null,
      }
      setState(baseState)

      if (baseState.audioUrl || !result.audio_job_id) return
      // The API returns the text answer as soon as it is ready; TTS runs on its
      // own worker and is polled separately so the user can read while it runs.
      for (let attempt = 0; attempt < 180; attempt += 1) {
        await delay(2000)
        if (cancelledRef.current) return
        let job: AudioJobResponse
        try {
          job = await getAudioJob(result.audio_job_id)
        } catch {
          setState((previous) => ({ ...previous, status: 'success', error: 'tts_failed' }))
          return
        }
        if (job.status === 'complete' && job.audio_url) {
          setState((previous) => ({ ...previous, status: 'success', audioUrl: apiAudioUrl(job.audio_url!) }))
          return
        }
        if (job.status === 'failed') {
          setState((previous) => ({ ...previous, status: 'success', error: 'tts_failed' }))
          return
        }
      }
      setState((previous) => ({ ...previous, status: 'success', error: 'tts_failed' }))
    } catch (error) {
      if (cancelledRef.current) return
      setState((previous) => ({
        ...previous,
        status: 'error',
        error: error instanceof BackendUnavailableError ? 'backend_unavailable' : 'unknown',
      }))
    }
  }, [])

  const startRecording = useCallback(async () => {
    cancelledRef.current = false
    setState({ ...initialState, status: 'recording' })
    const error = await recorder.start()
    if (error) {
      setState({ ...initialState, status: 'error', error: error === 'permission_denied' ? 'mic_permission_denied' : 'recording_failed' })
    }
  }, [recorder])

  const stopRecordingAndProcess = useCallback(async () => {
    const blob = await recorder.stop()
    if (cancelledRef.current) return
    if (!blob) {
      setState({ ...initialState, status: 'error', error: 'recording_failed' })
      return
    }
    await processAudio(blob)
  }, [processAudio, recorder])

  const cancel = useCallback(() => {
    cancelledRef.current = true
    void recorder.stop()
    recorder.reset()
  }, [recorder])

  const reset = useCallback(() => {
    cancelledRef.current = true
    recorder.reset()
    setState(initialState)
  }, [recorder])

  return {
    state,
    micStream: recorder.stream,
    recordingDuration: recorder.recordingDuration,
    startRecording,
    stopRecordingAndProcess,
    processAudio,
    cancel,
    reset,
  }
}
