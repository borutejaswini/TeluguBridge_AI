import { useCallback, useEffect, useRef, useState } from 'react'

export type RecorderError = 'permission_denied' | 'unsupported' | 'start_failed'

interface UseAudioRecorderResult {
  isRecording: boolean
  recordingDuration: number
  error: RecorderError | null
  /** Live microphone stream, exposed only while recording, for waveform visualization. */
  stream: MediaStream | null
  /** Returns the error synchronously (as a return value, not just state) so callers don't read stale state after the await. */
  start: () => Promise<RecorderError | null>
  stop: () => Promise<Blob | null>
  reset: () => void
}

const CANDIDATE_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
  'audio/ogg;codecs=opus',
  'audio/wav',
]

function pickSupportedMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') return undefined
  return CANDIDATE_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type))
}

export function useAudioRecorder(): UseAudioRecorderResult {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [error, setError] = useState<RecorderError | null>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<number | null>(null)

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const cleanupStream = useCallback(() => {
    clearTimer()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setStream(null)
  }, [clearTimer])

  const start = useCallback(async (): Promise<RecorderError | null> => {
    setError(null)
    chunksRef.current = []
    setRecordingDuration(0)
    clearTimer()

    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('unsupported')
      return 'unsupported'
    }

    let micStream: MediaStream
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
    } catch {
      setError('permission_denied')
      return 'permission_denied'
    }

    try {
      const mimeType = pickSupportedMimeType()
      const recorder = mimeType ? new MediaRecorder(micStream, { mimeType }) : new MediaRecorder(micStream)
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      mediaRecorderRef.current = recorder
      streamRef.current = micStream
      setStream(micStream)

      // Collect audio chunks every 100ms so stopping is immediate without buffer flushing delay
      recorder.start(100)
      setIsRecording(true)

      // Start duration counter
      const startMs = Date.now()
      timerRef.current = window.setInterval(() => {
        setRecordingDuration(Math.floor((Date.now() - startMs) / 1000))
      }, 500)

      return null
    } catch {
      micStream.getTracks().forEach((track) => track.stop())
      setError('start_failed')
      return 'start_failed'
    }
  }, [clearTimer])

  const stop = useCallback((): Promise<Blob | null> => {
    clearTimer()
    return new Promise((resolve) => {
      const recorder = mediaRecorderRef.current
      if (!recorder || recorder.state === 'inactive') {
        cleanupStream()
        setIsRecording(false)
        resolve(null)
        return
      }
      recorder.onstop = () => {
        const blob = chunksRef.current.length > 0
          ? new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
          : null
        chunksRef.current = []
        cleanupStream()
        setIsRecording(false)
        resolve(blob)
      }
      recorder.stop()
    })
  }, [cleanupStream, clearTimer])

  const reset = useCallback(() => {
    clearTimer()
    mediaRecorderRef.current = null
    chunksRef.current = []
    cleanupStream()
    setIsRecording(false)
    setRecordingDuration(0)
    setError(null)
  }, [cleanupStream, clearTimer])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearTimer()
      cleanupStream()
    }
  }, [clearTimer, cleanupStream])

  return { isRecording, recordingDuration, error, stream, start, stop, reset }
}
