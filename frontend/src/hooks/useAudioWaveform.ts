import { useEffect, useRef } from 'react'

/**
 * Draws a live organic waveform of `stream` onto `canvasRef`'s canvas using the
 * native Web Audio API (AnalyserNode) + Canvas 2D. No external waveform library.
 * Fully tears down the AudioContext, analyser and animation frame on cleanup so
 * no microphone stream or audio node is ever leaked.
 */
export function useAudioWaveform(canvasRef: React.RefObject<HTMLCanvasElement | null>, stream: MediaStream | null, active: boolean) {
  const frameRef = useRef<number | null>(null)

  useEffect(() => {
    if (!active || !stream) return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let audioContext: AudioContext | null = null
    let source: MediaStreamAudioSourceNode | null = null
    let analyser: AnalyserNode | null = null

    try {
      const AudioCtx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      audioContext = new AudioCtx()
      analyser = audioContext.createAnalyser()
      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.75
      source = audioContext.createMediaStreamSource(stream)
      source.connect(analyser)
    } catch {
      return
    }

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const dpr = Math.min(window.devicePixelRatio || 1, 2)

    const draw = () => {
      if (!analyser || !ctx) return
      const width = canvas.clientWidth
      const height = canvas.clientHeight
      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr
        canvas.height = height * dpr
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      analyser.getByteTimeDomainData(dataArray)

      ctx.clearRect(0, 0, width, height)
      ctx.lineWidth = 2.5
      ctx.strokeStyle = '#2f8f6f'
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.beginPath()

      const sliceWidth = width / bufferLength
      let x = 0
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0
        const y = (v * height) / 2
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
        x += sliceWidth
      }
      ctx.stroke()

      frameRef.current = requestAnimationFrame(draw)
    }

    frameRef.current = requestAnimationFrame(draw)

    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
      frameRef.current = null
      source?.disconnect()
      analyser?.disconnect()
      if (audioContext && audioContext.state !== 'closed') {
        void audioContext.close()
      }
      const finalCtx = canvas.getContext('2d')
      finalCtx?.clearRect(0, 0, canvas.width, canvas.height)
    }
  }, [stream, active, canvasRef])
}
