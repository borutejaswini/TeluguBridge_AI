import { useRef } from 'react'
import { useAudioWaveform } from '../../hooks/useAudioWaveform'

interface AudioWaveformProps {
  stream: MediaStream | null
  active: boolean
}

/** Only rendered while recording; unmounts (and its hook tears down) otherwise. */
export default function AudioWaveform({ stream, active }: AudioWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useAudioWaveform(canvasRef, stream, active)

  if (!active) return null

  return (
    <div className="waveform" role="status" aria-label="Listening, recording in progress">
      <canvas ref={canvasRef} className="waveform__canvas" />
    </div>
  )
}
