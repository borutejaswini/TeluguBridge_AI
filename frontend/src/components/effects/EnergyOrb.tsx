import { useMemo, type KeyboardEvent } from 'react'
import { GlobeCollection } from '@designcodeio/threeui'
import '@designcodeio/threeui/style.css'
import type { PipelineStatus } from '../../types/api'
import { useReducedMotion } from '../../hooks/useReducedMotion'

// Official ThreeUI adapter. Renders the authored GlobeCollection "energy-orb"
// variant unmodified. We only ever set the exposed props (hue, saturation, glow,
// brightness, speed, opacity) to harmonize the orb with the site's emerald/teal/
// cyan palette and to express pipeline state — the shader internals are never
// touched, per the integration rule in the master spec.
//
// The orb is ALSO the primary interaction target: it's rendered as a
// role="button" wrapper so clicking/tapping/pressing Enter or Space on it
// starts or stops recording directly, with no separate microphone button
// competing for attention.

interface EnergyOrbProps {
  status: PipelineStatus
  onActivate: () => void
  label: { te: string; en: string }
}

const BUSY_STATUSES: PipelineStatus[] = ['uploading', 'transcribing', 'translating', 'retrieving', 'generating', 'synthesizing']

// Every state stays in the emerald/teal/cyan family — nothing here ever drifts
// toward yellow/brown/gold, so the orb reads as an energy core, not a moon.
// Error/out-of-domain use a restrained warm accent, kept dim and desaturated
// rather than turning the whole orb red or gold.
function paramsForStatus(status: PipelineStatus) {
  switch (status) {
    case 'idle':
      return { speed: 0.45, glow: 0.9, brightness: 0.95, saturation: 0.85, hue: 172 }
    case 'recording':
      return { speed: 1.4, glow: 1.35, brightness: 1.2, saturation: 1, hue: 178 }
    case 'uploading':
      return { speed: 0.9, glow: 1.0, brightness: 1.0, saturation: 0.9, hue: 175 }
    case 'transcribing':
    case 'translating':
      return { speed: 0.75, glow: 0.95, brightness: 0.98, saturation: 0.88, hue: 176 }
    case 'retrieving':
    case 'generating':
      return { speed: 0.55, glow: 0.9, brightness: 0.95, saturation: 0.85, hue: 180 }
    case 'synthesizing':
      return { speed: 0.65, glow: 1.0, brightness: 1.02, saturation: 0.9, hue: 174 }
    case 'success':
      return { speed: 0.5, glow: 1.2, brightness: 1.1, saturation: 0.95, hue: 168 }
    case 'error':
      return { speed: 0.4, glow: 0.55, brightness: 0.7, saturation: 0.3, hue: 30 }
    case 'out_of_domain':
      return { speed: 0.45, glow: 0.6, brightness: 0.75, saturation: 0.35, hue: 34 }
    default:
      return { speed: 0.45, glow: 0.9, brightness: 0.95, saturation: 0.85, hue: 172 }
  }
}

export default function EnergyOrb({ status, onActivate, label }: EnergyOrbProps) {
  const reducedMotion = useReducedMotion()
  const params = useMemo(() => paramsForStatus(status), [status])
  const speed = reducedMotion ? Math.min(params.speed, 0.2) : params.speed
  const isBusy = BUSY_STATUSES.includes(status)
  const isRecording = status === 'recording'

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (isBusy) return
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onActivate()
    }
  }

  return (
    <div
      className={`energy-orb-shell${isRecording ? ' energy-orb-shell--recording' : ''}${isBusy ? ' energy-orb-shell--busy' : ''}`}
      role="button"
      tabIndex={isBusy ? -1 : 0}
      aria-pressed={isRecording}
      aria-disabled={isBusy}
      aria-label={label ? `${label.en} - ${label.te}` : (isRecording ? 'Stop recording' : 'Start voice recording')}
      onClick={() => !isBusy && onActivate()}
      onKeyDown={handleKeyDown}
    >
      <div className="energy-orb-canvas-wrap">
        <GlobeCollection
          variant="energy-orb"
          speed={speed}
          scale={1}
          smokeScale={0.85}
          smokeStrength={0.7}
          smokeSpeed={speed}
          hue={params.hue}
          saturation={params.saturation}
          glow={params.glow}
          starDensity={0.5}
          starSpeed={reducedMotion ? 0.1 : 0.45}
          starSize={0.85}
          brightness={params.brightness}
          opacity={1}
        />
      </div>
      <div className="energy-orb-label" aria-hidden="true">
        <span className="energy-orb-label__icon">🎙️</span>
        <span className="energy-orb-label__te">{label.te}</span>
        <span className="energy-orb-label__en">{label.en}</span>
      </div>
    </div>
  )
}
