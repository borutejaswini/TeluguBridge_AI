import type { PipelineStatus } from '../../types/api'

interface PipelineProgressProps {
  status: PipelineStatus
}

const STEPS: Array<{ key: PipelineStatus[]; label: string; icon: string }> = [
  { key: ['recording', 'uploading'], label: 'తెలుగు మాట', icon: '🎙' },
  { key: ['transcribing'], label: 'Telugu వచనం', icon: '📝' },
  { key: ['translating'], label: 'అనువాదం', icon: '🌐' },
  { key: ['retrieving'], label: 'జ్ఞాన అన్వేషణ', icon: '🔎' },
  { key: ['generating'], label: 'సమాధాన రూపకల్పన', icon: '🤖' },
  { key: ['synthesizing', 'success'], label: 'తెలుగు స్పందన', icon: '🔊' },
]

const ORDER: PipelineStatus[] = ['recording', 'uploading', 'transcribing', 'translating', 'retrieving', 'generating', 'synthesizing', 'success']

export default function PipelineProgress({ status }: PipelineProgressProps) {
  if (status === 'idle' || status === 'error' || status === 'out_of_domain') return null

  const currentIndex = ORDER.indexOf(status)

  return (
    <ol className="pipeline" aria-label="Processing steps">
      {STEPS.map((step) => {
        const stepMaxIndex = Math.max(...step.key.map((k) => ORDER.indexOf(k)))
        const isActive = step.key.includes(status)
        const isDone = currentIndex > stepMaxIndex

        return (
          <li
            key={step.label}
            className={`pipeline__step${isActive ? ' pipeline__step--active' : ''}${isDone ? ' pipeline__step--done' : ''}`}
          >
            <span className="pipeline__icon" aria-hidden="true">
              {step.icon}
            </span>
            <span className="pipeline__label">{step.label}</span>
          </li>
        )
      })}
    </ol>
  )
}
