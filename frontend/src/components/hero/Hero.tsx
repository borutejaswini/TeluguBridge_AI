import EnergyOrb from '../effects/EnergyOrb'
import CondensationBackground from '../effects/CondensationBackground'
import AudioWaveform from './AudioWaveform'
import type { RecorderError } from '../../hooks/useAudioRecorder'
import type { PipelineStatus } from '../../types/api'

interface HeroProps {
  status: PipelineStatus
  micStream: MediaStream | null
  recorderError: RecorderError | null
  recordingDuration?: number
  onStart: () => void
  onStop: () => void
  onRetryPermission: () => void
}

const ORB_LABEL: Partial<Record<PipelineStatus, { te: string; en: string }>> = {
  idle: { te: 'మాట్లాడండి', en: 'Speak' },
  recording: { te: 'వింటున్నాను…', en: 'Listening…' },
  uploading: { te: 'అర్థం చేసుకుంటున్నాను…', en: 'Understanding…' },
  transcribing: { te: 'అర్థం చేసుకుంటున్నాను…', en: 'Understanding…' },
  translating: { te: 'అనువదిస్తున్నాను…', en: 'Translating…' },
  retrieving: { te: 'సమాచారం వెతుకుతున్నాను…', en: 'Retrieving…' },
  generating: { te: 'సమాధానం సిద్ధమవుతోంది…', en: 'Preparing answer…' },
  synthesizing: { te: 'సమాధానం సిద్ధమవుతోంది…', en: 'Preparing answer…' },
  success: { te: 'సిద్ధంగా ఉంది', en: 'Ready' },
  error: { te: 'మళ్లీ ప్రయత్నించండి', en: 'Try again' },
  out_of_domain: { te: 'మళ్లీ అడగండి', en: 'Ask again' },
}

const RESTARTABLE: PipelineStatus[] = ['idle', 'success', 'error', 'out_of_domain']

function formatDuration(seconds: number = 0): string {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

export default function Hero({
  status,
  micStream,
  recorderError,
  recordingDuration = 0,
  onStart,
  onStop,
  onRetryPermission,
}: HeroProps) {
  const label = ORB_LABEL[status] ?? ORB_LABEL.idle!
  const isBusy = ['uploading', 'transcribing', 'translating', 'retrieving', 'generating', 'synthesizing'].includes(status)

  const handleOrbActivate = () => {
    if (status === 'recording') {
      onStop()
    } else if (RESTARTABLE.includes(status)) {
      onStart()
    }
  }

  return (
    <section id="top" className="hero">
      <div className="hero__atmosphere" aria-hidden="true">
        <CondensationBackground />
      </div>

      <div className="container hero__inner">
        <p className="hero__eyebrow">TELUGUBRIDGE AI</p>
        <h1 className="hero__title">మీ ప్రశ్నను తెలుగులో అడగండి.</h1>
        <p className="hero__title-en">Ask your question in Telugu.</p>
        <p className="hero__subtitle-te">
          మీ మాటలను అర్థం చేసుకుని, సంబంధిత సమాచారాన్ని వెతికి, సమాధానాన్ని మీకు తెలుగులో అందిస్తుంది.
        </p>
        <p className="hero__subtitle-en">Speak in Telugu, get relevant, source-grounded answers.</p>

        <div className="hero__orb-stage">
          <div className="hero__orb-glow" aria-hidden="true" />

          <div className="hero__float-card hero__float-card--tl" aria-hidden="true">
            <p className="hero__float-card-te">తెలుగు మాటలు</p>
            <p className="hero__float-card-en">Telugu speech input</p>
          </div>
          <div className="hero__float-card hero__float-card--bl" aria-hidden="true">
            <p className="hero__float-card-te">అనువాదం</p>
            <p className="hero__float-card-en">Translation</p>
          </div>
          <div className="hero__float-card hero__float-card--tr" aria-hidden="true">
            <p className="hero__float-card-te">జ్ఞాన ఆధారం</p>
            <p className="hero__float-card-en">Knowledge retrieval</p>
          </div>
          <div className="hero__float-card hero__float-card--br" aria-hidden="true">
            <p className="hero__float-card-te">తెలుగు సమాధానం</p>
            <p className="hero__float-card-en">Telugu response</p>
          </div>

          <EnergyOrb status={status} onActivate={handleOrbActivate} label={label} />
        </div>

        <AudioWaveform stream={micStream} active={status === 'recording'} />

        {/* Dedicated Voice Control Bar */}
        <div className="hero__voice-controls">
          {status === 'recording' ? (
            <div className="voice-controls__recording-group">
              <div className="recording-status-badge">
                <span className="recording-status-badge__dot" aria-hidden="true" />
                <span className="recording-status-badge__text">వింటున్నాను…</span>
                <span className="recording-status-badge__timer">{formatDuration(recordingDuration)}</span>
              </div>
              <button
                type="button"
                className="btn-mic-action btn-mic-action--stop"
                onClick={onStop}
                title="రికార్డింగ్ ఆపి సమాధానం పొందండి"
              >
                <span className="btn-mic-action__icon" aria-hidden="true">⏹️</span>
                <span className="btn-mic-action__content">
                  <span className="btn-mic-action__primary">ఆపండి &amp; పంపండి</span>
                  <span className="btn-mic-action__secondary">Stop &amp; Submit</span>
                </span>
              </button>
            </div>
          ) : isBusy ? (
            <div className="voice-controls__busy-group">
              <button type="button" className="btn-mic-action btn-mic-action--busy" disabled>
                <span className="btn-mic-action__spinner" aria-hidden="true" />
                <span className="btn-mic-action__content">
                  <span className="btn-mic-action__primary">{label.te}</span>
                  <span className="btn-mic-action__secondary">{label.en}</span>
                </span>
              </button>
            </div>
          ) : (
            <div className="voice-controls__idle-group">
              <button
                type="button"
                className="btn-mic-action btn-mic-action--speak"
                onClick={onStart}
                title="మాట్లాడటం ప్రారంభించండి"
              >
                <span className="btn-mic-action__icon" aria-hidden="true">🎙️</span>
                <span className="btn-mic-action__content">
                  <span className="btn-mic-action__primary">మాట్లాడండి</span>
                  <span className="btn-mic-action__secondary">Click to Speak</span>
                </span>
              </button>
            </div>
          )}
        </div>

        <p className="hero__helper">
          {status === 'recording' ? (
            <>
              <span className="hero__helper-te">మీ ప్రశ్న మాట్లాడటం పూర్తయ్యాక <strong>'ఆపండి &amp; పంపండి'</strong> నొక్కండి</span>
              <span className="hero__helper-en">Click 'Stop &amp; Submit' when you finish asking</span>
            </>
          ) : isBusy ? (
            <>
              <span className="hero__helper-te">మీ ప్రశ్నకు సమాధానం సిద్ధమవుతోంది, దయచేసి వేచి ఉండండి…</span>
              <span className="hero__helper-en">Processing your query, please wait…</span>
            </>
          ) : (
            <>
              <span className="hero__helper-te"><strong>'మాట్లాడండి'</strong> బటన్ లేదా ఆర్బ్ నొక్కి ప్రశ్నను అడగండి</span>
              <span className="hero__helper-en">Click button or orb to start speaking</span>
            </>
          )}
        </p>

        {recorderError === 'permission_denied' && (
          <div className="inline-alert" role="alert">
            <p className="inline-alert__title">మైక్రోఫోన్ అనుమతి అవసరం.</p>
            <p>మీ బ్రౌజర్‌లో microphone permission‌ను అనుమతించి మళ్లీ ప్రయత్నించండి.</p>
            <button type="button" className="btn btn--secondary btn--compact" onClick={onRetryPermission}>
              మళ్లీ ప్రయత్నించండి
            </button>
          </div>
        )}
        {recorderError === 'unsupported' && (
          <div className="inline-alert" role="alert">
            <p>ఈ బ్రౌజర్‌లో వాయిస్ రికార్డింగ్ మద్దతు లేదు. దయచేసి కొత్త బ్రౌజర్‌ను ఉపయోగించండి.</p>
          </div>
        )}

        {status === 'idle' && (
          <p className="hero__example">
            <span className="hero__example-label">ఉదాహరణ:</span>{' '}
            <span className="hero__example-te">"వరి పంటలో పురుగులను ఎలా నియంత్రించాలి?"</span>
            <br />
            <span className="hero__example-en">Example: "How can I manage pests in a rice crop?"</span>
          </p>
        )}
      </div>
    </section>
  )
}
