import { useState } from 'react'
import AudioPlayer from './AudioPlayer'
import WordByWordReader from './WordByWordReader'
import ChunksDrawer from './ChunksDrawer'
import type { VoiceInteractionState } from '../../types/api'
import { env } from '../../config/environment'

interface AnswerCardProps {
  state: VoiceInteractionState
  onReset: () => void
  onRetryPermission?: () => void
}

const ERROR_COPY: Record<string, string> = {
  mic_permission_denied: 'మైక్రోఫోన్ అనుమతి అవసరం. మీ బ్రౌజర్‌లో అనుమతించి మళ్లీ ప్రయత్నించండి.',
  recording_failed: 'రికార్డింగ్ ప్రారంభించడం సాధ్యం కాలేదు.',
  backend_unavailable: 'సేవ అందుబాటులో లేదు. దయచేసి మళ్లీ ప్రయత్నించండి లేదా డెమో మోడ్‌ను ఉపయోగించండి.',
  transcription_failed: 'మీ గొంతు స్పష్టంగా అర్థం కాలేదు.',
  translation_failed: 'అనువాదం పూర్తి కాలేదు.',
  retrieval_failed: 'సంబంధిత సమాచారం లభించలేదు.',
  generation_failed: 'సమాధానం రూపొందించడం సాధ్యం కాలేదు.',
  unknown: 'ఏదో సమస్య వచ్చింది. దయచేసి మళ్లీ ప్రయత్నించండి.',
}

export default function AnswerCard({ state, onReset }: AnswerCardProps) {
  const [showEnglish, setShowEnglish] = useState(false)
  const [showChunksDrawer, setShowChunksDrawer] = useState(false)

  // Audio synchronization state for word-by-word reading
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [audioEnded, setAudioEnded] = useState(false)

  const handleTimeUpdate = (cur: number, dur: number) => {
    setCurrentTime(cur)
    setDuration(dur)
    setAudioEnded(false)
  }

  const handlePlayStateChange = (playing: boolean) => {
    setIsPlaying(playing)
    if (playing) {
      setAudioEnded(false)
    }
  }

  const handleAudioEnded = () => {
    setIsPlaying(false)
    setAudioEnded(true)
  }

  if (state.status === 'idle' || state.status === 'recording') return null

  if (state.status === 'error' && state.error) {
    return (
      <div className="panel-card panel-card--error" role="alert">
        <p>{ERROR_COPY[state.error] ?? ERROR_COPY.unknown}</p>
        <button type="button" className="btn btn--secondary btn--compact" onClick={onReset}>
          మరొక ప్రశ్న అడగండి
        </button>
      </div>
    )
  }

  if (state.status === 'out_of_domain') {
    return (
      <div className="panel-card panel-card--muted">
        <p className="panel-card__title">ఈ ప్రశ్నకు సంబంధించిన సమాచారం ప్రస్తుతం మా జ్ఞాన ఆధారంలో లేదు.</p>
        <p>వ్యవసాయం లేదా సంబంధిత సమాచారంపై ప్రశ్న అడగండి.</p>
        <p className="panel-card__note">మా వద్ద ధృవీకరించిన సమాచారం లేనప్పుడు సమాధానం ఇవ్వకుండా ఉండటానికి ఈ వ్యవస్థ రూపొందించబడింది.</p>
        <button type="button" className="btn btn--secondary btn--compact" onClick={onReset}>
          మరొక ప్రశ్న అడగండి
        </button>
      </div>
    )
  }

  return (
    <div className="answer-stack">
      {/* User's spoken/submitted question */}
      {state.transcript && (
        <div className="panel-card">
          <p className="panel-card__label">మీరు అడిగింది (Your Question)</p>
          <p className="panel-card__telugu">{state.transcript}</p>
          {state.translation && (
            <>
              <button type="button" className="link-toggle" onClick={() => setShowEnglish((v) => !v)}>
                {showEnglish ? 'Hide English interpretation' : 'English interpretation'}
              </button>
              {showEnglish && <p className="panel-card__english">{state.translation}</p>}
            </>
          )}
        </div>
      )}

      {/* Loading states */}
      {(state.status === 'transcribing' ||
        state.status === 'translating' ||
        state.status === 'retrieving' ||
        state.status === 'generating') && (
        <div className="panel-card panel-card--muted" aria-live="polite">
          <p>సమాధానం సిద్ధమవుతోంది…</p>
        </div>
      )}

      {/* Answer panel */}
      {(state.status === 'success' || state.status === 'synthesizing') && state.answer && (
        <div className="panel-card panel-card--answer">
          {/* Top header row with label & Side Chunks Button — only for grounded KB answers */}
          <div className="panel-card__header-row">
            <p className="panel-card__label">సమాధానం (Answer)</p>
            {state.answerGrounded === true && state.sources.length > 0 && (
              <button
                type="button"
                className="btn-chunks-side"
                onClick={() => setShowChunksDrawer(true)}
                title="View retrieved document chunks"
              >
                <span className="btn-chunks-side__icon">📑</span>
                <span className="btn-chunks-side__text">జ్ఞాన భాగాలు (Chunks)</span>
                <span className="btn-chunks-side__count">{state.sources.length}</span>
              </button>
            )}
          </div>

          {/* Grounded vs General LLM Advisory Flag Banner */}
          {state.answerGrounded === true ? (
            <div className="panel-badge-flag panel-badge-flag--grounded" role="status">
              <span className="badge-flag__icon" aria-hidden="true">📑</span>
              <div className="badge-flag__content">
                <span className="badge-flag__title">ధృవీకరించిన వ్యవసాయ సమాచారం (Verified Knowledge Base)</span>
                <span className="badge-flag__desc">ఈ సమాధానం అధికారిక వ్యవసాయ డాక్యుమెంట్ (paddy.pdf) ఆధారంగా ధృవీకరించబడింది.</span>
              </div>
            </div>
          ) : (
            <div className="panel-badge-flag panel-badge-flag--general" role="status">
              <span className="badge-flag__icon" aria-hidden="true">🤖</span>
              <div className="badge-flag__content">
                <span className="badge-flag__title">సాధారణ AI సమాధానం (General LLM Advisory)</span>
                <span className="badge-flag__desc">ఈ నిర్దిష్ట అంశం స్థానిక డాక్యుమెంట్‌లో లేనందున, AI సాధారణ విజ్ఞానం ద్వారా మార్గదర్శకాలు ఇవ్వబడ్డాయి. రసాయన మోతాదులను స్థానిక వ్యవసాయ అధికారితో నిర్ధారించుకోండి.</span>
              </div>
            </div>
          )}

          {/* Word-by-word animated Telugu answer display synchronized with audio */}
          <WordByWordReader
            text={state.answer}
            isPlaying={isPlaying}
            currentTime={currentTime}
            duration={duration}
            audioEnded={audioEnded}
          />

          {/* English translation collapsible */}
          {state.answerTranslation && (
            <details className="answer-collapsible" open={false}>
              <summary>English interpretation</summary>
              <div className="answer-collapsible__body">
                <p>{state.answerTranslation}</p>
              </div>
            </details>
          )}

          {/* Synthesizing status indicator */}
          {state.status === 'synthesizing' && (
            <div className="audio-synthesizing-pill" role="status">
              <span className="audio-synthesizing-pill__spinner" aria-hidden="true" />
              <span>తెలుగు వాయిస్ సిద్ధమవుతోంది… సిద్ధం కాగానే ఆడియో స్వయంచాలకంగా ప్లే అవుతుంది</span>
            </div>
          )}

          {/* Single stable Audio Player with automatic playback */}
          <AudioPlayer
            audioUrl={state.audioUrl}
            fallbackText={env.demoMode ? state.answer : null}
            ttsFailed={state.error === 'tts_failed'}
            onTimeUpdate={handleTimeUpdate}
            onPlayStateChange={handlePlayStateChange}
            onAudioEnded={handleAudioEnded}
          />

          {/* Bottom chunks notice — only for grounded KB answers */}
          {state.answerGrounded === true && state.sources.length > 0 && (
            <div className="chunks-hidden-notice">
              <button
                type="button"
                className="chunks-hidden-notice__btn"
                onClick={() => setShowChunksDrawer(true)}
              >
                📑 ఈ సమాధానం కోసం {state.sources.length} ఆధార భాగాలు (Chunks) ఉపయోగించబడ్డాయి. చూడటానికి ఇక్కడ క్లిక్ చేయండి.
              </button>
            </div>
          )}

          <button type="button" className="btn btn--secondary btn--compact" onClick={onReset}>
            మరొక ప్రశ్న అడగండి
          </button>
        </div>
      )}

      {/* Side Drawer for Knowledge Chunks (hidden by default) */}
      <ChunksDrawer
        isOpen={showChunksDrawer}
        onClose={() => setShowChunksDrawer(false)}
        sources={state.sources}
      />
    </div>
  )
}
