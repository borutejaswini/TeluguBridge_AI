import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import Header from './components/layout/Header'
import Footer from './components/layout/Footer'
import Hero from './components/hero/Hero'
import PipelineProgress from './components/pipeline/PipelineProgress'
import AnswerCard from './components/answer/AnswerCard'
import HowItWorks from './components/sections/HowItWorks'
import KnowledgeBase from './components/sections/KnowledgeBase'
import Evaluation from './components/sections/Evaluation'
import Limitations from './components/sections/Limitations'
import Privacy from './components/sections/Privacy'
import About from './components/sections/About'
import { useVoiceInteraction } from './hooks/useVoiceInteraction'
import { env } from './config/environment'
import './App.css'

function App() {
  const { state, micStream, recordingDuration, startRecording, stopRecordingAndProcess, processAudio, reset } = useVoiceInteraction()
  const [upload, setUpload] = useState<File | null>(null)
  const interactionRef = useRef<HTMLElement>(null)

  // Automatically scroll to the interaction/answer area when the user finishes speaking or audio starts processing
  useEffect(() => {
    if (['uploading', 'transcribing', 'translating', 'retrieving', 'generating', 'synthesizing', 'success'].includes(state.status)) {
      interactionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [state.status])

  const handleStart = useCallback(() => {
    void startRecording()
  }, [startRecording])

  const handleStop = useCallback(() => {
    void stopRecordingAndProcess()
  }, [stopRecordingAndProcess])

  const handleRetryPermission = useCallback(() => {
    reset()
    void startRecording()
  }, [reset, startRecording])

  const handleUpload = useCallback((event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (upload) void processAudio(upload, upload.name)
  }, [processAudio, upload])

  const isBusy = ['uploading', 'transcribing', 'translating', 'retrieving', 'generating', 'synthesizing'].includes(state.status)

  return (
    <>
      {env.demoMode && (
        <div className="demo-strip">
          <span className="demo-badge">● Demo Mode</span>
          <span>No backend connected — responses are simulated for demonstration.</span>
        </div>
      )}

      <Header onVoiceCtaClick={handleStart} />

      <main>
        <Hero
          status={state.status}
          micStream={micStream}
          recordingDuration={recordingDuration}
          recorderError={
            state.error === 'mic_permission_denied' ? 'permission_denied' : state.error === 'recording_failed' ? 'start_failed' : null
          }
          onStart={handleStart}
          onStop={handleStop}
          onRetryPermission={handleRetryPermission}
        />

        <section className="container upload-recording" aria-label="Upload an audio recording">
          <form className="upload-recording__form" onSubmit={handleUpload}>
            <label htmlFor="recording-file">లేదా రికార్డింగ్‌ను ఎంచుకోండి <span>Upload an existing recording</span></label>
            <input
              id="recording-file"
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm,.mp4"
              onChange={(event) => setUpload(event.target.files?.[0] ?? null)}
              disabled={isBusy}
            />
            <button type="submit" className="btn btn--primary" disabled={!upload || isBusy}>
              {isBusy ? 'ప్రాసెస్ అవుతోంది…' : 'రికార్డింగ్‌ను ప్రాసెస్ చేయండి'}
            </button>
            <p>WAV, MP3, M4A, FLAC, OGG, WebM లేదా MP4 · గరిష్ఠం 50 MB</p>
          </form>
        </section>

        <section ref={interactionRef} className="interaction-panel container" aria-live="polite">
          <PipelineProgress status={state.status} />
          <AnswerCard state={state} onReset={reset} onRetryPermission={handleRetryPermission} />
        </section>

        <HowItWorks />
        <KnowledgeBase />
        <Evaluation />
        <Limitations />
        <Privacy />
        <About />
      </main>

      <Footer />
    </>
  )
}

export default App
