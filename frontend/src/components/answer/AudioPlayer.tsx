import { useEffect, useRef, useState } from 'react'

const DEFAULT_PLAYBACK_RATE = 1.25

interface AudioPlayerProps {
  audioUrl: string | null
  /** Text to speak via browser SpeechSynthesis when no real audioUrl exists (demo mode only). */
  fallbackText?: string | null
  ttsFailed?: boolean
  onTimeUpdate?: (currentTime: number, duration: number) => void
  onPlayStateChange?: (isPlaying: boolean) => void
  onAudioEnded?: () => void
}

export default function AudioPlayer({
  audioUrl,
  fallbackText,
  ttsFailed,
  onTimeUpdate,
  onPlayStateChange,
  onAudioEnded,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [metaReady, setMetaReady] = useState(false)
  const playAttemptedRef = useRef(false)
  const canUseSpeechSynthesis = typeof window !== 'undefined' && 'speechSynthesis' in window

  // Reset state whenever the audio URL changes (new answer)
  useEffect(() => {
    setIsPlaying(false)
    setMetaReady(false)
    playAttemptedRef.current = false
    onPlayStateChange?.(false)
  }, [audioUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-play at 1.25x ONCE after metadata is loaded
  useEffect(() => {
    if (!audioUrl || !metaReady || playAttemptedRef.current) return
    const el = audioRef.current
    if (!el) return
    playAttemptedRef.current = true
    // Set playback rate BEFORE play so it's in effect from the first frame
    el.playbackRate = DEFAULT_PLAYBACK_RATE
    const promise = el.play()
    if (promise !== undefined) {
      promise
        .then(() => {
          setIsPlaying(true)
          onPlayStateChange?.(true)
        })
        .catch((err) => {
          // Browser autoplay policy blocked — user can click play
          console.log('Autoplay deferred by browser policy:', err)
          setIsPlaying(false)
          onPlayStateChange?.(false)
        })
    }
  }, [audioUrl, metaReady]) // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup speech synthesis on unmount
  useEffect(() => {
    return () => {
      window.speechSynthesis?.cancel()
    }
  }, [])

  if (ttsFailed) {
    return <p className="audio-player__note">ఆడియో సృష్టించడం సాధ్యం కాలేదు. దయచేసి పైన ఉన్న సమాధానాన్ని చదవండి.</p>
  }

  // --- Speech synthesis fallback (demo mode only) ---
  const handleSpeechPlay = () => {
    if (!fallbackText || !canUseSpeechSynthesis) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(fallbackText)
    utterance.lang = 'te-IN'
    utterance.rate = DEFAULT_PLAYBACK_RATE  // match 1.25x for speech synthesis too
    utterance.onend = () => {
      setIsPlaying(false)
      onPlayStateChange?.(false)
      onAudioEnded?.()
    }
    utterance.onerror = () => {
      setIsPlaying(false)
      onPlayStateChange?.(false)
    }
    window.speechSynthesis.speak(utterance)
    setIsPlaying(true)
    onPlayStateChange?.(true)
  }

  const handleSpeechPause = () => {
    window.speechSynthesis?.cancel()
    setIsPlaying(false)
    onPlayStateChange?.(false)
  }

  if (!audioUrl && !fallbackText) return null

  return (
    <div className="audio-player">
      {audioUrl ? (
        <audio
          ref={audioRef}
          src={audioUrl}
          controls
          preload="auto"
          // NO autoPlay attribute — playback triggered via useEffect after metadata loads
          onLoadedMetadata={(e) => {
            // Set 1.25x immediately when metadata is known — before autoplay fires
            e.currentTarget.playbackRate = DEFAULT_PLAYBACK_RATE
            setMetaReady(true)
          }}
          onRateChange={(e) => {
            // If user changes the speed via the native controls, keep word-sync working.
            // The effective speed for sync purposes is: currentTime / duration (unchanged)
            // — HTML audio currentTime is always file-position, not wall-clock.
            e.currentTarget.playbackRate = e.currentTarget.playbackRate // no-op, just let it be
          }}
          onTimeUpdate={(e) => {
            const el = e.currentTarget
            // currentTime = position in the audio file (seconds), always in file-time regardless of playbackRate.
            // duration = total file length (seconds). The ratio currentTime/duration is correct at any speed.
            if (el.duration && isFinite(el.duration)) {
              onTimeUpdate?.(el.currentTime, el.duration)
            }
          }}
          onPlay={() => {
            // Re-assert 1.25x in case browser resets rate on play
            if (audioRef.current) audioRef.current.playbackRate = DEFAULT_PLAYBACK_RATE
            setIsPlaying(true)
            onPlayStateChange?.(true)
          }}
          onPause={() => {
            setIsPlaying(false)
            onPlayStateChange?.(false)
          }}
          onEnded={() => {
            setIsPlaying(false)
            onPlayStateChange?.(false)
            onAudioEnded?.()
          }}
          onError={() => {
            setIsPlaying(false)
            onPlayStateChange?.(false)
          }}
          aria-label="తెలుగు సమాధాన ఆడియో"
          style={metaReady ? {} : { opacity: 0.4, pointerEvents: 'none' }}
        />
      ) : (
        fallbackText && canUseSpeechSynthesis && (
          <button
            type="button"
            className="btn btn--secondary btn--compact"
            onClick={isPlaying ? handleSpeechPause : handleSpeechPlay}
            aria-label={isPlaying ? 'Pause answer audio' : 'Play answer audio'}
          >
            {isPlaying ? '⏸ ఆపండి' : '🔊 వినండి'}
          </button>
        )
      )}
    </div>
  )
}
