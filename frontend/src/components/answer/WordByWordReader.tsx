import { useMemo, useState, useEffect } from 'react'

interface WordByWordReaderProps {
  text: string
  isPlaying: boolean
  currentTime: number
  duration: number
  audioEnded: boolean
}

export default function WordByWordReader({
  text,
  isPlaying,
  currentTime,
  duration,
  audioEnded,
}: WordByWordReaderProps) {
  // Split text into words while preserving Telugu characters and punctuation
  const words = useMemo(() => {
    return text.trim().split(/\s+/).filter(Boolean)
  }, [text])

  // Allow user to toggle showing all text directly if they don't want to wait
  const [forceShowAll, setForceShowAll] = useState(false)

  // Reset forceShowAll when new answer arrives
  useEffect(() => {
    setForceShowAll(false)
  }, [text])

  // Calculate word highlighting based on audio playback progress
  const { revealedCount, activeIndex } = useMemo(() => {
    if (forceShowAll || audioEnded) {
      return { revealedCount: words.length, activeIndex: -1 }
    }

    if (isPlaying && duration > 0) {
      const progress = Math.min(1, Math.max(0, currentTime / duration))
      const count = Math.min(words.length, Math.floor(progress * words.length) + 1)
      const active = Math.min(words.length - 1, count - 1)
      return { revealedCount: count, activeIndex: active }
    }

    // Default when idle or before playback: show all words naturally
    return { revealedCount: words.length, activeIndex: -1 }
  }, [words.length, isPlaying, currentTime, duration, audioEnded, forceShowAll])

  const isReading = isPlaying && duration > 0 && !audioEnded

  return (
    <div className="word-by-word-container">
      {/* Control bar */}
      <div className="word-by-word__toolbar">
        {isReading ? (
          <span className="reading-indicator">
            <span className="reading-indicator__dot" />
            ఆడియో చదువుతోంది… ({revealedCount}/{words.length} పదాలు)
          </span>
        ) : (
          <span className="reading-hint">
            🌾 సమాధానం (Answer)
          </span>
        )}
      </div>

      {/* Words display */}
      <p className="panel-card__telugu word-by-word__text" aria-live="polite">
        {words.map((word, index) => {
          const isCurrentActive = isReading && index === activeIndex
          const isAlreadyRead = isReading && index < activeIndex

          return (
            <span
              key={`${word}-${index}`}
              className={`reader-word ${isCurrentActive ? 'reader-word--active' : ''} ${isAlreadyRead ? 'reader-word--read' : ''}`}
            >
              {word}{' '}
            </span>
          )
        })}
      </p>
    </div>
  )
}
