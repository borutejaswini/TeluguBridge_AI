import { useState } from 'react'
import type { SourceReference } from '../../types/api'

interface ChunksDrawerProps {
  isOpen: boolean
  onClose: () => void
  sources: SourceReference[]
}

export default function ChunksDrawer({ isOpen, onClose, sources }: ChunksDrawerProps) {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)

  if (!isOpen) return null

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedIndex(index)
      setTimeout(() => setCopiedIndex(null), 2000)
    })
  }

  return (
    <div className="chunks-modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label="Retrieved Knowledge Chunks">
      <div className="chunks-drawer" onClick={(e) => e.stopPropagation()}>
        {/* Drawer Header */}
        <div className="chunks-drawer__header">
          <div className="chunks-drawer__title-group">
            <div className="chunks-drawer__icon-wrap">
              <span className="chunks-drawer__icon">📚</span>
            </div>
            <div>
              <h3 className="chunks-drawer__title">జ్ఞాన ఆధార భాగాలు (Knowledge Chunks)</h3>
              <p className="chunks-drawer__subtitle">
                {sources.length} chunks retrieved from local agriculture knowledge base
              </p>
            </div>
          </div>
          <button
            type="button"
            className="chunks-drawer__close-btn"
            onClick={onClose}
            aria-label="Close chunks drawer"
          >
            ✕
          </button>
        </div>

        {/* Drawer Content */}
        <div className="chunks-drawer__content">
          {sources.length === 0 ? (
            <p className="chunks-drawer__empty">ఈ ప్రశ్నకు సంబంధించిన భాగాలు అందుబాటులో లేవు.</p>
          ) : (
            <div className="chunks-list">
              {sources.map((source, index) => {
                const faissPercent = source.relevance ? Math.round(source.relevance * 100) : null
                const rerankerPercent = source.rerankerScore !== undefined ? Math.round(source.rerankerScore * 100) : null

                return (
                  <div key={source.chunkId || `${source.title}-${index}`} className="chunk-card">
                    {/* Chunk Meta Header */}
                    <div className="chunk-card__header">
                      <div className="chunk-card__badge-row">
                        <span className="chunk-badge chunk-badge--index">Chunk #{index + 1}</span>
                        {source.page && (
                          <span className="chunk-badge chunk-badge--page">పేజీ {source.page}</span>
                        )}
                        {source.section && (
                          <span className="chunk-badge chunk-badge--section">{source.section}</span>
                        )}
                      </div>
                      <div className="chunk-card__scores">
                        {rerankerPercent !== null && (
                          <span className="score-tag score-tag--reranker" title="Cross-Encoder Reranker confidence score">
                            Reranker: {rerankerPercent}%
                          </span>
                        )}
                        {faissPercent !== null && (
                          <span className="score-tag score-tag--faiss" title="FAISS Vector similarity">
                            Vector Match: {faissPercent}%
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Source File */}
                    <div className="chunk-card__file">
                      <span className="chunk-card__file-icon">📄</span>
                      <span className="chunk-card__file-name">{source.title}</span>
                      {source.demo && <span className="demo-tag">Demo</span>}
                    </div>

                    {/* Chunk Text Content */}
                    <div className="chunk-card__text-wrap">
                      <p className="chunk-card__text">
                        {source.text || 'పూర్తి టెక్స్ట్ సమాచారం ఇక్కడ భద్రపరచబడింది.'}
                      </p>
                    </div>

                    {/* Actions */}
                    {source.text && (
                      <div className="chunk-card__actions">
                        <button
                          type="button"
                          className="btn-copy-chunk"
                          onClick={() => handleCopy(source.text!, index)}
                        >
                          {copiedIndex === index ? '✓ కాపీ చేయబడింది (Copied)' : '📋 ఈ భాగాన్ని కాపీ చేయండి'}
                        </button>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
