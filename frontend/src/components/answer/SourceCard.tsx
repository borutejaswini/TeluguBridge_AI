import type { SourceReference } from '../../types/api'

export default function SourceCard({ source }: { source: SourceReference }) {
  return (
    <li className="source-card">
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" className="source-card__icon">
        <rect x="3" y="2" width="12" height="14" rx="1.5" fill="none" stroke="var(--emerald)" strokeWidth="1.4" />
        <line x1="5.5" y1="6" x2="12.5" y2="6" stroke="var(--emerald)" strokeWidth="1.2" />
        <line x1="5.5" y1="9" x2="12.5" y2="9" stroke="var(--emerald)" strokeWidth="1.2" />
        <line x1="5.5" y1="12" x2="10" y2="12" stroke="var(--emerald)" strokeWidth="1.2" />
      </svg>
      <div className="source-card__body">
        <p className="source-card__title">
          {source.title} {source.demo && <span className="demo-tag">Demo</span>}
        </p>
        <p className="source-card__meta">
          {source.section && <span>{source.section}</span>}
          {source.page && <span>Page {source.page}</span>}
          {source.date && <span>{source.date}</span>}
        </p>
      </div>
    </li>
  )
}
