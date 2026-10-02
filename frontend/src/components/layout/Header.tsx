import { useState } from 'react'

interface HeaderProps {
  onVoiceCtaClick: () => void
}

const NAV_LINKS = [
  { href: '#how-it-works', te: 'ఎలా పనిచేస్తుంది', en: 'How it works' },
  { href: '#knowledge-base', te: 'జ్ఞాన ఆధారం', en: 'Knowledge' },
  { href: '#evaluation', te: 'మూల్యాంకనం', en: 'Evaluation' },
  { href: '#limitations', te: 'పరిమితులు', en: 'Limitations' },
  { href: '#privacy', te: 'గోప్యత', en: 'Privacy' },
  { href: '#about', te: 'ప్రాజెక్ట్ గురించి', en: 'About' },
]

export default function Header({ onVoiceCtaClick }: HeaderProps) {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <header className="site-header">
      <div className="container site-header__inner">
        <a href="#top" className="wordmark" aria-label="TeluguBridge AI home">
          <svg width="26" height="26" viewBox="0 0 30 30" aria-hidden="true">
            <path d="M4 20c4-8 8-8 11-8s7 0 11 8" fill="none" stroke="var(--emerald)" strokeWidth="2.4" strokeLinecap="round" />
            <path d="M4 12c4 6 8 6 11 6s7 0 11-6" fill="none" stroke="var(--gold)" strokeWidth="2.4" strokeLinecap="round" opacity="0.85" />
          </svg>
          <span>
            TeluguBridge <em>AI</em>
          </span>
        </a>

        <nav className="site-nav site-nav--desktop" aria-label="Primary">
          {NAV_LINKS.map((link) => (
            <a key={link.href} href={link.href}>
              <span className="nav-link__te">{link.te}</span>
              <span className="nav-link__en">{link.en}</span>
            </a>
          ))}
        </nav>

        <div className="site-header__actions">
          <button type="button" className="btn btn--primary btn--compact" onClick={onVoiceCtaClick} aria-label="Start voice recording">
            🎙 మాట్లాడండి <span className="btn__en">Ask a question</span>
          </button>
          <button
            type="button"
            className="nav-toggle"
            aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={menuOpen}
            aria-controls="mobile-nav"
            onClick={() => setMenuOpen((v) => !v)}
          >
            <span />
            <span />
            <span />
          </button>
        </div>
      </div>

      {menuOpen && (
        <nav id="mobile-nav" className="site-nav site-nav--mobile" aria-label="Mobile">
          {NAV_LINKS.map((link) => (
            <a key={link.href} href={link.href} onClick={() => setMenuOpen(false)}>
              <span className="nav-link__te">{link.te}</span>
              <span className="nav-link__en">{link.en}</span>
            </a>
          ))}
        </nav>
      )}
    </header>
  )
}
