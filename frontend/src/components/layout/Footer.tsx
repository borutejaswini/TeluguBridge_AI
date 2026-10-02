export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="container site-footer__inner">
        <div>
          <div className="wordmark wordmark--footer">
            TeluguBridge <em>AI</em>
          </div>
          <p className="site-footer__tagline">"తెలుగులో అడగండి. సంబంధిత సమాచారాన్ని పొందండి."</p>
          <p className="site-footer__tagline-en">Ask in Telugu. Access relevant domain knowledge.</p>
          <p className="site-footer__meta">Voice-first • Telugu-first • Knowledge-grounded</p>
        </div>

        <nav aria-label="Footer">
          <a href="#how-it-works">How it works</a>
          <a href="#knowledge-base">Knowledge</a>
          <a href="#evaluation">Evaluation</a>
          <a href="#about">About</a>
        </nav>
      </div>
      <p className="site-footer__note container">
        Research/college project. Results not yet benchmarked. No production deployment or accuracy guarantees implied.
      </p>
    </footer>
  )
}
