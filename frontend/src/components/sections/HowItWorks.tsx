import Section from '../layout/Section'

const STEPS = [
  { te: 'మాట్లాడండి', en: 'Speak in Telugu' },
  { te: 'మాటను పాఠ్యంగా', en: 'Speech-to-text' },
  { te: 'అనువాదం', en: 'Translation' },
  { te: 'సమాచారం వెతకడం', en: 'Knowledge retrieval' },
  { te: 'LLM సమాధానం', en: 'LLM response' },
  { te: 'తెలుగులో సమాధానం', en: 'Telugu response' },
]

export default function HowItWorks() {
  return (
    <Section id="how-it-works" eyebrow="Pipeline" title="ఎలా పనిచేస్తుంది" titleEn="How it works" tone="secondary">
      <ol className="timeline">
        {STEPS.map((step, i) => (
          <li key={step.te} className="timeline__step">
            <span className="timeline__index">{i + 1}</span>
            <h3>{step.te}</h3>
            <p className="timeline__en">{step.en}</p>
            {i < STEPS.length - 1 && <span className="timeline__arrow" aria-hidden="true">→</span>}
          </li>
        ))}
      </ol>
    </Section>
  )
}
