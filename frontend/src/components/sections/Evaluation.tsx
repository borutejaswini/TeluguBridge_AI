import Section from '../layout/Section'

const METRICS = [
  { name: 'WER', full: 'Word Error Rate', domain: 'Speech Recognition', desc: 'Measures how many words the transcription gets wrong.' },
  { name: 'CER', full: 'Character Error Rate', domain: 'Speech Recognition', desc: 'Character-level transcription accuracy, useful for Telugu script.' },
  { name: 'BLEU', full: 'Translation quality', domain: 'Translation', desc: 'Compares machine translation output against reference translations.' },
  { name: 'MOS', full: 'Speech quality', domain: 'Speech Quality', desc: 'Human-rated naturalness of synthesized speech.' },
  { name: 'Latency', full: 'End-to-end response time', domain: 'System Performance', desc: 'Time from end of recording to first audio playback.' },
]

export default function Evaluation() {
  return (
    <Section
      id="evaluation"
      eyebrow="Research"
      title="మూల్యాంకనం"
      titleEn="Evaluation"
      intro="Final benchmark values will be reported after evaluation on the selected Telugu datasets (e.g. Kathbath Telugu) and custom recordings."
      tone="secondary"
    >
      <div className="metric-grid">
        {METRICS.map((m) => (
          <div className="metric-card" key={m.name}>
            <p className="metric-card__domain">{m.domain}</p>
            <p className="metric-card__name">{m.name}</p>
            <p className="metric-card__full">{m.full}</p>
            <p className="metric-card__desc">{m.desc}</p>
            <span className="metric-card__status">To be evaluated</span>
          </div>
        ))}
      </div>
    </Section>
  )
}
