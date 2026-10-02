import Section from '../layout/Section'

const LIMITATIONS = [
  'Speech recognition can make mistakes, especially with accents, background noise, or unclear speech.',
  'Translation between Telugu and English can introduce errors or lose nuance.',
  'Retrieval quality depends entirely on what documents exist in the knowledge base.',
  'Agricultural information can become outdated as practices and conditions change.',
  'Large language models can occasionally produce plausible-sounding but incorrect answers (hallucination).',
  'Questions outside the supported domain should be rejected rather than answered speculatively.',
  'Final accuracy figures depend on evaluation that has not yet been completed.',
  'Latency depends on the model, backend infrastructure, and network conditions in use.',
]

export default function Limitations() {
  return (
    <Section id="limitations" eyebrow="Transparency" title="పరిమితులు" titleEn="Limitations" tone="primary">
      <ul className="checklist">
        {LIMITATIONS.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </Section>
  )
}
