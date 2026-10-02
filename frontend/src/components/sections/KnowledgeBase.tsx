import Section from '../layout/Section'

const STAGES = [
  'PDFs / Telugu Documents',
  'Extraction',
  'Semantic Chunking',
  'BGE-M3 Embeddings',
  'FAISS Vector Search',
  'Relevant Context',
  'Gemini 2.5 Flash',
  'Grounded Answer',
]

export default function KnowledgeBase() {
  return (
    <Section
      id="knowledge-base"
      eyebrow="Retrieval-augmented generation"
      title="జ్ఞాన ఆధారం"
      titleEn="Knowledge Base"
      intro="Answers are grounded in retrieved documents rather than generated from the model's general knowledge alone."
      tone="primary"
    >
      <div className="flow">
        {STAGES.map((stage, i) => (
          <div className="flow__step" key={stage}>
            <span className="flow__card">{stage}</span>
            {i < STAGES.length - 1 && (
              <span className="flow__arrow" aria-hidden="true">
                ↓
              </span>
            )}
          </div>
        ))}
      </div>
      <p className="section__footnote">
        Retrieval quality depends on the documents available in the knowledge base and on how well BGE-M3 embeddings perform for
        Telugu/English content — this is part of ongoing evaluation, not a solved problem.
      </p>
    </Section>
  )
}
