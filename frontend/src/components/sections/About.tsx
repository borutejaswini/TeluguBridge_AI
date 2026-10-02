import Section from '../layout/Section'

export default function About() {
  return (
    <Section
      id="about"
      eyebrow="Project"
      title="ప్రాజెక్ట్ గురించి"
      titleEn="About"
      intro="TeluguBridge AI is a voice-to-prompt framework designed to make domain-specific AI information more accessible to Telugu-speaking users, particularly farmers and rural communities."
      tone="primary"
    >
      <div className="about-grid">
        <div className="about-card">
          <h3>Telugu-first</h3>
          <p>Interactions are designed around spoken Telugu as the primary input and output modality, not as an afterthought.</p>
        </div>
        <div className="about-card">
          <h3>Voice-first</h3>
          <p>The interface is built around speaking and listening rather than typing and reading, for accessibility in low-literacy contexts.</p>
        </div>
        <div className="about-card">
          <h3>Retrieval-grounded</h3>
          <p>Answers are intended to be grounded in retrieved domain documents rather than generated from unconstrained model knowledge.</p>
        </div>
        <div className="about-card">
          <h3>Research-oriented</h3>
          <p>This is an academic project. Its pipeline combines speech recognition, translation, retrieval, LLM generation, and speech synthesis, and is under active evaluation.</p>
        </div>
      </div>
      <p className="section__footnote">
        Base research context: a Telugu–English speech translation approach related to TeluguST-46 (2025), and a document-grounded RAG
        workflow (PDF extraction → semantic chunking → embeddings → vector search → LLM generation, 2024). Implementation choices such
        as Whisper Large V3 Turbo, NLLB, and IndicTTS are being explored; details are subject to change as evaluation progresses.
      </p>
    </Section>
  )
}
