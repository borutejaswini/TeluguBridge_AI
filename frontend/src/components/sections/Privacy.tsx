import Section from '../layout/Section'

export default function Privacy() {
  return (
    <Section id="privacy" eyebrow="Privacy" title="గోప్యత" titleEn="Privacy" tone="secondary">
      <div className="privacy-card">
        <p>The frontend requests microphone access only when voice input is activated by the user, and only for the duration of a recording.</p>
        <p>
          Voice data handling — including whether recordings are stored, for how long, and how they are deleted — depends on the
          connected backend implementation. The frontend itself does not make guarantees it cannot enforce.
        </p>
        <p>No account or personal profile is required to use the voice interface.</p>
        <p className="section__footnote">Specific data retention and deletion guarantees will be documented once the backend's data handling is finalized.</p>
      </div>
    </Section>
  )
}
