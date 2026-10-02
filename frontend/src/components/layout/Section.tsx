import type { ReactNode } from 'react'

interface SectionProps {
  id: string
  title: string
  titleEn?: string
  eyebrow?: string
  intro?: ReactNode
  children: ReactNode
  tone?: 'primary' | 'secondary'
}

export default function Section({ id, title, titleEn, eyebrow, intro, children, tone = 'primary' }: SectionProps) {
  return (
    <section id={id} className={`section section--${tone}`} aria-labelledby={`${id}-title`}>
      <div className="container">
        <div className="section__head">
          {eyebrow && <span className="section__eyebrow">{eyebrow}</span>}
          <h2 id={`${id}-title`}>{title}</h2>
          {titleEn && <p className="section__title-en">{titleEn}</p>}
          {intro && <p className="section__intro">{intro}</p>}
        </div>
        {children}
      </div>
    </section>
  )
}
