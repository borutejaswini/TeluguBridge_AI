// Simulated pipeline used when no backend is configured (VITE_DEMO_MODE=true).
// Every value returned here is clearly demo content — never presented as real
// model output, real citations, or real backend behavior.

import type { QueryResponse, SourceReference, SynthesizeResponse, TranscribeResponse, TranslateResponse } from '../types/api'

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

// A handful of realistic-but-clearly-fake agricultural questions and cautious answers.
// Chemical/dosage instructions are intentionally avoided; answers defer to local experts.
const DEMO_ENTRIES: Array<{ transcript: string; translation: string; answer: string; answerTranslation: string; sources: SourceReference[] }> = [
  {
    transcript: 'వరి పంటలో పురుగులను ఎలా నియంత్రించాలి?',
    translation: 'How can I control pests in a paddy (rice) crop?',
    answer:
      'పంటలో కనిపిస్తున్న పురుగు రకం, దాని తీవ్రత మరియు పంట దశను బట్టి నియంత్రణ పద్ధతి మారుతుంది. తెగులు కనిపించిన వెంటనే స్థానిక వ్యవసాయ విస్తరణ అధికారిని లేదా కృషి విజ్ఞాన కేంద్రాన్ని సంప్రదించడం మంచిది. సేంద్రీయ ఎర పంటలు, ఎర ఉచ్చులు వంటి సమీకృత సస్యరక్షణ పద్ధతులు మొదటి అడుగుగా పరిగణించవచ్చు.',
    answerTranslation:
      'The right control method depends on the exact pest, its severity, and the crop stage. It is best to consult your local agricultural extension officer or Krishi Vigyan Kendra as soon as damage is noticed. Integrated pest management steps such as trap crops and pheromone traps are often a reasonable first step.',
    sources: [
      { title: 'Demo Agricultural Knowledge Base', section: 'Pest Management — Paddy', relevance: 0.86, demo: true },
      { title: 'Demo Agricultural Knowledge Base', section: 'Integrated Pest Management Basics', relevance: 0.71, demo: true },
    ],
  },
  {
    transcript: 'వేసవిలో పంటలకు నీరు ఎప్పుడు పెట్టాలి?',
    translation: 'When should crops be watered during summer?',
    answer:
      'సాధారణంగా ఉదయం లేదా సాయంత్రం సమయాల్లో నీరు పెట్టడం వల్ల నీటి ఆవిరి తగ్గుతుంది. ఖచ్చితమైన నీటి అవసరం నేల రకం, పంట రకం మరియు స్థానిక వాతావరణంపై ఆధారపడి ఉంటుంది — వీటికి స్థానిక వ్యవసాయ శాఖ సలహా ఉపయోగకరంగా ఉంటుంది.',
    answerTranslation:
      'Watering in the early morning or evening generally reduces evaporation loss. The exact requirement depends on soil type, crop, and local weather — local agriculture department guidance is useful here.',
    sources: [{ title: 'Demo Agricultural Knowledge Base', section: 'Irrigation Scheduling', relevance: 0.78, demo: true }],
  },
]

function pickDemoEntry(transcript: string) {
  const found = DEMO_ENTRIES.find((e) => e.transcript === transcript)
  return found ?? DEMO_ENTRIES[0]
}

// A short list of phrases that trigger the out-of-domain UX in the demo, so reviewers
// can see that state without needing a real classifier.
const OUT_OF_DOMAIN_HINTS = ['సినిమా', 'క్రికెట్', 'రాజకీయ', 'movie', 'cricket']

export async function demoTranscribeAudio(_blob: Blob): Promise<TranscribeResponse> {
  await wait(800 + Math.random() * 400)
  // Demo mode cannot actually run speech recognition in-browser; it returns a
  // representative sample question so the rest of the pipeline can be demonstrated.
  const entry = DEMO_ENTRIES[Math.floor(Math.random() * DEMO_ENTRIES.length)]
  return { transcript: entry.transcript }
}

export async function demoTranslateText(text: string): Promise<TranslateResponse> {
  await wait(500 + Math.random() * 400)
  const entry = pickDemoEntry(text)
  return { translation: entry.translation }
}

export async function demoQueryKnowledgeBase(text: string): Promise<QueryResponse> {
  await wait(700 + Math.random() * 400)
  const isOutOfDomain = OUT_OF_DOMAIN_HINTS.some((hint) => text.toLowerCase().includes(hint.toLowerCase()))
  if (isOutOfDomain) {
    return { answer: '', sources: [], outOfDomain: true }
  }
  await wait(1000 + Math.random() * 600)
  const entry = pickDemoEntry(text)
  return { answer: entry.answer, answerTranslation: entry.answerTranslation, sources: entry.sources, outOfDomain: false }
}

export async function demoSynthesizeSpeech(_text: string): Promise<SynthesizeResponse> {
  await wait(600 + Math.random() * 400)
  // No real IndicTTS audio file exists in demo mode. Callers should treat an empty
  // audioUrl as "use the optional browser SpeechSynthesis fallback, if available".
  return { audioUrl: '' }
}
