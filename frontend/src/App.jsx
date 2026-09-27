import { useState } from "react";

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch(`${API}/api/pipeline/run`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data?.error?.message || `Backend returned ${response.status}`);
      setResult(data);
    } catch (err) {
      setError(err.message || "Could not process this recording.");
    } finally {
      setBusy(false);
    }
  }

  return <main className="page">
    <header><p className="eyebrow">AGRICULTURE VOICE ASSISTANT</p><h1>తెలుగుBridge</h1><p className="intro">Ask an agriculture question in Telugu. Get an evidence based answer in Telugu.</p></header>
    <form className="upload" onSubmit={submit}>
      <label htmlFor="audio">Choose a Telugu recording</label>
      <input id="audio" type="file" accept="audio/*,.wav" onChange={(event) => setFile(event.target.files?.[0] || null)} />
      <button disabled={!file || busy}>{busy ? "Processing recording…" : "Ask about this recording"}</button>
      <p className="hint">Audio is processed by the local TeluguBridge services.</p>
    </form>
    {error && <section className="error" role="alert"><strong>Could not complete the request</strong><p>{error}</p></section>}
    {result && <div className="results" aria-live="polite">
      <section><h2>మీ ప్రశ్న</h2><p lang="te">{result.transcript_telugu}</p></section>
      <section><h2>సమాధానం</h2><p lang="te">{result.answer_telugu}</p></section>
      <section><h2>Sources</h2><ul>{result.sources.map((source) => <li key={source.chunk_id}>{source.source_filename}, page {source.page_number}{source.section ? ` — ${source.section}` : ""}</li>)}</ul></section>
      <section><h2>Answer audio</h2><audio controls src={`${API}${result.audio_url}`} /></section>
      <details><summary>English retrieval query and processing details</summary><p>{result.translated_query}</p><pre>{JSON.stringify(result.timings, null, 2)}</pre></details>
    </div>}
    <footer>Answers are based on the available agriculture documents. For urgent crop loss, consult a local agriculture officer.</footer>
  </main>;
}
