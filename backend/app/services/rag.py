from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import AppError

INSUFFICIENT_EVIDENCE_TE = (
    "క్షమించండి. అందుబాటులో ఉన్న వ్యవసాయ సమాచారంలో దీనికి సరిపడిన ఆధారాలు లేవు. "
    "దయచేసి స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."
)


class RagService:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or httpx.Client(timeout=self.settings.service_timeout_seconds)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _post_service(self, stage: str, url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        try:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:1000]
            try:
                obj = exc.response.json()
                detail = obj.get("error", {}).get("message", body) if isinstance(obj, dict) else body
            except ValueError:
                detail = body
            raise AppError(f"{stage.upper()}_FAILED", f"{stage} returned HTTP {exc.response.status_code}: {detail}", 502) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise AppError(f"{stage.upper()}_UNAVAILABLE", f"{stage} service unavailable: {type(exc).__name__}: {exc}", 503) from exc
        if not isinstance(data, dict):
            raise AppError(f"{stage.upper()}_INVALID_RESPONSE", f"{stage} returned an invalid response", 502)
        return data, (time.perf_counter() - started) * 1000

    # ── Translation: Telugu → English via Groq LLM (with Gemini fallback) ─────

    def translate_te_to_en(self, text_te: str, *, preserve_quantities: bool = True) -> tuple[str, float]:
        """Translate Telugu text to English using Groq LLM with automatic Gemini fallback.
        Supports Telugu script (తెలుగు), Romanized phonetic Telugu (Tanglish, e.g. 'rice gurinchi cheppu'),
        and English queries.
        """
        provider = self.settings.llm_provider.strip().lower() or "groq"
        started = time.perf_counter()

        system_prompt = (
            "You are an expert Telugu-English agricultural translator. "
            "Translate the user query into a concise, search-optimized English query for an agricultural knowledge base. "
            "The user query may be in Telugu script (తెలుగు), Romanized phonetic Telugu (e.g. 'rice gurinchi cheppu', 'vari pantalo purugulu'), or English. "
            "Key agricultural terms: వరి/రైస్ = rice/paddy, మొక్కజొన్న/మక్క/జొన్న = maize/corn, కంకి = ear/cob, పంట = crop, తెగులు = disease, నివారణ = control, "
            "పురుగు = pest, మందులు = pesticides, ఎరువులు = fertilizers, ఆకు = leaf, కాండం = stem. "
            "CRITICAL RULES: "
            "1. Spoken Telugu Speech-to-Text often has colloquial phonetic spelling variations (e.g. 'ముక్క జన్న' = maize/corn, 'కాంకి' = cob/ear, 'గురంచి' = about). "
            "Always interpret the farmer's intended meaning and translate accurately (e.g. 'ముక్క జన్న కాంకి గురంచి చెప్పు' -> 'maize cob overview and management'). "
            "2. If the user mentions rice/paddy (e.g. 'వరి గురించి చెప్పు'), translate as 'rice paddy cultivation overview and management'. "
            "3. If the user asks about a non-agricultural topic (e.g. 'చందమామ' - Moon, movies, politics), translate it accurately (e.g. 'tell me about the moon'). "
            "4. ONLY output UNCLEAR_QUERY if the input is purely random unmeaningful letters or unintelligible noise (e.g. 'asldkfj asdf', random syllables). "
            "If any crop, plant, pest, or agricultural meaning can be understood, NEVER output UNCLEAR_QUERY! "
            "Output ONLY the concise English query — no explanations, notes, or quotes."
        )
        user_text = text_te.strip()
        # Pre-normalize common spoken Telugu Whisper transcriptions
        user_text = re.sub(r"ముక్క\s*జన్న", "మొక్కజొన్న", user_text)
        user_text = re.sub(r"\bకాంకి\b", "కంకి", user_text)
        user_text = re.sub(r"\bగురంచి\b", "గురించి", user_text)
        user_text = re.sub(r"\bరైసేల\b", "రైస్ ఎలా", user_text)

        # Helper to try Groq with fallback model
        def try_groq(model_override: str | None = None) -> str:
            api_key = self.settings.groq_api_key.strip()
            model = model_override or self.settings.groq_model
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set")
            url = f"{self.settings.groq_url.rstrip('/')}/chat/completions"
            resp = self.client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
                timeout=12.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        # Helper to try Gemini
        def try_gemini() -> str:
            api_key = self.settings.gemini_api_key.strip()
            model = self.settings.gemini_model
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set")
            url = f"{self.settings.gemini_url.rstrip('/')}/models/{model}:generateContent"
            resp = self.client.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
                },
                timeout=12.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return "".join(
                part.get("text", "")
                for candidate in data.get("candidates", [])
                for part in candidate.get("content", {}).get("parts", [])
                if isinstance(part, dict)
            ).strip()

        translation = ""
        # Try primary Groq model, then high-speed Qwen fallback, then 20b, then Gemini
        try:
            translation = try_groq()
        except Exception:
            try:
                translation = try_groq("qwen/qwen3.8-27b")
            except Exception:
                try:
                    translation = try_groq("openai/gpt-oss-20b")
                except Exception:
                    try:
                        translation = try_gemini()
                    except Exception:
                        # Fallback if all fail: use cleaned input
                        translation = re.sub(r"[^\w\s]", "", user_text)

        # Clean thinking tags, markdown, and quotes
        translation = re.sub(r"<think>.*?</think>", "", translation, flags=re.DOTALL)
        translation = re.sub(r"^```[a-z]*\s*", "", translation)
        translation = re.sub(r"\s*```$", "", translation)
        translation = translation.strip().strip('"\'')
        if not translation:
            translation = "rice paddy crop management"

        elapsed_ms = (time.perf_counter() - started) * 1000
        return translation, elapsed_ms

    def translate_en_to_te_summary(self, answer_te: str) -> tuple[str, float]:
        """Translate Telugu answer into fluent, accurate English for the English summary field.
        Uses Groq with Gemini fallback. Token budget scales with answer length to prevent cut-off.
        """
        provider = self.settings.llm_provider.strip().lower() or "groq"
        started = time.perf_counter()

        system_prompt = (
            "You are a professional agricultural translator. "
            "Translate the provided Telugu agricultural answer into clear, accurate, and fluent English. "
            "CRITICAL RULES: "
            "1. The output MUST be 100% in English. Never leave any Telugu characters or Telugu script. "
            "2. Keep agricultural facts, crop names, pest names, and chemical treatments exact. "
            "3. Finish every sentence completely. Do not cut off mid-thought. "
            "4. Output PLAIN PROSE ONLY — no asterisks (**), no pound signs (#), no bullet dashes, no markdown formatting whatsoever. "
            "5. Output only the English translation — no preamble, no commentary, no quotes."
        )
        text_to_translate = answer_te.strip()
        if not text_to_translate:
            return "No answer text available.", 0.0

        # Estimate token budget: ~1.5 English tokens per Telugu character, capped at 2000
        estimated_tokens = min(2000, max(300, int(len(text_to_translate) * 1.5)))

        def try_groq_answer(model_override: str | None = None) -> str:
            api_key = self.settings.groq_api_key.strip()
            model = model_override or self.settings.groq_model
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set")
            url = f"{self.settings.groq_url.rstrip('/')}/chat/completions"
            resp = self.client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text_to_translate},
                    ],
                    "temperature": 0.0,
                    "max_tokens": estimated_tokens,
                },
                timeout=18.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        def try_gemini_answer() -> str:
            api_key = self.settings.gemini_api_key.strip()
            model = self.settings.gemini_model
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set")
            url = f"{self.settings.gemini_url.rstrip('/')}/models/{model}:generateContent"
            resp = self.client.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": text_to_translate}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": estimated_tokens},
                },
                timeout=18.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return "".join(
                part.get("text", "")
                for candidate in data.get("candidates", [])
                for part in candidate.get("content", {}).get("parts", [])
                if isinstance(part, dict)
            ).strip()

        translation = ""
        try:
            translation = try_groq_answer()
        except Exception:
            try:
                translation = try_groq_answer("qwen/qwen3.8-27b")
            except Exception:
                try:
                    translation = try_groq_answer("openai/gpt-oss-20b")
                except Exception:
                    try:
                        translation = try_gemini_answer()
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).warning("Failed to translate answer to English: %s", exc)

        # Strip thinking tags, markdown bold/headers/bullets, and code fences
        translation = re.sub(r"<think>.*?</think>", "", translation, flags=re.DOTALL)
        translation = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", translation)   # **bold** → plain
        translation = re.sub(r"(?m)^#{1,4}\s+", "", translation)               # ## Headings
        translation = re.sub(r"(?m)^\s*[-*]\s+", " ", translation)             # bullet lists → inline
        translation = re.sub(r"^```[a-z]*\s*", "", translation)
        translation = re.sub(r"\s*```$", "", translation)
        translation = re.sub(r"\n{3,}", "\n\n", translation)                   # collapse extra blank lines
        translation = translation.strip()
        if "agricultural officer" not in translation.lower() and ("వ్యవసాయ" in telugu_text or "సంప్రదించండి" in telugu_text):
            translation = f"{translation.rstrip('. ')}. Consult the local agricultural officer."
        translation = re.sub(r"(?:Consult the local agricultural officer[\.\s]*){2,}", "Consult the local agricultural officer.", translation, flags=re.IGNORECASE).strip()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not translation:
            translation = "English summary is temporarily unavailable. Please refer to the primary Telugu answer above."
        return translation, elapsed_ms

    def retrieve(self, query_en: str) -> tuple[dict[str, Any], float]:
        result, elapsed = self._post_service(
            "retrieval", f"{self.settings.retrieval_url}/retrieve", {"query": query_en, "include_candidates": True}
        )
        results = result.get("results")
        candidates = result.get("faiss_candidates", results)
        if not isinstance(results, list) or not isinstance(candidates, list):
            raise AppError("RETRIEVAL_INVALID_RESPONSE", "Retrieval service returned malformed results", 502)
        return result, elapsed

    @staticmethod
    def _clean_answer(text: str) -> str:
        """Remove simple Markdown and obvious model looping before display/TTS."""
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = text.replace("`", "")
        text = re.sub(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", "", text)
        words = text.split()
        cleaned: list[str] = []
        previous = None
        repeat_count = 0
        for word in words:
            normalized = word.strip(".,!?;:。！？;:,()[]{}\"'").casefold()
            if normalized and normalized == previous:
                repeat_count += 1
                if repeat_count >= 2:
                    continue
            else:
                repeat_count = 0
            cleaned.append(word)
            previous = normalized
        return " ".join(cleaned).strip()

    @staticmethod
    def _correct_agriculture_query(question_te: str, translated: str) -> str:
        """Correct common Whisper confusions and standardize agricultural pest entities."""
        source = question_te.casefold()
        query = translated

        # Rice crop context
        rice_mentioned = bool(re.search(r"(?:వరి|ఒరి)\s*పంట", source))
        bph_mentioned = bool(re.search(r"(?:బ్రౌన్\s*ప్లాంట్\s*హాపర్|సుడి\s*దోమ)", source))
        if rice_mentioned and bph_mentioned:
            query = re.sub(r"\borange(?:\s+peel)?\b", "rice", query, flags=re.IGNORECASE)
            query = re.sub(r"\bon\s+(?:(?:a|an)\s+)?rice(?:\s+crop)?\b", "in rice crop", query, flags=re.IGNORECASE)
            if not re.search(r"\bbrown\s+plant\s+hopper\b", query, flags=re.IGNORECASE):
                query = f"{query.rstrip(' ?.,;:')} brown plant hopper"
            if not re.search(r"\brice\b", query, flags=re.IGNORECASE):
                query = f"{query.rstrip(' ?.,;:')} rice"

        # Leaf folder mapping: "ఆకు ముడుత", "ఆకు ముడత", "ఆకు చుట్టు" -> leaf folder
        if re.search(r"ఆకు\s*(?:ముడుత|ముడత|చుట్టు)", source):
            query = re.sub(r"\bleaf(?:\s*-\s*|\s+)curl\b", "leaf folder", query, flags=re.IGNORECASE)
            if not re.search(r"\bleaf\s+folder\b", query, flags=re.IGNORECASE):
                query = f"{query.rstrip(' ?.,;:')} leaf folder"
            if not re.search(r"\brice|paddy\b", query, flags=re.IGNORECASE):
                query = f"rice {query}"

        # Stem borer mapping: "కాండం తొలుచు" -> stem borer
        if re.search(r"కాండం\s*తొలుచు", source):
            if not re.search(r"\bstem\s+borer\b", query, flags=re.IGNORECASE):
                query = f"{query.rstrip(' ?.,;:')} stem borer"

        # Gall midge / Whorl maggot: "ఉల్లికోడు", "సుడి ఈగ"
        if re.search(r"(?:ఉల్లికోడు|సుడి\s*ఈగ|గాల్\s*మిడ్జ్)", source):
            if not re.search(r"\bgall\s+midge|whorl\s+maggot\b", query, flags=re.IGNORECASE):
                query = f"{query.rstrip(' ?.,;:')} gall midge whorl maggot"

        # Rice Blast / అగ్గి తెగులు
        if re.search(r"(?:బ్లాస్ట్|అగ్గి\s*తెగులు)", source):
            if not re.search(r"\brice\s+blast\b", query, flags=re.IGNORECASE):
                query = f"{query.rstrip(' ?.,;:')} rice blast"

        # Rice Hispa / రైస్ హిస్పా
        if re.search(r"హిస్పా", source):
            if not re.search(r"\brice\s+hispa\b", query, flags=re.IGNORECASE):
                query = f"{query.rstrip(' ?.,;:')} rice hispa"

        return query

    @staticmethod
    def _has_evidence(retrieval: dict[str, Any], query_en: str = "") -> bool:
        """Evaluate if retrieved chunks genuinely ground the query.
        Real paddy knowledge base hits score reranker 0.80 - 0.99 and FAISS >= 0.50.
        Paddy planting/agronomy queries score reranker ~0.30 - 0.48 on pest chunks and must NOT be grounded.
        Non-paddy queries score reranker < 0.06.
        """
        results = retrieval.get("results") or []
        if not results:
            return False
        best = results[0]
        try:
            reranker_score = float(best.get("reranker_score") or 0)
            faiss_score = float(best.get("faiss_score") or 0)

            # Strict threshold: genuine hits on paddy.pdf manual score >= 0.60 reranker and >= 0.45 FAISS
            if reranker_score < 0.60 or faiss_score < 0.45:
                return False

            # Topic alignment: paddy.pdf is exclusively a pest and disease manual.
            # If the user asks how to plant/sow/grow without asking about pests/diseases,
            # do not force pest chunks as evidence!
            query_lower = query_en.lower()
            planting_terms = {"plant", "planting", "sow", "sowing", "transplant", "transplanting", "cultivation", "grow", "growing"}
            pest_terms = {"pest", "disease", "borer", "bug", "hopper", "blast", "blight", "midge", "caterpillar", "mite", "thrip", "worm", "rot", "smut"}
            words = set(re.findall(r"[a-z]+", query_lower))
            if (words & planting_terms) and not (words & pest_terms):
                return False

            return True
        except (TypeError, ValueError):
            return False

    def _generate_answer(
        self,
        question_te: str,
        query_en: str,
        context: list[dict[str, Any]],
        *,
        grounded: bool = True,
    ) -> tuple[str, float]:
        provider = self.settings.llm_provider.strip().lower() or "groq"
        if provider == "groq":
            api_key = self.settings.groq_api_key.strip()
            model = self.settings.groq_model
            if not api_key:
                raise AppError("GROQ_NOT_CONFIGURED", "GROQ_API_KEY is not configured in the project .env file", 503)
        else:
            provider = "gemini"
            api_key = self.settings.gemini_api_key.strip()
            model = self.settings.gemini_model
            if not api_key:
                raise AppError("GEMINI_NOT_CONFIGURED", "GEMINI_API_KEY is not configured in the project .env file", 503)

        if grounded:
            system_instruction = (
                "You are an expert Telugu agricultural advisory assistant.\n"
                "CRITICAL MANDATE: Directly and specifically answer the farmer's EXACT question based on the verified evidence.\n"
                "1. If the question asks about a specific pest or disease: State identification symptoms, economic threshold (ETL), cultural control, and chemical active ingredients with dosage from the evidence, then conclude with: 'స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.'\n"
                "2. If the question asks about planting, cultivation, agronomy, or general practices: Synthesize practical agronomic guidance directly answering what was asked, then conclude with: 'స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.'\n"
                "MANDATORY CLOSING LINE: Always conclude the answer with the exact sentence: 'స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.'\n"
                "RULES: Write in 4-6 sentences of clean, flowing natural Telugu prose. No bullet points, no asterisks, no headers, no dashes. Never invent or wander into an unrelated pest or topic not asked by the user."
            )
        else:
            system_instruction = (
                "You are a helpful Telugu agricultural and horticultural advisory assistant.\n"
                "CRITICAL MANDATE: Directly and practically answer the farmer's EXACT question in natural Telugu.\n"
                "1. If the question is about PLANTING / SOWING / CULTIVATION (e.g. 'రైస్ ఎలా నాటాలి' / how to plant rice/crops): Explain the key practical steps: nursery preparation, seedling age (20-25 days), field puddling/leveling, spacing (15-20 cm), and water management. Do NOT talk about pests unless specifically asked! Always conclude with: 'స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.'\n"
                "2. If the question is about PEST OR DISEASE CONTROL: Give symptom identification, one key cultural practice, recommended active chemical ingredient with dosage, and conclude with: 'స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.'\n"
                "3. If the question is about a REAL CROP / PLANT OVERVIEW: Give the plant's key features, uses, basic cultivation steps, and conclude with: 'స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.'\n"
                "4. If the question is NON-AGRICULTURAL / OUT OF DOMAIN (e.g. the Moon / చందమామ, cinema, etc.): Politely state that you are an agricultural assistant and invite them to ask a farming question.\n"
                "MANDATORY CLOSING LINE: For every agricultural answer, always conclude the answer with the exact sentence: 'స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.'\n"
                "RULES: Answer in 4-6 sentences of clean, flowing natural Telugu prose. No bullet points, no asterisks, no headers, no dashes. Complete every sentence fully."
            )

        evidence = [
            {
                "source": item.get("source_filename"),
                "page": item.get("page_number"),
                "section": item.get("section"),
                "text": item.get("text"),
                "faiss_score": item.get("faiss_score"),
                "reranker_score": item.get("reranker_score"),
            }
            for item in context
        ]

        user_content = (
            f"రైతు అడిగిన ప్రశ్న (Farmer's Question): {question_te}\n"
            f"English Query Translation: {query_en}\n\n"
            f"Verified Manual Evidence:\n"
            f"{json.dumps(evidence, ensure_ascii=False) if grounded else 'No specific manual record. Provide general agricultural advisory.'}\n\n"
            f"సూచన: దయచేసి రైతు అడిగిన ప్రశ్న '{question_te}' కు మాత్రమే నేరుగా, స్పష్టంగా మరియు పూర్తి వాక్యాలలో తెలుగులో సమాధానం ఇవ్వండి. సంబంధం లేని ఇతర పురుగులు లేదా అంశాల గురించి మాట్లాడవద్దు."
        )

        started = time.perf_counter()

        # Budget: grounded answers can be detailed; general/unclear answers must be concise but complete (Telugu chars need ~3-4 tokens each)
        max_answer_tokens = 2200 if grounded else 1200

        def call_groq(model_override: str | None = None) -> str:
            g_key = self.settings.groq_api_key.strip()
            g_model = model_override or self.settings.groq_model
            if not g_key:
                raise ValueError("GROQ_API_KEY is not configured")
            url = f"{self.settings.groq_url.rstrip('/')}/chat/completions"
            resp = self.client.post(
                url,
                headers={"Authorization": f"Bearer {g_key}", "Content-Type": "application/json"},
                json={
                    "model": g_model,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.1,
                    "max_tokens": max_answer_tokens,
                },
                timeout=25.0,
            )
            resp.raise_for_status()
            res_obj = resp.json()
            choices = res_obj.get("choices", [])
            return choices[0].get("message", {}).get("content", "") if choices else ""

        def call_gemini() -> str:
            gem_key = self.settings.gemini_api_key.strip()
            gem_model = self.settings.gemini_model
            if not gem_key:
                raise ValueError("GEMINI_API_KEY is not configured")
            url = f"{self.settings.gemini_url.rstrip('/')}/models/{gem_model}:generateContent"
            resp = self.client.post(
                url,
                headers={"x-goog-api-key": gem_key, "Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system_instruction}]},
                    "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_answer_tokens},
                },
                timeout=25.0,
            )
            resp.raise_for_status()
            res_obj = resp.json()
            return "".join(
                part.get("text", "")
                for candidate in res_obj.get("candidates", [])
                for part in candidate.get("content", {}).get("parts", [])
                if isinstance(part, dict)
            )

        text = ""
        try:
            text = call_groq()
        except Exception:
            try:
                text = call_groq("qwen/qwen3.8-27b")
            except Exception:
                try:
                    text = call_groq("openai/gpt-oss-20b")
                except Exception:
                    try:
                        text = call_gemini()
                    except Exception as exc:
                        raise AppError("GENERATION_FAILED", f"LLM generation failed: {type(exc).__name__}: {exc}", 502) from exc

        elapsed = (time.perf_counter() - started) * 1000
        text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL)
        text = text.strip()
        closing_sentence = "స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."
        if "సంప్రదించండి" not in text and "సహాయకుడు" not in text and "చందమామ" not in question_te:
            text = f"{text.rstrip('. ')}. {closing_sentence}"
        text = re.sub(r"(?:స్థానిక\s+)?వ్యవసాయ[ా\s]*ధికారిని\s*సంప్రదించండి[\.\s]*(?:స్థానిక\s+)?వ్యవసాయ[ా\s]*ధికారిని\s*సంప్రదించండి[\.]?", "స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.", text)
        if not text:
            raise AppError(f"{provider.upper()}_EMPTY_RESPONSE", f"Model returned no answer text", 502)
        return text, elapsed

    def query(self, question_te: str) -> dict[str, Any]:
        if not isinstance(question_te, str) or not question_te.strip():
            raise AppError("EMPTY_QUERY", "query must be a non-empty string", 422)
        question = question_te.strip()
        total_started = time.perf_counter()

        # ── Fast path: Greetings & clearly non-agricultural input ────────────
        # Detect common greetings / pleasantries in Telugu, Tanglish, or English.
        # Return a short, friendly nudge — never launch the full pipeline.
        _greeting_patterns = [
            r"^(?:hello|hi|hey|hii+|hai|howdy|sup|yo)\b",
            r"^(?:నమస్కారం?|నమస్కారాలు|నమస్కరిస్తున్నాను|హలో|హాయ్)\b",
            r"^(?:good\s*(?:morning|afternoon|evening|night))\b",
            r"^(?:how\s+are\s+you|what'?s\s+up|wassup)\b",
            r"^(?:thanks?|thank\s+you|dhanyavaadalu|ధన్యవాదాలు|thanks?)\b",
            r"^(?:ok|okay|hmm+|ohh?|acha|అచ్చా)\b",
            r"^(?:bye|goodbye|bbye|see\s+you|cya|అల్విదా)\b",
        ]
        _non_agri_test = question.lower().strip().rstrip("!.,?")
        if any(re.match(p, _non_agri_test, re.IGNORECASE) for p in _greeting_patterns):
            total_ms = (time.perf_counter() - total_started) * 1000
            provider = self.settings.llm_provider.strip().lower() or "groq"
            return {
                "status": "greeting",
                "original_query": question,
                "translated_query": "greeting",
                "answer_telugu": (
                    "నమస్కారం! నేను TeluguBridge వ్యవసాయ సహాయకుడిని. "
                    "వరి పంట, పురుగులు, రోగాలు లేదా పంట నిర్వహణ గురించి "
                    "తెలుగులో మీ ప్రశ్న అడగండి — నేను సహాయం చేస్తాను!"
                ),
                "answer_english": (
                    "Hello! I'm TeluguBridge, your Telugu agriculture assistant. "
                    "Please ask me a question about paddy crops, pests, diseases, "
                    "or crop management — I'm happy to help!"
                ),
                "answer_grounded": False,
                "answer_basis": "greeting",
                "retrieved_context": [],
                "sources": [],
                "faiss_candidates": [],
                "faiss_candidate_count": 0,
                "reranker_used": False,
                "translation": {"model": "none", "device": "none", "input_language": "tel_Telu", "output_language": "eng_Latn", "method": "shortcircuit"},
                "timings_ms": {"nllb": 0.0, "answer_translation": 0.0, "embedding": 0.0, "faiss": 0.0, "reranker": 0.0, "llm": 0.0, "gemini": 0.0, "retrieval_http": 0.0, "total": total_ms},
                "llm_provider": provider,
                "llm_model": self.settings.groq_model if provider == "groq" else self.settings.gemini_model,
                "gemini_model": None,
            }
        # ─────────────────────────────────────────────────────────────────────

        # Step 1: Translate Telugu → English via LLM (replaces slow NLLB)
        translated, translation_ms = self.translate_te_to_en(question)

        # ── Garbled/unclear speech detection ─────────────────────────────────
        # If the translation LLM returned UNCLEAR_QUERY, the input was garbled
        # (e.g. noisy mic, wrong language, or unclear Whisper transcription).
        # Return a friendly "please repeat" message instead of hallucinating.
        if translated.strip().upper().startswith("UNCLEAR_QUERY"):
            total_ms = (time.perf_counter() - total_started) * 1000
            provider = self.settings.llm_provider.strip().lower() or "groq"
            return {
                "status": "unclear_query",
                "original_query": question,
                "translated_query": "UNCLEAR_QUERY",
                "answer_telugu": (
                    "క్షమించండి, మీ ప్రశ్న స్పష్టంగా అర్థం కాలేదు. "
                    "దయచేసి మళ్లీ స్పష్టంగా తెలుగులో మాట్లాడండి — "
                    "ఉదాహరణకు: 'వరి పంటలో స్టెమ్ బోరర్ నివారణ ఏమిటి?'"
                ),
                "answer_english": (
                    "Sorry, your question was not clear. "
                    "Please speak clearly in Telugu again — "
                    "for example: 'What is the control for stem borer in paddy?'"
                ),
                "answer_grounded": False,
                "answer_basis": "unclear_query",
                "retrieved_context": [],
                "sources": [],
                "faiss_candidates": [],
                "faiss_candidate_count": 0,
                "reranker_used": False,
                "translation": {"model": "none", "device": "none", "input_language": "tel_Telu", "output_language": "eng_Latn", "method": "shortcircuit"},
                "timings_ms": {"nllb": translation_ms, "answer_translation": 0.0, "embedding": 0.0, "faiss": 0.0, "reranker": 0.0, "llm": 0.0, "gemini": 0.0, "retrieval_http": 0.0, "total": total_ms},
                "llm_provider": provider,
                "llm_model": self.settings.groq_model if provider == "groq" else self.settings.gemini_model,
                "gemini_model": None,
            }
        # ─────────────────────────────────────────────────────────────────────

        translated = self._correct_agriculture_query(question, translated)

        # Step 2: Retrieve from FAISS vector store
        retrieved, retrieval_elapsed = self.retrieve(translated)
        timings_raw = retrieved.get("timings_ms", {})
        evidence = retrieved.get("results", [])
        has_evidence = self._has_evidence(retrieved, translated)

        # ── Crop scope guard ─────────────────────────────────────────────────
        # The knowledge base only covers paddy / rice (paddy.pdf).
        # If the query is clearly about a DIFFERENT crop, do NOT treat paddy
        # chunks as valid evidence — that would be factually misleading.
        _NON_PADDY_CROPS = {
            # Telugu crop names
            "జొన్న", "మొక్కజొన్న", "మక్క", "మొక్క", "గోధుమ", "వేరుసెనగ", "పత్తి",
            "చెరకు", "సోయా", "సోయాబీన్", "పప్పుదినుసులు", "శనగ", "కంది", "పెసర",
            "మినుము", "ఉలవ", "అవిసె", "ఆముదాలు", "సజ్జ", "జనుము", "మిరప",
            "టమాట", "బ్రింజాల్", "వంకాయ", "బెండ", "దోస", "ద్రాక్ష", "అరటి",
            "మామిడి", "బొప్పాయి", "దానిమ్మ", "జామ", "రోజ్", "గులాబీ", "తులసి", "తులశి", "మల్లె", "చామంతి",
            # English / transliterated crop names
            "maize", "corn", "wheat", "cotton", "sugarcane", "soybean", "groundnut",
            "peanut", "sorghum", "jowar", "bajra", "pearl millet", "chili", "chilli",
            "tomato", "brinjal", "eggplant", "okra", "cucumber", "mango", "banana",
            "papaya", "pomegranate", "guava", "grapes", "onion", "potato", "carrot",
            "cabbage", "cauliflower", "turmeric", "ginger", "garlic", "rose", "tulsi", "jasmine", "marigold",
        }
        question_lower = question.lower()
        translated_lower = translated.lower()
        if has_evidence and any(
            crop in question_lower or crop in translated_lower
            for crop in _NON_PADDY_CROPS
        ):
            # Query is about a crop not in the KB — override to general LLM
            has_evidence = False
        # ─────────────────────────────────────────────────────────────────────

        # Step 3: Generate Telugu answer via LLM
        answer, llm_ms = self._generate_answer(
            question,
            translated,
            evidence if has_evidence else [],
            grounded=has_evidence,
        )
        answer = self._clean_answer(answer)

        # Step 4: Translate answer → English for the English summary field
        # (LLM translation of its own output, very fast)
        answer_english, answer_translation_ms = self.translate_en_to_te_summary(answer)

        status = "answered" if has_evidence else "general_answer"
        total_ms = (time.perf_counter() - total_started) * 1000

        sources = [
            {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "source_filename": item.get("source_filename"),
                "source_relative_path": item.get("source_relative_path"),
                "page_number": item.get("page_number"),
                "section": item.get("section"),
                "text": item.get("text"),
                "faiss_score": item.get("faiss_score"),
                "reranker_score": item.get("reranker_score"),
            }
            for item in evidence if has_evidence
        ]

        return {
            "status": status,
            "original_query": question,
            "translated_query": translated,
            "answer_telugu": answer,
            "answer_english": answer_english,
            "answer_grounded": has_evidence,
            "answer_basis": "knowledge_base" if has_evidence else f"general_{self.settings.llm_provider.strip().lower() or 'groq'}",
            "retrieved_context": evidence,
            "sources": sources,
            "faiss_candidates": retrieved.get("faiss_candidates", []),
            "faiss_candidate_count": retrieved.get("candidate_count", 0),
            "reranker_used": retrieved.get("reranker_used", False),
            "translation": {
                "model": self.settings.groq_model if self.settings.llm_provider.strip().lower() == "groq" else self.settings.gemini_model,
                "device": "api",
                "input_language": "tel_Telu",
                "output_language": "eng_Latn",
                "method": "llm",
            },
            "timings_ms": {
                "nllb": translation_ms,
                "answer_translation": answer_translation_ms,
                "embedding": float(timings_raw.get("query_embedding_ms", 0.0)),
                "faiss": float(timings_raw.get("faiss_search_ms", 0.0)),
                "reranker": float(timings_raw.get("reranker_ms", 0.0)),
                "llm": llm_ms,
                "gemini": llm_ms,
                "retrieval_http": retrieval_elapsed,
                "total": total_ms,
            },
            "llm_provider": self.settings.llm_provider.strip().lower() or "groq",
            "llm_model": self.settings.groq_model if (self.settings.llm_provider.strip().lower() == "groq") else self.settings.gemini_model,
            "gemini_model": self.settings.gemini_model if (self.settings.llm_provider.strip().lower() != "groq") else None,
        }


rag_service = RagService()
