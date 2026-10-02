// Public runtime configuration only. Never place secrets (API keys, tokens) here —
// this module is bundled into client-side JavaScript and is fully visible to users.

const rawBaseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined

export const env = {
  /** Base URL of the backend API. Empty string means "no backend configured". */
  apiBaseUrl: (rawBaseUrl ?? 'http://127.0.0.1:8000').trim().replace(/\/$/, ''),
  demoMode: false,
} as const
