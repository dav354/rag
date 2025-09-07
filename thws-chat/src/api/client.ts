// src/api/client.ts
import type { AskResponse } from '@/types';

const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

export async function ask(query: string): Promise<AskResponse> {
  const res = await fetch(`${BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API-Fehler ${res.status}: ${text || res.statusText}`);
  }
  const data = (await res.json()) as AskResponse;
  if (!data?.answer?.answer) throw new Error('Ungültige API-Antwort: Feld answer.answer fehlt.');
  return data;
}
