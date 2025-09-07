// src/components/Chat.tsx
import React, { useCallback, useEffect, useRef, useState } from 'react';
import MessageBubble from './MessageBubble';
import SourcesList from './SourcesList';
import { ask } from '@/api/client';
import type { AskResponse } from '@/types';
import clsx from 'clsx';

type Msg =
  | { id: string; role: 'user'; text: string }
  | {
      id: string;
      role: 'assistant';
      text: string; // markdown
      response: AskResponse;
    }
  | { id: string; role: 'error'; text: string };

export default function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [inlineError, setInlineError] = useState<string | null>(null);
  const taRef = useRef<HTMLTextAreaElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  // Auto-resize textarea
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = '0px';
    el.style.height = Math.min(el.scrollHeight, 240) + 'px';
  }, [input]);

  // Scroll to bottom on new message
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  const canSend = input.trim().length > 0 && !loading;

  const send = useCallback(async () => {
    const q = input.trim();
    if (!q) return;
    setInlineError(null);
    setLoading(true);
    setInput('');
    const id = crypto.randomUUID();
    setMessages((m) => [...m, { id: id + ':u', role: 'user', text: q }]);

    try {
      const res = await ask(q);
      const text = res.answer?.answer ?? '';
      setMessages((m) => [
        ...m,
        {
          id: id + ':a',
          role: 'assistant',
          text,
          response: res,
        },
      ]);
    } catch (err: any) {
      const msg = err?.message || 'Unbekannter Fehler.';
      setMessages((m) => [...m, { id: id + ':e', role: 'error', text: msg }]);
      setInlineError(msg);
    } finally {
      setLoading(false);
    }
  }, [input]);

  // Keyboard: Enter = senden, Shift+Enter = Zeilenumbruch
  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (canSend) void send();
    }
  };

  const placeholder = 'Frage eingeben … (Enter = senden, Shift+Enter = Zeilenumbruch)';
  const hasMessages = messages.length > 0;

  return (
    <section aria-label="Chatbereich" className="mx-auto mt-4 flex min-h-[80vh] w-full max-w-chat flex-col gap-3 sm:mt-6">
      <div
        ref={listRef}
        className="flex-1 overflow-auto rounded-xl border border-neutral-200 bg-white p-3"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
      >
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-3">
          {!hasMessages && (
            <div className="rounded-lg border border-dashed border-neutral-300 p-6 text-center text-neutral-600">
              Stelle eine Frage, z. B.: <span className="font-medium">„Wann haben die THWS Gebäude geschlossen?“</span>
            </div>
          )}
          {messages.map((m) => (
            <div key={m.id} className={clsx('flex', m.role === 'user' ? 'justify-end' : 'justify-start')}>
              <div className="max-w-full">
                <MessageBubble
                  role={m.role}
                  durationSeconds={m.role === 'assistant' ? m.response.duration_seconds : undefined}
                >
                  {m.text}
                </MessageBubble>
                {m.role === 'assistant' && m.response.answer?.sources?.length ? (
                  <SourcesList sources={m.response.answer.sources} />
                ) : null}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bubble bubble-bot text-neutral-600">Antwort wird geladen …</div>
            </div>
          )}
        </div>
      </div>

      {inlineError && (
        <div className="inline-error" role="alert">
          {inlineError}
        </div>
      )}

      <form
        className="sticky bottom-0 mx-auto w-full max-w-3xl space-y-2 rounded-xl"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSend) void send();
        }}
        aria-label="Eingabebereich"
      >
        <label htmlFor="chat-input" className="sr-only">
          Frage an den ChatBot
        </label>
        <textarea
          id="chat-input"
          ref={taRef}
          className="chat-input"
          placeholder={placeholder}
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          aria-label="Nachricht eingeben"
        />
        <div className="flex items-center justify-end">
          <button
            type="submit"
            className="btn btn-primary"
            aria-label="Senden"
            aria-busy={loading}
            disabled={!canSend}
          >
            {loading ? <span className="spinner" aria-hidden="true"></span> : SendIcon}
            <span>Senden</span>
          </button>
        </div>
      </form>
    </section>
  );
}

const SendIcon = (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
    <path fill="currentColor" d="M3.4 20.4L22 12L3.4 3.6L3 10l11 2l-11 2z"/>
  </svg>
);