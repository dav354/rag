// src/components/SourcesList.tsx
import React, { useMemo, useState } from 'react';
import { Disclosure } from '@headlessui/react';
import type { Source } from '@/types';
import clsx from 'clsx';

function isUrl(s: string) {
  return /^https?:\/\//i.test(s);
}

type Props = { sources: Source[] };

export default function SourcesList({ sources }: Props) {
  const unique = useMemo(() => {
    const map = new Map<string, Source>();
    for (const s of sources || []) {
      if (!map.has(s.id)) map.set(s.id, s);
    }
    return Array.from(map.values());
  }, [sources]);

  if (!unique.length) return null;

  return (
    <div className="mt-2">
      <Disclosure>
        {({ open }) => (
          <>
            <Disclosure.Button
              className={clsx(
                'w-full rounded-md border px-3 py-2 text-left text-sm font-medium',
                open
                  ? 'border-thws-primary bg-white text-thws-primary'
                  : 'border-neutral-300 bg-white text-neutral-800 hover:border-thws-primary',
              )}
              aria-label="Quellen ein- oder ausklappen"
            >
              Quellen ({unique.length})
            </Disclosure.Button>
            <Disclosure.Panel className="mt-2 space-y-3">
              {unique.map((s) => (
                <SourceItem key={s.id} source={s} />
              ))}
            </Disclosure.Panel>
          </>
        )}
      </Disclosure>
    </div>
  );
}

function SourceItem({ source }: { source: Source }) {
  const [expanded, setExpanded] = useState(false);
  const { id, content, file_path } = source;
  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-3">
      <div className="mb-1 text-xs font-semibold text-neutral-600">
        Quelle #{id}
      </div>
      <p className={clsx('whitespace-pre-wrap text-sm text-neutral-800', !expanded && 'line-clamp-4')}>
        {content || '—'}
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-3">
        <button
          type="button"
          className="text-sm font-medium text-thws-primary underline underline-offset-2"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
        >
          {expanded ? 'Weniger' : 'Mehr anzeigen'}
        </button>
        {file_path && isUrl(file_path) && (
          <a
            href={file_path}
            target="_blank"
            rel="noopener noreferrer nofollow"
            className="text-sm text-neutral-700 underline underline-offset-2"
          >
            Quelle öffnen
          </a>
        )}
      </div>
    </div>
  );
}