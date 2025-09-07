// src/components/MessageBubble.tsx
import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import clsx from 'clsx';

type Props = {
  role: 'user' | 'assistant' | 'system' | 'error';
  children: string;
  durationSeconds?: number;
};

export default function MessageBubble({ role, children, durationSeconds }: Props) {
  const isUser = role === 'user';
  const isError = role === 'error';

  const bubbleClasses = clsx('bubble', {
    'bubble-user': isUser,
    'bubble-bot': !isUser,
    'border-red-300 bg-red-50 text-red-800': isError,
  });

  const timeHint =
    typeof durationSeconds === 'number'
      ? `${durationSeconds.toFixed(2)}s`
      : undefined;

  return (
    <div className={bubbleClasses} title={timeHint ? `Antwortzeit: ${timeHint}` : undefined}>
      {isUser || isError ? (
        <p className="whitespace-pre-wrap">{children}</p>
      ) : (
        <div className="prose-chat prose-sm sm:prose">
          <ReactMarkdown
            rehypePlugins={[rehypeHighlight]}
            components={{
              a: ({ node, ...props }) => (
                <a
                  {...props}
                  target="_blank"
                  rel="noopener noreferrer nofollow"
                />
              ),
              // FIX: 'inline' entfernt – gibt es in v9 nicht mehr
              code: ({ node, className, children, ...props }) => (
                <code className={clsx(className)} {...props}>
                  {children}
                </code>
              ),
            }}
          >
            {children}
          </ReactMarkdown>
        </div>
      )}
      {!isUser && timeHint && (
        <div className="mt-1 text-xs text-neutral-500" aria-hidden="true">
          Antwortzeit: {timeHint}
        </div>
      )}
    </div>
  );
}