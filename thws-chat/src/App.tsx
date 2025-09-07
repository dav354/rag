// src/App.tsx
import React from 'react';
import Header from './components/Header';
import Chat from './components/Chat';

export default function App() {
  return (
    <div className="min-h-screen text-neutral-900">
      <Header />
      <main className="mx-auto w-full max-w-chat px-4 sm:px-6">
        <Chat />
      </main>
    </div>
  );
}
