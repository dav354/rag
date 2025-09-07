// src/components/Header.tsx
import React from 'react';

export default function Header() {
  return (
    <header
      className="w-full bg-thws-primary text-white"
      role="banner"
      aria-label="THWS Kopfbereich"
    >
      <div className="mx-auto flex w-full max-w-chat items-center gap-3 px-4 py-3 sm:px-6">
        <img
          src="/src/assets/thws-logo.png"
          alt="THWS Logo"
          className="h-8 w-auto select-none"
          draggable={false}
        />
        <h1 className="text-lg font-semibold">THWS ChatBot</h1>
      </div>
    </header>
  );
}