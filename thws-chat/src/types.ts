// src/types.ts
export type AskRequest = { query: string };

export type Source = {
  id: string;
  content: string;
  file_path: string; // evtl. URL
};

export type AskAnswer = {
  answer: string;
  sources: Source[];
};

export type AskResponse = {
  question: string;
  answer: AskAnswer;
  mode: string;
  duration_seconds: number;
};
