// tailwind.config.cjs
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'thws-primary': '#FF6900', // <— flacher Key
      },
      maxWidth: { chat: '48rem' },
      boxShadow: { subtle: '0 1px 2px rgba(0,0,0,0.06)' },
      borderRadius: { bubble: '1.25rem' },
    },
  },
  plugins: [],
};