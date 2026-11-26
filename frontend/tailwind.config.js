/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        tech: ['"Chakra Petch"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
        sans: ['"Plus Jakarta Sans"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      colors: {
        yt: {
          red: '#ff0033',
          redHover: '#e6002e',
          dark: '#000000',
          panel: '#0a0a0c',
          surface: '#121214',
          card: '#18181b',
          border: '#27272a',
          muted: '#71717a',
        },
      },
      boxShadow: {
        'yt-red': '0 0 20px -3px rgba(255, 0, 51, 0.35)',
        'yt-subtle': '0 1px 3px rgba(0, 0, 0, 0.5), 0 1px 2px rgba(0, 0, 0, 0.3)',
      },
    },
  },
  plugins: [],
};
