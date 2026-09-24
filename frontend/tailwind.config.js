/** @type {import('tailwindcss').Config} */

/**
 * Design tokens.
 *
 * Colours are declared as CSS custom properties in `index.css` and mapped
 * to Tailwind names here, rather than being listed twice as `x` and
 * `dark:x`. That means a component says `bg-surface` once and is correct
 * in both themes — the alternative, a `dark:` variant on every colour
 * utility, is where theme drift comes from: one forgotten variant and a
 * panel is white-on-white for half the users.
 *
 * `<alpha-value>` keeps Tailwind's opacity modifiers working through the
 * variables, so `text-muted/70` still does what it looks like.
 */
const token = (name) => `rgb(var(${name}) / <alpha-value>)`

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  // Class strategy rather than media: the user's explicit choice has to be
  // able to override the OS preference, and it has to persist.
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // --- semantic surfaces, in depth order ---
        canvas: token('--canvas'),
        surface: token('--surface'),
        raised: token('--raised'),
        sunken: token('--sunken'),

        // --- text, in emphasis order ---
        strong: token('--text-strong'),
        body: token('--text-body'),
        muted: token('--text-muted'),
        faint: token('--text-faint'),

        line: token('--line'),
        'line-strong': token('--line-strong'),
        control: token('--control'),

        // One accent, used only for interactive and "this is selected"
        // states. Everything else is neutral, so the papers are what the
        // eye lands on rather than the chrome.
        accent: {
          50: token('--accent-50'),
          100: token('--accent-100'),
          200: token('--accent-200'),
          300: token('--accent-300'),
          400: token('--accent-400'),
          500: token('--accent-500'),
          600: token('--accent-600'),
          700: token('--accent-700'),
          fg: token('--accent-fg'),
        },

        // Per-source identity, so a badge is recognisable before it is read.
        source: {
          pubmed: token('--source-pubmed'),
          arxiv: token('--source-arxiv'),
          crossref: token('--source-crossref'),
        },

        positive: token('--positive'),
        caution: token('--caution'),
        critical: token('--critical'),
      },

      fontFamily: {
        sans: ['Inter var', 'Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },

      // A real scale rather than Tailwind's defaults at every step: the
      // gap between a title and its metadata is what creates hierarchy,
      // and the stock 14/16 pairing is too close to read as one.
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.01em' }],
        xs: ['0.75rem', { lineHeight: '1.125rem' }],
        sm: ['0.8125rem', { lineHeight: '1.25rem' }],
        base: ['0.875rem', { lineHeight: '1.5rem' }],
        md: ['0.9375rem', { lineHeight: '1.5rem' }],
        lg: ['1.0625rem', { lineHeight: '1.5rem', letterSpacing: '-0.011em' }],
        xl: ['1.25rem', { lineHeight: '1.75rem', letterSpacing: '-0.017em' }],
        '2xl': ['1.5rem', { lineHeight: '2rem', letterSpacing: '-0.021em' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem', letterSpacing: '-0.023em' }],
      },

      borderRadius: { md: '0.375rem', lg: '0.5rem', xl: '0.75rem', '2xl': '1rem' },

      // Two-layer shadows: a tight contact shadow plus a wider ambient one.
      // Sharpened for a modern crisp look.
      boxShadow: {
        subtle: '0 1px 2px 0 rgb(var(--shadow) / 0.05)',
        raised: '0 1px 3px 0 rgb(var(--shadow) / 0.1), 0 1px 2px -1px rgb(var(--shadow) / 0.1)',
        float: '0 4px 6px -1px rgb(var(--shadow) / 0.1), 0 2px 4px -2px rgb(var(--shadow) / 0.1)',
        dialog: '0 10px 15px -3px rgb(var(--shadow) / 0.1), 0 4px 6px -4px rgb(var(--shadow) / 0.1)',
      },

      keyframes: {
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(3px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'scale-in': {
          from: { opacity: '0', transform: 'scale(0.98) translateY(6px)' },
          to: { opacity: '1', transform: 'scale(1) translateY(0)' },
        },
        shimmer: { '100%': { transform: 'translateX(100%)' } },
        'pulse-dot': {
          '0%, 100%': { opacity: '0.25', transform: 'scale(0.85)' },
          '50%': { opacity: '1', transform: 'scale(1)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 200ms cubic-bezier(0.22, 1, 0.36, 1) both',
        'scale-in': 'scale-in 180ms cubic-bezier(0.22, 1, 0.36, 1) both',
        shimmer: 'shimmer 1.6s infinite',
        'pulse-dot': 'pulse-dot 1.2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
