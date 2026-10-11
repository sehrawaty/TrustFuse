/** Design tokens live in src/styles.css as CSS variables; Tailwind maps utilities onto them. */
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  // classes built from data at runtime (modality / series / status names)
  safelist: [{ pattern: /^(bg|text|border|ring|fill|stroke)-(bvp|acc|eda|temp|fixed|defended|attack|good|flag)$/ }],
  theme: {
    extend: {
      colors: {
        paper: v("paper"),
        surface: v("surface"),
        sunken: v("sunken"),
        line: v("line"),
        "line-strong": v("line-strong"),
        ink: v("ink"),
        "ink-2": v("ink-2"),
        muted: v("muted"),
        faint: v("faint"),
        focus: v("focus"),
        attack: v("attack"),
        "attack-soft": v("attack-soft"),
        flag: v("flag"),
        "flag-ink": v("flag-ink"),
        "flag-soft": v("flag-soft"),
        good: v("good"),
        "good-ink": v("good-ink"),
        "good-soft": v("good-soft"),
        fixed: v("fixed"),
        defended: v("defended"),
        "defended-soft": v("defended-soft"),
        bvp: v("bvp"),
        acc: v("acc"),
        eda: v("eda"),
        temp: v("temp"),
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "system-ui", "-apple-system", '"Segoe UI"', "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      boxShadow: {
        card: "0 1px 2px rgb(16 24 40 / 0.04), 0 1px 3px rgb(16 24 40 / 0.04)",
        pop: "0 12px 32px -8px rgb(16 24 40 / 0.18), 0 2px 6px rgb(16 24 40 / 0.06)",
      },
      borderRadius: { xl: "0.75rem", "2xl": "1rem" },
      keyframes: {
        pulseDot: { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.35" } },
        flow: { to: { strokeDashoffset: "-24" } },
      },
      animation: { pulseDot: "pulseDot 1.6s ease-in-out infinite", flow: "flow 1.2s linear infinite" },
    },
  },
  plugins: [],
};
