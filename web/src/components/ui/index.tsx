import * as RSlider from "@radix-ui/react-slider";
import * as RSwitch from "@radix-ui/react-switch";
import * as RTooltip from "@radix-ui/react-tooltip";
import * as RSelect from "@radix-ui/react-select";
import clsx from "clsx";
import { AlertTriangle, Check, ChevronDown, Info, Loader2, ShieldCheck, Zap } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { forwardRef } from "react";

export { clsx };

/* ---------------------------------------------------------------- layout */

export function Card({ className, children, ...rest }: { className?: string; children: ReactNode } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={clsx("card", className)} {...rest}>
      {children}
    </div>
  );
}

export function CardHeader({ title, eyebrow, actions, sub }: { title: ReactNode; eyebrow?: ReactNode; actions?: ReactNode; sub?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-3.5">
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow mb-0.5">{eyebrow}</div>}
        <div className="text-[13.5px] font-semibold text-ink">{title}</div>
        {sub && <div className="mt-0.5 text-xs text-muted">{sub}</div>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        {eyebrow && <div className="eyebrow mb-1.5">{eyebrow}</div>}
        <h1 className="text-[22px] font-semibold leading-tight text-ink">{title}</h1>
        {description && <p className="mt-1.5 text-[13.5px] leading-relaxed text-ink-2">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

/* ---------------------------------------------------------------- controls */

type BtnVariant = "primary" | "secondary" | "ghost" | "danger";

export const Button = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & { variant?: BtnVariant; size?: "sm" | "md"; loading?: boolean; icon?: ReactNode }>(
  function Button({ variant = "secondary", size = "md", loading, icon, className, children, disabled, ...rest }, ref) {
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={clsx(
          "inline-flex select-none items-center justify-center gap-1.5 rounded-lg font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
          size === "sm" ? "h-7 px-2.5 text-xs" : "h-8 px-3 text-[13px]",
          variant === "primary" && "bg-ink text-white hover:bg-ink-2",
          variant === "secondary" && "border border-line-strong bg-surface text-ink hover:bg-sunken",
          variant === "ghost" && "text-ink-2 hover:bg-sunken hover:text-ink",
          variant === "danger" && "border border-attack/30 bg-surface text-attack hover:bg-attack-soft",
          className,
        )}
        {...rest}
      >
        {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : icon}
        {children}
      </button>
    );
  },
);

export function Segmented<T extends string | number>({
  value,
  onChange,
  options,
  size = "md",
  className,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  size?: "sm" | "md";
  className?: string;
}) {
  return (
    <div role="radiogroup" className={clsx("inline-flex rounded-lg border border-line bg-sunken p-0.5", className)}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          role="radio"
          aria-checked={o.value === value}
          title={o.title}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
          className={clsx(
            "rounded-md font-medium transition-all disabled:opacity-40",
            size === "sm" ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-[12.5px]",
            o.value === value ? "bg-surface text-ink shadow-card ring-1 ring-line" : "text-muted hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Slider({ value, onChange, min = 0, max = 1, step = 0.01, disabled, label }: { value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; disabled?: boolean; label?: string }) {
  return (
    <RSlider.Root
      className="relative flex h-5 w-full touch-none select-none items-center"
      value={[value]}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onValueChange={(v) => onChange(v[0])}
      aria-label={label}
    >
      <RSlider.Track className="relative h-1 grow rounded-full bg-line">
        <RSlider.Range className="absolute h-full rounded-full bg-ink" />
      </RSlider.Track>
      <RSlider.Thumb className="block h-4 w-4 rounded-full border-2 border-ink bg-surface shadow-card transition-transform hover:scale-110 focus:outline-none focus-visible:ring-4 focus-visible:ring-focus/25" />
    </RSlider.Root>
  );
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: ReactNode }) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-2 text-[12.5px] text-ink-2">
      <RSwitch.Root
        checked={checked}
        onCheckedChange={onChange}
        className="relative h-[18px] w-8 rounded-full bg-line-strong transition-colors data-[state=checked]:bg-ink"
      >
        <RSwitch.Thumb className="block h-3.5 w-3.5 translate-x-0.5 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-[15px]" />
      </RSwitch.Root>
      {label}
    </label>
  );
}

export function Select<T extends string>({ value, onChange, options, className, label }: { value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; hint?: ReactNode }[]; className?: string; label?: string }) {
  return (
    <RSelect.Root value={value} onValueChange={(v) => onChange(v as T)}>
      <RSelect.Trigger
        aria-label={label}
        className={clsx("inline-flex h-8 items-center justify-between gap-2 rounded-lg border border-line-strong bg-surface px-2.5 text-[13px] text-ink hover:bg-sunken", className)}
      >
        <RSelect.Value />
        <RSelect.Icon>
          <ChevronDown className="h-3.5 w-3.5 text-muted" />
        </RSelect.Icon>
      </RSelect.Trigger>
      <RSelect.Portal>
        <RSelect.Content position="popper" sideOffset={4} className="z-50 max-h-80 min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-lg border border-line bg-surface p-1 shadow-pop">
          <RSelect.Viewport>
            {options.map((o) => (
              <RSelect.Item
                key={o.value}
                value={o.value}
                className="relative flex cursor-pointer select-none items-center justify-between gap-6 rounded-md py-1.5 pl-7 pr-2 text-[13px] text-ink outline-none data-[highlighted]:bg-sunken"
              >
                <RSelect.ItemIndicator className="absolute left-2">
                  <Check className="h-3.5 w-3.5" />
                </RSelect.ItemIndicator>
                <RSelect.ItemText>{o.label}</RSelect.ItemText>
                {o.hint && <span className="text-xs text-muted">{o.hint}</span>}
              </RSelect.Item>
            ))}
          </RSelect.Viewport>
        </RSelect.Content>
      </RSelect.Portal>
    </RSelect.Root>
  );
}

export function Tip({ content, children, side = "top" }: { content: ReactNode; children: ReactNode; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <RTooltip.Root delayDuration={150}>
      <RTooltip.Trigger asChild>{children}</RTooltip.Trigger>
      <RTooltip.Portal>
        <RTooltip.Content side={side} sideOffset={6} className="z-50 max-w-xs rounded-md bg-ink px-2.5 py-1.5 text-xs leading-relaxed text-white shadow-pop">
          {content}
          <RTooltip.Arrow className="fill-ink" />
        </RTooltip.Content>
      </RTooltip.Portal>
    </RTooltip.Root>
  );
}

export function InfoTip({ children }: { children: ReactNode }) {
  return (
    <Tip content={children}>
      <button className="inline-flex text-faint hover:text-muted" aria-label="More information">
        <Info className="h-3.5 w-3.5" />
      </button>
    </Tip>
  );
}

/* ---------------------------------------------------------------- display */

export function Badge({ tone = "neutral", children, icon, className }: { tone?: "neutral" | "attack" | "flag" | "good" | "defended" | "fixed" | "dark"; children: ReactNode; icon?: ReactNode; className?: string }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-md px-1.5 py-0.5 text-[11.5px] font-medium",
        tone === "neutral" && "bg-sunken text-ink-2 ring-1 ring-inset ring-line",
        tone === "attack" && "bg-attack-soft text-attack ring-1 ring-inset ring-attack/20",
        tone === "flag" && "bg-flag-soft text-flag-ink ring-1 ring-inset ring-flag/40",
        tone === "good" && "bg-good-soft text-good-ink ring-1 ring-inset ring-good/25",
        tone === "defended" && "bg-defended-soft text-defended ring-1 ring-inset ring-defended/25",
        tone === "fixed" && "bg-[#fdf0ea] text-[#b5431a] ring-1 ring-inset ring-fixed/25",
        tone === "dark" && "bg-ink text-white",
        className,
      )}
    >
      {icon}
      {children}
    </span>
  );
}

export function FlagBadge({ on, compact }: { on: boolean; compact?: boolean }) {
  return on ? (
    <Badge tone="flag" icon={<AlertTriangle className="h-3 w-3" />}>
      {compact ? "Flagged" : "Low-confidence flag raised"}
    </Badge>
  ) : (
    <Badge tone="good" icon={<ShieldCheck className="h-3 w-3" />}>
      {compact ? "No flag" : "Evidence consistent"}
    </Badge>
  );
}

export function AttackBadge({ label }: { label: string }) {
  return (
    <Badge tone="attack" icon={<Zap className="h-3 w-3" />}>
      {label}
    </Badge>
  );
}

export function Stat({ label, value, sub, tone, className }: { label: ReactNode; value: ReactNode; sub?: ReactNode; tone?: "fixed" | "defended" | "attack" | "good"; className?: string }) {
  return (
    <div className={clsx("min-w-0", className)}>
      <div className="eyebrow mb-1 flex items-center gap-1.5">
        {tone && <span className={clsx("h-2 w-2 rounded-full", `bg-${tone}`)} />}
        {label}
      </div>
      <div className="text-[22px] font-semibold leading-none tracking-tight text-ink">{value}</div>
      {sub && <div className="mt-1.5 text-xs text-muted">{sub}</div>}
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-line-strong bg-surface px-1 font-mono text-[10.5px] text-muted shadow-[0_1px_0_rgb(var(--line-strong))]">{children}</kbd>;
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded-md bg-line/70", className)} />;
}

export function Empty({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      {icon && <div className="mb-3 text-faint">{icon}</div>}
      <div className="text-[13.5px] font-medium text-ink">{title}</div>
      {children && <div className="mt-1 max-w-md text-[13px] text-muted">{children}</div>}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className="flex items-start gap-2 rounded-lg border border-attack/25 bg-attack-soft px-3 py-2 text-[13px] text-attack">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{msg}</span>
    </div>
  );
}

export function ModalityDot({ m, className }: { m: string; className?: string }) {
  return <span className={clsx("inline-block h-2 w-2 shrink-0 rounded-full", `bg-${m}`, className)} />;
}

export function CiText({ ci, digits = 3, pct }: { ci?: { mean: number | null; lo: number | null; hi: number | null } | null; digits?: number; pct?: boolean }) {
  if (!ci || ci.mean == null) return <span className="text-faint">—</span>;
  const f = (v: number) => (pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(digits));
  return (
    <span className="num">
      <span className="text-ink">{f(ci.mean)}</span>
      {ci.lo != null && ci.hi != null && ci.lo !== ci.hi && (
        <span className="ml-1 text-[11px] text-muted">
          [{f(ci.lo)}, {f(ci.hi)}]
        </span>
      )}
    </span>
  );
}

export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-3 mt-8 flex items-baseline justify-between first:mt-0">
      <h2 className="text-[15px] font-semibold text-ink">{children}</h2>
      {aside}
    </div>
  );
}
