import katex from "katex";
import { useMemo } from "react";

export default function Tex({ children, block }: { children: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: !!block, throwOnError: false, strict: false }), [children, block]);
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
