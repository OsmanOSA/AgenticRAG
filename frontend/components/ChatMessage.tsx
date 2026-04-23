"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import type { SourceItem } from "@/lib/api";
import { SourceCard } from "./SourceCard";

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
}

export function ChatMessage({ role, content, sources }: Message) {
  const isUser = role === "user";

  return (
    <div className={cn("flex flex-col gap-2", isUser && "items-end")}>
      <div
        className={cn(
          "rounded-2xl px-4 py-3 max-w-3xl text-sm",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground"
        )}
      >
        {isUser ? (
          <p>{content}</p>
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({ children }) => (
                <div className="overflow-x-auto my-2">
                  <table className="text-xs border-collapse w-full">{children}</table>
                </div>
              ),
              th: ({ children }) => (
                <th className="border border-border px-2 py-1 bg-muted font-semibold text-left">
                  {children}
                </th>
              ),
              td: ({ children }) => (
                <td className="border border-border px-2 py-1">{children}</td>
              ),
            }}
          >
            {content}
          </ReactMarkdown>
        )}
      </div>

      {sources && sources.length > 0 && (
        <details className="max-w-3xl w-full">
          <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground mb-2">
            {sources.length} source{sources.length > 1 ? "s" : ""} utilisée
            {sources.length > 1 ? "s" : ""}
          </summary>
          <div className="flex flex-col gap-2">
            {sources.map((s, i) => (
              <SourceCard key={i} source={s} index={i + 1} />
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
