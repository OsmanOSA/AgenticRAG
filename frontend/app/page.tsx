"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Link from "next/link";
import {
  Plus, Database, RefreshCw,
  ChevronDown, FileText, Table2, Menu, X,
  Paperclip, Mic, ArrowUp, Settings2, SlidersHorizontal,
} from "lucide-react";
import { fetchStatus, queryRAG, triggerIngest, type StatusResponse, type SourceItem } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
  loading?: boolean;
}

// ── Gemini-style Input ────────────────────────────────────────────────────────

function GeminiInput({ onSend, isLoading, placeholder = "Demander à AgenticRAG" }: {
  onSend: (text: string) => void;
  isLoading?: boolean;
  placeholder?: string;
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  function submit() {
    const q = value.trim();
    if (!q || isLoading) return;
    setValue("");
    onSend(q);
  }

  return (
    <div className="rounded-3xl border border-[#e0e0e0] bg-white shadow-[0_1px_6px_rgba(32,33,36,0.10)] hover:shadow-[0_1px_10px_rgba(32,33,36,0.15)] transition-shadow">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
        placeholder={placeholder}
        disabled={isLoading}
        rows={1}
        className="w-full resize-none bg-transparent px-5 pt-4 pb-2 text-[15px] text-[#1f1f1f] placeholder-[#80868b] focus:outline-none leading-relaxed disabled:opacity-50"
        style={{ minHeight: "52px" }}
      />
      <div className="flex items-center gap-2 px-4 pb-3 pt-1">
        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[#5f6368] hover:bg-[#f1f3f4] transition-colors text-sm">
          <Paperclip className="size-4" />
        </button>
        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[#5f6368] hover:bg-[#f1f3f4] transition-colors text-sm">
          <Settings2 className="size-4" />
          <span>Outils</span>
        </button>
        <div className="flex-1" />
        {isLoading ? (
          <button
            disabled
            className="size-9 rounded-full bg-[#0b57d0] flex items-center justify-center"
          >
            <svg className="size-4 animate-spin text-white" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
              <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
            </svg>
          </button>
        ) : value.trim() ? (
          <button
            onClick={submit}
            className="size-9 rounded-full bg-[#0b57d0] hover:bg-[#0842a0] flex items-center justify-center transition-colors"
          >
            <ArrowUp className="size-4 text-white" />
          </button>
        ) : (
          <button className="size-9 rounded-full bg-[#f0f4f9] hover:bg-[#e8eaed] flex items-center justify-center text-[#5f6368] transition-colors">
            <Mic className="size-4" />
          </button>
        )}
      </div>
    </div>
  );
}

// ── Source Card ───────────────────────────────────────────────────────────────

function SourceCard({ source, index }: { source: SourceItem; index: number }) {
  const [open, setOpen] = useState(false);
  const isTable = source.type === "Table";
  const filename = source.source?.split(/[\\/]/).pop() ?? "—";

  return (
    <div className="rounded-2xl border border-[#e0e0e0] bg-white overflow-hidden">
      <button
        onClick={() => setOpen((p) => !p)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-[#f8f9fa] transition-colors"
      >
        <span className="text-xs font-mono text-[#80868b]">[{index}]</span>
        {isTable
          ? <Table2 className="size-4 text-[#7c4dff] shrink-0" />
          : <FileText className="size-4 text-[#0b57d0] shrink-0" />}
        <span className="text-sm text-[#3c4043] truncate flex-1">{filename}</span>
        <span className="text-xs text-[#80868b] shrink-0">p.{source.page_number ?? "?"}</span>
        <span className="text-xs font-mono text-[#9aa0a6] shrink-0">{source.rerank_score.toFixed(3)}</span>
        <ChevronDown className={cn("size-4 text-[#80868b] shrink-0 transition-transform duration-200", open && "rotate-180")} />
      </button>
      {open && (
        <div className="px-4 pb-4 border-t border-[#f1f3f4]">
          <pre className="whitespace-pre-wrap font-mono text-xs text-[#5f6368] bg-[#f8f9fa] rounded-xl p-3 mt-3 max-h-48 overflow-y-auto leading-relaxed">
            {source.content.slice(0, 600)}{source.content.length > 600 ? "…" : ""}
          </pre>
        </div>
      )}
    </div>
  );
}

// ── Chat Message ──────────────────────────────────────────────────────────────

function ChatMessage({ role, content, sources, loading }: Message) {
  const isUser = role === "user";
  const [showSources, setShowSources] = useState(false);

  return (
    <div className={cn("flex flex-col gap-3", isUser ? "items-end" : "items-start")}>
      {loading ? (
        <div className="flex gap-1.5 items-center py-2 px-1">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="size-2 rounded-full bg-[#bdc1c6] animate-bounce"
              style={{ animationDelay: `${i * 160}ms` }}
            />
          ))}
        </div>
      ) : isUser ? (
        <div className="bg-[#e8f0fe] text-[#1f1f1f] rounded-3xl rounded-tr-md px-5 py-3 text-[15px] leading-relaxed max-w-[78%]">
          {content}
        </div>
      ) : (
        <div className="text-[#1f1f1f] text-[15px] leading-relaxed w-full">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({ children }) => (
                <div className="overflow-x-auto my-4 rounded-2xl border border-[#e0e0e0]">
                  <table className="text-sm border-collapse w-full">{children}</table>
                </div>
              ),
              th: ({ children }) => (
                <th className="px-4 py-3 bg-[#f8f9fa] font-semibold text-left text-[#3c4043] border-b border-[#e0e0e0] text-sm">{children}</th>
              ),
              td: ({ children }) => (
                <td className="px-4 py-3 text-[#5f6368] border-b border-[#f1f3f4] text-sm last:border-0">{children}</td>
              ),
              code: ({ children, className: cls }) => {
                const isBlock = cls?.includes("language-");
                return isBlock ? (
                  <code className="block bg-[#f8f9fa] border border-[#e0e0e0] rounded-xl p-4 text-sm font-mono text-[#1f1f1f] my-3 overflow-x-auto">{children}</code>
                ) : (
                  <code className="bg-[#f1f3f4] text-[#1f1f1f] px-1.5 py-0.5 rounded text-[13px] font-mono">{children}</code>
                );
              },
              p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
              ul: ({ children }) => <ul className="list-disc list-outside pl-5 mb-3 space-y-1.5 text-[#3c4043]">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal list-outside pl-5 mb-3 space-y-1.5 text-[#3c4043]">{children}</ol>,
              li: ({ children }) => <li>{children}</li>,
              h1: ({ children }) => <h1 className="text-xl font-semibold mb-3 text-[#1f1f1f]">{children}</h1>,
              h2: ({ children }) => <h2 className="text-lg font-semibold mb-2.5 text-[#1f1f1f]">{children}</h2>,
              h3: ({ children }) => <h3 className="text-base font-semibold mb-2 text-[#1f1f1f]">{children}</h3>,
              strong: ({ children }) => <strong className="font-semibold text-[#1f1f1f]">{children}</strong>,
              hr: () => <hr className="border-[#e0e0e0] my-5" />,
            }}
          >
            {content}
          </ReactMarkdown>
        </div>
      )}

      {sources && sources.length > 0 && (
        <div className="w-full">
          <button
            onClick={() => setShowSources((p) => !p)}
            className="flex items-center gap-2 text-sm text-[#0b57d0] hover:text-[#0842a0] transition-colors mb-2"
          >
            <Database className="size-3.5" />
            {sources.length} source{sources.length > 1 ? "s" : ""} utilisée{sources.length > 1 ? "s" : ""}
            <ChevronDown className={cn("size-3.5 transition-transform duration-200", showSources && "rotate-180")} />
          </button>
          {showSources && (
            <div className="flex flex-col gap-2">
              {sources.map((s, i) => <SourceCard key={i} source={s} index={i + 1} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function Sidebar({ status, onIngest, ingesting, onNewChat, open, onClose }: {
  status: StatusResponse | null;
  onIngest: () => void;
  ingesting: boolean;
  onNewChat: () => void;
  open: boolean;
  onClose: () => void;
}) {
  return (
    <>
      {open && <div className="fixed inset-0 bg-black/20 z-20 lg:hidden" onClick={onClose} />}
      <aside className={cn(
        "fixed inset-y-0 left-0 z-30 flex flex-col w-72 bg-[#f0f4f9] transition-transform duration-300 lg:relative lg:translate-x-0",
        open ? "translate-x-0" : "-translate-x-full"
      )}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 h-16 shrink-0">
          <span className="text-[22px] font-medium text-[#1f1f1f] tracking-tight">AgenticRAG</span>
          <button onClick={onClose} className="lg:hidden text-[#5f6368] hover:text-[#1f1f1f] transition-colors p-2 rounded-full hover:bg-[#e8eaed]">
            <X className="size-5" />
          </button>
        </div>

        {/* New chat */}
        <div className="px-4 pb-2 shrink-0">
          <button
            onClick={onNewChat}
            className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-white hover:bg-[#e8f0fe] border border-[#c7d2da] text-sm text-[#3c4043] hover:text-[#0b57d0] transition-all shadow-sm w-full"
          >
            <Plus className="size-4" />
            Nouvelle discussion
          </button>
        </div>

        {/* Dashboard link */}
        <div className="px-4 pb-4 shrink-0">
          <Link
            href="/settings"
            className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-[#3c4043] hover:bg-[#e8eaed] transition-all"
          >
            <SlidersHorizontal className="size-4 text-[#9aa0a6]" />
            Tableaux de bord
          </Link>
        </div>

        <div className="flex-1" />
      </aside>
    </>
  );
}

// ── Suggestions ───────────────────────────────────────────────────────────────

const SUGGESTIONS = [
  "Résume les points clés du document",
  "Quels sont les tableaux principaux ?",
  "Explique la méthodologie utilisée",
  "Liste les conclusions importantes",
];

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const pendingRef = useRef<string | null>(null);
  const hasMessages = messages.length > 0;

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(text: string) {
    const q = text.trim();
    if (!q) return;

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: q };
    const loadingMsg: Message = { id: crypto.randomUUID(), role: "assistant", content: "", loading: true };
    setMessages((prev) => [...prev, userMsg, loadingMsg]);
    pendingRef.current = loadingMsg.id;

    try {
      const res = await queryRAG(q);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingRef.current
            ? { ...m, content: res.answer, sources: res.sources, loading: false }
            : m
        )
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingRef.current
            ? { ...m, content: `Erreur : ${err instanceof Error ? err.message : "Inconnu"}`, loading: false }
            : m
        )
      );
    }
  }

  async function handleIngest() {
    setIngesting(true);
    try {
      await triggerIngest();
      setStatus(await fetchStatus());
    } finally {
      setIngesting(false);
    }
  }

  const isLoading = messages.some((m) => m.loading);

  return (
    <div className="flex h-full bg-white">
      <Sidebar
        status={status}
        onIngest={handleIngest}
        ingesting={ingesting}
        onNewChat={() => { setMessages([]); setSidebarOpen(false); }}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Mobile header */}
        <header className="flex items-center gap-3 px-4 h-14 border-b border-[#e0e0e0] lg:hidden shrink-0">
          <button onClick={() => setSidebarOpen(true)} className="text-[#5f6368] hover:text-[#1f1f1f] p-2 rounded-full hover:bg-[#f1f3f4] transition-colors">
            <Menu className="size-5" />
          </button>
          <span className="font-medium text-[#1f1f1f]">AgenticRAG</span>
          {status?.ready && (
            <span className="ml-auto text-xs px-2.5 py-1 rounded-full bg-[#e6f4ea] text-[#137333]">
              {status.point_count} pts
            </span>
          )}
        </header>

        {/* ── EMPTY STATE ── */}
        {!hasMessages && (
          <div className="flex-1 flex flex-col items-center justify-center px-6 gap-8 overflow-y-auto">
            <div className="text-center max-w-xl">
              <h1 className="text-[32px] font-normal text-[#1f1f1f] mb-3 tracking-tight">
                Comment puis-je vous aider ?
              </h1>
              <p className="text-[15px] text-[#5f6368] leading-relaxed">
                Posez une question sur vos documents indexés.
              </p>
            </div>

            <div className="w-full max-w-2xl">
              <GeminiInput onSend={handleSend} isLoading={isLoading} />
            </div>

            <div className="grid grid-cols-2 gap-3 w-full max-w-2xl">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSend(s)}
                  className="px-4 py-3.5 rounded-2xl border border-[#e0e0e0] bg-white hover:bg-[#f8f9fa] hover:border-[#c7d2da] text-sm text-[#3c4043] transition-all text-left leading-snug shadow-sm"
                >
                  {s}
                </button>
              ))}
            </div>

            <p className="text-xs text-[#9aa0a6]">
              {status?.point_count ?? "…"} chunks indexés · Retrieval hybride + Gemma 4
            </p>
          </div>
        )}

        {/* ── CHAT STATE ── */}
        {hasMessages && (
          <>
            <div className="flex-1 overflow-y-auto">
              <div className="max-w-3xl mx-auto px-6 py-8 flex flex-col gap-8">
                {messages.map((m) => <ChatMessage key={m.id} {...m} />)}
                <div ref={bottomRef} />
              </div>
            </div>

            <div className="shrink-0 px-6 pb-6 pt-3">
              <div className="max-w-3xl mx-auto">
                <GeminiInput onSend={handleSend} isLoading={isLoading} />
                <p className="text-center text-xs text-[#9aa0a6] mt-2">
                  AgenticRAG · {status?.point_count ?? "…"} chunks · Gemma 4
                </p>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
