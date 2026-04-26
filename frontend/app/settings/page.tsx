"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  RefreshCw, Table2, Image,
  Hash, Coins, FolderOpen, Plus, SlidersHorizontal, Menu, X,
  MessageSquare,
} from "lucide-react";
import { fetchStats, triggerIngest, type StatsResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Conversation {
  id: string;
  title: string;
  updatedAt: number;
}

const STORAGE_KEY = "agenticrag_convs";

function relativeTime(ts: number): string {
  const diff = Date.now() - ts;
  if (diff < 60_000) return "À l'instant";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} min`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} h`;
  return `${Math.floor(diff / 86_400_000)} j`;
}

function fmt(n: number) {
  return n.toLocaleString("fr-FR");
}

// ── Stat Card ─────────────────────────────────────────────────────────────────

function StatCard({
  label, value, sub, icon, color,
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon: React.ReactNode;
  color: string;
}) {
  return (
    <div className="rounded-xl border border-[#e0e0e0] bg-white px-4 py-3 shadow-sm flex items-center gap-3">
      <div className={cn("size-8 rounded-lg flex items-center justify-center shrink-0", color)}>
        {icon}
      </div>
      <div>
        <p className="text-[20px] font-semibold text-[#1f1f1f] tabular-nums leading-none">
          {typeof value === "number" ? fmt(value) : value}
        </p>
        <p className="text-xs text-[#5f6368] mt-0.5">{label}</p>
        {sub && <p className="text-[11px] text-[#9aa0a6]">{sub}</p>}
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function SettingsPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);

  useEffect(() => {
    try {
      setConversations(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]"));
    } catch {}
  }, []);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setStats(await fetchStats());
    } catch {
      setError("Impossible de récupérer les statistiques. Le backend est-il démarré ?");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function handleIngest() {
    setIngesting(true);
    try {
      await triggerIngest();
      await load();
    } finally {
      setIngesting(false);
    }
  }

  return (
    <div className="flex h-full bg-white">

      {/* Sidebar */}
      <>
        {sidebarOpen && (
          <div className="fixed inset-0 bg-black/20 z-20 lg:hidden" onClick={() => setSidebarOpen(false)} />
        )}
        <aside className={cn(
          "fixed inset-y-0 left-0 z-30 flex flex-col w-72 bg-[#f0f4f9] transition-transform duration-300 lg:relative lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}>
          <div className="flex items-center justify-between px-4 h-16 shrink-0">
            <span className="text-[22px] font-medium text-[#1f1f1f] tracking-tight">AgenticRAG</span>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden text-[#5f6368] hover:text-[#1f1f1f] transition-colors p-2 rounded-full hover:bg-[#e8eaed]"
            >
              <X className="size-5" />
            </button>
          </div>

          <div className="px-4 pb-2 shrink-0">
            <Link
              href="/"
              className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-white hover:bg-[#e8f0fe] border border-[#c7d2da] text-sm text-[#3c4043] hover:text-[#0b57d0] transition-all shadow-sm w-full"
            >
              <Plus className="size-4" />
              Nouvelle discussion
            </Link>
          </div>

          <div className="px-4 pb-3 shrink-0">
            <Link
              href="/settings"
              className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-[#0b57d0] bg-[#e8f0fe] font-medium transition-all"
            >
              <SlidersHorizontal className="size-4" />
              Tableaux de bord
            </Link>
          </div>

          {conversations.length > 0 && (
            <div className="flex-1 overflow-y-auto px-2 pb-4">
              <p className="px-2 pb-1 text-[11px] font-medium text-[#9aa0a6] uppercase tracking-wide">
                Récent
              </p>
              <div className="flex flex-col gap-0.5">
                {conversations.map((conv) => (
                  <Link
                    key={conv.id}
                    href={`/?s=${conv.id}`}
                    className="flex items-start gap-2.5 px-3 py-2.5 rounded-xl text-left text-[#3c4043] hover:bg-[#e8eaed] transition-colors"
                  >
                    <MessageSquare className="size-3.5 shrink-0 mt-0.5 text-[#9aa0a6]" />
                    <div className="min-w-0 flex-1">
                      <p className="text-[13px] truncate leading-snug">{conv.title}</p>
                      <p className="text-[11px] text-[#9aa0a6] mt-0.5">{relativeTime(conv.updatedAt)}</p>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          )}

          {conversations.length === 0 && <div className="flex-1" />}
        </aside>
      </>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Mobile header */}
        <header className="flex items-center gap-3 px-4 h-14 border-b border-[#e0e0e0] lg:hidden shrink-0">
          <button
            onClick={() => setSidebarOpen(true)}
            className="text-[#5f6368] hover:text-[#1f1f1f] p-2 rounded-full hover:bg-[#f1f3f4] transition-colors"
          >
            <Menu className="size-5" />
          </button>
          <span className="font-medium text-[#1f1f1f]">Tableaux de bord</span>
        </header>

        <div className="flex-1 overflow-y-auto">
          {/* Page header */}
          <div className="sticky top-0 z-10 bg-white border-b border-[#e0e0e0]">
            <div className="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
              <h1 className="text-[18px] font-medium text-[#1f1f1f]">Tableaux de bord</h1>
              <div className="flex items-center gap-2">
                <button
                  onClick={load}
                  disabled={loading}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm text-[#5f6368] hover:bg-[#f1f3f4] transition-colors disabled:opacity-50"
                >
                  <RefreshCw className={cn("size-4", loading && "animate-spin")} />
                  Actualiser
                </button>
                <button
                  onClick={handleIngest}
                  disabled={ingesting || loading}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-[#0b57d0] hover:bg-[#0842a0] text-sm text-white transition-colors disabled:opacity-50"
                >
                  <RefreshCw className={cn("size-4", ingesting && "animate-spin")} />
                  {ingesting ? "Indexation…" : "Re-indexer"}
                </button>
              </div>
            </div>
          </div>

          <div className="max-w-4xl mx-auto px-6 py-8 flex flex-col gap-8">

            {error && (
              <div className="rounded-2xl bg-[#fce8e6] border border-[#f28b82] px-5 py-4 text-sm text-[#c5221f]">
                {error}
              </div>
            )}

            {loading && !stats && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[...Array(8)].map((_, i) => (
                  <div key={i} className="h-28 rounded-2xl bg-[#f1f3f4] animate-pulse" />
                ))}
              </div>
            )}

            {stats && (
              <>
                <section>
                  <h2 className="text-sm font-medium text-[#5f6368] uppercase tracking-wider mb-4">Vue d'ensemble</h2>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <StatCard
                      label="Documents"
                      value={stats.document_count}
                      icon={<FolderOpen className="size-5 text-[#0b57d0]" />}
                      color="bg-[#e8f0fe]"
                    />
                    <StatCard
                      label="Chunks totaux"
                      value={stats.total_chunks}
                      icon={<Hash className="size-5 text-[#7c4dff]" />}
                      color="bg-[#f3e8ff]"
                    />
                    <StatCard
                      label="Tokens estimés"
                      value={stats.estimated_tokens}
                      sub={`${fmt(stats.total_chars)} caractères`}
                      icon={<Coins className="size-5 text-[#ea8600]" />}
                      color="bg-[#fef3e2]"
                    />
                    <StatCard
                      label="Tableaux"
                      value={stats.table_chunks}
                      icon={<Table2 className="size-5 text-[#34a853]" />}
                      color="bg-[#e6f4ea]"
                    />
                    <StatCard
                      label="Images"
                      value={stats.image_chunks}
                      icon={<Image className="size-5 text-[#f29900]" />}
                      color="bg-[#fef3e2]"
                    />
                  </div>
                </section>

                <section>
                  <h2 className="text-sm font-medium text-[#5f6368] uppercase tracking-wider mb-4">Répartition des chunks</h2>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { label: "Texte", value: stats.text_chunks, pct: stats.total_chunks ? Math.round(stats.text_chunks / stats.total_chunks * 100) : 0, color: "bg-[#0b57d0]", bg: "bg-[#e8f0fe]", text: "text-[#0b57d0]" },
                      { label: "Tableaux", value: stats.table_chunks, pct: stats.total_chunks ? Math.round(stats.table_chunks / stats.total_chunks * 100) : 0, color: "bg-[#7c4dff]", bg: "bg-[#f3e8ff]", text: "text-[#7c4dff]" },
                      { label: "Images", value: stats.image_chunks, pct: stats.total_chunks ? Math.round(stats.image_chunks / stats.total_chunks * 100) : 0, color: "bg-[#34a853]", bg: "bg-[#e6f4ea]", text: "text-[#34a853]" },
                    ].map((item) => (
                      <div key={item.label} className="rounded-2xl border border-[#e0e0e0] bg-white p-4 shadow-sm">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-sm text-[#5f6368]">{item.label}</span>
                          <span className={cn("text-xs font-medium px-2 py-0.5 rounded-full", item.bg, item.text)}>
                            {item.pct}%
                          </span>
                        </div>
                        <p className="text-2xl font-semibold text-[#1f1f1f] tabular-nums">{fmt(item.value)}</p>
                        <div className="mt-3 h-1.5 rounded-full bg-[#f1f3f4] overflow-hidden">
                          <div className={cn("h-full rounded-full", item.color)} style={{ width: `${item.pct}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
