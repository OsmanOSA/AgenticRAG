"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft, RefreshCw, Table2, Image,
  Hash, Coins, FolderOpen,
} from "lucide-react";
import { fetchStats, triggerIngest, type StatsResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

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

// ── Document Row ──────────────────────────────────────────────────────────────

// ── Page ──────────────────────────────────────────────────────────────────────

export default function SettingsPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    <div className="min-h-full bg-white">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white border-b border-[#e0e0e0]">
        <div className="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="size-9 rounded-full hover:bg-[#f1f3f4] flex items-center justify-center text-[#5f6368] transition-colors"
            >
              <ArrowLeft className="size-5" />
            </Link>
            <h1 className="text-[18px] font-medium text-[#1f1f1f]">Paramètres de l'index</h1>
          </div>
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

        {/* Error */}
        {error && (
          <div className="rounded-2xl bg-[#fce8e6] border border-[#f28b82] px-5 py-4 text-sm text-[#c5221f]">
            {error}
          </div>
        )}

        {/* Loading skeleton */}
        {loading && !stats && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="h-28 rounded-2xl bg-[#f1f3f4] animate-pulse" />
            ))}
          </div>
        )}

        {stats && (
          <>
            {/* Global stats */}
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

            {/* Breakdown */}
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
  );
}
