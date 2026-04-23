"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import type { SourceItem } from "@/lib/api";

interface Props {
  source: SourceItem;
  index: number;
}

export function SourceCard({ source, index }: Props) {
  const isTable = source.type === "Table";
  const filename = source.source?.split(/[\\/]/).pop() ?? "—";

  return (
    <Card className="text-sm border-border/50">
      <CardHeader className="py-2 px-3 flex flex-row items-center gap-2 space-y-0">
        <span className="text-muted-foreground font-mono">[{index}]</span>
        <Badge variant={isTable ? "default" : "secondary"} className="text-xs">
          {isTable ? "📊 Table" : "📄 Text"}
        </Badge>
        <span className="truncate text-muted-foreground flex-1">{filename}</span>
        <span className="text-muted-foreground shrink-0">
          p.{source.page_number ?? "?"}
        </span>
        <span className="font-mono text-xs text-muted-foreground shrink-0">
          {source.rerank_score.toFixed(4)}
        </span>
      </CardHeader>
      <CardContent className="px-3 pb-3">
        <pre className="whitespace-pre-wrap font-mono text-xs bg-muted/50 rounded p-2 max-h-40 overflow-y-auto">
          {source.content.slice(0, 400)}
          {source.content.length > 400 ? "…" : ""}
        </pre>
      </CardContent>
    </Card>
  );
}
