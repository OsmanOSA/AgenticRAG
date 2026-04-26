from dataclasses import dataclass, field
from typing import (Dict, Any, Optional, 
                    List, TypedDict)


@dataclass
class BaseChunk:
    id: str
    doc_id: str
    type: str
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class TextChunk(BaseChunk):
    type: str = field(default="Text", init=False)


@dataclass
class TableChunk(BaseChunk):
    type: str = field(default="Table", init=False)
    table_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["table_id"] = self.table_id
        return d


@dataclass
class ImageChunk(BaseChunk):
    type: str = field(default="Image", init=False)
    image_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["image_id"] = self.image_id
        return d

class DocChunks(TypedDict):
    text:   List[TextChunk]
    tables: List[TableChunk]
    images: List[ImageChunk]




# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class JudgeScore:
    score: float
    reasoning: str


@dataclass
class JudgeResult:
    faithfulness: JudgeScore
    relevance:    JudgeScore
    completeness: JudgeScore

    @property
    def overall(self) -> float:
        """Score moyen pondéré (faithfulness a plus de poids)."""
        return round(
            self.faithfulness.score * 0.4
            + self.relevance.score    * 0.35
            + self.completeness.score * 0.25,
            3,
        )

    def to_dict(self) -> dict:
        return {
            "faithfulness":  {"score": self.faithfulness.score,  "reasoning": self.faithfulness.reasoning},
            "relevance":     {"score": self.relevance.score,     "reasoning": self.relevance.reasoning},
            "completeness":  {"score": self.completeness.score,  "reasoning": self.completeness.reasoning},
            "overall":       self.overall,
        }
