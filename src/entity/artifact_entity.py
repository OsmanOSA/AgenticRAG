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