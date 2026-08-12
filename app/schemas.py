from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    filename: str
    content_type: str
    limit: int
    category: str | None = None


class JewelrySearchResult(BaseModel):
    id: int
    km_code: str
    category: str
    image_path: str
    image_url: str
    similarity: float = Field(description="Cosine similarity score")


class SearchResponse(BaseModel):
    query: SearchQuery
    results: list[JewelrySearchResult]
