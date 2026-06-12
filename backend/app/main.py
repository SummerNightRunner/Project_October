from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


app = FastAPI(title="Project October API")


class RecommendationsRequest(BaseModel):
    liked_movie_ids: list[str] = Field(..., min_length=1)
    include_adult: bool = False
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("liked_movie_ids")
    @classmethod
    def liked_movie_ids_must_not_be_blank(cls, value: list[str]) -> list[str]:
        normalized_movie_ids = []
        for movie_id in value:
            normalized_movie_id = movie_id.strip()
            if not normalized_movie_id:
                raise ValueError("liked_movie_ids must contain non-empty movie IDs.")
            normalized_movie_ids.append(normalized_movie_id)
        return normalized_movie_ids


class RecommendationItem(BaseModel):
    id: str
    title: str
    vote_average: float | None = None
    site_user_rating: float | str | None = None


class RecommendationsResponse(BaseModel):
    items: list[RecommendationItem]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def build_recommendations(
    liked_movie_ids: list[str], include_adult: bool, limit: int
) -> list[dict[str, Any]]:
    from backend.recommendations_func import get_recommendations

    return get_recommendations(
        selected_movie_ids=liked_movie_ids,
        include_adult=include_adult,
        top_n=limit,
    )


@app.post("/recommendations", response_model=RecommendationsResponse)
def recommendations(request: RecommendationsRequest) -> RecommendationsResponse:
    try:
        items = build_recommendations(
            liked_movie_ids=request.liked_movie_ids,
            include_adult=request.include_adult,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Recommendation data is not available. "
                "Build data/processed/processed_metadata.csv first."
            ),
        ) from exc

    return RecommendationsResponse(items=items)
