import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator


app = FastAPI(title="Project October API")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "processed_metadata.csv"
)
PROCESSED_METADATA_ENV = "PROJECT_OCTOBER_PROCESSED_METADATA"


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


class MovieSearchItem(BaseModel):
    id: str
    title: str
    vote_average: float | None = None


class MovieSearchResponse(BaseModel):
    items: list[MovieSearchItem]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def get_processed_metadata_path() -> Path:
    return Path(os.environ.get(PROCESSED_METADATA_ENV, DEFAULT_PROCESSED_METADATA_PATH))


def search_movies_in_catalog(query: str, limit: int) -> list[dict[str, Any]]:
    import pandas as pd

    metadata_path = get_processed_metadata_path()
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    movies_df = pd.read_csv(metadata_path, low_memory=False)
    title_column = "original_title" if "original_title" in movies_df.columns else "title"
    required_columns = {"id", title_column, "vote_average"}
    if not required_columns.issubset(movies_df.columns):
        raise ValueError("Movie catalog data has an unsupported schema.")

    titles = movies_df[title_column].fillna("").astype(str)
    matches = movies_df[
        titles.str.casefold().str.contains(query.casefold(), regex=False, na=False)
    ].head(limit)

    items = []
    for _, row in matches.iterrows():
        vote_average = row["vote_average"]
        items.append(
            {
                "id": str(row["id"]),
                "title": str(row[title_column]),
                "vote_average": None
                if pd.isna(vote_average)
                else float(vote_average),
            }
        )
    return items


@app.get("/movies/search", response_model=MovieSearchResponse)
def search_movies(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
) -> MovieSearchResponse:
    normalized_query = query.strip()
    if not normalized_query:
        raise HTTPException(status_code=422, detail="query must not be empty.")

    try:
        items = search_movies_in_catalog(query=normalized_query, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Movie catalog data is not available. "
                "Build data/processed/processed_metadata.csv first."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return MovieSearchResponse(items=items)


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
