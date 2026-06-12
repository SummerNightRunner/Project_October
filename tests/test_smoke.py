import csv
import os
import subprocess
import sys

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommendations_example_runs_with_fixture(tmp_path):
    processed_metadata_path = tmp_path / "processed_metadata.csv"
    fieldnames = [
        "id",
        "original_title",
        "overview",
        "adult",
        "vote_average",
        "genres_list",
        "animation",
        "keywords_list",
        "avg_people_rating",
        "collection_id",
    ]
    rows = [
        {
            "id": "862",
            "original_title": "Toy Story",
            "overview": "toy adventure friendship rescue",
            "adult": "0",
            "vote_average": "7.7",
            "genres_list": "['Animation', 'Adventure']",
            "animation": "1",
            "keywords_list": "['toy', 'friendship']",
            "avg_people_rating": "4.0",
            "collection_id": "101",
        },
        {
            "id": "8844",
            "original_title": "Jumanji",
            "overview": "game adventure friendship jungle",
            "adult": "0",
            "vote_average": "6.9",
            "genres_list": "['Adventure', 'Fantasy']",
            "animation": "0",
            "keywords_list": "['game', 'jungle']",
            "avg_people_rating": "3.6",
            "collection_id": "",
        },
    ]
    for index in range(10):
        rows.append(
            {
                "id": str(9000 + index),
                "original_title": f"Fixture Movie {index}",
                "overview": "adventure friendship rescue game",
                "adult": "0",
                "vote_average": str(8.5 - index * 0.1),
                "genres_list": "['Adventure', 'Family']",
                "animation": "0",
                "keywords_list": "['adventure', 'friendship']",
                "avg_people_rating": "3.5",
                "collection_id": "",
            }
        )

    with processed_metadata_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    env = {
        **os.environ,
        "PROJECT_OCTOBER_PROCESSED_METADATA": str(processed_metadata_path),
    }

    result = subprocess.run(
        [sys.executable, "backend/recommendations_func.py"],
        cwd=os.getcwd(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Fixture Movie" in result.stdout
    assert "(Rating:" in result.stdout
