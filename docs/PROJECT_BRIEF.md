# Project Brief

## One-Line Description

Project October is a movie recommendation platform with a user-facing web app and a public API for third-party projects.

## Problem

People often know what they have watched and liked, but it is hard to turn that history into useful movie recommendations. Existing recommendation feeds are usually closed, hard to tune, or locked inside one platform.

## Product Goal

Build a service that accepts a user's movie history, ratings, and preferences, then returns explainable movie recommendations through both a website and an API.

## Primary Users

- Viewers who want personalized movie suggestions.
- Developers who want recommendation results through an API.
- Product builders who need a movie discovery component for another app.

## MVP Scope

- Search movies in the local dataset.
- Let a user manually add watched, liked, disliked, and rated movies.
- Generate recommendations from selected movies and stored history.
- Exclude already watched movies.
- Provide a basic web interface.
- Provide a documented API endpoint for recommendations.

## Later Scope

- User accounts and persistent profiles.
- CSV import for watched history.
- TMDB metadata enrichment.
- Optional Kinopoisk research/import path if legally and technically stable.
- API keys, rate limits, usage logs, and developer documentation.
- Hybrid recommendation models using collaborative signals.

## Non-Goals For The MVP

- Streaming or hosting movie video.
- Scraping private user accounts without explicit user consent.
- Production-grade billing.
- Full mobile app.
