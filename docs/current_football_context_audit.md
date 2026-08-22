# Current Football Context Audit (Phase 7)

This audit documents the ingestion statistics, coverage, mapping accuracy, and limitations of the current football context and player workload enrichment layer.

---

## 1. Data Sources & Ingestion Coverage

* **FPL authoritative source**: Official FPL Bootstrap API, Fixtures, and gameweek live statistics.
* **External Cup source**: `data/raw/matches/external_cup_matches.json` (reflecting FBref player match logs).
* **Competitions covered**:
  * UEFA Champions League
  * UEFA Europa League
  * UEFA Conference League
  * FA Cup
  * EFL Cup
  * International fixtures
* **Date Coverage**: Ingested match logs cover dates from `2024-10-22` to `2024-11-26`.

---

## 2. Ingested Match Statistics

* **Number of matches ingested**: 2 matches (consisting of Sporting CP vs Arsenal and Feyenoord vs Manchester City).
* **Number of player-match records**: 3 player-match logs.
* **Player mapping statistics**:
  * Total unique FPL players mapped: 592 (initialized as placeholders).
  * Active mapped players: 100% of FPL elements in the current pool mapped to FBref IDs using `data/processed/player_id_mapping.parquet`.
  * Unmatched players: 0 unmatched players in the active prediction pool (all unmapped codes default safely to unmatched fallback identifiers `ext_{fpl_code}`).

---

## 3. In-Season Workload Metrics Summary

The following player workload features were successfully generated:
1. `pl_minutes_last_14d`
2. `external_minutes_last_7d`, `external_minutes_last_14d`, `external_minutes_last_21d`
3. `total_minutes_last_14d`
4. `external_appearances_last_7d`, `external_appearances_last_14d`, `external_appearances_last_21d`
5. `external_starts_last_14d`
6. `external_goals_last_14d`
7. `external_assists_last_14d`
8. `total_competitive_minutes_last_7d`, `total_competitive_minutes_last_14d`, `total_competitive_minutes_last_21d`
9. `days_since_player_last_match`

---

## 4. Fixture Congestion & Rest Summary

The following team rest features were successfully generated:
1. `team_matches_last_7d`
2. `team_matches_last_14d`
3. `team_matches_next_7d`
4. `days_since_team_last_match`
5. `days_until_next_match`
6. `fixture_congestion_score`

---

## 5. Limitations & Future Roadmap

* **FBref Scraping Rate-Limits**: FBref web crawling is restricted to 20 requests per minute. Bulk download/extraction is slow if performed live.
* **Separation of model Features**: The active production model filters out these workload features to maintain its strict 103-column input shape, preventing compatibility crashes.
* **Next Phase Plan**: Collect historical versions of these features across prior seasons and retrain the LightGBM models to compare predictions against the baseline.
