# Galaxy Tool Recommender — PoC

Seeded a Neo4j DB with synthetic Galaxy histories and built two models:

* **HSP**: sequence-based next tool prediction.
* **RIC**: session-memory blending of confidence scores.

## Files

* `synthetic_galaxy_data.cypher` — seeds Neo4j.
* `ric_demo_from_json.py` / `ric_demo_output.py` — demo scripts.
* `neo4j_query_table_data_2025-9-21.json` — sample JSON input.
* `ric_demo_output.csv`, `ric_demo_output.html` — demo outputs.
* `.env.template` — template for DB credentials.

## Run demo (no DB)

```bash
python ric_demo_from_json.py --json neo4j_query_table_data_2025-9-21.json
```

## Run demo (with DB)

1. Copy `.env.example` to `.env` and fill with Aura creds.
2. Run:

```bash
python ric_demo_from_json.py
```

## Sample Cypher

Top co-occurrence for `tool_fastqc`:

```cypher
MATCH (start:Tool {id:'tool_fastqc'})-[r:TOOL_CO_OCCURRENCE]->(rec:Tool)
RETURN rec.id, r.weight ORDER BY r.weight DESC LIMIT 10;
```
