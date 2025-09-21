#!/usr/bin/env python3
"""
RIC demo that can run in two modes:
 - --json <path> : read precomputed Neo4j query JSON (dry-run/demo mode)
 - (default)     : connect to Neo4j using env vars NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD

Usage:
  python ric_demo_from_json.py --json /mnt/data/neo4j_query_table_data_2025-9-21.json
  OR (live)
  python ric_demo_from_json.py
"""
import argparse
import json
import os
import sys
from dotenv import load_dotenv
from typing import Dict, Optional

import pandas as pd
from neo4j import GraphDatabase

# --- Load env for live mode ---
load_dotenv()
AURA_URI = os.getenv("NEO4J_URI")
AURA_USER = os.getenv("NEO4J_USER")
AURA_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ---- Helper: load JSON "confidence table" ----
def load_json_conf_table(path: str) -> pd.DataFrame:
    """
    Accepts JSON file with rows like:
    { "recommendedTool": "tool_x", "confidence_score": 0.12 }
    OR rows like:
    { "from": "tool_x", "confidence": 0.12, "weight": 5 }
    Returns DataFrame with columns: ['tool_from','recommendedTool','confidence']
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)

    # Normalize column names
    if "recommendedTool" in df.columns and "confidence_score" in df.columns:
        df = df.rename(columns={"recommendedTool": "recommendedTool", "confidence_score": "confidence"})
        df["tool_from"] = df.get("from", df.get("tool_from", None))
    elif "from" in df.columns and "confidence" in df.columns:
        df = df.rename(columns={"from": "tool_from"})
        # if the rows are aggregated edges they may have 'from','to','confidence','weight'
        if "to" in df.columns:
            df = df.rename(columns={"to": "recommendedTool"})
    else:
        # Try best effort: find two columns that look like tool and confidence
        possible_tool_cols = [c for c in df.columns if c.lower() in ("from","tool","tool_from","recommendedtool","to")]
        possible_conf_cols = [c for c in df.columns if "conf" in c.lower() or "weight" in c.lower()]
        if not possible_tool_cols or not possible_conf_cols:
            raise ValueError("JSON format not recognized. Please provide rows with recommendedTool/confidence_score or from/confidence.")
        df = df.rename(columns={possible_tool_cols[0]: "tool_from", possible_conf_cols[0]: "confidence"})
        # if there's a 'to' or 'recommended' column, set recommendedTool
        if len(possible_tool_cols) > 1:
            df = df.rename(columns={possible_tool_cols[1]: "recommendedTool"})

    # Make sure required columns exist
    if "recommendedTool" not in df.columns:
        # For some JSON shapes we only have edges with 'from' and 'to'; try to compute recommendedTool from 'to'
        if "to" in df.columns:
            df = df.rename(columns={"to": "recommendedTool"})
    df = df[["tool_from", "recommendedTool", "confidence"]].dropna(subset=["recommendedTool"]).reset_index(drop=True)
    return df

# ---- Fuzzy helper to match tool keys (accept 'FastQC' vs 'tool_fastqc') ----
def find_tool_key(candidate: str, all_tools: pd.Index) -> Optional[str]:
    """
    Try to find the best matching tool id in all_tools for the candidate.
    Matching rules (in order):
     - exact match
     - case-insensitive match
     - normalized match (lower, non-alnum -> _)
     - substring match (contains)
    Returns the matched key or None.
    """
    if candidate in all_tools:
        return candidate
    lower_map = {t.lower(): t for t in all_tools}
    if candidate.lower() in lower_map:
        return lower_map[candidate.lower()]

    def normalize(name: str):
        import re
        return re.sub(r'[^0-9a-z]', '_', name.lower())

    norm_map = {normalize(t): t for t in all_tools}
    if normalize(candidate) in norm_map:
        return norm_map[normalize(candidate)]

    # substring match
    for t in all_tools:
        if candidate.lower() in t.lower():
            return t
    return None

# ---- The JSON-backed "query" function (replaces live db call) ----
class JsonConfidenceSource:
    def __init__(self, df_edges: pd.DataFrame):
        """
        df_edges: DataFrame with columns tool_from,recommendedTool,confidence
                  tool_from is the 'start' tool (or sometimes missing). If missing, we fallback to treating recommendedTool as the main axis.
        """
        self.df = df_edges.copy()
        # produce mapping: for each start-tool -> list of (recommended, confidence)
        # If df contains tool_from, use it; otherwise, global fallback per recommendedTool
        if "tool_from" in self.df.columns and not self.df["tool_from"].isnull().all():
            self.df["tool_from"] = self.df["tool_from"].astype(str)
            self.df["recommendedTool"] = self.df["recommendedTool"].astype(str)
            self.map = {}
            for from_tool, group in self.df.groupby("tool_from"):
                # sort by confidence desc
                self.map[from_tool] = group.sort_values("confidence", ascending=False)[["recommendedTool", "confidence"]].to_dict("records")
        else:
            # fallback global mapping: treat recommendedTool as axis (use aggregate confidences)
            global_df = self.df.groupby("recommendedTool").agg(confidence=("confidence","mean")).reset_index()
            self.map = {"__global__": global_df.sort_values("confidence", ascending=False).to_dict("records")}

    def get_top_k(self, last_tool_id: str, k: int = 10, all_tools: Optional[pd.Index] = None):
        # attempt direct lookup; if not found, try normalized matching using all_tools
        if last_tool_id in self.map:
            recs = self.map[last_tool_id][:k]
            return pd.DataFrame(recs).rename(columns={"recommendedTool":"recommendedTool","confidence":"confidence"})
        if all_tools is not None:
            matched = find_tool_key(last_tool_id, all_tools)
            if matched and matched in self.map:
                recs = self.map[matched][:k]
                return pd.DataFrame(recs).rename(columns={"recommendedTool":"recommendedTool","confidence":"confidence"})
        # fallback to global
        if "__global__" in self.map:
            recs = self.map["__global__"][:k]
            return pd.DataFrame(recs).rename(columns={"recommendedTool":"recommendedTool","confidence":"confidence"})
        # nothing
        return pd.DataFrame(columns=["recommendedTool","confidence"])


# ---- The original RIC recommender class, slightly adapted to accept a "confidence source" ----
class UserSessionRecommender:
    def __init__(self, confidence_source, all_tool_ids: list, alpha: float = 0.3):
        """
        confidence_source: object with method get_top_k(last_tool_id, k, all_tools)
        all_tool_ids: list of known tool ids
        """
        self.source = confidence_source
        self.alpha = float(alpha)
        self.session_weights = pd.DataFrame({'tool': all_tool_ids, 'weight': 0.0}).set_index('tool')
        print("--- New User Session Started ---")

    def update_recommendations(self, last_tool_run: str, top_k: int = 5):
        """
        Updates session weights and returns top recommendations.
        """
        # Query confidence scores from the source (json or db)
        confidence_scores = self.source.get_top_k(last_tool_run, k=50, all_tools=self.session_weights.index).set_index('recommendedTool')
        # If empty, we still proceed (no changes)
        print(f"\nStep Details for '{last_tool_run}':")
        print("1. Fading old weights (multiplying by alpha={})...".format(self.alpha))
        self.session_weights['weight'] *= self.alpha

        print("2. Blending in new confidence scores...")
        for tool, row in confidence_scores.iterrows():
            if tool not in self.session_weights.index:
                # try to match human name to known id (best effort)
                matched = find_tool_key(tool, self.session_weights.index)
                if matched:
                    tool = matched
                else:
                    # unknown tool, skip
                    continue
            new_score = (1.0 - self.alpha) * float(row['confidence'])
            self.session_weights.loc[tool, 'weight'] += new_score

        recommendations = self.session_weights.drop(last_tool_run, errors='ignore')
        return recommendations.sort_values('weight', ascending=False).head(top_k)


# ---- Main driver: choose JSON or live DB mode ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", "-j", help="Path to Neo4j JSON results file (dry-run/demo mode).")
    parser.add_argument("--alpha", type=float, default=0.3, help="RIC alpha (forgetting factor).")
    parser.add_argument("--out", default="ric_demo_output.csv", help="CSV file to write demo recommendations.")
    args = parser.parse_args()

    driver = None
    json_src = None

    try:
        if args.json:
            print("Running in JSON dry-run mode using:", args.json)
            df = load_json_conf_table(args.json)
            json_src = JsonConfidenceSource(df)
            # Derive the set of known tools from JSON (both from and recommendedTool)
            tools = set()
            if "tool_from" in df.columns:
                tools.update(df["tool_from"].dropna().unique().tolist())
            tools.update(df["recommendedTool"].dropna().unique().tolist())
            ALL_TOOLS = sorted(list(tools))
            source = json_src
        else:
            # Live Neo4j mode
            if not all([AURA_URI, AURA_USER, AURA_PASSWORD]):
                print("FATAL: Neo4j credentials not found in env. Either provide --json FILE or set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in .env.")
                sys.exit(1)
            driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))
            with driver.session() as s:
                result = s.run("MATCH (t:Tool) RETURN t.id as toolId")
                ALL_TOOLS = [record["toolId"] for record in result]
            # Build a tiny DB-backed source with the same get_top_k signature
            class DbSource:
                def __init__(self, driver):
                    self.driver = driver
                def get_top_k(self, last_tool_id, k=10, all_tools=None):
                    q = """
                    WITH $last_tool_id AS lastToolId
                    MATCH (lastTool:Tool {id: lastToolId})
                    MATCH (lastTool)<-[:EXECUTED]-(:Job)-[:IN_SESSION]->(s:Session)
                    WITH COLLECT(DISTINCT s) AS sessions_with_last_tool, lastTool
                    WITH size(sessions_with_last_tool) AS last_tool_session_count, sessions_with_last_tool, lastTool
                    UNWIND sessions_with_last_tool AS s
                    MATCH (s)<-[:IN_SESSION]-(:Job)-[:EXECUTED]->(otherTool:Tool)
                    WHERE otherTool <> lastTool
                    WITH last_tool_session_count, otherTool, count(DISTINCT s) AS joint_session_count
                    RETURN otherTool.id AS recommendedTool,
                           toFloat(joint_session_count) / last_tool_session_count AS confidence_score
                    ORDER BY confidence_score DESC
                    LIMIT $k;
                    """
                    with self.driver.session() as sess:
                        res = sess.run(q, last_tool_id=last_tool_id, k=k)
                        rows = [{"recommendedTool": r["recommendedTool"], "confidence": r["confidence_score"]} for r in res]
                    return pd.DataFrame(rows)
            source = DbSource(driver)

        # Initialize recommender
        recommender = UserSessionRecommender(confidence_source=source, all_tool_ids=ALL_TOOLS, alpha=args.alpha)
        print(f"Initialized recommender with {len(ALL_TOOLS)} tools.")

        # Demo: three steps. Use names that likely exist in your dataset; try matching:
        demo_sequence = ["FastQC", "Trimmomatic", "MultiQC"]

        # If JSON tools use normalized ids (like tool_fastqc), map human names to known keys
        demo_sequence_mapped = []
        for name in demo_sequence:
            matched = find_tool_key(name, pd.Index(ALL_TOOLS))
            if matched:
                demo_sequence_mapped.append(matched)
            else:
                print(f"Warning: demo tool '{name}' not found in known tools; using literal '{name}' (may be ignored).")
                demo_sequence_mapped.append(name)

        # Run the 3-step demo and save intermediate recommendation tables
        records = []
        step_no = 1
        for t in demo_sequence_mapped:
            print("\n" + "="*25 + f" STEP {step_no} " + "="*25)
            print(f">>> User runs '{t}'")
            recs = recommender.update_recommendations(t, top_k=10)
            display_df = recs.reset_index().rename(columns={"tool":"tool_id","weight":"score"})
            print("\nRECOMMENDATIONS AFTER STEP {}:".format(step_no))
            print(display_df.to_string(index=False))
            # store for CSV
            for rank, row in enumerate(display_df.itertuples(index=False), start=1):
                records.append({
                    "step": step_no,
                    "last_tool_run": t,
                    "rank": rank,
                    "recommended_tool": row.tool_id,
                    "score": float(row.score)
                })
            step_no += 1

        out_df = pd.DataFrame(records)
        out_csv = args.out
        out_df.to_csv(out_csv, index=False)
        print(f"\nSaved demo recommendations to: {out_csv}")

        # Also create a very small HTML summary that can be sent to your mentor
        html_path = out_csv.replace(".csv", ".html")
        with open(html_path, "w", encoding="utf-8") as fo:
            fo.write("<html><body><h2>RIC Demo Recommendations (3 steps)</h2>\n")
            fo.write(out_df.to_html(index=False))
            fo.write("<p>Notes: scores are the internal RIC session weights after each step.</p></body></html>")
        print(f"Saved HTML summary to: {html_path}")

        # Print sample queries your mentor can paste (if you later provide Aura credentials)
        print("\nSample Aura queries you can share with your mentor (paste into Aura browser):")
        print("1) Show top tool-to-tool co-occurrence for 'FastQC'")
        print("MATCH (start:Tool {id:'FastQC'})-[r:TOOL_CO_OCCURRENCE]->(rec:Tool) RETURN rec.id AS recommendation, r.weight AS support ORDER BY support DESC LIMIT 5;")
        print("\n2) Show a sample user session (graph view) -- replace user id with one in your DB:")
        print("MATCH p=(u:User)-[:BELONGS_TO]->(s:Session)-[:IN_SESSION]->(j:Job)-[:EXECUTED]->(t:Tool) WHERE u.id = 'user_<uuid_here>' RETURN p LIMIT 1;")

    except Exception as exc:
        print("\nAn error occurred:", exc)
        sys.exit(1)
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()
