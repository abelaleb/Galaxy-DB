import pandas as pd
from neo4j import GraphDatabase
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read Aura credentials
AURA_URI = os.getenv("NEO4J_URI")
AURA_USER = os.getenv("NEO4J_USER")
AURA_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Check if the environment variables are set
if not all([AURA_URI, AURA_USER, AURA_PASSWORD]):
    print("FATAL: Please set the NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables.")
    sys.exit(1)


def get_ric_confidence_scores(driver, last_tool_id: str) -> pd.DataFrame:
    """
    Connects to Neo4j, runs the confidence score query, and returns a DataFrame.
    """
    print(f"\n--- Querying Neo4j for confidence scores based on '{last_tool_id}' ---")
    query = """
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
    LIMIT 10;
    """
    records = []
    with driver.session() as session:
        result = session.run(query, last_tool_id=last_tool_id)
        # Note: Neo4j integers are returned as objects with 'low' and 'high' keys in some drivers.
        # We assume the session count conversion happens correctly inside the query.
        records = [{"recommendedTool": record["recommendedTool"], "confidence_score": record["confidence_score"]} for record in result]
    print(f"--- Found {len(records)} results from the database ---")
    return pd.DataFrame(records)


class UserSessionRecommender:
    def __init__(self, driver, all_tool_ids: list, alpha: float = 0.3):
        self.driver = driver
        self.alpha = alpha
        self.session_weights = pd.DataFrame(
            {'tool': all_tool_ids, 'weight': 0.0}
        ).set_index('tool')
        print("--- New User Session Started ---")

    def update_recommendations(self, last_tool_run: str):
        """
        Updates session weights and returns top recommendations.
        """
        confidence_scores = get_ric_confidence_scores(self.driver, last_tool_run).set_index('recommendedTool')
        
        print(f"\nStep Details for '{last_tool_run}':")
        print("1. Fading old weights (multiplying by alpha={})...".format(self.alpha))
        self.session_weights['weight'] *= self.alpha
        
        print("2. Blending in new confidence scores...")
        for tool, row in confidence_scores.iterrows():
            new_score = (1 - self.alpha) * row['confidence_score']
            # Use .get() with a default value of 0.0 to handle tools not yet in session_weights (shouldn't happen with all_tool_ids, but good practice)
            # The .loc access is safer since all_tool_ids should contain the tool.
            self.session_weights.loc[tool, 'weight'] += new_score
            
        recommendations = self.session_weights.drop(last_tool_run, errors='ignore')
        return recommendations.sort_values('weight', ascending=False).head(5)

# --- Main Simulation Logic (3 STEPS) ---
if __name__ == "__main__":
    
    # Define the new, verified pipeline based on the provided co-occurrence data and workflow structure.
    STEP_1_TOOL = 'Trimmomatic'
    STEP_2_TOOL = 'featureCounts'
    STEP_3_TOOL = 'Bedtools_intersect'
    
    try:
        db_driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))
        with db_driver.session() as session:
            result = session.run("MATCH (t:Tool) RETURN t.id as toolId")
            ALL_TOOLS = [record["toolId"] for record in result]

        # 1. Start a new session
        recommender = UserSessionRecommender(driver=db_driver, all_tool_ids=ALL_TOOLS, alpha=0.3)
        print(f"Initialized recommender with {len(ALL_TOOLS)} tools.")

        # =================== STEP 1 ===================
        print("\n" + "="*25 + " STEP 1 " + "="*25)
        print(f">>> User runs '{STEP_1_TOOL}'") 
        recommendations_1 = recommender.update_recommendations(STEP_1_TOOL)
        print("\nRECOMMENDATIONS AFTER STEP 1:")
        print(recommendations_1)
        print("\nInternal Session Memory (Top 5):")
        print(recommender.session_weights.sort_values('weight', ascending=False).head())

        # =================== STEP 2 ===================
        print("\n" + "="*25 + " STEP 2 " + "="*25)
        print(f">>> User now runs '{STEP_2_TOOL}'")
        recommendations_2 = recommender.update_recommendations(STEP_2_TOOL)
        print("\nRECOMMENDATIONS AFTER STEP 2:")
        print(recommendations_2)
        print("\nInternal Session Memory (Top 5):")
        print(recommender.session_weights.sort_values('weight', ascending=False).head())

        # =================== STEP 3 ===================
        print("\n" + "="*25 + " STEP 3 " + "="*25)
        print(f">>> User finally runs '{STEP_3_TOOL}'")
        recommendations_3 = recommender.update_recommendations(STEP_3_TOOL)
        print("\nRECOMMENDATIONS AFTER STEP 3:")
        print(recommendations_3)
        print("\nInternal Session Memory (Top 5):")
        print(recommender.session_weights.sort_values('weight', ascending=False).head())

        db_driver.close()
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print(f"Please check your Aura credentials and that the tools ('{STEP_1_TOOL}', '{STEP_2_TOOL}', '{STEP_3_TOOL}') exist in your database.")