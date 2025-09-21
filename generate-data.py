import random
import uuid
import re
from datetime import datetime, timedelta

# --- Configuration (SMALLER VERSION) ---
NUM_USERS = 10
NUM_SESSIONS_PER_USER = random.randint(2, 5)
NUM_TOOLS = 25
MIN_STEPS_PER_SESSION = 3
MAX_STEPS_PER_SESSION = 8

# --- Sample Tool Names (human-friendly) ---
TOOLS = [
    "Galaxy_Upload", "FastQC", "Trimmomatic", "BWA-MEM", "Samtools_view",
    "Samtools_sort", "Samtools_index", "BCFtools_mpileup", "BCFtools_call",
    "GATK_HaplotypeCaller", "Picard_MarkDuplicates", "Bedtools_intersect",
    "featureCounts", "STAR", "HISAT2", "Cufflinks", "DESeq2", "MultiQC",
    "BLASTn", "ClustalW", "Cut1", "Filter_by_quality", "ggplot2",
    "PCA_Plot", "Python_script"
]

# --- Small addition: produce safe tool IDs for the graph (minimal change) ---
def sanitize_tool_name(name: str) -> str:
    """
    Normalize a tool name for use as an 'id' in Neo4j:
      - lowercase
      - non-alphanumeric -> underscore
      - prefix with 'tool_'
    Example: "BWA-MEM" -> "tool_bwa_mem"
    """
    safe = re.sub(r"[^0-9A-Za-z]", "_", name).lower()
    return f"tool_{safe}"

# Derived list used when creating nodes / sampling — keeps TOOLS unchanged for readability.
TOOL_IDS = [sanitize_tool_name(t) for t in TOOLS]

def generate_cypher_script():
    """Generates the full Cypher script to create the graph."""
    cypher_commands = []
    cypher_commands.append("MERGE (g:Galaxy {name: 'Galaxy Platform'});")

    # --- Create Tool Nodes using TOOL_IDS (sanitized) ---
    for tool_id in TOOL_IDS:
        cypher_commands.append("MERGE (:Tool {{id: '{}'}})-[:IS_PART_OF]->(g);".format(tool_id))

    # --- Create User and Session Data ---
    for i in range(NUM_USERS):
        user_id = "user_{}".format(uuid.uuid4().hex[:8])
        cypher_commands.append("MERGE (:User {{id: '{}'}});".format(user_id))

        for j in range(NUM_SESSIONS_PER_USER):
            session_id = "session_{}".format(uuid.uuid4().hex[:12])
            cypher_commands.append("MERGE (s:Session {{id: '{}'}})-[:BELONGS_TO]->(:User {{id: '{}'}});".format(session_id, user_id))

            num_steps = random.randint(MIN_STEPS_PER_SESSION, MAX_STEPS_PER_SESSION)
            # sample from sanitized TOOL_IDS
            session_tools = random.sample(TOOL_IDS, num_steps)
            start_time = datetime.now() - timedelta(days=random.randint(1, 365))
            
            previous_job_id = None

            for k in range(num_steps):
                job_id = "job_{}".format(uuid.uuid4().hex[:10])
                tool_id = session_tools[k]
                timestamp_iso = (start_time + timedelta(minutes=k * 5)).isoformat()

                # --- This section is rewritten to be more robust ---
                # 1. Create Job and link to Session
                job_creation_query = "MATCH (s:Session {{id: '{}'}}) MERGE (j:Job {{id: '{}', timestamp: datetime('{}')}})-[:IN_SESSION]->(s);"
                cypher_commands.append(job_creation_query.format(session_id, job_id, timestamp_iso))
                
                # 2. Link Job to the Tool it used
                tool_link_query = "MATCH (j:Job {{id: '{}'}}), (t:Tool {{id: '{}'}}) MERGE (j)-[:EXECUTED]->(t);"
                cypher_commands.append(tool_link_query.format(job_id, tool_id))

                # 3. If not the first job, link to the previous one
                if previous_job_id:
                    precedes_link_query = "MATCH (prev:Job {{id: '{}'}}), (curr:Job {{id: '{}'}}) MERGE (prev)-[:PRECEDES]->(curr);"
                    cypher_commands.append(precedes_link_query.format(previous_job_id, job_id))
                
                previous_job_id = job_id

    return "\n".join(cypher_commands)

# --- Main Execution ---
if __name__ == "__main__":
    cypher_script = generate_cypher_script()
    with open("synthetic_galaxy_data.cypher", "w") as f:
        f.write(cypher_script)
    print("Successfully generated FINAL 'synthetic_galaxy_data.cypher'")
