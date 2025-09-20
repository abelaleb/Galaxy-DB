# seed_galaxy_graph.py

import logging
from neo4j import GraphDatabase
from datetime import datetime, timedelta
import random
import os 
from dotenv import load_dotenv
load_dotenv()


AURA_URI = os.getenv("NEO4J_URI")
AURA_USER = os.getenv("NEO4J_USER")
AURA_PASSWORD = os.getenv("NEO4J_PASSWORD")

class GalaxyGraphSeeder:
    """
    Connects to a Neo4j AuraDB instance and seeds it with dummy data
    simulating Galaxy user activity for a recommendation engine.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def generate_dummy_data(self):
        """
        Generates a list of job records mimicking the Galaxy 'job' table.
        This data is structured to create meaningful tool sequences.
        """
        print("STEP 1: Generating dummy data...")
        
        # Define our "items" - the tools
        tools = [
            'fastqc', 'trimmomatic', 'bwa_mem', 'samtools_view', 
            'samtools_sort', 'bcftools_call', 'cutadapt', 'hisat2'
        ]
        
        # Define common user workflows (tool sequences)
        workflows = [
            ['fastqc', 'trimmomatic', 'bwa_mem', 'samtools_view', 'samtools_sort', 'bcftools_call'],
            ['fastqc', 'cutadapt', 'hisat2', 'samtools_view', 'samtools_sort'],
            ['bwa_mem', 'samtools_view'],
            ['fastqc', 'trimmomatic', 'hisat2', 'samtools_view']
        ]

        jobs = []
        job_id_counter = 1
        num_sessions = 20 # Total number of histories/sessions to create

        for session_id in range(1, num_sessions + 1):
            # Pick a random workflow for this session
            workflow = random.choice(workflows)
            # Slightly vary the workflow length
            session_length = random.randint(max(2, len(workflow) - 2), len(workflow))
            current_tools = workflow[:session_length]
            
            # Use a consistent timestamp for the session start
            current_time = datetime.now() - timedelta(days=random.randint(1, 30))

            for tool_id in current_tools:
                jobs.append({
                    "job_id": job_id_counter,
                    "session_id": f"session_{session_id}",
                    "tool_id": tool_id,
                    "create_time": current_time.isoformat()
                })
                job_id_counter += 1
                # Increment time to ensure sequential order
                current_time += timedelta(minutes=random.randint(1, 5))
        
        print(f"-> Generated {len(jobs)} total jobs across {num_sessions} sessions.")
        return jobs

    def seed_database(self, jobs):
        """
        Executes Cypher queries to create the graph structure based on the
        Hierarchical Sequence Probability (HSP) and Recurrent Item Co-occurrence (RIC) models.
        """
        print("\nSTEP 2: Seeding the Neo4j database...")
        with self.driver.session() as session:
            # Clean up database from previous runs
            print("-> Cleaning up existing data...")
            session.run("MATCH (n) DETACH DELETE n")

            # Create uniqueness constraints for faster lookups
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tool) REQUIRE t.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE")

            # Create :Tool and :Session nodes from the job data
            print("-> Creating Tool and Session nodes...")
            create_nodes_query = """
            UNWIND $jobs as job
            MERGE (s:Session {id: job.session_id})
            MERGE (t:Tool {id: job.tool_id})
            """
            session.run(create_nodes_query, jobs=jobs)

            # Create :CONTAINS relationships
            # This links sessions to the tools used within them.
            print("-> Creating :CONTAINS relationships...")
            create_contains_rel_query = """
            UNWIND $jobs as job
            MATCH (s:Session {id: job.session_id})
            MATCH (t:Tool {id: job.tool_id})
            // Using MERGE prevents duplicate relationships if a tool is used multiple times
            MERGE (s)-[r:CONTAINS]->(t)
            ON CREATE SET r.timestamps = [job.create_time]
            ON MATCH SET r.timestamps = r.timestamps + job.create_time
            """
            session.run(create_contains_rel_query, jobs=jobs)

            # Create :NEXT relationships for tool sequences
            # This is the key relationship for sequential recommendations[cite: 146, 210].
            print("-> Creating :NEXT relationships...")
            create_next_rel_query = """
            // Group jobs by session and sort by time to establish the sequence
            UNWIND $jobs as job
            WITH job.session_id AS sessionId, collect(job) AS jobList
            UNWIND jobList AS jobData
            WITH sessionId, jobData
            ORDER BY jobData.create_time ASC
            WITH sessionId, collect(jobData.tool_id) AS toolSequence
            
            // Iterate through the sequence to create NEXT relationships
            UNWIND range(0, size(toolSequence) - 2) AS i
            WITH toolSequence[i] AS currentTool, toolSequence[i+1] AS nextTool
            
            MATCH (t1:Tool {id: currentTool})
            MATCH (t2:Tool {id: nextTool})
            
            MERGE (t1)-[r:NEXT]->(t2)
            ON CREATE SET r.count = 1
            ON MATCH SET r.count = r.count + 1
            """
            session.run(create_next_rel_query, jobs=jobs)
            print("-> Seeding complete! ✨")


if __name__ == "__main__":
    try:
        seeder = GalaxyGraphSeeder(AURA_URI, AURA_USER, AURA_PASSWORD)
        job_data = seeder.generate_dummy_data()
        seeder.seed_database(job_data)
        seeder.close()
        print("\n✅ Database is seeded and ready for the next step.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print("\n❌ Failed to seed the database. Please check your credentials and connection.")