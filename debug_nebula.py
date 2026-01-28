"""Script to add missing indexes to NebulaGraph"""
from knowledge_graph.nebula_handler import NebulaGraphClient
import time

client = NebulaGraphClient()
session = client._get_session()
if session is None:
    raise RuntimeError("Failed to get NebulaGraph session")
session.execute('USE extraction_db')

print("Adding missing indexes...")

# Add missing tag indexes
tag_types = ['Date']
for tag in tag_types:
    result = session.execute(f"CREATE TAG INDEX IF NOT EXISTS idx_{tag.lower()} ON `{tag}`()")
    print(f"  Tag index {tag}: {result.error_msg() if not result.is_succeeded() else 'OK'}")

# Add missing edge indexes
edge_types = ['SIGNED_ON', 'HAS_AMOUNT', 'EFFECTIVE_ON', 'HAS_VALUE', 'ISSUED', 'DUE_ON', 'BILLED_TO']
for edge in edge_types:
    result = session.execute(f"CREATE EDGE INDEX IF NOT EXISTS idx_{edge.lower()} ON `{edge}`()")
    print(f"  Edge index {edge}: {result.error_msg() if not result.is_succeeded() else 'OK'}")

print("\nRebuilding indexes...")
time.sleep(2)

for tag in tag_types:
    result = session.execute(f"REBUILD TAG INDEX idx_{tag.lower()}")
    print(f"  Rebuild {tag}: {result.as_primitive() if result.is_succeeded() else result.error_msg()}")

for edge in edge_types:
    result = session.execute(f"REBUILD EDGE INDEX idx_{edge.lower()}")
    print(f"  Rebuild {edge}: {result.as_primitive() if result.is_succeeded() else result.error_msg()}")

print("\nWaiting for indexes to be ready...")
time.sleep(5)

# Test the indexes
print("\nTesting indexes:")
stats = client.get_graph_stats()
print(f"  Stats: {stats}")

entities = client.get_all_entities()
print(f"  Entities: {len(entities)}")

relationships = client.get_all_relationships()
print(f"  Relationships: {len(relationships)}")

session.release()
print("\nDone!")
