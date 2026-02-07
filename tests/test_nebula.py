"""Test NebulaGraph edge insertion after warmup"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

config = Config()
pool = ConnectionPool()
pool.init([('127.0.0.1', 9669)], config)
session = pool.get_session('root', 'nebula')

session.execute('USE extraction_db')

# Test vertex insertion
r1 = session.execute('INSERT VERTEX Organization(name, description) VALUES "test_org_1":("Test Org 1", "A test organization")')
print(f"Vertex insert 1: {r1.is_succeeded()}")

r2 = session.execute('INSERT VERTEX Organization(name, description) VALUES "test_org_2":("Test Org 2", "Another test org")')
print(f"Vertex insert 2: {r2.is_succeeded()}")

# Test edge insertion
r3 = session.execute('INSERT EDGE RELATED_TO() VALUES "test_org_1"->"test_org_2":()')
print(f"Edge insert: {r3.is_succeeded()} - Error: {r3.error_msg()}")

# Verify data
r4 = session.execute('MATCH (o:Organization) RETURN o.Organization.name as name LIMIT 5')
print(f"\nOrganizations in DB:")
print(r4)

session.release()
pool.close()
