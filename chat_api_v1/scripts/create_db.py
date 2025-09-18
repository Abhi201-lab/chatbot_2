import sys
from dotenv import load_dotenv
import os
import psycopg2
from psycopg2 import sql
from sqlalchemy.engine import make_url

# load .env from the chat_api folder (script lives in scripts/)
base = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(base, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("DATABASE_URL not set in .env"); sys.exit(1)

url = make_url(DATABASE_URL)
target_db = url.database or "chatdb"

conn_kwargs = {
    "host": url.host or "localhost",
    "port": url.port or 5432,
    "user": url.username,
    "password": url.password,
    "dbname": "postgres",  # connect to default DB to create target DB
}

conn = None
try:
    # do not use "with psycopg2.connect(...) as conn" because that starts a transaction block
    conn = psycopg2.connect(**conn_kwargs)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
        exists = cur.fetchone() is not None
        if exists:
            print(f"Database '{target_db}' already exists.")
        else:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db)))
            print(f"Database '{target_db}' created.")
except Exception as e:
    print("Failed to create database:", e)
    raise
finally:
    if conn:
        conn.close()