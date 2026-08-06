"""
Run this script once to create the application-level database tables.

For anything beyond a prototype use Alembic migrations instead:

    pip install alembic
    alembic init app/db/migrations
    # Edit alembic.ini sqlalchemy.url, then:
    alembic revision --autogenerate -m "create projects table"
    alembic upgrade head
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app` is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.db.models import Base
from app.db.session import engine

Base.metadata.create_all(engine)
print("Application tables created successfully.")
