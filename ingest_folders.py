#!/usr/bin/env python3
"""
CLI script to ingest product folders into the database.

Usage:
    python ingest_folders.py --root /data/raw_products
    python ingest_folders.py --root /data/raw_products --init-db

This script scans the root directory for product subfolders and ingests
all images into the PostgreSQL database.

Environment variables required:
- DATABASE_URL: PostgreSQL connection string
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import db


def main():
    parser = argparse.ArgumentParser(
        description="Ingest product folders into database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all folders
  python ingest_folders.py --root /data/raw_products
  
  # Initialize database tables first, then ingest
  python ingest_folders.py --root /data/raw_products --init-db
  
  # Use environment variable
  export RAW_ROOT_DIR=/data/raw_products
  python ingest_folders.py
        """
    )
    
    parser.add_argument(
        '--root',
        type=str,
        help='Root directory containing product folders (can also use RAW_ROOT_DIR env var)'
    )
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database tables before ingestion'
    )
    
    args = parser.parse_args()
    
    # Get root directory
    import os
    root_dir = args.root or os.getenv('RAW_ROOT_DIR')
    if not root_dir:
        print("Error: --root argument or RAW_ROOT_DIR environment variable is required")
        sys.exit(1)
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        sys.exit(1)
    
    print(f"Ingesting from: {root_dir}")
    print("=" * 60)
    
    try:
        # Initialize database if requested
        if args.init_db:
            print("Initializing database tables...")
            db.init_db()
            print("✓ Database initialized\n")
        
        # Check database connectivity
        if not db.healthcheck():
            print("Error: Cannot connect to database. Check DATABASE_URL.")
            sys.exit(1)
        print("✓ Database connection OK\n")
        
        # Ingest dataset
        print("Starting ingestion...")
        result = db.ingest_dataset(root_dir)
        
        print("\n" + "=" * 60)
        print("Ingestion Summary:")
        print(f"  Products ingested: {result['products']}")
        print(f"  Images ingested:   {result['images']}")
        
        # Show stats
        stats = db.get_stats()
        print("\nDatabase Statistics:")
        print(f"  Total products:    {stats['total_products']}")
        print(f"  Total images:      {stats['total_images']}")
        print(f"  Unlabeled images:  {stats['unlabeled_images']}")
        print(f"  Labeled images:    {stats['labeled_images']}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
