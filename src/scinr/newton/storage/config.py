"""
storage/config.py — Storage backend configuration from environment variables.

All variables are read at import time.  Change them via a ``.env`` file or
by setting them in the process environment before importing this module.

Variables
---------
STORAGE_BACKEND
    Which backend to use.  ``"none"`` (default), ``"mongodb"``, or ``"custom"``.
    Default: ``"none"``.
MONGODB_URI
    MongoDB connection URI.
    Default: ``"mongodb://localhost:27017"``.
MONGODB_DATABASE
    Name of the MongoDB database.
    Default: ``"scinr"``.
RAW_FILES_COLLECTION
    Name of the MongoDB collection that stores raw-file metadata documents.
    Default: ``"raw_files"``.
PAGES_COLLECTION
    Name of the MongoDB collection that stores converted-page documents.
    Default: ``"converted_pages"``.
GRIDFS_BUCKET
    Name of the GridFS bucket used for binary file storage.
    Default: ``"raw_binaries"``.
"""

import os

STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "none")
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "scinr")
RAW_FILES_COLLECTION: str = os.getenv("MONGODB_RAW_FILES_COLLECTION", "raw_files")
PAGES_COLLECTION: str = os.getenv("MONGODB_PAGES_COLLECTION", "converted_pages")
GRIDFS_BUCKET: str = os.getenv("MONGODB_GRIDFS_BUCKET", "raw_binaries")
