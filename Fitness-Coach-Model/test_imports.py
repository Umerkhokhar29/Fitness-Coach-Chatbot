#!/usr/bin/env python3
"""
Quick test script to verify all packages are properly installed
"""

def test_imports():
    """Test if all required packages can be imported"""
    
    print("🔍 Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
        print(f"   pandas version: {pd.__version__}")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
    
    try:
        import openai
        print("✅ openai imported successfully")
        print(f"   openai version: {openai.__version__}")
    except ImportError as e:
        print(f"❌ openai import failed: {e}")
    
    try:
        import chromadb
        print("✅ chromadb imported successfully")
        print(f"   chromadb version: {chromadb.__version__}")
    except ImportError as e:
        print(f"❌ chromadb import failed: {e}")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"❌ sentence-transformers import failed: {e}")
    
    try:
        import sys
        print(f"\n🐍 Python executable: {sys.executable}")
        print(f"🐍 Python version: {sys.version}")
    except Exception as e:
        print(f"❌ Error getting Python info: {e}")

if __name__ == "__main__":
    test_imports()