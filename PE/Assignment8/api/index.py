import sys
import os

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

# Export FastAPI app for Vercel Serverless Function
handler = app
