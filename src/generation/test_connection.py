import vertexai
from dotenv import load_dotenv
import os

load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
vertexai.init(project=project_id, location="us-central1")

print(f"Connected to project: {project_id}")
print("Vertex AI initialized successfully.")