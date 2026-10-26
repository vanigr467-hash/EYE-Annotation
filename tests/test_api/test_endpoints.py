"""
API Endpoint Unit Tests.
Tests FastAPI routes using Starlette TestClient.
"""

import os
import sys
import uuid
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app
from backend.database import init_db

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    init_db()
    yield

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "api_prefix" in data

def test_project_crud():
    uid = uuid.uuid4().hex[:6]
    proj_data = {
        "name": f"API Test Project {uid}",
        "description": "Project for automated API testing",
        "owner": "tester"
    }
    create_resp = client.post("/api/v1/projects/", json=proj_data)
    assert create_resp.status_code == 201, create_resp.text
    created = create_resp.json()
    assert created["name"] == proj_data["name"]
    proj_id = created["id"]

    # Get project
    get_resp = client.get(f"/api/v1/projects/{proj_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == proj_id

    # List projects
    list_resp = client.get("/api/v1/projects/")
    assert list_resp.status_code == 200
    assert any(p["id"] == proj_id for p in list_resp.json())

    # Update project
    update_resp = client.put(f"/api/v1/projects/{proj_id}", json={"description": "Updated description"})
    assert update_resp.status_code == 200
    assert update_resp.json()["description"] == "Updated description"

    # Clean up
    del_resp = client.delete(f"/api/v1/projects/{proj_id}")
    assert del_resp.status_code in (200, 204)

def test_annotation_crud_and_versioning():
    uid = uuid.uuid4().hex[:6]
    p_resp = client.post("/api/v1/projects/", json={"name": f"Ann Test Proj {uid}", "owner": "test"})
    assert p_resp.status_code == 201, p_resp.text
    p_id = p_resp.json()["id"]

    # Create annotation
    ann_data = {
        "label_type": "pupil",
        "cx": 100.0,
        "cy": 100.0,
        "rx": 25.0,
        "ry": 25.0,
        "angle": 0.0,
        "source": "human",
        "status": "approved",
        "created_by": "tester"
    }
    ann_resp = client.post("/api/v1/annotations/", json=ann_data)
    assert ann_resp.status_code == 201, ann_resp.text
    ann = ann_resp.json()
    assert "id" in ann, f"Expected id in response: {ann}"
    ann_id = ann["id"]
    assert ann["label_type"] == "pupil"
    assert ann["cx"] == 100.0

    # Verify version was created
    ver_resp = client.get(f"/api/v1/annotations/{ann_id}/versions")
    assert ver_resp.status_code == 200
    versions = ver_resp.json()
    assert len(versions) >= 1
    assert versions[0]["action"] == "create"

    # Update annotation
    up_resp = client.put(f"/api/v1/annotations/{ann_id}", json={"cx": 105.0, "cy": 102.0})
    assert up_resp.status_code == 200
    updated = up_resp.json()
    assert updated["cx"] == 105.0

    # Verify second version exists
    ver_resp2 = client.get(f"/api/v1/annotations/{ann_id}/versions")
    assert len(ver_resp2.json()) >= 2

    # Clean up
    client.delete(f"/api/v1/annotations/{ann_id}")
    client.delete(f"/api/v1/projects/{p_id}")
