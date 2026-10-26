"""
Full End-to-End Integration Test for the Eye Annotation Platform.
Tests the complete lifecycle:
Project Creation -> Media Upload -> Frame Ingestion -> AI Auto-Detection ->
Human-in-the-Loop Review -> Versioning Audit Trail -> Dataset Export (COCO/CSV/YOLO).
"""

import os
import sys
import io
import uuid
import cv2
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app
from backend.database import init_db
from utils.synthetic_eye import generate_synthetic_eye

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield

def test_full_annotation_lifecycle():
    uid = uuid.uuid4().hex[:6]
    # -------------------------------------------------------------
    # 1. Create a Project
    # -------------------------------------------------------------
    proj_resp = client.post("/api/v1/projects/", json={
        "name": f"Integration Test E2E Project {uid}",
        "description": "Validating full upload -> detect -> review -> export pipeline",
        "owner": "integration_tester"
    })
    assert proj_resp.status_code == 201, proj_resp.text
    project = proj_resp.json()
    proj_id = project["id"]

    # -------------------------------------------------------------
    # 2. Upload an Image Asset (Synthetic Eye Image)
    # -------------------------------------------------------------
    syn_rgb, gt = generate_synthetic_eye(pupil_offset=(5, -4))
    syn_bgr = cv2.cvtColor(syn_rgb, cv2.COLOR_RGB2BGR)
    success, img_encoded = cv2.imencode(".png", syn_bgr)
    assert success
    img_bytes = img_encoded.tobytes()

    upload_resp = client.post(
        "/api/v1/media/upload",
        data={"project_id": str(proj_id)},
        files={"file": ("synthetic_eye.png", io.BytesIO(img_bytes), "image/png")}
    )
    assert upload_resp.status_code == 201, upload_resp.text
    asset = upload_resp.json()
    asset_id = asset["id"]
    assert asset["media_type"] == "image"
    assert asset["status"] == "ready"

    # -------------------------------------------------------------
    # 3. Verify Frame Ingestion
    # -------------------------------------------------------------
    frames_resp = client.get(f"/api/v1/frames/{asset_id}")
    assert frames_resp.status_code == 200
    frames = frames_resp.json()
    assert len(frames) == 1
    frame_id = frames[0]["id"]
    assert frames[0]["width"] == syn_rgb.shape[1]
    assert frames[0]["height"] == syn_rgb.shape[0]

    # Verify frame image retrieval
    img_get_resp = client.get(f"/api/v1/frames/{frame_id}/image")
    assert img_get_resp.status_code == 200
    assert len(img_get_resp.content) > 0

    # -------------------------------------------------------------
    # 4. Trigger AI Auto-Detection on Frame
    # -------------------------------------------------------------
    infer_resp = client.post(f"/api/v1/inference/detect/{frame_id}")
    assert infer_resp.status_code == 200, infer_resp.text
    infer_result = infer_resp.json()
    assert infer_result["detected"] >= 1
    annotations = infer_result["annotations"]

    pupil_ann = next((a for a in annotations if a["label_type"] == "pupil"), None)
    iris_ann = next((a for a in annotations if a["label_type"] == "iris"), None)

    assert pupil_ann is not None, "Pupil detection should be created"
    assert iris_ann is not None, "Iris detection should be created"

    assert pupil_ann["source"] == "ai"
    assert pupil_ann["cx"] > 0 and pupil_ann["cy"] > 0
    assert pupil_ann["rx"] > 0 and pupil_ann["ry"] > 0

    # -------------------------------------------------------------
    # 5. Review Workflow & Audit Trail
    # -------------------------------------------------------------
    # Get review queue
    queue_resp = client.get("/api/v1/review/queue", params={"project_id": proj_id})
    assert queue_resp.status_code == 200

    # If pupil was pending review, approve it
    if pupil_ann["status"] == "pending_review":
        app_resp = client.put(f"/api/v1/review/{pupil_ann['id']}/approve", params={"reviewed_by": "lead_doctor"})
        assert app_resp.status_code == 200
        assert app_resp.json()["status"] == "approved"

    # Edit iris annotation
    edit_resp = client.put(
        f"/api/v1/review/{iris_ann['id']}/edit",
        json={"cx": iris_ann["cx"] + 1.0, "cy": iris_ann["cy"], "rx": iris_ann["rx"], "ry": iris_ann["ry"]},
        params={"reviewed_by": "expert_annotator"}
    )
    assert edit_resp.status_code == 200
    edited_iris = edit_resp.json()
    assert edited_iris["status"] == "approved"
    assert edited_iris["source"] == "human"

    # Verify version audit trail on iris
    ver_resp = client.get(f"/api/v1/annotations/{iris_ann['id']}/versions")
    assert ver_resp.status_code == 200
    versions = ver_resp.json()
    assert len(versions) >= 2  # create + edit
    assert versions[0]["action"] == "edit"
    assert versions[0]["edited_by"] == "expert_annotator"

    # -------------------------------------------------------------
    # 6. Dataset Export Verification
    # -------------------------------------------------------------
    # COCO JSON
    coco_resp = client.get(f"/api/v1/export/{proj_id}/coco")
    assert coco_resp.status_code == 200
    coco_data = coco_resp.json()
    assert "images" in coco_data
    assert "annotations" in coco_data
    assert "categories" in coco_data
    assert len(coco_data["annotations"]) >= 1

    # CSV
    csv_resp = client.get(f"/api/v1/export/{proj_id}/csv")
    assert csv_resp.status_code == 200
    csv_text = csv_resp.text
    assert "label_type" in csv_text
    assert "cx" in csv_text

    # YOLO format
    yolo_resp = client.get(f"/api/v1/export/{proj_id}/yolo")
    assert yolo_resp.status_code == 200
    assert len(yolo_resp.text) > 0

    # Summary
    sum_resp = client.get(f"/api/v1/export/{proj_id}/summary")
    assert sum_resp.status_code == 200
    summary = sum_resp.json()
    assert summary["approved"] >= 1

    # -------------------------------------------------------------
    # 7. Cleanup
    # -------------------------------------------------------------
    client.delete(f"/api/v1/projects/{proj_id}")
