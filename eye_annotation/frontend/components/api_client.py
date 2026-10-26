"""
API Client for Streamlit frontend to communicate with FastAPI backend.
Provides typed methods for all backend endpoints.
"""

import io
import httpx
from typing import Any, Dict, List, Optional, Tuple

API_BASE = "http://localhost:8000/api/v1"
TIMEOUT = 60.0


class APIClient:
    """HTTP client wrapper for the FastAPI backend."""

    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=TIMEOUT)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def is_backend_running(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------
    def list_projects(self) -> List[Dict[str, Any]]:
        r = self._client.get("/projects/")
        r.raise_for_status()
        return r.json()

    def create_project(self, name: str, description: str = "", owner: str = "default") -> Dict[str, Any]:
        r = self._client.post("/projects/", json={
            "name": name, "description": description, "owner": owner
        })
        r.raise_for_status()
        return r.json()

    def get_project(self, project_id: int) -> Dict[str, Any]:
        r = self._client.get(f"/projects/{project_id}")
        r.raise_for_status()
        return r.json()

    def delete_project(self, project_id: int) -> bool:
        r = self._client.delete(f"/projects/{project_id}")
        return r.status_code in (200, 204)

    # ------------------------------------------------------------------
    # Media Assets
    # ------------------------------------------------------------------
    def upload_media(self, project_id: int, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        files = {"file": (filename, io.BytesIO(file_bytes), "application/octet-stream")}
        r = self._client.post(
            "/media/upload",
            data={"project_id": str(project_id)},
            files=files,
        )
        r.raise_for_status()
        return r.json()

    def list_media(self, project_id: Optional[int] = None, media_type: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if project_id is not None:
            params["project_id"] = project_id
        if media_type is not None:
            params["media_type"] = media_type
        r = self._client.get("/media/", params=params)
        r.raise_for_status()
        return r.json()

    def get_media(self, media_id: int) -> Dict[str, Any]:
        r = self._client.get(f"/media/{media_id}")
        r.raise_for_status()
        return r.json()

    def get_media_image_bytes(self, media_id: int) -> bytes:
        r = self._client.get(f"/media/{media_id}/image")
        r.raise_for_status()
        return r.content

    def delete_media(self, media_id: int) -> bool:
        r = self._client.delete(f"/media/{media_id}")
        return r.status_code in (200, 204)

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    def extract_frames(self, media_asset_id: int, fps: float = 5.0) -> Dict[str, Any]:
        r = self._client.post(
            f"/frames/extract/{media_asset_id}",
            params={"fps": fps},
            timeout=300.0  # video extraction can be slow
        )
        r.raise_for_status()
        return r.json()

    def list_frames(self, media_asset_id: int) -> List[Dict[str, Any]]:
        r = self._client.get(f"/frames/{media_asset_id}")
        r.raise_for_status()
        return r.json()

    def get_frame_image_bytes(self, frame_id: int) -> bytes:
        r = self._client.get(f"/frames/{frame_id}/image")
        r.raise_for_status()
        return r.content

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    def create_annotation(
        self,
        label_type: str,
        cx: float, cy: float,
        rx: float, ry: float,
        angle: float = 0.0,
        frame_id: Optional[int] = None,
        media_asset_id: Optional[int] = None,
        source: str = "human",
        confidence: Optional[float] = None,
        boundary_points: Optional[List[List[float]]] = None,
        created_by: str = "annotator",
        status: str = "approved"
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "label_type": label_type,
            "cx": cx, "cy": cy, "rx": rx, "ry": ry, "angle": angle,
            "source": source, "status": status, "created_by": created_by,
        }
        if frame_id is not None:
            body["frame_id"] = frame_id
        if media_asset_id is not None:
            body["media_asset_id"] = media_asset_id
        if confidence is not None:
            body["confidence"] = confidence
        if boundary_points is not None:
            body["boundary_points_json"] = boundary_points
        r = self._client.post("/annotations/", json=body)
        r.raise_for_status()
        return r.json()

    def list_annotations(
        self,
        frame_id: Optional[int] = None,
        media_asset_id: Optional[int] = None,
        label_type: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if frame_id is not None:
            params["frame_id"] = frame_id
        if media_asset_id is not None:
            params["media_asset_id"] = media_asset_id
        if label_type is not None:
            params["label_type"] = label_type
        if status is not None:
            params["status"] = status
        if source is not None:
            params["source"] = source
        r = self._client.get("/annotations/", params=params)
        r.raise_for_status()
        return r.json()

    def get_annotation(self, annotation_id: int) -> Dict[str, Any]:
        r = self._client.get(f"/annotations/{annotation_id}")
        r.raise_for_status()
        return r.json()

    def update_annotation(
        self, annotation_id: int, cx: float, cy: float,
        rx: float, ry: float, angle: float = 0.0,
        edited_by: str = "annotator"
    ) -> Dict[str, Any]:
        r = self._client.put(f"/annotations/{annotation_id}", json={
            "cx": cx, "cy": cy, "rx": rx, "ry": ry, "angle": angle,
            "created_by": edited_by,
        })
        r.raise_for_status()
        return r.json()

    def delete_annotation(self, annotation_id: int) -> bool:
        r = self._client.delete(f"/annotations/{annotation_id}")
        return r.status_code in (200, 204)

    def get_annotation_versions(self, annotation_id: int) -> List[Dict[str, Any]]:
        r = self._client.get(f"/annotations/{annotation_id}/versions")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Inference (Auto-Detection)
    # ------------------------------------------------------------------
    def detect_single_frame(self, frame_id: int) -> Dict[str, Any]:
        r = self._client.post(f"/inference/detect/{frame_id}", timeout=120.0)
        r.raise_for_status()
        return r.json()

    def detect_batch(self, media_asset_id: int) -> Dict[str, Any]:
        r = self._client.post(
            f"/inference/detect-batch/{media_asset_id}",
            timeout=600.0  # batch can be slow
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------
    def get_review_queue(self, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if project_id is not None:
            params["project_id"] = project_id
        r = self._client.get("/review/queue", params=params)
        r.raise_for_status()
        return r.json()

    def approve_annotation(self, annotation_id: int, reviewed_by: str = "reviewer") -> Dict[str, Any]:
        r = self._client.put(
            f"/review/{annotation_id}/approve",
            params={"reviewed_by": reviewed_by}
        )
        r.raise_for_status()
        return r.json()

    def reject_annotation(self, annotation_id: int, reviewed_by: str = "reviewer") -> Dict[str, Any]:
        r = self._client.put(
            f"/review/{annotation_id}/reject",
            params={"reviewed_by": reviewed_by}
        )
        r.raise_for_status()
        return r.json()

    def edit_and_approve(
        self, annotation_id: int,
        cx: float, cy: float, rx: float, ry: float, angle: float = 0.0,
        reviewed_by: str = "reviewer"
    ) -> Dict[str, Any]:
        r = self._client.put(
            f"/review/{annotation_id}/edit",
            json={"cx": cx, "cy": cy, "rx": rx, "ry": ry, "angle": angle},
            params={"reviewed_by": reviewed_by}
        )
        r.raise_for_status()
        return r.json()

    def batch_approve(self, threshold: float, project_id: Optional[int] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"threshold": threshold}
        if project_id is not None:
            body["project_id"] = project_id
        r = self._client.post("/review/batch-approve", json=body)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_coco(self, project_id: int) -> Dict[str, Any]:
        r = self._client.get(f"/export/{project_id}/coco")
        r.raise_for_status()
        return r.json()

    def export_csv(self, project_id: int) -> str:
        r = self._client.get(f"/export/{project_id}/csv")
        r.raise_for_status()
        return r.text

    def export_yolo(self, project_id: int) -> str:
        r = self._client.get(f"/export/{project_id}/yolo")
        r.raise_for_status()
        return r.text

    def export_summary(self, project_id: int) -> Dict[str, Any]:
        r = self._client.get(f"/export/{project_id}/summary")
        r.raise_for_status()
        return r.json()


# Singleton for easy import
api = APIClient()
