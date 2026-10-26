"""
Master Launcher for the Enterprise Eye Annotation Platform.
Starts both the FastAPI backend (port 8000) and the Streamlit frontend (port 8501).

Usage:
    python run.py                   # Start both backend and frontend
    python run.py --backend-only    # Start only the FastAPI backend
    python run.py --frontend-only   # Start only the Streamlit frontend
"""

import sys
import os
import time
import signal
import subprocess
import argparse
import urllib.request
import urllib.error

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

def wait_for_backend(url: str = "http://localhost:8000/health", timeout_sec: float = 15.0) -> bool:
    """Poll the backend health endpoint until it responds or times out."""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def main():
    parser = argparse.ArgumentParser(description="Eye Annotation Platform Runner")
    parser.add_argument("--backend-only", action="store_true", help="Run only the FastAPI backend")
    parser.add_argument("--frontend-only", action="store_true", help="Run only the Streamlit frontend")
    parser.add_argument("--api-port", type=int, default=8000, help="FastAPI backend port (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=8501, help="Streamlit frontend port (default: 8501)")
    args = parser.parse_args()

    processes = []

    def shutdown(sig=None, frame=None):
        print("\n🛑 Shutting down Eye Annotation Platform services...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("✅ Shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    # 1. Start Backend
    if not args.frontend_only:
        print(f"🚀 Starting FastAPI backend on http://localhost:{args.api_port}...")
        backend_cmd = [
            PYTHON_EXE, "-m", "uvicorn", "backend.main:app",
            "--host", "0.0.0.0",
            "--port", str(args.api_port),
            "--log-level", "info"
        ]
        backend_proc = subprocess.Popen(backend_cmd, cwd=PROJECT_ROOT, env=env)
        processes.append(backend_proc)

        if not args.backend_only:
            print("⏳ Waiting for backend to become ready...")
            if wait_for_backend(f"http://localhost:{args.api_port}/health", timeout_sec=20.0):
                print("🟢 Backend is ready!")
            else:
                print("⚠️ Backend took longer than expected to report healthy, continuing...")

    # 2. Start Frontend
    if not args.backend_only:
        print(f"🚀 Starting Streamlit frontend on http://localhost:{args.ui_port}...")
        frontend_cmd = [
            PYTHON_EXE, "-m", "streamlit", "run",
            os.path.join("frontend", "app.py"),
            "--server.port", str(args.ui_port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        frontend_proc = subprocess.Popen(frontend_cmd, cwd=PROJECT_ROOT, env=env)
        processes.append(frontend_proc)

    print("\n" + "=" * 60)
    print("👁️ EYE ANNOTATION PLATFORM IS RUNNING")
    if not args.backend_only:
        print(f"   Frontend UI:  http://localhost:{args.ui_port}")
    if not args.frontend_only:
        print(f"   API Docs:     http://localhost:{args.api_port}/docs")
        print(f"   API Base:     http://localhost:{args.api_port}/api/v1")
    print("   Press Ctrl+C to stop all services.")
    print("=" * 60 + "\n")

    try:
        while True:
            for p in processes:
                code = p.poll()
                if code is not None:
                    print(f"⚠️ A service exited with return code {code}. Stopping other services...")
                    shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()

if __name__ == "__main__":
    main()
