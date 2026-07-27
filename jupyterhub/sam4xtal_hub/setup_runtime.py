"""Notebook-driven setup for JupyterHub: sidecar + Next.js, no Docker, no CLI.

Runtime data (HF cache, venv, logs, pids) lives next to the setup notebook.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .client import SidecarClient, SidecarError


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "sidecar" / "pyproject.toml").is_file() and (
            candidate / "web" / "package.json"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find sam4xtal repo root (need sidecar/ + web/). "
        "Open setup.ipynb from the cloned repo."
    )


def notebook_dir(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    # Prefer cwd when the user opened notebooks/setup.ipynb
    cwd = Path.cwd().resolve()
    if (cwd / "setup.ipynb").is_file() or cwd.name == "notebooks":
        return cwd
    root = find_repo_root(cwd)
    nb = root / "notebooks"
    return nb if nb.is_dir() else cwd


class HubRuntime:
    """Paths + process control for one JupyterHub session."""

    def __init__(
        self,
        *,
        notebook_directory: Path | str | None = None,
        sidecar_port: int = 9001,
        web_port: int = 3000,
        mock: bool = False,
    ) -> None:
        self.notebook_dir = notebook_dir(notebook_directory)
        self.repo_root = find_repo_root(self.notebook_dir)
        self.runtime_dir = self.notebook_dir / "sam4xtal-runtime"
        self.hf_home = self.runtime_dir / "hf-cache"
        self.uv_cache = self.runtime_dir / "uv-cache"
        self.venv_dir = self.runtime_dir / "sidecar-venv"
        # Vite+ (vp) home: standalone CLI + managed Node/pnpm shims (writable)
        self.vp_home = self.runtime_dir / "vite-plus"
        self.log_dir = self.runtime_dir / "logs"
        self.sidecar_port = int(sidecar_port)
        self.web_port = int(web_port)
        self.mock = mock

        self.sidecar_pid_file = self.runtime_dir / "sidecar.pid"
        self.web_pid_file = self.runtime_dir / "web.pid"
        self.sidecar_log = self.log_dir / "sidecar.log"
        self.web_log = self.log_dir / "web.log"
        self.env_file = self.runtime_dir / "hub.env"

    # --- layout ----------------------------------------------------------------

    def prepare_dirs(self) -> None:
        for p in (
            self.runtime_dir,
            self.hf_home,
            self.uv_cache,
            self.vp_home,
            self.log_dir,
            self.hf_home / "hub",
            self.hf_home / "transformers",
            self.runtime_dir / "bin",
        ):
            p.mkdir(parents=True, exist_ok=True)
        print(f"notebook dir : {self.notebook_dir}")
        print(f"repo root    : {self.repo_root}")
        print(f"runtime dir  : {self.runtime_dir}")
        print(f"HF cache     : {self.hf_home}")
        print(f"sidecar venv : {self.venv_dir}")
        print(f"vite-plus    : {self.vp_home}")

    def save_hf_token(self, token: str) -> Path:
        token = (token or "").strip()
        if not token or token.startswith("hf_…") or token == "PASTE_YOUR_TOKEN_HERE":
            raise ValueError(
                "Paste a real Hugging Face token into HF_TOKEN in the notebook cell "
                "(https://huggingface.co/settings/tokens). Accept the facebook/sam3 license."
            )
        # Persist for restart cells without re-pasting
        token_path = self.runtime_dir / "hf_token"
        token_path.write_text(token + "\n", encoding="utf-8")
        try:
            token_path.chmod(0o600)
        except OSError:
            pass
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print(f"HF token saved to {token_path} (mode 600 if supported)")
        return token_path

    def load_hf_token(self) -> str | None:
        token_path = self.runtime_dir / "hf_token"
        if token_path.is_file():
            t = token_path.read_text(encoding="utf-8").strip()
            if t:
                os.environ["HF_TOKEN"] = t
                os.environ["HUGGING_FACE_HUB_TOKEN"] = t
                return t
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        token = self.load_hf_token()
        if token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token
        env["HF_HOME"] = str(self.hf_home)
        env["HUGGINGFACE_HUB_CACHE"] = str(self.hf_home / "hub")
        env["TRANSFORMERS_CACHE"] = str(self.hf_home / "transformers")
        env["UV_CACHE_DIR"] = str(self.uv_cache)
        env["UV_PROJECT_ENVIRONMENT"] = str(self.venv_dir)
        env["VIRTUAL_ENV"] = str(self.venv_dir)
        env["VP_HOME"] = str(self.vp_home)
        env["SAM3_BACKEND"] = "mock" if self.mock else "transformers"
        env["PORT"] = str(self.sidecar_port)
        env["INFERENCE_URL"] = f"http://127.0.0.1:{self.sidecar_port}"
        env["HOSTNAME"] = "0.0.0.0"
        env["NEXT_TELEMETRY_DISABLED"] = "1"
        # Runtime bins first: uv, then Vite+ shims (node/pnpm), then sidecar venv.
        # Must beat the read-only module Node under /software/opt/...
        path_parts: list[str] = []
        for p in (
            self.runtime_dir / "bin",
            self.vp_home / "bin",
            self.venv_dir / ("Scripts" if os.name == "nt" else "bin"),
        ):
            if p.is_dir():
                path_parts.append(str(p))
        env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
        return env

    def write_hub_env(self) -> Path:
        env = self.base_env()
        lines = [
            f"HF_HOME={self.hf_home}",
            f"UV_CACHE_DIR={self.uv_cache}",
            f"UV_PROJECT_ENVIRONMENT={self.venv_dir}",
            f"SAM3_BACKEND={env['SAM3_BACKEND']}",
            f"PORT={self.sidecar_port}",
            f"INFERENCE_URL={env['INFERENCE_URL']}",
            f"WEB_PORT={self.web_port}",
        ]
        if env.get("HF_TOKEN"):
            lines.append(f"HF_TOKEN={env['HF_TOKEN']}")
        self.env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {self.env_file}")
        return self.env_file

    # --- tools -----------------------------------------------------------------

    def _run(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        print("+", " ".join(cmd))
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env or self.base_env(),
            check=check,
            text=True,
            capture_output=False,
        )

    def ensure_uv(self) -> str:
        """Always use a uv binary under sam4xtal-runtime/bin (never system PATH).

        Downloads the official standalone build into the runtime folder next to
        the notebook so every Hub user gets the same tool, even if a broken or
        ancient ``uv`` is on PATH. Reuses a previous download when present.
        Falls back to pip install --target only if the download fails.
        """
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        bin_dir = self.runtime_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        local_uv = bin_dir / ("uv.exe" if os.name == "nt" else "uv")

        if local_uv.is_file() and os.access(local_uv, os.X_OK):
            print(f"uv (runtime): {local_uv}")
            return str(local_uv)

        # Official standalone → runtime/bin only (ignore any system uv)
        if self._install_uv_standalone(bin_dir, local_uv):
            return str(local_uv)

        # Last resort: pip install into runtime (still not system PATH)
        print("standalone uv download failed — pip installing into runtime …")
        pip_target = self.runtime_dir / "pip-uv"
        self._run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(pip_target),
                "uv",
            ],
            check=False,
        )
        for name in ("uv", "uv.exe"):
            candidate = pip_target / "bin" / name
            if not candidate.is_file():
                candidate = pip_target / "Scripts" / name
            if candidate.is_file():
                shutil.copy2(candidate, local_uv)
                try:
                    local_uv.chmod(local_uv.stat().st_mode | 0o755)
                except OSError:
                    pass
                print(f"uv (pip→runtime): {local_uv}")
                return str(local_uv)

        raise RuntimeError(
            "Could not install uv into sam4xtal-runtime/bin. "
            "Check network access to GitHub (astral-sh/uv releases)."
        )

    def _install_uv_standalone(self, bin_dir: Path, dest: Path) -> bool:
        """Download Astral's uv binary into bin_dir. Returns True on success."""
        import platform
        import tarfile
        import tempfile
        import zipfile

        system = platform.system().lower()  # linux, darwin, windows
        machine = platform.machine().lower()  # x86_64, aarch64, amd64, arm64
        if machine in ("x86_64", "amd64"):
            arch = "x86_64"
        elif machine in ("aarch64", "arm64"):
            arch = "aarch64"
        else:
            print(f"unsupported arch for standalone uv: {machine}")
            return False

        if system == "linux":
            # musl vs gnu: prefer gnu; fall back handled by URL 404 → False
            target = f"uv-{arch}-unknown-linux-gnu"
            asset = f"{target}.tar.gz"
        elif system == "darwin":
            target = f"uv-{arch}-apple-darwin"
            asset = f"{target}.tar.gz"
        elif system == "windows":
            target = f"uv-{arch}-pc-windows-msvc"
            asset = f"{target}.zip"
        else:
            print(f"unsupported OS for standalone uv: {system}")
            return False

        # Latest release asset (Astral redirects /latest/)
        url = f"https://github.com/astral-sh/uv/releases/latest/download/{asset}"
        print(f"downloading uv from {url} …")
        try:
            with tempfile.TemporaryDirectory(prefix="uv-dl-") as tmp:
                tmp_path = Path(tmp)
                archive = tmp_path / asset
                urllib.request.urlretrieve(url, archive)
                extract_dir = tmp_path / "out"
                extract_dir.mkdir()
                if asset.endswith(".zip"):
                    with zipfile.ZipFile(archive, "r") as zf:
                        zf.extractall(extract_dir)
                else:
                    with tarfile.open(archive, "r:gz") as tf:
                        tf.extractall(extract_dir)

                # tarball usually contains uv-.../uv or just uv
                candidates = list(extract_dir.rglob("uv.exe" if os.name == "nt" else "uv"))
                candidates = [p for p in candidates if p.is_file()]
                if not candidates:
                    print("uv binary not found inside archive")
                    return False
                src = candidates[0]
                shutil.copy2(src, dest)
                try:
                    dest.chmod(dest.stat().st_mode | 0o755)
                except OSError:
                    pass
            if dest.is_file():
                print(f"uv installed to {dest}")
                return True
        except Exception as e:
            print(f"standalone uv download failed: {e}")
        return False

    def setup_sidecar_venv(self) -> Path:
        uv = self.ensure_uv()
        env = self.base_env()
        sidecar = self.repo_root / "sidecar"
        py_name = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        py = self.venv_dir / py_name

        self._run([uv, "python", "install", "3.12"], cwd=sidecar, env=env, check=False)

        # Explicit venv next to the notebook (not sidecar/.venv)
        if not py.is_file():
            self._run(
                [uv, "venv", str(self.venv_dir), "--python", "3.12"],
                cwd=sidecar,
                env=env,
            )
        env = self.base_env()  # PATH picks up new venv
        self._run(
            [uv, "sync", "--python", str(py)],
            cwd=sidecar,
            env=env,
        )
        if not py.is_file():
            raise RuntimeError(f"venv python missing at {py}")
        print(f"sidecar python: {py}")
        # torch may be huge; import check proves the env works
        self._run(
            [
                str(py),
                "-c",
                "import fastapi; print('fastapi ok'); "
                "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())",
            ],
            cwd=sidecar,
            env=env,
        )
        return self.venv_dir

    def _vp_bin(self) -> Path:
        name = "vp.exe" if os.name == "nt" else "vp"
        return self.vp_home / "bin" / name

    def ensure_vp(self) -> str:
        """Install Vite+ (`vp`) into sam4xtal-runtime/vite-plus (always writable).

        Faculty Hub Node lives under read-only /software/opt — corepack/npm -g
        cannot install pnpm there. Vite+ downloads its own Node + package-manager
        shims under VP_HOME so we never write into the module tree.
        """
        self.vp_home.mkdir(parents=True, exist_ok=True)
        vp = self._vp_bin()
        env = self.base_env()

        if not (vp.is_file() and os.access(vp, os.X_OK)):
            print(f"installing Vite+ (vp) into {self.vp_home} …")
            if os.name == "nt":
                # Windows: official PowerShell installer with custom VP_HOME
                ps = (
                    f'$env:VP_HOME = "{self.vp_home}"; '
                    f'$env:VP_NODE_MANAGER = "yes"; '
                    f'$env:CI = "true"; '
                    "irm https://vite.plus/ps1 | iex"
                )
                self._run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                    env=env,
                )
            else:
                # curl | bash; non-interactive: manage Node, no shell prompts
                install_env = env.copy()
                install_env["VP_HOME"] = str(self.vp_home)
                install_env["VP_NODE_MANAGER"] = "yes"
                install_env["CI"] = "true"
                # Avoid writing into the user's real home shell configs if possible;
                # we inject PATH via base_env() for all subprocesses.
                self._run(
                    ["bash", "-lc", "curl -fsSL https://vite.plus | bash"],
                    env=install_env,
                )
            if not vp.is_file():
                # some installs put vp under current/bin via symlink
                alt = self.vp_home / "current" / "bin" / ("vp.exe" if os.name == "nt" else "vp")
                if alt.is_file():
                    bin_dir = self.vp_home / "bin"
                    bin_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        if not vp.exists():
                            vp.symlink_to(alt)
                    except OSError:
                        shutil.copy2(alt, vp)
                        try:
                            vp.chmod(vp.stat().st_mode | 0o755)
                        except OSError:
                            pass
            if not vp.is_file():
                raise RuntimeError(
                    f"Vite+ install finished but {vp} is missing. "
                    "Check network to vite.plus / npm registry."
                )
            print(f"vp installed: {vp}")
        else:
            print(f"vp (runtime): {vp}")

        env = self.base_env()
        # Ensure Node shims exist under VP_HOME/bin (node, npm, pnpm, corepack)
        self._run(
            [str(vp), "env", "setup", "--refresh"],
            env=env,
            check=False,
        )
        # Prefer a modern LTS for Next 16 (system module is often Node 18)
        for node_ver in ("22", "20"):
            r = self._run(
                [str(vp), "env", "use", node_ver],
                env=env,
                check=False,
            )
            if r.returncode == 0:
                break

        env = self.base_env()
        node = shutil.which("node", path=env["PATH"])
        pnpm = shutil.which("pnpm", path=env["PATH"])
        if node:
            try:
                ver = subprocess.check_output(
                    [node, "-v"], text=True, env=env
                ).strip()
            except Exception:
                ver = "?"
            print(f"node (via vp): {node} ({ver})")
        else:
            print("warn: node shim not on PATH yet — vp install may still work")
        if pnpm:
            print(f"pnpm (via vp): {pnpm}")
        else:
            print("pnpm shim not found yet — will use `vp install` / `vp run`")

        return str(vp)

    def ensure_node_pnpm(self) -> tuple[str, str]:
        """Back-compat: ensure Vite+ toolchain; return (node, pnpm-or-vp)."""
        vp = self.ensure_vp()
        env = self.base_env()
        node = shutil.which("node", path=env["PATH"]) or vp
        pnpm = shutil.which("pnpm", path=env["PATH"]) or vp
        return node, pnpm

    def setup_web(self) -> None:
        vp = self.ensure_vp()
        web = self.repo_root / "web"
        env = self.base_env()
        # Vite+ picks pnpm from pnpm-lock.yaml / packageManager field
        r = self._run(
            [vp, "install", "--config.minimumReleaseAge=0"],
            cwd=web,
            env=env,
            check=False,
        )
        if r.returncode != 0:
            print("retry vp install without minimumReleaseAge …")
            self._run([vp, "install"], cwd=web, env=env)
        # Native builds (sharp) — only if real pnpm is on PATH
        pnpm = shutil.which("pnpm", path=env["PATH"])
        if pnpm:
            self._run([pnpm, "approve-builds", "--all"], cwd=web, env=env, check=False)
        print("building Next.js (this can take a few minutes) …")
        # Prefer package.json script via vp run; fall back to pnpm / npx
        r = self._run([vp, "run", "build"], cwd=web, env=env, check=False)
        if r.returncode != 0:
            if pnpm:
                self._run([pnpm, "build"], cwd=web, env=env)
            else:
                self._run([vp, "exec", "next", "build"], cwd=web, env=env)
        print("web build complete")

    # --- processes -------------------------------------------------------------

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _read_pid(self, path: Path) -> int | None:
        if not path.is_file():
            return None
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            return None

    def _stop_pidfile(self, path: Path, name: str) -> None:
        pid = self._read_pid(path)
        if pid and self._pid_alive(pid):
            print(f"stopping {name} pid={pid}")
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            for _ in range(20):
                if not self._pid_alive(pid):
                    break
                time.sleep(0.25)
            if self._pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if path.is_file():
            path.unlink(missing_ok=True)

    def stop_sidecar(self) -> None:
        self._stop_pidfile(self.sidecar_pid_file, "sidecar")

    def stop_web(self) -> None:
        self._stop_pidfile(self.web_pid_file, "web")

    def stop_all(self) -> None:
        self.stop_web()
        self.stop_sidecar()
        print("stopped web + sidecar")

    def start_sidecar(self, *, restart: bool = True) -> SidecarClient:
        if restart:
            self.stop_sidecar()
        client = SidecarClient(f"http://127.0.0.1:{self.sidecar_port}")
        try:
            h = client.health()
            if h.get("ok"):
                print("sidecar already responding:", h)
                return client
        except SidecarError:
            pass

        env = self.base_env()
        py = self.venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not py.is_file():
            # fallback after uv sync into sidecar/.venv
            py = self.repo_root / "sidecar" / ".venv" / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
        if not py.is_file():
            raise RuntimeError("Run the sidecar venv setup cell first")

        uvicorn = [
            str(py),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.sidecar_port),
        ]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_f = open(self.sidecar_log, "ab", buffering=0)
        proc = subprocess.Popen(
            uvicorn,
            cwd=str(self.repo_root / "sidecar"),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.sidecar_pid_file.write_text(str(proc.pid), encoding="utf-8")
        print(f"sidecar pid={proc.pid}  log={self.sidecar_log}")
        print("waiting for /health (model download can take many minutes the first time) …")
        try:
            status = client.wait_ready(timeout_s=1800.0, poll_s=3.0)
        except SidecarError:
            print("--- last sidecar log lines ---")
            self.tail_log(self.sidecar_log, n=40)
            raise
        print("sidecar ready:", status)
        return client

    def start_web(self, *, restart: bool = True) -> None:
        if restart:
            self.stop_web()
        vp = self.ensure_vp()
        env = self.base_env()
        env["PORT"] = str(self.web_port)
        env["HOSTNAME"] = "0.0.0.0"
        env["INFERENCE_URL"] = f"http://127.0.0.1:{self.sidecar_port}"

        web = self.repo_root / "web"
        # Prefer Vite+ shims; fall back to `vp exec next`
        pnpm = shutil.which("pnpm", path=env["PATH"])
        next_bin = shutil.which("next", path=env["PATH"])
        if pnpm:
            cmd = [
                pnpm,
                "exec",
                "next",
                "start",
                "--hostname",
                "0.0.0.0",
                "--port",
                str(self.web_port),
            ]
        elif next_bin:
            cmd = [
                next_bin,
                "start",
                "--hostname",
                "0.0.0.0",
                "--port",
                str(self.web_port),
            ]
        else:
            cmd = [
                vp,
                "exec",
                "next",
                "start",
                "--hostname",
                "0.0.0.0",
                "--port",
                str(self.web_port),
            ]

        log_f = open(self.web_log, "ab", buffering=0)
        proc = subprocess.Popen(
            cmd,
            cwd=str(web),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.web_pid_file.write_text(str(proc.pid), encoding="utf-8")
        print(f"web pid={proc.pid}  log={self.web_log}")
        # wait until port answers
        deadline = time.time() + 120
        while time.time() < deadline:
            if not self._pid_alive(proc.pid):
                print("--- last web log lines ---")
                self.tail_log(self.web_log, n=40)
                raise RuntimeError("Next.js process exited early")
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.web_port}/",
                    timeout=2,
                )
                print("Next.js is answering on port", self.web_port)
                return
            except Exception:
                time.sleep(1)
        print("warning: port open timeout — check log; may still be starting")
        self.tail_log(self.web_log, n=30)

    def tail_log(self, path: Path, n: int = 30) -> None:
        if not path.is_file():
            print(f"(no log at {path})")
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            print(f"(cannot read log: {e})")
            return
        for line in lines[-n:]:
            print(line)

    # --- URLs ------------------------------------------------------------------

    def access_urls(self) -> dict[str, Any]:
        host = socket.gethostname()
        try:
            fqdn = socket.getfqdn()
        except Exception:
            fqdn = host

        hub_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "")
        # e.g. /user/jane.doe/
        public_hub = (
            os.environ.get("JUPYTERHUB_PUBLIC_URL")
            or os.environ.get("JUPYTER_HUB_API_URL")
            or ""
        )
        # Best-effort Hub browser origin from common env vars
        hub_base = os.environ.get("JUPYTERHUB_BASE_URL") or ""

        proxy_paths = []
        if hub_prefix:
            proxy_paths.append(f"{hub_prefix.rstrip('/')}/proxy/{self.web_port}/")
            proxy_paths.append(f"{hub_prefix.rstrip('/')}/proxy/{self.sidecar_port}/")

        info = {
            "hostname": host,
            "fqdn": fqdn,
            "web_port": self.web_port,
            "sidecar_port": self.sidecar_port,
            "local_web": f"http://127.0.0.1:{self.web_port}",
            "local_sidecar": f"http://127.0.0.1:{self.sidecar_port}",
            "node_web": f"http://{host}:{self.web_port}",
            "jupyterhub_service_prefix": hub_prefix or None,
            "proxy_paths": proxy_paths,
            "hub_public_url": public_hub or None,
            "hub_base_url": hub_base or None,
        }
        return info

    def _guess_hub_origin(self) -> str | None:
        """Best-effort public Hub origin for proxy links."""
        val = (os.environ.get("JUPYTERHUB_PUBLIC_URL") or "").strip().rstrip("/")
        if val.startswith("http"):
            return val
        try:
            from jupyter_server import serverapp

            for srv in serverapp.list_running_servers():
                url = (srv.get("url") or "").rstrip("/")
                if url:
                    return url
        except Exception:
            pass
        try:
            from notebook import notebookapp  # classic

            for srv in notebookapp.list_running_servers():
                url = (srv.get("url") or "").rstrip("/")
                if url:
                    return url
        except Exception:
            pass
        host = (
            os.environ.get("JUPYTERHUB_HOST")
            or os.environ.get("JUPYTERHUB_DOMAIN")
            or ""
        ).strip()
        if host:
            if not host.startswith("http"):
                host = "https://" + host
            return host.rstrip("/")
        return None

    def print_access_instructions(self) -> None:
        info = self.access_urls()
        origin = self._guess_hub_origin()
        print("=" * 60)
        print("OPEN THE NEXT.JS APP")
        print("=" * 60)
        print()
        print(f"Web port:     {info['web_port']}")
        print(f"Sidecar port: {info['sidecar_port']} (API only, used by the website)")
        print(f"This node:    {info['fqdn']}")
        print()
        print("── Option A: JupyterHub proxy (only if server-proxy is enabled) ──")
        if info["proxy_paths"]:
            web_path = info["proxy_paths"][0]
            print(f"   path:  {web_path}")
            if origin:
                # origin may already include /user/x/lab — strip to host when needed
                print(f"   try:   {origin.rstrip('/')}{web_path}")
            else:
                print(
                    "   try:   https://jupyter.physik.uni-muenchen.de"
                    f"{web_path}"
                )
            print("   If that 404s, use Option B.")
        else:
            print("   (no JUPYTERHUB_SERVICE_PREFIX — skip to Option B)")
        print()
        print("── Option B: SSH tunnel (most reliable) ──")
        print(f"   ssh -L {info['web_port']}:127.0.0.1:{info['web_port']} YOU@{info['fqdn']}")
        print(f"   then open →  http://127.0.0.1:{info['web_port']}")
        print()
        print("── Option C: direct node URL (often blocked off-node) ──")
        print(f"   {info['node_web']}")
        print(f"   on-node only: {info['local_web']}")
        print()
        print("Logs:")
        print(f"   {self.sidecar_log}")
        print(f"   {self.web_log}")
        print("=" * 60)
        print("Leave this Hub session running while you use the app.")
        print("When finished: run the Stop cell, then shut down the Hub server.")

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "runtime_dir": str(self.runtime_dir),
            "sidecar_pid": self._read_pid(self.sidecar_pid_file),
            "web_pid": self._read_pid(self.web_pid_file),
        }
        try:
            out["sidecar_health"] = SidecarClient(
                f"http://127.0.0.1:{self.sidecar_port}"
            ).health()
        except Exception as e:
            out["sidecar_health"] = {"error": str(e)}
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.web_port}/", timeout=2)
            out["web_ok"] = True
        except Exception as e:
            out["web_ok"] = False
            out["web_error"] = str(e)
        return out
