"""Exercise deployment disk handling without Docker or AWS access."""

import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    "free_kb,pull_fails", [(1, False), (10_000_000, False), (10_000_000, True)]
)
def test_disk_preflight_preserves_referenced_images(free_kb, pull_fails):
    workflow = yaml.safe_load(Path(".github/workflows/deploy.yml").read_text())
    script = workflow["jobs"]["build-push-deploy"]["steps"][-1]["with"]["script"]
    # Only run cleanup/preflight/pull, never the service replacement commands.
    start = script.index("# Remove only superseded")
    end = script.index("docker rm -f lesson-plan-qdrant")
    mock = r"""
set -euo pipefail
IMAGE_URI=registry/app:new
IMAGE_SIZE_BYTES=1000000000
ECR_REGISTRY=registry
ECR_REPOSITORY=app
docker() {
  case "$1 $2" in
    'ps -aq') printf 'running\nstopped\n' ;;
    'inspect --format') printf 'sha256:%s\n' "$4" ;;
    'image ls')
      [[ "$*" == *'reference=registry/app:*'* ]]
      printf 'sha256:running registry/app:current\nsha256:stopped registry/app:rollback\nsha256:unused registry/app:old\nsha256:new registry/app:new\n'
      ;;
    'image rm') echo "REMOVE $3" ;;
    'image inspect') return 1 ;;
    'info --format') echo /docker ;;
    'system df') echo 'Disk usage' ;;
    'pull registry/app:new') echo PULL; return "$PULL_FAILS" ;;
    *) echo "Unexpected docker command: $*"; return 2 ;;
  esac
}
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted\n/dev/mock 20000000 1 %s 1%% /docker\n' "$FREE_KB"; }
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"FREE_KB={free_kb}\nPULL_FAILS={int(pull_fails)}\n" + mock + script[start:end],
        ],
        capture_output=True,
        text=True,
    )
    assert "REMOVE registry/app:old" in result.stdout
    assert "REMOVE registry/app:current" not in result.stdout
    assert "REMOVE registry/app:rollback" not in result.stdout
    assert "REMOVE registry/app:new" not in result.stdout
    if free_kb == 1:
        assert result.returncode == 1
        assert "Insufficient Docker disk space" in result.stdout
        assert "PULL" not in result.stdout
    elif pull_fails:
        assert result.returncode == 1
        assert "Existing services were not stopped" in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert "PULL" in result.stdout
