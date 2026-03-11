#!/bin/sh

set -eu

usage() {
    cat <<'EOF'
Usage: scripts/release_fork_linux_local.sh <tag> [repo]

Build Linux release artifacts locally and upload them to a GitHub release.

Examples:
  scripts/release_fork_linux_local.sh pnoch-v0.17.7-p3
  scripts/release_fork_linux_local.sh pnoch-v0.17.7-p3 pnoch/ollama

Environment overrides:
  PLATFORM=linux/amd64,linux/arm64
  OLLAMA_DOWNLOAD_URL=https://github.com/pnoch/ollama/releases/latest/download
EOF
}

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit "${1:+0}"
fi

TAG="$1"
REPO="${2:-pnoch/ollama}"
VERSION="${TAG#pnoch-v}"
ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
DOWNLOAD_URL="${OLLAMA_DOWNLOAD_URL:-https://github.com/${REPO}/releases/latest/download}"

require() {
    missing=""
    for tool in "$@"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing="$missing $tool"
        fi
    done

    if [ -n "$missing" ]; then
        echo "missing required tools:$missing" >&2
        exit 1
    fi
}

require docker gh sed zstd sha256sum tar

cd "$ROOT_DIR"

gh auth status >/dev/null

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

echo "Building local Linux release for $TAG"
VERSION="$VERSION" ./scripts/build_linux.sh

echo "Preparing installer"
sed "s|https://ollama.com/download|$DOWNLOAD_URL|g" scripts/install.sh > "$DIST_DIR/install.sh"
chmod +x "$DIST_DIR/install.sh"

echo "Generating checksums"
(
    cd "$DIST_DIR"
    find . -maxdepth 1 -type f ! -name 'sha256sum.txt' -printf '%P\n' | sort | xargs sha256sum > sha256sum.txt
)

echo "Creating release if needed"
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    gh release edit "$TAG" --repo "$REPO" --title "$TAG" >/dev/null
else
    gh release create "$TAG" --repo "$REPO" --title "$TAG" --generate-notes >/dev/null
fi

echo "Uploading artifacts"
gh release upload "$TAG" --repo "$REPO" \
    "$DIST_DIR/install.sh" \
    "$DIST_DIR/sha256sum.txt" \
    "$DIST_DIR"/*.tar.zst \
    --clobber

echo "Release updated: https://github.com/$REPO/releases/tag/$TAG"
