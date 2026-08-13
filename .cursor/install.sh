#!/usr/bin/env bash
# Cloud Agent install script for the SGLang CPU development environment.
#
# Idempotent: safe to run repeatedly and on either the Cursor default base
# image or a snapshot that already has the toolchain baked in. It prepares the
# exact CPU dev/test flow that CI uses (see .github/workflows/_pr-test-stage-cpu.yml):
# a Python 3.10 venv with `sglang[dev]` installed editable, plus the Rust +
# protoc toolchain needed to compile the bundled native extensions and routers.
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Where the project virtualenv lives. Kept outside the repo tree so it survives
# a fresh source checkout on agents that boot from a prebuilt environment.
VENV="${SGLANG_VENV:-$HOME/.venvs/sglang}"

# ---------------------------------------------------------------------------
# sudo helper (install runs as an unprivileged user with passwordless sudo).
# ---------------------------------------------------------------------------
if [ "$(id -u)" = "0" ]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

# ---------------------------------------------------------------------------
# uv (Python package/dependency manager used by the project).
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:${CARGO_HOME:-$HOME/.cargo}/bin:/usr/local/cargo/bin:$PATH"

# ---------------------------------------------------------------------------
# System build dependencies. build-essential provides gcc/g++/make; the Rust
# routers (sgl-model-gateway / sgl-router) link against OpenSSL via openssl-sys.
# ---------------------------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  ${SUDO} apt-get update -qq || true
  ${SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential pkg-config libssl-dev ca-certificates curl git || true
fi

# ---------------------------------------------------------------------------
# Prefer gcc/g++ over clang. Some base images alias cc/c++ to clang, which
# cannot locate libstdc++ headers here and breaks both the cc-rs C++ compile
# (esaxx-rs) and the final rustc link (-lstdc++) of the native extensions.
# ---------------------------------------------------------------------------
if command -v update-alternatives >/dev/null 2>&1 && [ -x /usr/bin/gcc ]; then
  ${SUDO} update-alternatives --set cc /usr/bin/gcc  2>/dev/null || true
  ${SUDO} update-alternatives --set c++ /usr/bin/g++ 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# protoc + Rust toolchain. The repo's helper installs protoc system-wide and a
# rustup toolchain, and pre-installs the workspace-pinned channel from
# rust/rust-toolchain.toml. It does NOT change the default toolchain when the
# base image already ships one, so pin the default explicitly afterwards:
# setuptools-rust runs `cargo` from python/ (outside rust/'s toolchain-file
# scope), so only the default toolchain governs the extension build.
# ---------------------------------------------------------------------------
bash scripts/ci/utils/install_rust_protoc.sh
RUST_CHANNEL="$(sed -n 's/^channel *= *"\([^"]*\)".*/\1/p' rust/rust-toolchain.toml 2>/dev/null || true)"
RUST_CHANNEL="${RUST_CHANNEL:-1.92}"
rustup default "${RUST_CHANNEL}"
export RUSTUP_TOOLCHAIN="${RUST_CHANNEL}"

# ---------------------------------------------------------------------------
# Python 3.10 venv (matches the CPU CI interpreter) + editable install with the
# dev/test extras. This also compiles the bundled Rust extension modules
# (sglang.srt.{grpc,multimodal,server}._core) via setuptools-rust.
# ---------------------------------------------------------------------------
uv python install 3.10
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.10 "$VENV"
fi

uv pip install --python "$VENV/bin/python" -e "python[dev]" \
  --index-strategy unsafe-best-match --prerelease allow

echo "SGLang CPU dev environment ready."
echo "Activate it with:  source ${VENV}/bin/activate"
