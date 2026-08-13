#!/usr/bin/env bash
# INSTALL.sh — Install EXO build dependencies for macOS.
#
# What it does:
#   1. Verifies macOS + x86_64/arm64.
#   2. Installs Xcode Command Line Tools (xcode-select --install) if missing.
#   3. Installs Homebrew if missing.
#   4. Installs uv, just, rust, node, pkg-config, libssl via Homebrew.
#   5. Installs Python 3.13 via uv and pins it locally.
#   6. Installs Pillow (system Python) for the DMG background generator.
#
# After this script, run `just sync --all-packages --group dev --extra mlx`
# to install the MLX/Mistral/vLM/mflux extras that exo needs at runtime.
#
# Idempotent: every step is a no-op if already satisfied.
#
# Usage:
#   ./INSTALL.sh
#
# To check status without installing anything:
#   ./INSTALL.sh --check
#
set -euo pipefail

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Colours
if [[ -t 1 ]]; then
  RED=$'\033[0;31m'
  GREEN=$'\033[0;32m'
  YELLOW=$'\033[0;33m'
  BLUE=$'\033[0;34m'
  BOLD=$'\033[1m'
  RESET=$'\033[0m'
else
  RED=""; GREEN=""; YELLOW=""; BLUE=""; BOLD=""; RESET=""
fi

say()   { printf "%b\n" "${BLUE}==>${RESET} ${BOLD}$*${RESET}"; }
warn()  { printf "%b\n" "${YELLOW}==> WARNING:${RESET} $*"; }
err()   { printf "%b\n" "${RED}==> ERROR:${RESET} $*" >&2; }
ok()    { printf "%b\n" "${GREEN}==>${RESET} $*"; }
hr()    { printf "%b\n" "${BLUE}─────────────────────────────────────────────────${RESET}"; }

# ── 1. macOS guard ────────────────────────────────────────────────────────────
hr
say "Step 1/6: macOS guard"
hr

if [[ "$(uname -s)" != "Darwin" ]]; then
  err "This script is macOS-only. Detected: $(uname -s)."
  exit 1
fi
ok "macOS $(sw_vers -productVersion) ($(uname -m))"

# ── 2. Xcode Command Line Tools ────────────────────────────────────────────────
hr
say "Step 2/6: Xcode Command Line Tools"
hr

if xcode-select -p >/dev/null 2>&1; then
  ok "xcode-select: $(xcode-select -p)"
else
  if [[ $CHECK_ONLY -eq 1 ]]; then
    warn "Xcode Command Line Tools not installed."
    exit 1
  fi
  say "Installing Command Line Tools..."
  xcode-select --install
  say "After the popup completes, re-run $0."
  exit 0
fi

# Full Xcode is recommended for the Swift app build (the `EXO` scheme uses
# `xcodebuild` from the full app). Skip-check: just print whether the path
# looks like a full app or just `usr/bin`.
if [[ "$(xcode-select -p)" == "/Library/Developer/CommandLineTools" ]]; then
  warn "Only CommandLineTools installed. Install full Xcode for the Swift app build:"
  warn "  App Store → Xcode. Or: https://developer.apple.com/xcode/"
else
  ok "Xcode detected at $(xcode-select -p)"
fi

# ── 3. Homebrew ────────────────────────────────────────────────────────────────
hr
say "Step 3/6: Homebrew"
hr

if command -v brew >/dev/null 2>&1; then
  ok "brew $(brew --version | head -1 | awk '{print $1}')"
else
  if [[ $CHECK_ONLY -eq 1 ]]; then
    err "Homebrew not installed."
    exit 1
  fi
  say "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add to PATH for the rest of this script.
  if [[ -f /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -f /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
fi

# Ensure brew is in PATH for the rest of the session.
if [[ -f /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# ── 4. Tooling via Homebrew ───────────────────────────────────────────────────
hr
say "Step 4/6: Brew packages (uv, just, rust, node, pkg-config, openssl)"
hr

BREW_PACKAGES=(
  uv           # Python package manager
  just         # Task runner for the justfile
  rust         # cargo + rustc
  node         # Node.js + npm
  pkg-config   # Needed by some Rust crates
  openssl@3    # Needed by Rust crates that link OpenSSL
)

if [[ $CHECK_ONLY -eq 1 ]]; then
  for pkg in "${BREW_PACKAGES[@]}"; do
    if brew list "$pkg" >/dev/null 2>&1; then
      ok "$pkg"
    else
      warn "missing: $pkg"
    fi
  done
else
  brew update
  brew install "${BREW_PACKAGES[@]}"
fi

# ── 5. Python 3.13 (via uv) ────────────────────────────────────────────────────
hr
say "Step 5/6: Python 3.13 via uv"
hr

# uv needs to be in PATH.
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version)"
else
  err "uv not on PATH. Restart your shell or source brew shellenv."
  exit 1
fi

PYTHON_313=$(uv python find 3.13 || true)
if [[ -n "$PYTHON_313" && -x "$PYTHON_313" ]]; then
  ok "Python 3.13: $PYTHON_313"
else
  if [[ $CHECK_ONLY -eq 1 ]]; then
    err "Python 3.13 not installed via uv."
    exit 1
  fi
  say "Installing Python 3.13..."
  uv python install 3.13
  PYTHON_313=$(uv python find 3.13)
fi

# Pin 3.13 in this repo (pyproject.toml pins `requires-python = "==3.13.*"`).
if [[ $CHECK_ONLY -eq 0 ]]; then
  uv python pin 3.13 || warn "uv python pin failed (already pinned?)"
fi

# ── 6. Pillow (for the DMG background generator) ────────────────────────────
# Pillow is used by `packaging/dmg/create-dmg.sh` to render the drag-to-Applications
# background.png. We install it into whichever `python3` is on PATH first.
# That may be the Apple system Python (/usr/bin/python3) or the Homebrew one
# (/opt/homebrew/bin/python3) — both are fine; create-dmg.sh just calls `python3`.
#
# We use `--break-system-packages` because Homebrew Python is marked
# "externally managed" (PEP 668). The Apple system Python accepts
# `pip install --user` without the flag, so the flag is a no-op there.
hr
say "Step 6/6: Pillow for the DMG background generator"
hr

PYTHON3="$(command -v python3 || true)"
if [[ -z "$PYTHON3" ]]; then
  err "python3 not on PATH. Install Python (e.g., 'brew install python') and re-run."
  exit 1
fi

PIL_VERSION="$("$PYTHON3" -c "import PIL; print(PIL.__version__)" 2>/dev/null || true)"
if [[ -n "$PIL_VERSION" ]]; then
  ok "Pillow $PIL_VERSION via $PYTHON3"
else
  if [[ $CHECK_ONLY -eq 1 ]]; then
    warn "Pillow not installed for $PYTHON3."
    exit 1
  fi
  say "Installing Pillow via $PYTHON3..."
  "$PYTHON3" -m pip install --user --break-system-packages --quiet Pillow
  PIL_VERSION="$("$PYTHON3" -c "import PIL; print(PIL.__version__)" 2>/dev/null || true)"
  if [[ -z "$PIL_VERSION" ]]; then
    err "Pillow install failed. Try: $PYTHON3 -m pip install --user --break-system-packages Pillow"
    exit 1
  fi
  ok "Pillow $PIL_VERSION installed via $PYTHON3"
fi

# Sanity check the DMG generator.
DMG_GENERATOR="${REPO_ROOT}/packaging/dmg/generate-background.py"
if [[ -f "$DMG_GENERATOR" ]]; then
  if "$PYTHON3" "$DMG_GENERATOR" /tmp/exo-dmg-check.png 2>/dev/null; then
    ok "DMG background generator: $PYTHON3"
    rm -f /tmp/exo-dmg-check.png
  else
    warn "DMG background generator failed. Inspect $DMG_GENERATOR by hand."
  fi
fi

# ── Done ───────────────────────────────────────────────────────────────────────
hr
say "Summary"
hr

printf "%b\n" "${GREEN}OK${RESET}  All install steps satisfied."
printf "\n%b\n" "${BOLD}Next:${RESET}"
printf "%b\n" "  uv sync --all-packages --group dev --extra mlx"
printf "%b\n" "  just rust-rebuild"
printf "%b\n" "  just package"
printf "%b\n" "  ./BUILD.sh"