#!/usr/bin/env bash
# BUILD.sh — Build EXO from source.
#
# Pipeline:
#   0. Verify deps via `./INSTALL.sh --check`.
#   1. `uv sync --all-packages --group dev --extra mlx` (Python runtime).
#   2. `just rust-rebuild`   (Rust → PyO3 bindings).
#   3. `just package`        (dashboard + PyInstaller bundle → dist/exo/).
#   4. `xcodebuild`          (Swift app → app/EXO/build/.../EXO.app, Release config).
#   5. `./packaging/dmg/create-dmg.sh` (DMG installer → output/EXO-VERSION.dmg).
#
# Steps 1-4 are the "app build". Step 5 is the "DMG build". Skip step 5 with
# `--app-only`. Skip the dashboard rebuild with `EXO_SKIP_DASHBOARD=1` (use it
# after dependency-only changes that don't touch dashboard sources).
#
# Environment variables:
#   EXO_VERSION     Override the version (default: `git describe --tags --abbrev=0`).
#   EXO_OUTPUT_DIR  Where to write the DMG (default: ./output).
#   EXO_SKIP_DASHBOARD=1  Skip `npm install` + `npm run build` (faster re-builds).
#   EXO_BUILD_MLX_KERNELS=1  Build the vendored oMLX GLM-5.2 native Metal kernels
#                           (Darwin + full Xcode required; opt-in). The built
#                           _ext.so + metallib are bundled by PyInstaller.
#
# Usage:
#   ./BUILD.sh                   # build the app + DMG.
#   ./BUILD.sh --app-only        # build the app, no DMG.
#   ./BUILD.sh --clean           # `just clean` first, then full rebuild.
#
# After this script:
#   open output/EXO-VERSION.dmg
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Argument parsing ──────────────────────────────────────────────────────────
APP_ONLY=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --app-only) APP_ONLY=1 ;;
    --clean)    CLEAN=1 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "==> ERROR: unknown argument: $arg"; exit 1 ;;
  esac
done

# Colours
if [[ -t 1 ]]; then
  RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'
  BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
  RED=""; GREEN=""; YELLOW=""; BLUE=""; BOLD=""; RESET=""
fi
say()   { printf "%b\n" "${BLUE}==>${RESET} ${BOLD}$*${RESET}"; }
warn()  { printf "%b\n" "${YELLOW}==> WARNING:${RESET} $*"; }
err()   { printf "%b\n" "${RED}==> ERROR:${RESET} $*" >&2; }
ok()    { printf "%b\n" "${GREEN}==>${RESET} $*"; }
hr()    { printf "%b\n" "${BLUE}─────────────────────────────────────────────────${RESET}"; }

# ── 0. Verify deps ────────────────────────────────────────────────────────────
hr
say "Step 0/5: Verifying dependencies"
hr

if ! ./INSTALL.sh --check; then
  err "Dependencies missing. Run ./INSTALL.sh first."
  exit 1
fi

# Ensure brew shellenv is active so `uv`, `just`, etc. are on PATH.
if [[ -f /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# uv local bin must be on PATH (uv-managed Python).
export PATH="$HOME/.local/bin:$PATH"

# ── Resolve version ───────────────────────────────────────────────────────────
if [[ -z "${EXO_VERSION:-}" ]]; then
  # Use the latest tag (cleanly), or fall back to short SHA.
  EXO_VERSION="$(git describe --tags --abbrev=0 2>/dev/null || git rev-parse --short HEAD)"
fi
ok "Version: $EXO_VERSION"

# ── Clean if requested ────────────────────────────────────────────────────────
if [[ $CLEAN -eq 1 ]]; then
  say "Cleaning previous builds..."
  rm -rf dist/ build/ app/EXO/build/
  say "Running 'just clean'..."
  just clean || warn "just clean failed; continuing."
fi

# ── 1. Python deps ────────────────────────────────────────────────────────────
hr
say "Step 1/5: uv sync (Python runtime)"
hr

if [[ ! -d .venv ]]; then
  say "Creating virtualenv with uv..."
  uv venv --python 3.13
fi

uv sync --all-packages --group dev --extra mlx
ok "Python deps in sync."

# ── 2. Rust bindings ──────────────────────────────────────────────────────────
hr
say "Step 2/5: just rust-rebuild"
hr

just rust-rebuild
ok "Rust bindings rebuilt."

# `just rust-rebuild` calls `uv sync --reinstall-package exo_rs`, which clears
# non-default extras (e.g. `mlx`). Re-sync to put them back.
say "Re-syncing with --extra mlx (just rust-rebuild reset the venv)..."
uv sync --all-packages --group dev --extra mlx

# ── 2b. Optional: native Metal kernels (GLM-5.2) ────────────────────────────
# Opt-in via EXO_BUILD_MLX_KERNELS=1 (Darwin + full Xcode only). Builds the
# vendored oMLX nanobind _ext + metallib into the source tree so PyInstaller
# bundles them and `fast.native_available()` returns True at runtime.
if [[ "${EXO_BUILD_MLX_KERNELS:-0}" == "1" ]]; then
  if [[ "$(uname)" != "Darwin" ]]; then
    warn "EXO_BUILD_MLX_KERNELS=1 set but not on Darwin; skipping native kernels."
  else
    hr
    say "Step 2b/5: just mlx-kernels (native Metal GLM-5.2 kernels)"
    hr
    uv run --extra build python mlx_kernels/build_kernels.py --force
    ok "Native Metal kernels built."
  fi
fi

# ── 3. Dashboard + PyInstaller bundle ────────────────────────────────────────
hr
say "Step 3/5: just package"
hr

# PyInstaller's COLLECT refuses to write into an existing output directory.
# Wipe `dist/exo/` so a re-run works.
rm -rf dist/exo build/exo

if [[ "${EXO_SKIP_DASHBOARD:-0}" -eq 1 ]]; then
  say "Skipping dashboard rebuild (EXO_SKIP_DASHBOARD=1)."
else
  just package
fi

if [[ ! -x dist/exo/exo ]]; then
  err "PyInstaller bundle missing: dist/exo/exo"
  exit 1
fi
ok "PyInstaller bundle: $(du -sh dist/exo | cut -f1) at dist/exo/"

# ── 4. Swift app (Release config) ─────────────────────────────────────────────
hr
say "Step 4/5: xcodebuild (Release)"
hr

MARKETING_VERSION="$EXO_VERSION"
EXO_BUILD_TAG="$EXO_VERSION"
EXO_BUILD_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"

XCODE_PROJ="app/EXO/EXO.xcodeproj"
XCODE_FLAGS=(
  -project "$XCODE_PROJ"
  -scheme EXO
  -configuration Release
  -derivedDataPath app/EXO/build
  MARKETING_VERSION="$MARKETING_VERSION"
  EXO_BUILD_TAG="$EXO_BUILD_TAG"
  EXO_BUILD_COMMIT="$EXO_BUILD_COMMIT"
  CODE_SIGN_IDENTITY=-
  CODE_SIGN_STYLE=Manual
  CODE_SIGN_INJECT_BASE_ENTITLEMENTS=NO
)

xcodebuild build "${XCODE_FLAGS[@]}"

APP_PATH="app/EXO/build/Build/Products/Release/EXO.app"
if [[ ! -d "$APP_PATH" ]]; then
  err "EXO.app missing at $APP_PATH"
  exit 1
fi

# Copy the PyInstaller bundle into the .app (Xcode normally does this via the
# PBXResourcesBuildPhase, but a manual run may need a forced refresh).
RESOURCES_EXO="$APP_PATH/Contents/Resources/exo"
if [[ ! -d "$RESOURCES_EXO/_internal" ]]; then
  say "Embedding PyInstaller bundle into EXO.app..."
  rm -rf "$RESOURCES_EXO"
  mkdir -p "$RESOURCES_EXO"
  cp -R dist/exo/_internal "$RESOURCES_EXO/_internal"
  cp dist/exo/exo "$RESOURCES_EXO/exo"
fi

ok "EXO.app: $(du -sh "$APP_PATH" | cut -f1)"

# Verify Info.plist is stamped with the right version.
PLIST="$APP_PATH/Contents/Info.plist"
PLIST_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$PLIST" 2>/dev/null || true)"
if [[ -n "$PLIST_VERSION" && "$PLIST_VERSION" != "$EXO_VERSION" ]]; then
  err "EXO.app reports version '$PLIST_VERSION', expected '$EXO_VERSION'."
  exit 1
fi
ok "EXO.app version: $PLIST_VERSION"

# ── 5. DMG installer ─────────────────────────────────────────────────────────
if [[ $APP_ONLY -eq 1 ]]; then
  hr
  say "Step 5/5: skipped (--app-only)"
  hr
else
  hr
  say "Step 5/5: create-dmg.sh"
  hr

  OUTPUT_DIR="${EXO_OUTPUT_DIR:-output}"
  mkdir -p "$OUTPUT_DIR"

  # Strip the leading 'v' so the DMG file name and volume name are clean.
  DMG_BASE_NAME="${EXO_VERSION#v}"
  OUTPUT_DMG="$OUTPUT_DIR/EXO-${DMG_BASE_NAME}.dmg"
  DMG_VOLUME_NAME="EXO ${DMG_BASE_NAME}"

  # The Finder layout step uses AppleScript and only works in interactive
  # shells (it drives the UI). In non-interactive shells (CI, agents, SSH)
  # it times out — skip with --no-finder.
  FINDER_FLAGS=()
  if [[ ! -t 1 ]]; then
    FINDER_FLAGS+=(--no-finder)
  fi

  ./packaging/dmg/create-dmg.sh "$APP_PATH" "$OUTPUT_DMG" "$DMG_VOLUME_NAME" "${FINDER_FLAGS[@]}"

  if [[ ! -f "$OUTPUT_DMG" ]]; then
    err "DMG missing at $OUTPUT_DMG"
    exit 1
  fi

  ok "DMG: $OUTPUT_DMG ($(du -h "$OUTPUT_DMG" | cut -f1))"
  shasum -a 256 "$OUTPUT_DMG" | awk '{print "SHA256:", $1}'
fi

# ── Done ──────────────────────────────────────────────────────────────────────
hr
say "Build summary"
hr

printf "%b\n" "${GREEN}OK${RESET}  Build complete."
printf "\n%b\n" "${BOLD}App:${RESET}      $APP_PATH"
if [[ $APP_ONLY -eq 0 ]]; then
  printf "%b\n" "${BOLD}DMG:${RESET}      $OUTPUT_DMG"
fi
printf "%b\n" "${BOLD}Version:${RESET}  $EXO_VERSION"
printf "%b\n" "${BOLD}Commit:${RESET}   $(git rev-parse --short HEAD 2>/dev/null || true)"
if [[ $APP_ONLY -eq 0 ]]; then
  printf "\n%b\n" "${BOLD}Run:${RESET}      open \"$OUTPUT_DMG\""
fi