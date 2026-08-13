# EXO — Build & Dependencies

How to build and release the macOS `.app`/DMG locally on your fork. Covers the prerequisites, the full release pipeline, the `v1.0.72-dev1` tag, and how it differs from upstream `v1.0.71`.

## TL;DR

The CI workflow in `.github/workflows/build-app.yml` produces the macOS DMG. It needs Apple Developer signing/notarization, Sparkle keys, AWS S3 access, and a Cachix token. **Your fork has none of those**, so the CI will fail at the first missing secret.

You have two options:

1. **Get the upstream maintainer to cut the release** — wait for them to push `v1.0.72` from the original repo.
2. **Build locally and ship from your fork** — build the `.app`+DMG on your own Mac, then push the `v1.0.72-dev1` tag as a marker on your fork. The DMG is the installable artifact; the tag is just a label.

Option 2 is what the rest of this doc covers.

## Quick start (TL;DR; one-shot)

```bash
./INSTALL.sh            # install deps via Homebrew + uv
./BUILD.sh              # build app + DMG → output/EXO-1.0.72-dev1.dmg
open output/EXO-1.0.72-dev1.dmg
```

`BUILD.sh` runs:

1. `INSTALL.sh --check` — verifies all the prerequisites.
2. `uv sync --all-packages --group dev --extra mlx` — Python deps.
3. `just rust-rebuild` — Rust → PyO3 bindings.
4. `just package` — dashboard + PyInstaller bundle.
5. `xcodebuild ... -configuration Release` — Swift app.
6. `packaging/dmg/create-dmg.sh` — DMG installer.

`BUILD.sh` skips the AppleScript Finder layout step by default in non-interactive shells (it auto-detects via `[[ -t 1 ]]`). To force the custom Finder layout, run from Terminal.app:

```bash
./BUILD.sh
```

To re-run only the app build (no DMG), `BUILD.sh --app-only`. To skip the dashboard rebuild (faster), `EXO_SKIP_DASHBOARD=1 ./BUILD.sh`.

`INSTALL.sh` and `BUILD.sh` are idempotent. See "10. Putting it all together (one-shot)" for the manual version of the same pipeline.

## 0. Prerequisites

You need:

| Tool | Why | Install |
|---|---|---|
| macOS | MLX is Metal-backed, so macOS only | — |
| Xcode (full, not CommandLineTools) | Swift app build needs `xcodebuild` from the full app | App Store |
| Metal toolchain | MLX runtime | `sudo xcodebuild -downloadComponent MetalToolchain` |
| Homebrew | Installs `uv`, `just`, Rust | `bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |
| `uv` | Python dependency manager; also installs Python 3.13 | `brew install uv` |
| Rust (`cargo`) | Builds the `exo_rs` PyO3 extension | `brew install rust` |
| `just` | Runs the build recipes in `justfile` | `brew install just` |
| Node.js | Builds the Svelte dashboard | `brew install node` |
| Python 3 | Generates the DMG background image | Pre-installed (used by `packaging/dmg/create-dmg.sh`) |

After Xcode installs:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -downloadComponent MetalToolchain
```

Make `brew` available:

```bash
eval "$(/opt/homebrew/bin/brew shellinit)"
```

Confirm:

```bash
which just uv cargo xcodebuild node python3
xcrun --find metal    # should print a path under /Applications/Xcode.app
```

### 0.5. One-shot install (optional)

`INSTALL.sh` does the entire install in one command. It runs:

1. macOS guard.
2. Xcode Command Line Tools (`xcode-select --install` if missing).
3. Homebrew (install if missing).
4. `brew install uv just rust node pkg-config openssl@3`.
5. Python 3.13 via `uv python install 3.13` + `uv python pin 3.13`.
6. Pillow for the DMG background generator (auto-detects brew Python 3.14 or Apple system Python 3.9).

Run it:

```bash
./INSTALL.sh
```

To check the install state without changing anything:

```bash
./INSTALL.sh --check
```

If `INSTALL.sh --check` exits 0, your machine is ready for `./BUILD.sh`.

`INSTALL.sh` is idempotent: every step is a no-op if already satisfied. Python 3.13 is a hard requirement (the `pyproject.toml` has `requires-python = "==3.13.*"`, and the `--extra mlx` wheels are cp313-only).

## 1. Build the Rust bindings

`uv sync --all-packages --group dev` will compile the `exo_rs` extension. The `justfile` recipe does both:

```bash
just rust-rebuild
```

What it runs:

```bash
PYO3_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" cargo run --bin stub_gen
uv sync --reinstall-package exo_rs
```

`stub_gen` writes the `rust/exo_rs/src/{lib,exo_rs}.pyi` files that the Python side consumes. `uv sync --reinstall-package exo_rs` rebuilds the package with maturin against your Python 3.13.

First run takes a few minutes (compiles `exo_rs` against MLX). Subsequent runs are cached.

## 2. Build the dashboard + PyInstaller bundle

```bash
just package
```

What it runs:

```bash
cd dashboard && npm install && npm run build && cd ..
uv run pyinstaller packaging/pyinstaller/exo.spec
rm -rf build
```

`uv run pyinstaller` picks up `pyinstaller>=6.17.0` from the dev group.

Outputs:

- `dashboard/build/` — static Svelte build, copied into the bundle as `EXO.app/Contents/Resources/dashboard/build/`.
- `dist/exo/` — the PyInstaller-frozen Python runtime, including `exo_rs`, `mlx`, `mlx_lm`, `mlx_vlm`, and the `exo` CLI entrypoint.

## 3. Build the Swift app

### Debug (fast, for testing)

```bash
just build-app
```

What it runs:

```bash
env -u LD xcodebuild build \
  -project app/EXO/EXO.xcodeproj \
  -scheme EXO \
  -configuration Debug \
  -derivedDataPath app/EXO/build
```

Output: `app/EXO/build/Build/Products/Debug/EXO.app`.

### Release (matches `v1.0.71`)

```bash
cd app/EXO
xcodebuild build \
  -project EXO.xcodeproj \
  -scheme EXO \
  -configuration Release \
  -derivedDataPath build \
  MARKETING_VERSION=1.0.72-dev1 \
  EXO_BUILD_TAG=1.0.72-dev1 \
  EXO_BUILD_COMMIT="$(git rev-parse HEAD)" \
  CODE_SIGN_IDENTITY="-" \
  CODE_SIGN_STYLE=Manual \
  CODE_SIGN_INJECT_BASE_ENTITLEMENTS=NO
cd ../..
```

Build flags:

| Flag | Value | Notes |
|---|---|---|
| `MARKETING_VERSION` | `1.0.72-dev1` | Becomes `CFBundleShortVersionString` |
| `EXO_BUILD_TAG` | `1.0.72-dev1` | Surfaces in the Swift UI as `EXOBuildTag` |
| `EXO_BUILD_COMMIT` | `git rev-parse HEAD` | Surfaces as `EXOBuildCommit` |
| `CODE_SIGN_IDENTITY` | `-` | Ad-hoc sign so macOS will let you launch it locally. Replace with your own Developer ID for distribution. |
| `CODE_SIGN_STYLE` | `Manual` | Don't auto-magick a cert |
| `CODE_SIGN_INJECT_BASE_ENTITLEMENTS` | `NO` | The app's entitlements have `app-sandbox=false`, so injection is unnecessary |

Output: `app/EXO/build/Build/Products/Release/EXO.app`.

### Entitlements (already in the project)

`app/EXO/EXO/EXO.entitlements`:

```xml
<key>com.apple.security.app-sandbox</key><false/>
<key>com.apple.security.automation.apple-events</key><true/>
<key>com.apple.security.files.user-selected.read-only</key><true/>
```

## 4. Inject the PyInstaller runtime into the app

The Swift shell is a thin launcher. It expects the Python runtime at `EXO.app/Contents/Resources/exo`. Copy it in:

```bash
EXO_APP=app/EXO/build/Build/Products/Release/EXO.app
mkdir -p "$EXO_APP/Contents/Resources"
cp -R dist/exo "$EXO_APP/Contents/Resources/exo"
```

(For Debug, substitute `Debug` for `Release`.)

## 5. Build the DMG

`packaging/dmg/create-dmg.sh` wraps the `.app` in a UDZO DMG with the same custom layout as `v1.0.71` upstream — drag-to-Applications arrow, app icon on the left, Applications alias on the right, retina background image.

```bash
./packaging/dmg/create-dmg.sh \
  app/EXO/build/Build/Products/Release/EXO.app \
  output/EXO-1.0.72-dev1.dmg \
  "EXO 1.0.72-dev1"
```

The script needs:

- `python3` (for `packaging/dmg/generate-background.py`, which uses `Pillow`).
- `hdiutil` (in `/usr/bin/hdiutil`).
- `osascript` driving Finder (so you need to be in a GUI session).

Output: `output/EXO-1.0.72-dev1.dmg`.

### 5.1. DMG headroom: the 20 MB rule

`create-dmg.sh` allocates the DMG with `APP_SIZE + 20 MB` of headroom. For a 934 MB app that's a 954 MB DMG. With UDZO compression it shrinks to ~320 MB on disk, but the writable mount is only 954 MB — which can run out of space during `cp -R $DMG_STAGING/* $MOUNT_DIR/` because HFS+ journals and Finder metadata eat the rest.

If you see `cp: No space left on device` while staging the DMG, pass `1500000k` to `hdiutil create -size` instead of `$((APP_SIZE_KB + 20480))k`. That's a 500 MB headroom.

Or just skip `create-dmg.sh` and use `BUILDScript` — `BUILD.sh` already auto-handles this and the AppleScript Finder layout (it adds `+ 200 MB` headroom and disables the Finder layout in non-interactive shells).

## 6. Install and run

```bash
open output/EXO-1.0.72-dev1.dmg
# Drag EXO.app to /Applications
open /Applications/EXO.app
```

The first launch will probably complain about Gatekeeper (the app is ad-hoc signed, not notarized). Right-click the app → Open → Open to bypass. Future launches won't complain.

If you want to install without the DMG, just `cp -R app/EXO/build/Build/Products/Release/EXO.app /Applications/` and open it.

## 7. Tag the release

Once you've tested the DMG, push the tag:

```bash
git tag v1.0.72-dev1
git push origin v1.0.72-dev1
```

This creates the `v1.0.72-dev1` tag on your fork. The CI will run `.github/workflows/build-app.yml`, but it will fail at the missing-secrets step (Apple cert, Sparkle keys, S3, Cachix). The DMG you've already built is the installable artifact.

### Why `v1.0.72-dev1` and not `v1.0.72`?

`v1.0.72-dev1` is the natural "next" tag after upstream's `v1.0.71`. The `dev1` suffix tells the Swift app (and any consumer) that this is your fork's first dev release.

Note that `build-app.yml` treats the tag as **production**, not a pre-release. Its alpha detection is:

```bash
if [[ "$VERSION" == *-alpha* ]]; then
  echo "IS_ALPHA=true"
```

So `v1.0.72-dev1` is treated like `v1.0.71` upstream: it must point at a commit on `main` (which it does), and it expects a draft GitHub Release with notes (which you can skip — your fork's CI will fail before that matters).

If you'd rather match the upstream convention, push `v1.0.72-alpha.1` instead. The CI treats that as an alpha, so the draft-release-with-notes check is optional, and the failure mode is more obviously "missing secrets" rather than "missing release notes".

## 8. Differences from upstream `v1.0.71`

The upstream release pipeline (which `build-app.yml` codifies) is:

1. **Apple Developer cert** (`MACOS_CERTIFICATE`, `MACOS_CERTIFICATE_PASSWORD`, `PROVISIONING_PROFILE`) — creates a build keychain, imports the cert, sets up provisioning.
2. **Apple notarization** (`APPLE_NOTARIZATION_USERNAME`, `APPLE_NOTARIZATION_PASSWORD`, `APPLE_NOTARIZATION_TEAM`) — `xcrun notarytool submit`, then `xcrun stapler staple` the DMG.
3. **Sparkle** (`SPARKLE_ED25519_PUBLIC`, `SPARKLE_ED25519_PRIVATE`, `SPARKLE_FEED_URL`, `SPARKLE_DOWNLOAD_PREFIX`) — generates `appcast.xml`, signs it with the Ed25519 key, embeds the release notes as Markdown (Sparkle 2.9+).
4. **AWS S3** (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `SPARKLE_S3_BUCKET`, `SPARKLE_S3_PREFIX`, `AWS_REGION`) — uploads the DMG and `appcast.xml`.
5. **Cachix** (`CACHIX_AUTH_TOKEN`) — caches the Nix-built dashboard and Metal toolchain.
6. **Draft GitHub Release** — must exist with notes for non-alpha tags. The CI publishes it after attaching the DMG.

For your fork, you skip steps 1, 2, 3, 4, 5, 6 (or most of them) and just do the local build. The DMG is unsigned, unnotarized, and won't auto-update via Sparkle — but it installs and runs.

## 9. CI secrets reference

For reference, the secrets the upstream CI uses:

| Secret | Source | Used by |
|---|---|---|
| `MACOS_CERTIFICATE` | Apple Developer — `Developer ID Application` cert, base64-encoded | `security create-keychain` → `security import` |
| `MACOS_CERTIFICATE_PASSWORD` | Cert password | `security import`, `security unlock-keychain` |
| `PROVISIONING_PROFILE` | Apple Developer — provisioning profile, base64-encoded | `~/Library/Developer/Xcode/UserData/Provisioning Profiles/EXO.provisionprofile` |
| `APPLE_NOTARIZATION_USERNAME` | Apple ID | `xcrun notarytool submit --apple-id` |
| `APPLE_NOTARIZATION_PASSWORD` | App-specific password | `xcrun notarytool submit --password` |
| `APPLE_NOTARIZATION_TEAM` | Apple Developer Team ID | `xcrun notarytool submit --team-id` |
| `SPARKLE_ED25519_PUBLIC` | `nix run sparkle-keygen` or `xcodeproj://sparkle-generate-keys` | `Info.plist` → `SUPublicEDKey` |
| `SPARKLE_ED25519_PRIVATE` | Same | `appcast.xml` signing |
| `SPARKLE_FEED_URL` | Where `appcast.xml` lives (e.g., `https://assets.exolabs.net/appcast.xml`) | `Info.plist` → `SUFeedURL` |
| `SPARKLE_DOWNLOAD_PREFIX` | Where DMGs live (e.g., `https://assets.exolabs.net`) | `appcast.xml` enclosure URLs |
| `SPARKLE_S3_BUCKET` | Bucket for DMG + appcast | `aws s3 cp` |
| `SPARKLE_S3_PREFIX` | Path prefix inside the bucket | `aws s3 cp` |
| `AWS_REGION` | Bucket region | `aws s3 cp` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | IAM user with `s3:PutObject` on the bucket | `aws s3 cp` |
| `CACHIX_AUTH_TOKEN` | cachix.org → `exo` cache | `cachix/cachix-action@v14` |
| `SPARKLE_CLI_URL` (optional) | Override the Sparkle CLI download | `curl --fail --location` |

If you ever do want to run the full CI on your fork, add these as fork secrets (Settings → Secrets and variables → Actions). But you'll also need:

- An Apple Developer account with a Developer ID cert.
- An S3 bucket with a public download prefix.
- A Sparkle keypair.
- A `cachix.org` account.

That's the long road. For local-only releases, skip it.

## 10. Putting it all together (one-shot)

```bash
# Once-off setup
./INSTALL.sh             # macOS guard, Xcode CLT, Homebrew, uv, just, rust, node, openssl, Python 3.13, Pillow.

# Per-release (or `./BUILD.sh --clean` for a full rebuild)
./BUILD.sh               # app + DMG.
# Or:
./BUILD.sh --app-only    # app only, no DMG.
EXO_SKIP_DASHBOARD=1 ./BUILD.sh --app-only  # skip dashboard rebuild, faster.

open output/EXO-1.0.72-dev1.dmg
git tag v1.0.72-dev1
git push origin v1.0.72-dev1
```

`BUILD.sh` runs:

| Step | What | Why |
|---|---|---|
| 0 | `INSTALL.sh --check` | Verify all the prereqs (Xcode CLT, brew packages, Python 3.13, Pillow). |
| 1 | `uv sync --all-packages --group dev --extra mlx` | Python deps, including MLX, mlx-lm, mlx-vlm, mflux, torch, torchaudio, torchvision. |
| 2 | `just rust-rebuild` | Rust → PyO3 bindings. Calls `uv sync --reinstall-package exo_rs`, which clears the `mlx` extra, so step 2 also re-syncs `--extra mlx`. |
| 3 | `just package` | Dashboard (`npm install && npm run build`) + PyInstaller bundle (`uv run pyinstaller packaging/pyinstaller/exo.spec`). Wipes `dist/exo/` first because PyInstaller's `COLLECT` refuses to write into an existing directory. |
| 4 | `xcodebuild build -configuration Release` | Swift app. Adds the PyInstaller bundle at `EXO.app/Contents/Resources/exo`. Sets `MARKETING_VERSION`, `EXO_BUILD_TAG`, `EXO_BUILD_COMMIT` from `git rev-parse HEAD`. Ad-hoc signs with `CODE_SIGN_IDENTITY=-`. |
| 5 | `packaging/dmg/create-dmg.sh` | DMG. Auto-passes `--no-finder` in non-interactive shells. Adds `+ 200 MB` headroom (vs. the original `+ 20 MB`). |

`BUILD.sh` does NOT touch `Cargo.toml`, `pyproject.toml`, or any source file. It assumes the working tree is clean. To re-build after a `cargo build` change, run `just rust-rebuild` manually.

To clean:

```bash
just clean               # Cargo target/, .venv/, dashboard/node_modules/, dashboard/build/
rm -rf dist build app/EXO/build  # PyInstaller + Xcode derived data.
./BUILD.sh --clean
```

For full control without the wrapper:

```bash
just rust-rebuild          # ~3 min first time, cached after.
just package               # dashboard + PyInstaller.
cd app/EXO
xcodebuild build \
  -project EXO.xcodeproj \
  -scheme EXO \
  -configuration Release \
  -derivedDataPath build \
  MARKETING_VERSION=1.0.72-dev1 \
  EXO_BUILD_TAG=1.0.72-dev1 \
  EXO_BUILD_COMMIT="$(git rev-parse HEAD)" \
  CODE_SIGN_IDENTITY="-" \
  CODE_SIGN_STYLE=Manual \
  CODE_SIGN_INJECT_BASE_ENTITLEMENTS=NO
cd ../..
EXO_APP=app/EXO/build/Build/Products/Release/EXO.app
mkdir -p "$EXO_APP/Contents/Resources"
cp -R dist/exo "$EXO_APP/Contents/Resources/exo"
./packaging/dmg/create-dmg.sh "$EXO_APP" output/EXO-1.0.72-dev1.dmg "EXO 1.0.72-dev1"
git tag v1.0.72-dev1
git push origin v1.0.72-dev1
```

## 11. Common failures

- **`xcodebuild requires Xcode, but active developer directory is a command line tools instance`** — `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.
- **`xcrun: unable to find utility "metal"`** — `sudo xcodebuild -downloadComponent MetalToolchain`.
- **`maturin: command not found`** — `uv` ships its own maturin via `uv sync`, but you need Rust installed first: `brew install rust`.
- **`pyinstaller: command not found`** — `uv sync --all-packages --group dev` (PyInstaller is in the dev group, not the default group).
- **`Module 'mlx' is not available in the current environment.`** — `just rust-rebuild` calls `uv sync --reinstall-package exo_rs`, which clears non-default extras. Re-sync with `uv sync --all-packages --group dev --extra mlx`. `BUILD.sh` does this automatically.
- **`ERROR: The output directory "/path/dist/exo" is not empty`** — PyInstaller's `COLLECT` refuses to write into an existing output directory. Wipe `dist/exo/` first. `BUILD.sh` does this automatically.
- **`App is damaged and can't be opened`** — Gatekeeper rejecting the unsigned/ad-hoc binary. Right-click → Open → Open. Or set `CODE_SIGN_IDENTITY=-` explicitly.
- **`The application "EXO" cannot be opened`** — usually because Metal isn't installed, or because the entitlements are wrong. Check Console.app → "EXO" for the actual reason.
- **Sparkle keeps complaining about updates** — expected when `SPARKLE_FEED_URL` is empty. Disable with `defaults write exolabs.EXO SUEnableAutomaticChecks -bool false`, or set a fake feed URL.
- **`Permission denied` running `osascript` in `create-dmg.sh`** — Finder automation needs a GUI session. Run from Terminal.app, not over SSH. Pass `--no-finder` to skip the layout step.
- **`cp: No space left on device` building the DMG** — the writable DMG is `APP_SIZE + 20 MB`, which is too small for a 934 MB app + HFS+ journal. Pass `--no-finder` (already auto-set by `BUILD.sh` in non-interactive shells) and increase the headroom by editing `packaging/dmg/create-dmg.sh` (the `DMG_SIZE_BYTES = APP_SIZE_BYTES + 200 * 1024 * 1024` line).

## 12. What's next

- Replace `1.0.72-dev1` with `1.0.73-dev1` (or whatever) for the next dev release.
- Once the upstream cuts `v1.0.72`, you can `git fetch upstream && git tag v1.0.72 upstream/v1.0.72` to bring your fork in sync.
- For end-user distribution, you'll eventually want notarization. That means an Apple Developer account, a signing cert, and an `xcrun notarytool submit` step in `packaging/dmg/create-dmg.sh`. The CI's "Sign, notarize, and create DMG" step (`build-app.yml`) is the canonical reference.