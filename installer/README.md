# Lynx native installers

One downloadable file per platform that installs Lynx and opens the guided
web UI. No terminal, no `git`, no manual Python setup. Aimed at people who
just want to use Lynx.

## Layout

```
installer/
  macos/
    launcher.sh          -> Lynx.app/Contents/MacOS/Lynx (opens Terminal)
    bootstrap.sh         -> install-if-needed via uv, then `manager ui`
    Info.plist.template  -> bundle metadata (version substituted at build)
    build-dmg.sh         -> assembles Lynx.app + the .dmg
    Lynx.png  (optional) -> source icon; converted to .icns if present
  windows/
    bootstrap.bat        -> Windows equivalent of bootstrap.sh
    lynx.iss             -> Inno Setup script -> Lynx-Setup-<ver>.exe
    Lynx.ico  (optional) -> app icon; used by Inno Setup if present
```

The `uv` binary is not committed. CI downloads it from the
[astral-sh/uv releases](https://github.com/astral-sh/uv/releases) and bundles
it into each installer. The pinned version lives in
`.github/workflows/release.yml` (`UV_VERSION`).

## Building

Both installers are produced by `.github/workflows/release.yml`:

- Push a tag `vX.Y.Z`: builds both and publishes a GitHub Release with the
  `.dmg` and `.exe` attached.
- Run the workflow manually (`workflow_dispatch`): builds the artifacts
  only (no Release), handy for testing.

### Local macOS build

```bash
# download a uv binary for your Mac, then:
UV_ARM64_BIN=/path/to/uv VERSION=1.0.0 bash installer/macos/build-dmg.sh
# -> dist/Lynx-1.0.0-macos.dmg
```

### Local Windows build

On Windows with [Inno Setup 6](https://jrsoftware.org/isdl.php) installed and
`uv.exe` copied next to `lynx.iss`:

```pwsh
ISCC /DMyAppVersion=1.0.0 installer\windows\lynx.iss
# -> dist\Lynx-Setup-1.0.0.exe
```

## First-launch warnings (unsigned builds)

The installers are not code-signed yet, so the OS shows a one-time
warning. This is expected:

- macOS (Gatekeeper): right-click `Lynx.app`, then **Open**, then **Open**
  again. After the first time it launches normally.
- Windows (SmartScreen): click **More info**, then **Run anyway**.

### Adding signing later

The CI workflow has commented slots for both:

- macOS: import a *Developer ID Application* cert, `codesign --options runtime`
  the app, then `xcrun notarytool submit` + staple the `.dmg`.
- Windows: `signtool sign` the `Lynx-Setup-*.exe` with a code-signing cert.

Both need secrets added to the repo. Once signed and notarized, the warnings
above disappear.

## How it works (thin bootstrapper)

The runtime stack (`onnxruntime` + `chromadb` + LlamaIndex) weighs about
470 MB installed, plus a ~130 MB embedding model. Shipping that inside a single
offline binary would be huge and awkward to sign and distribute. Instead the
installers are thin: they bundle only the [`uv`](https://docs.astral.sh/uv/)
binary (~30 MB) and download everything else on first launch.

```
download installer  ->  install (instant)  ->  first launch:
    uv installs lynx (~0.5 GB, once)  ->  init writes config + downloads model
    ->  browser opens http://127.0.0.1:8765 (LynxManager UI)
later launches: UI starts immediately, no download
```

Everything lives in a private per-user folder, so nothing touches the system
Python or PATH:

- macOS: `~/Library/Application Support/Lynx/`
- Windows: `%LOCALAPPDATA%\Lynx\`

To fully remove Lynx, uninstall the app and delete that folder.
