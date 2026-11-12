#!/bin/bash
# Create a Tempo (github.com/tempo-sim/Tempo) project that renders the LiveDirt simulation, so Tempo's cameras and
# lidars can see the dig and be streamed / controlled over Tempo's gRPC API.
#
#   unreal/make_tempo_project.sh [target_dir]      (default: ~/LiveDirtTempo)
#
# Then: Scripts/Build.sh and Scripts/Run.sh in the new project (see unreal/TEMPO.md).
# Needs: UE 5.8 (Epic Games Launcher), Xcode, git, jq. Tempo builds C++, so this one does need Xcode.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HOME/LiveDirtTempo}"
NAME="$(basename "$DEST")"

command -v jq >/dev/null || { echo "jq is required: brew install jq"; exit 1; }
[ -e "$DEST" ] && { echo "$DEST already exists; pass a different directory"; exit 1; }

# 1. TempoSample is the recommended starting point (UE 5.8, Tempo plugins as a submodule)
git clone --recurse-submodules https://github.com/tempo-sim/TempoSample.git "$DEST"
cd "$DEST"
./Scripts/Rename.sh "$NAME"

# 2. the plugins the LiveDirt renderer uses (Python editor scripting + Geometry Script for the dynamic terrain mesh)
jq '.Plugins += [{"Name":"PythonScriptPlugin","Enabled":true},{"Name":"EditorScriptingUtilities","Enabled":true},{"Name":"GeometryScripting","Enabled":true}]' \
  "$NAME.uproject" > "$NAME.uproject.tmp" && mv "$NAME.uproject.tmp" "$NAME.uproject"

# 3. the renderer script, auto-started when the editor opens (it builds /Game/LiveDirt/Maps/LiveDirt and follows the bridge)
mkdir -p Content/Python
cp "$HERE/LiveDirtUE/Content/Python/livedirt_ue.py" "$HERE/LiveDirtUE/Content/Python/init_livedirt.py" Content/Python/
printf '\n[/Script/PythonScriptPlugin.PythonScriptPluginSettings]\n+StartupScripts=init_livedirt.py\nbDeveloperMode=True\n' >> Config/DefaultEngine.ini

# 4. Tempo's one-time setup (third-party dependencies + git hooks)
./Plugins/Tempo/Setup.sh

echo
echo "Done: $DEST"
echo "Next:  cd \"$DEST\" && ./Scripts/Build.sh && ./Scripts/Run.sh   (keep unreal/run_bridge.sh running first)"
