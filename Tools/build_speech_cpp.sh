#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--archive" ) ]]; then
    printf '%s\n' "Usage: $0 [--archive]" >&2
    exit 2
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
work="$root/Frameworks/.build"
source="${HARU_SPEECH_CPP_SOURCE:-$work/source}"
revision=8d18924cd238b87c0297df112f7ff40783a6414f
ggml_revision=79b50d239d0b88852814c873db0d85d43ed66d9b
mkdir -p "$work"
if [[ ! -d "$source/.git" ]]; then
    git clone --depth 1 --single-branch --branch v0.11.0 \
        --recurse-submodules --shallow-submodules https://github.com/nyosegawa/speech.cpp.git "$source"
fi
[[ "$(git -C "$source" rev-parse HEAD)" == "$revision" ]]
git -C "$source" submodule update --init --depth 1
[[ "$(git -C "$source/ggml" rev-parse HEAD)" == "$ggml_revision" ]]
patch="$root/Tools/speech_cpp_haru_turns.patch"
if ! git -C "$source" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$source" apply "$patch"
fi

frameworks=()
for sdk in iphoneos iphonesimulator; do
    build="$work/$sdk"
    cmake -S "$source" -B "$build" -G Ninja \
        -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT="$sdk" \
        -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0 \
        -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
        -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_TARGET_OS=ios \
        -DGGML_BLAS=OFF -DGGML_ACCELERATE=ON \
        -DHARU_SPEECH_SYMBOL_HEADER="$root/Tools/HaruSpeechSymbols.h"
    cmake --build "$build" --target speech -j "${HARU_SPEECH_CPP_JOBS:-4}"

    framework="$build/HaruSpeechCpp.framework"
    mkdir -p "$framework/Headers" "$framework/Modules"
    cp -pL "$build/libspeech.dylib" "$framework/HaruSpeechCpp"
    install_name_tool -id '@rpath/HaruSpeechCpp.framework/HaruSpeechCpp' "$framework/HaruSpeechCpp"
    cp "$source/include/speech.h" "$framework/Headers/"
    cp "$root/Tools/HaruSpeechSymbols.h" "$framework/Headers/"
    cat >"$framework/Headers/HaruSpeechCpp.h" <<'HEADER'
#include "HaruSpeechSymbols.h"
#include "speech.h"
HEADER
    cat >"$framework/Modules/module.modulemap" <<'MODULE'
framework module HaruSpeechCpp {
    umbrella header "HaruSpeechCpp.h"
    export *
}
MODULE
    platform=iPhoneOS
    if [[ "$sdk" == iphonesimulator ]]; then platform=iPhoneSimulator; fi
    cat >"$framework/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>app.stackchan.haru.speechcpp</string>
<key>CFBundleExecutable</key><string>HaruSpeechCpp</string>
<key>CFBundleName</key><string>HaruSpeechCpp</string>
<key>CFBundlePackageType</key><string>FMWK</string>
<key>CFBundleShortVersionString</key><string>0.11.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>MinimumOSVersion</key><string>18.0</string>
<key>CFBundleSupportedPlatforms</key><array><string>$platform</string></array>
</dict></plist>
PLIST
    frameworks+=(-framework "$framework")
done

staged="$(mktemp -d "$work/package.XXXXXX")"
trap 'rm -rf "$staged"' EXIT
xcodebuild -create-xcframework "${frameworks[@]}" -output "$staged/HaruSpeechCpp.xcframework"
licenses="$staged/HaruSpeechCpp.xcframework/Licenses"
mkdir -p "$licenses"
cp "$root/Sources/HaruKit/Licenses/"{SpeechCpp,GGML,Arm-Optimized-Routines,YaRN}.txt "$licenses/"
cp "$root/LICENSE" "$licenses/HaruKit.txt"
rm -rf "$root/Frameworks/HaruSpeechCpp.xcframework"
mv "$staged/HaruSpeechCpp.xcframework" "$root/Frameworks/"

if [[ "${1:-}" == "--archive" ]]; then
    mkdir -p "$root/Artifacts"
    ditto -c -k --norsrc --keepParent "$root/Frameworks/HaruSpeechCpp.xcframework" "$staged/HaruSpeechCpp.xcframework.zip"
    mv "$staged/HaruSpeechCpp.xcframework.zip" "$root/Artifacts/HaruSpeechCpp.xcframework.zip"
fi
