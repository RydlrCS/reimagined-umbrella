#!/bin/bash
# Setup Michele Motion Library
# This script helps organize downloaded Michele animations

echo "🎬 Michele Motion Library Setup"
echo "================================"

MICHELE_DIR="/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/michele"
DOWNLOAD_DIR="$HOME/Downloads"

# Create directory structure
mkdir -p "$MICHELE_DIR"

# Check if any Michele FBX files exist in Downloads
MICHELE_COUNT=$(find "$DOWNLOAD_DIR" -maxdepth 1 -name "*.fbx" 2>/dev/null | wc -l)

if [ $MICHELE_COUNT -gt 0 ]; then
    echo "✅ Found $MICHELE_COUNT FBX files in Downloads"
    echo "📦 Moving to Michele library..."
    
    mv "$DOWNLOAD_DIR"/*.fbx "$MICHELE_DIR/" 2>/dev/null
    
    FINAL_COUNT=$(find "$MICHELE_DIR" -name "*.fbx" | wc -l)
    echo "✅ Michele library now contains $FINAL_COUNT animations"
    
    echo ""
    echo "📋 Animation List:"
    ls -1 "$MICHELE_DIR"/*.fbx | head -10
    if [ $FINAL_COUNT -gt 10 ]; then
        echo "... and $((FINAL_COUNT - 10)) more"
    fi
else
    echo "⚠️  No FBX files found in Downloads"
    echo ""
    echo "📥 Please download Michele animations first:"
    echo "   1. Go to https://www.mixamo.com"
    echo "   2. Log in with Adobe account"
    echo "   3. Open browser console (F12)"
    echo "   4. Copy and paste the download script"
    echo "   5. Run this script again after downloads complete"
    echo ""
    echo "See MICHELE_DOWNLOAD_INSTRUCTIONS.md for details"
fi

echo ""
echo "📁 Michele library location: $MICHELE_DIR"
