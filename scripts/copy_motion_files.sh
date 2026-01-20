#!/bin/bash
# Copy capoeira and breakdance related files from michelle.fbx to data directory

SOURCE_DIR="/workspaces/reimagined-umbrella/michelle.fbx"
DEST_DIR="/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/michele"

# Create destination directory
mkdir -p "$DEST_DIR"

echo "Finding relevant motion files..."

# Find files matching keywords
FILES=$(find "$SOURCE_DIR" -name "*.fbx" -type f | grep -iE "(capoeira|break|kick|flip|spin|dance)" | head -12)

echo "Found files:"
echo "$FILES" | nl

echo ""
echo "Copying files..."

COUNT=0
for FILE in $FILES; do
    BASENAME=$(basename "$FILE")
    if [ ! -f "$DEST_DIR/$BASENAME" ]; then
        cp "$FILE" "$DEST_DIR/"
        echo "✅ Copied: $BASENAME"
        COUNT=$((COUNT + 1))
    else
        echo "⏭️  Exists: $BASENAME"
    fi
done

echo ""
echo "Copied $COUNT new files to $DEST_DIR"
echo ""
echo "Files in destination:"
ls -1 "$DEST_DIR/"
