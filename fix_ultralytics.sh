# Save this as fix_ultralytics.sh in your internship folder
cat > ~/fix_ultralytics.sh << 'EOF'
#!/bin/bash
VENV_PATH=$1
FILE="$VENV_PATH/lib/python3.13/site-packages/ultralytics/utils/__init__.py"

if [ -f "$FILE" ]; then
    sed -i '' 's/yaml\.CSafeLoader/yaml.SafeLoader/g' "$FILE"
    sed -i '' 's/getattr(yaml, .CSafeLoader., yaml\.SafeLoader)/yaml.SafeLoader/g' "$FILE"
    echo "Fixed: $FILE"
    grep -n "SafeLoader" "$FILE" | head -5
else
    echo "File not found: $FILE"
fi
EOF
chmod +x ~/fix_ultralytics.sh