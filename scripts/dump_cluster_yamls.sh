#!/bin/bash
# Dump all resource YAMLs from a Kubernetes cluster.
# Discovers resource types dynamically. Skips secrets. Parallel fetches.
# Requires: kubectl configured with cluster access.
# Usage: ./scripts/dump_cluster_yamls.sh [output_dir]

set -e

OUTPUT_DIR="${1:-cluster-yamls}"
mkdir -p "$OUTPUT_DIR"

SKIP_KINDS="secrets|events|events.events.k8s.io"

echo "Dumping cluster resources to: $OUTPUT_DIR"
echo "Cluster: $(kubectl config current-context)"
echo "Skipping: $SKIP_KINDS"
echo ""

echo "Discovering resource types..."
RESOURCE_TYPES=$(kubectl api-resources --verbs=list -o name 2>/dev/null | grep -Ev "^($SKIP_KINDS)$" | sort)
NUM_TYPES=$(echo "$RESOURCE_TYPES" | wc -l)
echo "Found $NUM_TYPES resource types"
echo ""

total=0
cmdfile=$(mktemp)

for kind in $RESOURCE_TYPES; do
    kind_dir="$OUTPUT_DIR/$kind"
    items=$(kubectl get "$kind" --all-namespaces -o jsonpath='{range .items[*]}{.metadata.namespace}{"|"}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)

    [ -z "$items" ] && continue
    mkdir -p "$kind_dir"
    count=0

    while IFS='|' read -r ns name; do
        [ -z "$name" ] && continue
        # Sanitize name for filesystem
        safe_name=$(echo "$name" | tr '/' '_')
        if [ -z "$ns" ]; then
            echo "kubectl get '$kind' '$name' -o yaml > '$kind_dir/${safe_name}.yaml' 2>/dev/null" >> "$cmdfile"
        else
            echo "kubectl get '$kind' '$name' -n '$ns' -o yaml > '$kind_dir/${ns}_${safe_name}.yaml' 2>/dev/null" >> "$cmdfile"
        fi
        count=$((count + 1))
    done <<< "$items"

    if [ "$count" -gt 0 ]; then
        echo "  $kind: $count"
        total=$((total + count))
    fi
done

echo ""
echo "Fetching $total resources (10 parallel)..."
cat "$cmdfile" | xargs -P 10 -I {} sh -c {}
rm -f "$cmdfile"

# Count actual files written
actual=$(find "$OUTPUT_DIR" -name '*.yaml' | wc -l)
echo "Done. $actual YAML files saved to $OUTPUT_DIR/"
