#!/bin/bash
# Dump all resource YAMLs from a Kubernetes cluster.
# Requires: kubectl configured with cluster access.
# Usage: ./scripts/dump_cluster_yamls.sh [output_dir]

set -e

OUTPUT_DIR="${1:-cluster-yamls}"
mkdir -p "$OUTPUT_DIR"

echo "Dumping cluster resources to: $OUTPUT_DIR"
echo "Cluster: $(kubectl config current-context)"
echo ""

# Namespaced resource types to dump
NAMESPACED_KINDS=(
    deployments statefulsets daemonsets jobs cronjobs
    pods services ingresses
    configmaps secrets
    serviceaccounts roles rolebindings
    persistentvolumeclaims
    networkpolicies poddisruptionbudgets
    horizontalpodautoscalers
    limitranges resourcequotas
)

# Cluster-scoped resource types
CLUSTER_KINDS=(
    nodes namespaces
    clusterroles clusterrolebindings
    persistentvolumes storageclasses
    priorityclasses
    customresourcedefinitions
    validatingwebhookconfigurations mutatingwebhookconfigurations
)

total=0

# Dump namespaced resources across all namespaces
for kind in "${NAMESPACED_KINDS[@]}"; do
    echo "--- $kind ---"
    items=$(kubectl get "$kind" --all-namespaces -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}{"\n"}{end}' 2>/dev/null || true)
    if [ -z "$items" ]; then
        echo "  (none)"
        continue
    fi

    kind_dir="$OUTPUT_DIR/$kind"
    mkdir -p "$kind_dir"
    count=0

    while IFS= read -r item; do
        ns="${item%%/*}"
        name="${item##*/}"
        outfile="$kind_dir/${ns}_${name}.yaml"
        kubectl get "$kind" "$name" -n "$ns" -o yaml > "$outfile" 2>/dev/null && count=$((count + 1))
    done <<< "$items"

    echo "  $count resources"
    total=$((total + count))
done

# Dump cluster-scoped resources
for kind in "${CLUSTER_KINDS[@]}"; do
    echo "--- $kind (cluster-scoped) ---"
    items=$(kubectl get "$kind" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)
    if [ -z "$items" ]; then
        echo "  (none)"
        continue
    fi

    kind_dir="$OUTPUT_DIR/$kind"
    mkdir -p "$kind_dir"
    count=0

    while IFS= read -r name; do
        outfile="$kind_dir/${name}.yaml"
        kubectl get "$kind" "$name" -o yaml > "$outfile" 2>/dev/null && count=$((count + 1))
    done <<< "$items"

    echo "  $count resources"
    total=$((total + count))
done

echo ""
echo "Done. $total resources saved to $OUTPUT_DIR/"
