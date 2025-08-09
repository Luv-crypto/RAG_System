#!/usr/bin/env bash
# gke_shutdown.sh — Safely wind down costs for your GKE Autopilot + Jenkins setup.
# Usage examples:
#   ./gke_shutdown.sh --project peppy-sensor-464315-a8 --region us-central1 --cluster rag-autopilot \
#       --namespace prod --service rag-web-svc --stop-vm --yes
#
# Modes:
#  - default: delete LoadBalancer Service, delete HPA, scale Deployment to 0 (keeps cluster).
#  - --deep: also DELETE the GKE cluster (stops all cluster charges).
#  - --stop-vm: stop this VM at the end (stops compute charges; disk still bills).
#
set -euo pipefail

# Defaults (override via flags)
PROJECT=""
REGION="us-central1"
CLUSTER="rag-autopilot"
NAMESPACE="prod"
SERVICE="rag-web-svc"
MANIFEST_DIR="k8"           # or k8s
VM_NAME="$(hostname)"
ZONE=""
DEEP=0
STOP_VM=0
ASSUME_YES=0

red() { printf "\033[31m%s\033[0m\n" "$*"; }
grn() { printf "\033[32m%s\033[0m\n" "$*"; }
ylw() { printf "\033[33m%s\033[0m\n" "$*"; }
blu() { printf "\033[34m%s\033[0m\n" "$*"; }

usage() {
  cat <<EOF
gke_shutdown.sh
Flags:
  --project <id>          (RECOMMENDED) GCP project id
  --region <region>       default: ${REGION}
  --cluster <name>        default: ${CLUSTER}
  --namespace <ns>        default: ${NAMESPACE}
  --service <name>        default: ${SERVICE}
  --manifest-dir <dir>    default: ${MANIFEST_DIR}
  --zone <zone>           VM zone (needed for --stop-vm)
  --deep                  ALSO delete the GKE cluster
  --stop-vm               Stop this VM at the end (will cut your SSH/terminal)
  --yes                   Do not prompt for confirmations
  -h|--help               This help
EOF
}

confirm() {
  if [[ ${ASSUME_YES} -eq 1 ]]; then return 0; fi
  read -r -p "$1 [y/N]: " ans || true
  [[ "${ans:-}" =~ ^[Yy]$ ]]
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --cluster) CLUSTER="$2"; shift 2;;
    --namespace) NAMESPACE="$2"; shift 2;;
    --service) SERVICE="$2"; shift 2;;
    --manifest-dir) MANIFEST_DIR="$2"; shift 2;;
    --zone) ZONE="$2"; shift 2;;
    --deep) DEEP=1; shift;;
    --stop-vm) STOP_VM=1; shift;;
    --yes) ASSUME_YES=1; shift;;
    -h|--help) usage; exit 0;;
    *) red "Unknown flag: $1"; usage; exit 1;;
  esac
done

# Resolve project if empty
if [[ -z "$PROJECT" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT" ]]; then
  red "Project is required. Pass --project or set gcloud config."
  exit 1
fi

grn "=> Using project: ${PROJECT} | region: ${REGION} | cluster: ${CLUSTER} | ns: ${NAMESPACE}"

# Show current potential cost sources
ylw "-> Current GKE clusters:"
gcloud container clusters list --project "$PROJECT" || true
ylw "-> Current LoadBalancer Services in ${NAMESPACE}:"
kubectl get svc -n "$NAMESPACE" 2>/dev/null | awk '$4!="EXTERNAL-IP"{next} /LoadBalancer/ {print}' || true
ylw "-> External IP addresses (VPC):"
gcloud compute addresses list --project "$PROJECT" || true
ylw "-> Artifact Registry repositories (storage charges if images exist):"
gcloud artifacts repositories list --location="$REGION" --project "$PROJECT" || true

# Kube context (best-effort)
set +e
gcloud container clusters get-credentials "$CLUSTER" --region "$REGION" --project "$PROJECT" >/dev/null 2>&1
set -e

# 1) Delete LoadBalancer Service to release LB & IP
if kubectl get svc "$SERVICE" -n "$NAMESPACE" >/dev/null 2>&1; then
  if confirm "Delete Service ${SERVICE} (releases load balancer & IP)?"; then
    kubectl delete svc "$SERVICE" -n "$NAMESPACE"
    grn "Deleted Service ${SERVICE}."
  fi
else
  ylw "Service ${SERVICE} not found (ok)."
fi

# 2) Disable autoscaling (delete HPA) so it won’t recreate pods
if kubectl get hpa rag-web-hpa -n "$NAMESPACE" >/dev/null 2>&1; then
  if confirm "Delete HPA rag-web-hpa (prevents auto re-scaling)?"; then
    kubectl delete hpa rag-web-hpa -n "$NAMESPACE"
    grn "Deleted HPA rag-web-hpa."
  fi
else
  ylw "HPA rag-web-hpa not found (ok)."
fi

# 3) Scale app to zero replicas to stop pod charges
if kubectl get deploy rag-web -n "$NAMESPACE" >/dev/null 2>&1; then
  if confirm "Scale Deployment rag-web to 0 replicas?"; then
    kubectl scale deploy/rag-web -n "$NAMESPACE" --replicas=0
    grn "Scaled rag-web to 0."
  fi
else
  ylw "Deployment rag-web not found (ok)."
fi

# 4) Deep shutdown: delete the cluster
if [[ ${DEEP} -eq 1 ]]; then
  if confirm "DELETE cluster ${CLUSTER} in ${REGION}? (stops Autopilot cluster fee)"; then
    gcloud container clusters delete "$CLUSTER" --region "$REGION" --project "$PROJECT" --quiet
    grn "Cluster ${CLUSTER} deleted."
  fi
fi

# 5) Stop this VM (optional)
if [[ ${STOP_VM} -eq 1 ]]; then
  if [[ -z "$ZONE" ]]; then
    ZONE="$(gcloud compute instances list --filter="name=${VM_NAME}" --format="value(zone.basename())" --project "$PROJECT")"
  fi
  if [[ -z "$ZONE" ]]; then
    red "Could not determine VM zone; pass --zone ZONE to stop the VM."
    exit 1
  fi
  ylw "About to STOP VM ${VM_NAME} in ${ZONE}. Your session will end."
  if confirm "Proceed to stop VM now?"; then
    gcloud compute instances stop "$VM_NAME" --zone "$ZONE" --project "$PROJECT"
  fi
fi

grn "Shutdown sequence complete. Review above outputs to confirm resources are gone or paused."
