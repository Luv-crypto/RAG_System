#!/usr/bin/env bash
# gke_restart.sh — Bring environment back to the working state quickly.
# It will:
#   - Ensure Artifact Registry exists & docker is configured
#   - Ensure cluster exists (create Autopilot cluster if missing)
#   - Get kube credentials
#   - Recreate namespace, Service, HPA, PDB
#   - Ensure a Deployment exists using the LATEST image from Artifact Registry
#   - Scale up replicas and wait for readiness
#
# Usage example:
#   ./gke_restart.sh --project peppy-sensor-464315-a8 --region us-central1 --cluster rag-autopilot \
#       --namespace prod --service rag-web-svc --repo rag-repo --manifest-dir k8 --replicas 2
#
set -euo pipefail

PROJECT=""
REGION="us-central1"
CLUSTER="rag-autopilot"
NAMESPACE="prod"
SERVICE="rag-web-svc"
REPO="rag-repo"
AR_HOST=""                  # auto-derives from region
MANIFEST_DIR="k8"           # or k8s
REPLICAS=2
IMAGE_NAME="rag-app"        # image name within the repo (rag-repo/rag-app)

red() { printf "\033[31m%s\033[0m\n" "$*"; }
grn() { printf "\033[32m%s\033[0m\n" "$*"; }
ylw() { printf "\033[33m%s\033[0m\n" "$*"; }
blu() { printf "\033[34m%s\033[0m\n" "$*"; }

usage() {
  cat <<EOF
gke_restart.sh
Flags:
  --project <id>          (RECOMMENDED) GCP project id
  --region <region>       default: ${REGION}
  --cluster <name>        default: ${CLUSTER}
  --namespace <ns>        default: ${NAMESPACE}
  --service <name>        default: ${SERVICE}
  --repo <name>           Artifact Registry repo (default: ${REPO})
  --image-name <name>     Image name in AR (default: ${IMAGE_NAME})
  --manifest-dir <dir>    Path to k8 manifests (default: ${MANIFEST_DIR})
  --replicas <n>          Desired replicas after restart (default: ${REPLICAS})
  -h|--help               This help
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --cluster) CLUSTER="$2"; shift 2;;
    --namespace) NAMESPACE="$2"; shift 2;;
    --service) SERVICE="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --image-name) IMAGE_NAME="$2"; shift 2;;
    --manifest-dir) MANIFEST_DIR="$2"; shift 2;;
    --replicas) REPLICAS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) red "Unknown flag: $1"; usage; exit 1;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT" ]]; then
  red "Project is required. Pass --project or set gcloud config."
  exit 1
fi

# Derive AR host
AR_HOST="${REGION}-docker.pkg.dev"

grn "=> Project: ${PROJECT} | Region: ${REGION} | Cluster: ${CLUSTER} | NS: ${NAMESPACE}"
grn "=> AR: ${AR_HOST}/${PROJECT}/${REPO}/${IMAGE_NAME}"

# 0) Ensure AR repo exists and docker is configured
if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  ylw "Artifact Registry repo ${REPO} missing in ${REGION}. Creating..."
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker --location="$REGION" --project "$PROJECT" \
    --description="RAG images"
fi
gcloud auth configure-docker "${AR_HOST}" -q

# 1) Ensure Autopilot cluster exists (create if needed)
if ! gcloud container clusters describe "$CLUSTER" --region "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  ylw "Cluster ${CLUSTER} missing. Creating Autopilot cluster (this can take a few minutes)..."
  gcloud container clusters create-auto "$CLUSTER" --region "$REGION" --project "$PROJECT"
fi

# 2) Get kube credentials
gcloud container clusters get-credentials "$CLUSTER" --region "$REGION" --project "$PROJECT"

# 3) Ensure namespace exists
kubectl get ns "$NAMESPACE" >/dev/null 2>&1 || kubectl create ns "$NAMESPACE"

# 4) Apply base manifests (Service/HPA/PDB) if present
if [[ -d "$MANIFEST_DIR" ]]; then
  for f in service.yaml hpa.yaml pdb.yaml; do
    if [[ -f "${MANIFEST_DIR}/${f}" ]]; then
      kubectl apply -f "${MANIFEST_DIR}/${f}"
    fi
  done
else
  ylw "Manifest directory ${MANIFEST_DIR} not found. Skipping base applies."
fi

# 5) Ensure a Deployment exists pointing to the LATEST AR image
#    We fetch the newest image tag in AR and set it on the deployment.
LATEST_IMG="$(
  gcloud artifacts docker images list "${AR_HOST}/${PROJECT}/${REPO}/${IMAGE_NAME}" \
    --include-tags --project "$PROJECT" \
    --format='value(TAGS,UPDATE_TIME)' 2>/dev/null \
  | awk -F'\t' 'NF>=2 { if ($1!="") print $0 }' \
  | sort -k2 -r \
  | head -n1 \
  | awk '{print $1}'
)"
if [[ -z "$LATEST_IMG" ]]; then
  red "Could not determine latest image tag in AR. Build/push an image first."
  exit 1
fi

FULL_IMAGE="${AR_HOST}/${PROJECT}/${REPO}/${IMAGE_NAME}:${LATEST_IMG}"
grn "=> Using image: ${FULL_IMAGE}"

if kubectl get deploy rag-web -n "$NAMESPACE" >/dev/null 2>&1; then
  kubectl -n "$NAMESPACE" set image deploy/rag-web rag-web="${FULL_IMAGE}"
else
  # If no deployment file available, create a minimal one on the fly
  ylw "Deployment rag-web not found; creating a minimal one."
  cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-web
  namespace: ${NAMESPACE}
spec:
  replicas: ${REPLICAS}
  selector:
    matchLabels: { app: rag-web }
  template:
    metadata:
      labels: { app: rag-web }
    spec:
      containers:
      - name: rag-web
        image: "${FULL_IMAGE}"
        ports: [{ containerPort: 5000 }]
        env:
        - name: FLASK_RUN_HOST
          value: "0.0.0.0"
        - name: FLASK_RUN_PORT
          value: "5000"
        - name: FLASK_DEBUG
          value: "False"
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
EOF
fi

# 6) Scale up and wait for readiness
kubectl -n "$NAMESPACE" scale deploy/rag-web --replicas="${REPLICAS}"
kubectl -n "$NAMESPACE" rollout status deploy/rag-web --timeout=300s || true

# 7) Ensure Service exists (create basic LB service if missing)
if ! kubectl get svc "$SERVICE" -n "$NAMESPACE" >/dev/null 2>&1; then
  ylw "Service ${SERVICE} not found; creating a basic LoadBalancer service."
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE}
  namespace: ${NAMESPACE}
spec:
  type: LoadBalancer
  selector: { app: rag-web }
  ports:
  - name: http
    port: 80
    targetPort: 5000
EOF
fi

# 8) Recreate HPA if it was removed (optional: only if HPA manifest is present)
if [[ -f "${MANIFEST_DIR}/hpa.yaml" ]]; then
  kubectl apply -f "${MANIFEST_DIR}/hpa.yaml"
fi

# 9) Show external IP and HPA status
EXTERNAL_IP="$(kubectl get svc "$SERVICE" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)"
ylw "If EXTERNAL_IP is empty, it may take ~1-3 minutes to provision."
echo "EXTERNAL_IP: ${EXTERNAL_IP:-<pending>}"
kubectl get hpa -n "$NAMESPACE" || true

grn "Restart sequence complete."
