"""YAML-BERT Capability Tests.

Behavioral testing framework inspired by CheckList (Ribeiro et al., 2020).
Tests semantic understanding of Kubernetes YAML structure, not syntax or memorization.

Each capability represents a structural concept the model should understand.
Multiple test cases per capability. Track capability coverage.

Usage:
    python test_capabilities.py output_v1/checkpoints/yaml_bert_epoch_10.pt
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary, UNIVERSAL_ROOT_KEYS
from yaml_bert.types import NodeType


@dataclass
class TestCase:
    name: str
    yaml_text: str
    mask_token: str
    expect_in_top5: list[str] = field(default_factory=list)
    expect_not_top1: list[str] = field(default_factory=list)
    expect_confidence_above: float | None = None
    expect_confidence_below: float | None = None


@dataclass
class Capability:
    name: str
    description: str
    tests: list[TestCase]
    phase: str = "pretrain"  # "pretrain" or "finetune"


@dataclass
class TestResult:
    test_name: str
    passed: bool
    details: str
    predictions: list[tuple[str, float]]


def build_capabilities() -> list[Capability]:
    """Define all capabilities and their test cases."""
    capabilities: list[Capability] = []

    # ==========================================================
    # CAPABILITY 1: Parent-child validity
    # The model should know which keys are valid children of each parent.
    # ==========================================================
    capabilities.append(Capability(
        name="Parent-child validity",
        description="Model predicts keys that are valid children of the parent node",
        tests=[
            TestCase(
                name="spec children in Deployment",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
""",
                mask_token="replicas",
                expect_in_top5=["replicas", "selector", "template", "minReadySeconds", "strategy"],
            ),
            TestCase(
                name="metadata children",
                yaml_text="""\
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
  namespace: default
  labels:
    app: test
data:
  key: value
""",
                mask_token="namespace",
                expect_in_top5=["namespace", "name", "labels", "annotations"],
            ),
            TestCase(
                name="container children",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
    resources:
      limits:
        memory: 128Mi
""",
                mask_token="image",
                expect_in_top5=["image", "name", "command", "env", "resources"],
            ),
            TestCase(
                name="resources children",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    resources:
      limits:
        cpu: 500m
        memory: 128Mi
      requests:
        cpu: 250m
""",
                mask_token="limits",
                expect_in_top5=["limits", "requests"],
            ),
            TestCase(
                name="Service spec children",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
""",
                mask_token="selector",
                expect_in_top5=["selector", "ports", "type", "clusterIP"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 2: Kind conditioning
    # Different resource kinds should produce different predictions
    # for the same structural position.
    # ==========================================================
    capabilities.append(Capability(
        name="Kind conditioning",
        description="The kind value influences what keys appear under spec",
        tests=[
            TestCase(
                name="Deployment spec has replicas",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
""",
                mask_token="replicas",
                expect_in_top5=["replicas"],
            ),
            TestCase(
                name="Service spec has type",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  type: ClusterIP
""",
                mask_token="type",
                expect_in_top5=["type"],
            ),
            TestCase(
                name="ConfigMap has data",
                yaml_text="""\
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
data:
  key: value
""",
                mask_token="data",
                expect_in_top5=["data"],
            ),
            TestCase(
                name="CronJob spec has schedule",
                yaml_text="""\
apiVersion: batch/v1
kind: CronJob
metadata:
  name: job
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: worker
            image: busybox
""",
                mask_token="schedule",
                expect_in_top5=["schedule", "jobTemplate"],
            ),
            TestCase(
                name="ClusterRole has rules",
                yaml_text="""\
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
""",
                mask_token="rules",
                expect_in_top5=["rules"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 3: Depth sensitivity
    # Same structural position at different depths should yield
    # depth-appropriate predictions.
    # ==========================================================
    capabilities.append(Capability(
        name="Depth sensitivity",
        description="Model predictions change appropriately with depth",
        tests=[
            TestCase(
                name="Depth 0: top-level keys only",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: app
""",
                mask_token="kind",
                expect_in_top5=["kind", "apiVersion", "metadata", "spec"],
            ),
            TestCase(
                name="Depth 1 under metadata: metadata children",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: test
  labels:
    app: test
""",
                mask_token="labels",
                expect_in_top5=["labels", "annotations", "namespace", "name"],
                expect_not_top1=["apiVersion", "kind"],
            ),
            TestCase(
                name="Deep nesting: port fields",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
""",
                mask_token="targetPort",
                expect_in_top5=["targetPort", "port", "protocol", "name", "nodePort"],
                expect_not_top1=["apiVersion", "kind", "metadata"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 4: Sibling awareness
    # Model knows which keys commonly co-occur as siblings.
    # ==========================================================
    capabilities.append(Capability(
        name="Sibling awareness",
        description="Model predicts keys that are valid siblings of existing keys",
        tests=[
            TestCase(
                name="requests siblings with limits",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    resources:
      requests:
        memory: 64Mi
      limits:
        memory: 128Mi
""",
                mask_token="requests",
                expect_in_top5=["requests", "limits"],
            ),
            TestCase(
                name="readinessProbe siblings with livenessProbe",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
""",
                mask_token="readinessProbe",
                expect_in_top5=["readinessProbe", "livenessProbe", "startupProbe"],
            ),
            TestCase(
                name="matchLabels under selector",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  selector:
    matchLabels:
      app: web
    matchExpressions:
    - key: tier
      operator: In
      values: [frontend]
""",
                mask_token="matchLabels",
                expect_in_top5=["matchLabels", "matchExpressions"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 5: Required fields
    # Model knows which keys are almost always present.
    # ==========================================================
    capabilities.append(Capability(
        name="Required fields",
        description="Model predicts mandatory keys with high confidence",
        tests=[
            TestCase(
                name="apiVersion is required at root",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: test
""",
                mask_token="apiVersion",
                expect_in_top5=["apiVersion"],
                expect_confidence_above=0.90,
            ),
            TestCase(
                name="metadata is required",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: app
""",
                mask_token="metadata",
                expect_in_top5=["metadata"],
                expect_confidence_above=0.50,
            ),
            TestCase(
                name="name is required under metadata",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: test
  namespace: default
""",
                mask_token="name",
                expect_in_top5=["name"],
                expect_confidence_above=0.50,
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 6: Invalid structure rejection
    # Model shows low confidence when keys are in wrong positions.
    # ==========================================================
    capabilities.append(Capability(
        name="Invalid structure rejection",
        description="Model has low confidence when structure is wrong",
        phase="finetune",
        tests=[
            TestCase(
                name="containers under metadata (wrong)",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  containers:
  - name: app
""",
                mask_token="containers",
                expect_not_top1=["containers"],
            ),
            TestCase(
                name="replicas under metadata (wrong)",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  replicas: 3
""",
                mask_token="replicas",
                expect_not_top1=["replicas"],
            ),
            TestCase(
                name="apiVersion nested deep (wrong)",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: app
    apiVersion: wrong
""",
                mask_token="apiVersion",
                expect_not_top1=["apiVersion"],
            ),
            TestCase(
                name="Nonsense structure low confidence",
                yaml_text="""\
apiVersion: v1
kind: Pod
spec:
  metadata:
    containers:
      replicas:
        selector: wrong
""",
                mask_token="containers",
                expect_confidence_below=0.60,
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 7: Cross-kind discrimination
    # Model knows that different resource kinds have fundamentally
    # different structures.
    # ==========================================================
    capabilities.append(Capability(
        name="Cross-kind discrimination",
        description="Different resource types produce structurally appropriate predictions",
        tests=[
            TestCase(
                name="Secret has type field",
                yaml_text="""\
apiVersion: v1
kind: Secret
metadata:
  name: mysecret
type: Opaque
data:
  password: cGFzc3dvcmQ=
""",
                mask_token="type",
                expect_in_top5=["type"],
            ),
            TestCase(
                name="PVC has accessModes",
                yaml_text="""\
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
""",
                mask_token="accessModes",
                expect_in_top5=["accessModes", "resources", "storageClassName"],
            ),
            TestCase(
                name="Ingress has rules",
                yaml_text="""\
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ingress
spec:
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: svc
            port:
              number: 80
""",
                mask_token="rules",
                expect_in_top5=["rules", "tls", "ingressClassName"],
            ),
            TestCase(
                name="NetworkPolicy has podSelector",
                yaml_text="""\
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: policy
spec:
  podSelector:
    matchLabels:
      role: db
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: frontend
""",
                mask_token="podSelector",
                expect_in_top5=["podSelector", "ingress", "egress", "policyTypes"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 8: Value-context sensitivity
    # Even though we only predict keys, the VALUES in context
    # should influence predictions (proving values serve as context).
    # ==========================================================
    capabilities.append(Capability(
        name="Value-context sensitivity",
        description="Unmasked values influence key predictions (values serve as context)",
        tests=[
            TestCase(
                name="ports context: containerPort vs port",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    ports:
    - containerPort: 80
      protocol: TCP
""",
                mask_token="containerPort",
                expect_in_top5=["containerPort"],
                expect_not_top1=["port"],
            ),
            TestCase(
                name="env var structure: name and value",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    env:
    - name: DB_HOST
      value: localhost
""",
                mask_token="value",
                expect_in_top5=["value", "valueFrom"],
            ),
            TestCase(
                name="volume mount has mountPath",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: data
      mountPath: /var/data
      readOnly: true
""",
                mask_token="mountPath",
                expect_in_top5=["mountPath", "name", "readOnly", "subPath"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 9: Multi-container awareness
    # Model understands list semantics — multiple containers
    # have the same structure.
    # ==========================================================
    capabilities.append(Capability(
        name="Multi-container awareness",
        description="Model understands list item structure repeats",
        tests=[
            TestCase(
                name="Second container has same fields as first",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
  - name: sidecar
    image: envoy
""",
                mask_token="image",
                expect_in_top5=["image", "name", "command", "ports"],
            ),
            TestCase(
                name="initContainers have container fields",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  initContainers:
  - name: init
    image: busybox
    command: ["sh", "-c", "echo init"]
  containers:
  - name: app
    image: nginx
""",
                mask_token="command",
                expect_in_top5=["command", "image", "name", "args"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 10: RBAC structure
    # Role/ClusterRole rules have specific structure.
    # ==========================================================
    capabilities.append(Capability(
        name="RBAC structure",
        description="Model understands RBAC-specific key patterns",
        tests=[
            TestCase(
                name="Role rules have apiGroups",
                yaml_text="""\
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: reader
  namespace: default
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
""",
                mask_token="apiGroups",
                expect_in_top5=["apiGroups", "resources", "verbs", "resourceNames"],
            ),
            TestCase(
                name="RoleBinding has roleRef and subjects",
                yaml_text="""\
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: binding
  namespace: default
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: reader
subjects:
- kind: ServiceAccount
  name: default
""",
                mask_token="roleRef",
                expect_in_top5=["roleRef", "subjects"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 11: Volume semantics
    # Model understands the volume/volumeMount relationship
    # and different volume types.
    # ==========================================================
    capabilities.append(Capability(
        name="Volume semantics",
        description="Model understands volume types and mount structure",
        tests=[
            TestCase(
                name="volumes under pod spec",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  volumes:
  - name: data
    emptyDir: {}
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: data
      mountPath: /var/data
""",
                mask_token="volumes",
                expect_in_top5=["volumes", "containers", "serviceAccountName"],
            ),
            TestCase(
                name="emptyDir is a volume type",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  volumes:
  - name: cache
    emptyDir: {}
""",
                mask_token="emptyDir",
                expect_in_top5=["emptyDir", "configMap", "secret", "hostPath", "persistentVolumeClaim"],
            ),
            TestCase(
                name="configMap volume type",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  volumes:
  - name: config
    configMap:
      name: my-config
""",
                mask_token="configMap",
                expect_in_top5=["configMap", "secret", "emptyDir", "persistentVolumeClaim"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 12: StatefulSet structure
    # StatefulSets have unique fields not found in Deployments.
    # ==========================================================
    capabilities.append(Capability(
        name="StatefulSet structure",
        description="Model understands StatefulSet-specific patterns",
        tests=[
            TestCase(
                name="StatefulSet has serviceName",
                yaml_text="""\
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: db
spec:
  serviceName: db-headless
  replicas: 3
  selector:
    matchLabels:
      app: db
  template:
    metadata:
      labels:
        app: db
    spec:
      containers:
      - name: db
        image: postgres:14
""",
                mask_token="serviceName",
                expect_in_top5=["serviceName", "replicas", "selector", "template"],
            ),
            TestCase(
                name="StatefulSet has volumeClaimTemplates",
                yaml_text="""\
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: db
spec:
  serviceName: db
  replicas: 1
  selector:
    matchLabels:
      app: db
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
  template:
    metadata:
      labels:
        app: db
    spec:
      containers:
      - name: db
        image: postgres
""",
                mask_token="volumeClaimTemplates",
                expect_in_top5=["volumeClaimTemplates", "template", "serviceName"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 13: DaemonSet vs Deployment
    # DaemonSet has updateStrategy, not replicas.
    # ==========================================================
    capabilities.append(Capability(
        name="DaemonSet structure",
        description="Model distinguishes DaemonSet from Deployment",
        tests=[
            TestCase(
                name="DaemonSet has updateStrategy",
                yaml_text="""\
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: monitor
spec:
  updateStrategy:
    type: RollingUpdate
  selector:
    matchLabels:
      app: monitor
  template:
    metadata:
      labels:
        app: monitor
    spec:
      containers:
      - name: agent
        image: datadog/agent
""",
                mask_token="updateStrategy",
                expect_in_top5=["updateStrategy", "selector", "template"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 14: Job and CronJob structure
    # Jobs have completions, backoffLimit, etc.
    # ==========================================================
    capabilities.append(Capability(
        name="Job structure",
        description="Model understands Job-specific patterns",
        tests=[
            TestCase(
                name="Job has backoffLimit",
                yaml_text="""\
apiVersion: batch/v1
kind: Job
metadata:
  name: migrate
spec:
  backoffLimit: 4
  template:
    spec:
      containers:
      - name: migrate
        image: app:latest
        command: ["./migrate"]
      restartPolicy: Never
""",
                mask_token="backoffLimit",
                expect_in_top5=["backoffLimit", "template", "completions", "parallelism"],
            ),
            TestCase(
                name="restartPolicy in Job pod",
                yaml_text="""\
apiVersion: batch/v1
kind: Job
metadata:
  name: job
spec:
  template:
    spec:
      containers:
      - name: worker
        image: busybox
      restartPolicy: Never
""",
                mask_token="restartPolicy",
                expect_in_top5=["restartPolicy", "containers"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 15: Probe structure
    # Probes (liveness, readiness) have specific fields.
    # ==========================================================
    capabilities.append(Capability(
        name="Probe structure",
        description="Model understands probe configuration patterns",
        tests=[
            TestCase(
                name="httpGet probe fields",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
""",
                mask_token="httpGet",
                expect_in_top5=["httpGet", "exec", "tcpSocket", "initialDelaySeconds"],
            ),
            TestCase(
                name="probe timing fields",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 10
      timeoutSeconds: 3
""",
                mask_token="periodSeconds",
                expect_in_top5=["periodSeconds", "initialDelaySeconds", "timeoutSeconds", "failureThreshold", "successThreshold"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 16: Security context
    # Pod and container security settings.
    # ==========================================================
    capabilities.append(Capability(
        name="Security context",
        description="Model understands security configuration patterns",
        tests=[
            TestCase(
                name="pod securityContext fields",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  securityContext:
    runAsUser: 1000
    runAsGroup: 3000
    fsGroup: 2000
  containers:
  - name: app
    image: nginx
""",
                mask_token="runAsUser",
                expect_in_top5=["runAsUser", "runAsGroup", "fsGroup", "runAsNonRoot"],
            ),
            TestCase(
                name="container securityContext",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    securityContext:
      readOnlyRootFilesystem: true
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
""",
                mask_token="readOnlyRootFilesystem",
                expect_in_top5=["readOnlyRootFilesystem", "allowPrivilegeEscalation", "capabilities", "runAsUser", "privileged"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 17: Service types and port structure
    # Different service types have different fields.
    # ==========================================================
    capabilities.append(Capability(
        name="Service port structure",
        description="Model understands Service port and type patterns",
        tests=[
            TestCase(
                name="Service port has targetPort",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
""",
                mask_token="targetPort",
                expect_in_top5=["targetPort", "port", "protocol", "name", "nodePort"],
            ),
            TestCase(
                name="NodePort service has nodePort",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  type: NodePort
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 30080
""",
                mask_token="nodePort",
                expect_in_top5=["nodePort", "targetPort", "port", "protocol"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 18: Affinity and scheduling
    # Node affinity, pod affinity, tolerations.
    # ==========================================================
    capabilities.append(Capability(
        name="Scheduling and affinity",
        description="Model understands scheduling constraint patterns",
        tests=[
            TestCase(
                name="tolerations structure",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  tolerations:
  - key: node-role.kubernetes.io/master
    operator: Exists
    effect: NoSchedule
  containers:
  - name: app
    image: nginx
""",
                mask_token="tolerations",
                expect_in_top5=["tolerations", "nodeSelector", "affinity", "containers"],
            ),
            TestCase(
                name="toleration fields",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  tolerations:
  - key: dedicated
    operator: Equal
    value: gpu
    effect: NoSchedule
  containers:
  - name: app
    image: nginx
""",
                mask_token="operator",
                expect_in_top5=["operator", "key", "value", "effect"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 19: HPA structure
    # Horizontal Pod Autoscaler has specific fields.
    # ==========================================================
    capabilities.append(Capability(
        name="HPA structure",
        description="Model understands HPA-specific patterns",
        tests=[
            TestCase(
                name="HPA has scaleTargetRef",
                yaml_text="""\
apiVersion: autoscaling/v1
kind: HorizontalPodAutoscaler
metadata:
  name: hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app
  minReplicas: 1
  maxReplicas: 10
""",
                mask_token="scaleTargetRef",
                expect_in_top5=["scaleTargetRef", "minReplicas", "maxReplicas"],
            ),
            TestCase(
                name="HPA has minReplicas and maxReplicas",
                yaml_text="""\
apiVersion: autoscaling/v1
kind: HorizontalPodAutoscaler
metadata:
  name: hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app
  minReplicas: 2
  maxReplicas: 10
""",
                mask_token="maxReplicas",
                expect_in_top5=["maxReplicas", "minReplicas", "scaleTargetRef"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 20: Annotation patterns
    # Common annotations have specific keys.
    # ==========================================================
    capabilities.append(Capability(
        name="Annotation patterns",
        description="Model understands common annotation structures",
        tests=[
            TestCase(
                name="annotations under metadata",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
  annotations:
    description: "My service"
  labels:
    app: web
""",
                mask_token="annotations",
                expect_in_top5=["annotations", "labels", "name", "namespace"],
            ),
            TestCase(
                name="labels under metadata",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  labels:
    app: web
    version: v1
  annotations:
    description: "My app"
""",
                mask_token="labels",
                expect_in_top5=["labels", "annotations", "name", "namespace"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 21: Kind-specific spec children
    # Different resource kinds have different valid keys under spec.
    # ==========================================================
    capabilities.append(Capability(
        name="Kind-specific spec children",
        description="Different resource kinds have different valid keys under spec",
        tests=[
            TestCase(
                name="Pod spec has containers directly",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
""",
                mask_token="containers",
                expect_in_top5=["containers"],
                expect_not_top1=["template", "replicas"],
            ),
            TestCase(
                name="Deployment spec has template not containers",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: nginx
""",
                mask_token="template",
                expect_in_top5=["template"],
            ),
            TestCase(
                name="DaemonSet spec does not have replicas",
                yaml_text="""\
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: monitor
spec:
  updateStrategy:
    type: RollingUpdate
  selector:
    matchLabels:
      app: monitor
  template:
    spec:
      containers:
      - name: agent
        image: monitor:latest
""",
                mask_token="updateStrategy",
                expect_in_top5=["updateStrategy", "selector", "template"],
                expect_not_top1=["replicas"],
            ),
            TestCase(
                name="CronJob spec has schedule",
                yaml_text="""\
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cron
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: job
            image: busybox
""",
                mask_token="schedule",
                expect_in_top5=["schedule", "jobTemplate"],
            ),
            TestCase(
                name="ConfigMap has data not spec",
                yaml_text="""\
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
data:
  key1: value1
  key2: value2
""",
                mask_token="data",
                expect_in_top5=["data"],
                expect_not_top1=["spec"],
            ),
            TestCase(
                name="Secret has data not spec",
                yaml_text="""\
apiVersion: v1
kind: Secret
metadata:
  name: secret
type: Opaque
data:
  password: cGFzc3dvcmQ=
""",
                mask_token="data",
                expect_in_top5=["data"],
                expect_not_top1=["spec"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 22: Kind-specific invalid structure rejection
    # Model rejects keys that are wrong for the document's kind.
    # ==========================================================
    capabilities.append(Capability(
        name="Kind-specific invalid structure rejection",
        description="Model rejects keys that are wrong for the document's kind",
        phase="finetune",
        tests=[
            TestCase(
                name="replicas in Pod spec (WRONG)",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  replicas: 3
  containers:
  - name: app
    image: nginx
""",
                mask_token="replicas",
                expect_not_top1=["replicas"],
            ),
            TestCase(
                name="containers directly in Deployment spec (WRONG)",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 1
  containers:
  - name: app
    image: nginx
""",
                mask_token="containers",
                expect_not_top1=["containers"],
            ),
            TestCase(
                name="template in Pod spec (WRONG)",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  template:
    spec:
      containers:
      - name: app
        image: nginx
""",
                mask_token="template",
                expect_not_top1=["template"],
            ),
            TestCase(
                name="spec in ConfigMap (WRONG)",
                yaml_text="""\
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
spec:
  key1: value1
""",
                mask_token="spec",
                expect_not_top1=["spec"],
            ),
            TestCase(
                name="spec in Secret (WRONG)",
                yaml_text="""\
apiVersion: v1
kind: Secret
metadata:
  name: secret
type: Opaque
spec:
  password: secret123
""",
                mask_token="spec",
                expect_not_top1=["spec"],
            ),
            TestCase(
                name="schedule in Deployment (WRONG — CronJob field)",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  schedule: "*/5 * * * *"
  replicas: 1
""",
                mask_token="schedule",
                expect_not_top1=["schedule"],
            ),
            TestCase(
                name="serviceName in Deployment (WRONG — StatefulSet field)",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  serviceName: headless
  replicas: 1
""",
                mask_token="serviceName",
                expect_not_top1=["serviceName"],
            ),
            TestCase(
                name="replicas in Service spec (WRONG)",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  replicas: 3
  ports:
  - port: 80
""",
                mask_token="replicas",
                expect_not_top1=["replicas"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 23: Same structure, different kind
    # Identical tree position yields different predictions based on kind.
    # ==========================================================
    capabilities.append(Capability(
        name="Same structure different kind",
        description="Identical tree position yields different predictions based on kind",
        tests=[
            TestCase(
                name="First key under Deployment spec",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
""",
                mask_token="replicas",
                expect_in_top5=["replicas", "selector", "template"],
            ),
            TestCase(
                name="First key under Service spec",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  type: ClusterIP
""",
                mask_token="type",
                expect_in_top5=["type", "ports", "selector"],
            ),
            TestCase(
                name="First key under Pod spec",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
""",
                mask_token="containers",
                expect_in_top5=["containers", "volumes", "nodeSelector"],
            ),
            TestCase(
                name="First key under Job spec",
                yaml_text="""\
apiVersion: batch/v1
kind: Job
metadata:
  name: job
spec:
  template:
    spec:
      containers:
      - name: worker
        image: busybox
      restartPolicy: Never
""",
                mask_token="template",
                expect_in_top5=["template", "backoffLimit", "completions"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 24: Kind embedding does not harm valid structures
    # Adding kind embedding does not reduce accuracy on valid YAMLs.
    # ==========================================================
    capabilities.append(Capability(
        name="Kind embedding preserves valid structures",
        description="Adding kind embedding does not reduce accuracy on valid YAMLs",
        tests=[
            TestCase(
                name="Valid Deployment keys correct",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
""",
                mask_token="replicas",
                expect_in_top5=["replicas"],
                expect_confidence_above=0.50,
            ),
            TestCase(
                name="Valid Service keys correct",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
""",
                mask_token="selector",
                expect_in_top5=["selector"],
                expect_confidence_above=0.50,
            ),
            TestCase(
                name="Valid Pod keys correct",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
""",
                mask_token="containers",
                expect_in_top5=["containers"],
                expect_confidence_above=0.50,
            ),
            TestCase(
                name="Valid ConfigMap keys correct",
                yaml_text="""\
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
data:
  key: value
""",
                mask_token="data",
                expect_in_top5=["data"],
                expect_confidence_above=0.50,
            ),
            TestCase(
                name="Valid StatefulSet keys correct",
                yaml_text="""\
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: db
spec:
  serviceName: db
  replicas: 3
  selector:
    matchLabels:
      app: db
  template:
    spec:
      containers:
      - name: db
        image: postgres
""",
                mask_token="serviceName",
                expect_in_top5=["serviceName"],
                expect_confidence_above=0.30,
            ),
        ],
    ))

    # ----------------------------------------------------------
    # CAPABILITY 1 additions: env var children, label children,
    # volume children
    # ----------------------------------------------------------
    capabilities[0].tests.extend([
        TestCase(
            name="env var children (name, value)",
            yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    env:
    - name: DB_HOST
      value: localhost
    - name: DB_PORT
      valueFrom:
        configMapKeyRef:
          name: db-config
          key: port
""",
            mask_token="valueFrom",
            expect_in_top5=["valueFrom", "value", "name"],
        ),
        TestCase(
            name="label children (arbitrary label keys under labels)",
            yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  labels:
    app: web
    version: v1
    component: frontend
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: app
        image: nginx
""",
            mask_token="version",
            expect_in_top5=["version", "app", "component"],
        ),
        TestCase(
            name="volume children (name and volume type)",
            yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  volumes:
  - name: config-vol
    configMap:
      name: my-config
  - name: secret-vol
    secret:
      secretName: my-secret
  containers:
  - name: app
    image: nginx
""",
            mask_token="secret",
            expect_in_top5=["secret", "configMap", "emptyDir", "name", "persistentVolumeClaim"],
        ),
    ])

    # ----------------------------------------------------------
    # CAPABILITY 2 additions: StatefulSet serviceName,
    # DaemonSet updateStrategy, Namespace (no spec)
    # ----------------------------------------------------------
    capabilities[1].tests.extend([
        TestCase(
            name="StatefulSet spec has serviceName",
            yaml_text="""\
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: redis-headless
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7
""",
            mask_token="serviceName",
            expect_in_top5=["serviceName", "replicas", "selector", "template"],
        ),
        TestCase(
            name="DaemonSet spec has updateStrategy",
            yaml_text="""\
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      containers:
      - name: fluentd
        image: fluentd:v1
""",
            mask_token="updateStrategy",
            expect_in_top5=["updateStrategy", "selector", "template"],
        ),
        TestCase(
            name="Namespace has metadata but no spec",
            yaml_text="""\
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    env: prod
""",
            mask_token="name",
            expect_in_top5=["name", "labels", "annotations", "namespace"],
            expect_not_top1=["spec"],
        ),
    ])

    # ----------------------------------------------------------
    # CAPABILITY 6 additions: ports under metadata (wrong),
    # selector under containers (wrong)
    # ----------------------------------------------------------
    capabilities[5].tests.extend([
        TestCase(
            name="ports under metadata (wrong)",
            yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
  ports:
  - containerPort: 80
spec:
  containers:
  - name: app
    image: nginx
""",
            mask_token="ports",
            expect_not_top1=["ports"],
        ),
        TestCase(
            name="selector under containers list (wrong)",
            yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: app
        image: nginx
        selector:
          app: web
""",
            mask_token="selector",
            expect_not_top1=["selector"],
        ),
    ])

    # ----------------------------------------------------------
    # CAPABILITY 7 additions: PodDisruptionBudget, StorageClass,
    # ResourceQuota
    # ----------------------------------------------------------
    capabilities[6].tests.extend([
        TestCase(
            name="PodDisruptionBudget has minAvailable",
            yaml_text="""\
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: web
""",
            mask_token="minAvailable",
            expect_in_top5=["minAvailable", "maxUnavailable", "selector"],
        ),
        TestCase(
            name="StorageClass has provisioner",
            yaml_text="""\
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp2
reclaimPolicy: Retain
""",
            mask_token="provisioner",
            expect_in_top5=["provisioner", "parameters", "reclaimPolicy"],
        ),
        TestCase(
            name="ResourceQuota has hard",
            yaml_text="""\
apiVersion: v1
kind: ResourceQuota
metadata:
  name: quota
  namespace: default
spec:
  hard:
    pods: "10"
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
""",
            mask_token="hard",
            expect_in_top5=["hard"],
        ),
    ])

    # ----------------------------------------------------------
    # CAPABILITY 22 additions: extensive kind-specific invalid tests
    # ----------------------------------------------------------
    capabilities[21].tests.extend([
        # Pod should NOT have these (they belong to controllers)
        TestCase(
            name="selector in Pod spec (WRONG — controller field)",
            yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  selector:
    matchLabels:
      app: test
  containers:
  - name: app
    image: nginx
""",
            mask_token="selector",
            expect_not_top1=["selector"],
        ),
        TestCase(
            name="strategy in Pod spec (WRONG — Deployment field)",
            yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  strategy:
    type: RollingUpdate
  containers:
  - name: app
    image: nginx
""",
            mask_token="strategy",
            expect_not_top1=["strategy"],
        ),
        TestCase(
            name="jobTemplate in Pod spec (WRONG — CronJob field)",
            yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: app
            image: nginx
  containers:
  - name: app
    image: nginx
""",
            mask_token="jobTemplate",
            expect_not_top1=["jobTemplate"],
        ),
        # Service should NOT have these
        TestCase(
            name="containers in Service spec (WRONG)",
            yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  containers:
  - name: app
    image: nginx
  ports:
  - port: 80
""",
            mask_token="containers",
            expect_not_top1=["containers"],
        ),
        TestCase(
            name="template in Service spec (WRONG)",
            yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  template:
    spec:
      containers:
      - name: app
  ports:
  - port: 80
""",
            mask_token="template",
            expect_not_top1=["template"],
        ),
        # Job should NOT have these
        TestCase(
            name="replicas in Job spec (WRONG)",
            yaml_text="""\
apiVersion: batch/v1
kind: Job
metadata:
  name: job
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: worker
        image: busybox
      restartPolicy: Never
""",
            mask_token="replicas",
            expect_not_top1=["replicas"],
        ),
        TestCase(
            name="strategy in Job spec (WRONG — Deployment field)",
            yaml_text="""\
apiVersion: batch/v1
kind: Job
metadata:
  name: job
spec:
  strategy:
    type: RollingUpdate
  template:
    spec:
      containers:
      - name: worker
        image: busybox
      restartPolicy: Never
""",
            mask_token="strategy",
            expect_not_top1=["strategy"],
        ),
        # DaemonSet should NOT have replicas
        TestCase(
            name="replicas in DaemonSet spec (WRONG — DaemonSets run on all nodes)",
            yaml_text="""\
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: ds
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ds
  template:
    spec:
      containers:
      - name: agent
        image: monitor
""",
            mask_token="replicas",
            expect_not_top1=["replicas"],
        ),
        # Namespace should NOT have spec
        TestCase(
            name="spec in Namespace (WRONG — Namespaces have no spec)",
            yaml_text="""\
apiVersion: v1
kind: Namespace
metadata:
  name: test
spec:
  replicas: 1
""",
            mask_token="spec",
            expect_not_top1=["spec"],
        ),
        # CronJob spec should NOT have these
        TestCase(
            name="replicas in CronJob spec (WRONG)",
            yaml_text="""\
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cron
spec:
  replicas: 3
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: job
            image: busybox
""",
            mask_token="replicas",
            expect_not_top1=["replicas"],
        ),
        TestCase(
            name="containers in CronJob spec (WRONG — needs jobTemplate.spec.template.spec.containers)",
            yaml_text="""\
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cron
spec:
  schedule: "*/5 * * * *"
  containers:
  - name: job
    image: busybox
""",
            mask_token="containers",
            expect_not_top1=["containers"],
        ),
        # Original 3 additions below
        TestCase(
            name="data in Deployment (wrong - Deployment uses spec.template)",
            yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
data:
  key: value
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: app
        image: nginx
""",
            mask_token="data",
            expect_not_top1=["data"],
        ),
        TestCase(
            name="ports in ConfigMap (wrong)",
            yaml_text="""\
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
ports:
- containerPort: 80
data:
  key: value
""",
            mask_token="ports",
            expect_not_top1=["ports"],
        ),
        TestCase(
            name="volumeClaimTemplates in Deployment (wrong - StatefulSet field)",
            yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 1
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
  template:
    spec:
      containers:
      - name: app
        image: nginx
""",
            mask_token="volumeClaimTemplates",
            expect_not_top1=["volumeClaimTemplates"],
        ),
    ])

    # ==========================================================
    # CAPABILITY 25: Workload controller distinction
    # Deployment, StatefulSet, DaemonSet, Job have distinct spec structures.
    # ==========================================================
    capabilities.append(Capability(
        name="Workload controller distinction",
        description="Deployment, StatefulSet, DaemonSet, Job have distinct spec fields",
        tests=[
            TestCase(
                name="Deployment spec: replicas, selector, template, strategy",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  strategy:
    type: RollingUpdate
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: web
        image: nginx
""",
                mask_token="strategy",
                expect_in_top5=["strategy", "replicas", "selector", "template"],
            ),
            TestCase(
                name="StatefulSet spec: serviceName, replicas, volumeClaimTemplates",
                yaml_text="""\
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: db
spec:
  serviceName: db
  replicas: 3
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 5Gi
  selector:
    matchLabels:
      app: db
  template:
    metadata:
      labels:
        app: db
    spec:
      containers:
      - name: db
        image: postgres
""",
                mask_token="volumeClaimTemplates",
                expect_in_top5=["volumeClaimTemplates", "serviceName", "template"],
            ),
            TestCase(
                name="DaemonSet spec: updateStrategy, selector, template (NOT replicas)",
                yaml_text="""\
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: log-collector
spec:
  updateStrategy:
    type: RollingUpdate
  selector:
    matchLabels:
      app: log-collector
  template:
    metadata:
      labels:
        app: log-collector
    spec:
      containers:
      - name: agent
        image: log-agent:1.0
""",
                mask_token="updateStrategy",
                expect_in_top5=["updateStrategy", "selector", "template"],
                expect_not_top1=["replicas"],
            ),
            TestCase(
                name="Job spec: template, backoffLimit, completions (NOT replicas)",
                yaml_text="""\
apiVersion: batch/v1
kind: Job
metadata:
  name: data-process
spec:
  completions: 5
  parallelism: 2
  backoffLimit: 3
  template:
    spec:
      containers:
      - name: processor
        image: processor:latest
      restartPolicy: Never
""",
                mask_token="completions",
                expect_in_top5=["completions", "parallelism", "backoffLimit", "template"],
                expect_not_top1=["replicas"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 26: ConfigMap vs Secret
    # Both have data at root level but different associated fields.
    # ==========================================================
    capabilities.append(Capability(
        name="ConfigMap vs Secret",
        description="ConfigMap and Secret share data but have different associated fields",
        tests=[
            TestCase(
                name="ConfigMap has binaryData",
                yaml_text="""\
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
data:
  app.properties: |
    host=localhost
    port=8080
binaryData:
  logo.png: iVBORw0KGgo=
""",
                mask_token="binaryData",
                expect_in_top5=["binaryData", "data"],
            ),
            TestCase(
                name="Secret has stringData",
                yaml_text="""\
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
data:
  password: cGFzc3dvcmQ=
stringData:
  api-key: my-plaintext-key
""",
                mask_token="stringData",
                expect_in_top5=["stringData", "data", "type"],
            ),
            TestCase(
                name="Secret has type field at root",
                yaml_text="""\
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTi==
  tls.key: LS0tLS1CRUdJTi==
""",
                mask_token="type",
                expect_in_top5=["type", "data", "stringData"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 27: Container field ordering
    # Common container fields should be predicted correctly.
    # ==========================================================
    capabilities.append(Capability(
        name="Container field completeness",
        description="Model predicts all common container fields correctly",
        tests=[
            TestCase(
                name="Container args field",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: app:latest
    command: ["/bin/sh"]
    args:
    - "-c"
    - "echo hello && sleep 3600"
""",
                mask_token="args",
                expect_in_top5=["args", "command", "env", "ports", "name"],
            ),
            TestCase(
                name="Container resources field",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 500m
        memory: 256Mi
""",
                mask_token="resources",
                expect_in_top5=["resources", "ports", "image", "name", "env"],
            ),
            TestCase(
                name="Container securityContext field",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
    securityContext:
      runAsNonRoot: true
      readOnlyRootFilesystem: true
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
""",
                mask_token="securityContext",
                expect_in_top5=["securityContext", "livenessProbe", "readinessProbe", "image"],
            ),
            TestCase(
                name="Container volumeMounts field",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  volumes:
  - name: data
    emptyDir: {}
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: data
      mountPath: /var/data
    env:
    - name: ENV
      value: prod
""",
                mask_token="volumeMounts",
                expect_in_top5=["volumeMounts", "env", "image", "name"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 28: Ingress structure
    # Ingress has specific nested structure.
    # ==========================================================
    capabilities.append(Capability(
        name="Ingress structure",
        description="Model understands Ingress-specific nested structure",
        tests=[
            TestCase(
                name="Ingress spec.rules[].host",
                yaml_text="""\
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
spec:
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-svc
            port:
              number: 80
""",
                mask_token="host",
                expect_in_top5=["host", "http"],
            ),
            TestCase(
                name="Ingress spec.rules[].http.paths[].path",
                yaml_text="""\
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
spec:
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-svc
            port:
              number: 8080
""",
                mask_token="path",
                expect_in_top5=["path", "pathType", "backend"],
            ),
            TestCase(
                name="Ingress backend service name",
                yaml_text="""\
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ingress
spec:
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
  tls:
  - hosts:
    - example.com
    secretName: tls-cert
""",
                mask_token="tls",
                expect_in_top5=["tls", "rules", "ingressClassName"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 29: PV and PVC structure
    # PersistentVolume and PersistentVolumeClaim have distinct fields.
    # ==========================================================
    capabilities.append(Capability(
        name="PV and PVC structure",
        description="Model understands PersistentVolume and PersistentVolumeClaim fields",
        tests=[
            TestCase(
                name="PV has capacity and accessModes",
                yaml_text="""\
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-data
spec:
  capacity:
    storage: 50Gi
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  hostPath:
    path: /mnt/data
""",
                mask_token="capacity",
                expect_in_top5=["capacity", "accessModes", "persistentVolumeReclaimPolicy"],
            ),
            TestCase(
                name="PV has persistentVolumeReclaimPolicy",
                yaml_text="""\
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs
spec:
  capacity:
    storage: 100Gi
  accessModes:
  - ReadWriteMany
  persistentVolumeReclaimPolicy: Recycle
  nfs:
    path: /exports
    server: nfs-server.example.com
""",
                mask_token="persistentVolumeReclaimPolicy",
                expect_in_top5=["persistentVolumeReclaimPolicy", "capacity", "accessModes", "nfs"],
            ),
            TestCase(
                name="PVC has resources.requests.storage",
                yaml_text="""\
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: fast
  resources:
    requests:
      storage: 20Gi
""",
                mask_token="storageClassName",
                expect_in_top5=["storageClassName", "accessModes", "resources"],
            ),
        ],
    ))

    # ==========================================================
    # CAPABILITY 30: Label and annotation structure
    # Labels and annotations appear under metadata in most resources.
    # ==========================================================
    capabilities.append(Capability(
        name="Label and annotation structure",
        description="Labels and annotations appear under metadata, labels have common keys",
        tests=[
            TestCase(
                name="labels is a sibling of annotations",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  labels:
    app: web
    version: v2
  annotations:
    deployment.kubernetes.io/revision: "3"
    description: "production web app"
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: app
        image: nginx
""",
                mask_token="annotations",
                expect_in_top5=["annotations", "labels", "name", "namespace"],
            ),
            TestCase(
                name="Common label keys: app, version, component",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: pod
  labels:
    app: frontend
    version: v1
    component: web
spec:
  containers:
  - name: app
    image: nginx
""",
                mask_token="component",
                expect_in_top5=["component", "app", "version"],
            ),
        ],
    ))

    return capabilities


def _extract_key_from_target(target: str) -> str:
    """Extract the raw key name from a compound target.

    'spec::replicas' -> 'replicas'
    'Deployment::spec::replicas' -> 'replicas'
    'apiVersion' -> 'apiVersion'
    """
    return target.rsplit("::", 1)[-1]


def run_test(
    model: YamlBertModel,
    vocab: Vocabulary,
    test: TestCase,
) -> TestResult:
    """Run a single test case and return result."""
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    nodes = linearizer.linearize(test.yaml_text)
    if not nodes:
        return TestResult(test.name, False, "Failed to parse YAML", [])

    annotator.annotate(nodes)

    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    token_ids, node_types, depths, siblings = [], [], [], []
    mask_pos: int = -1

    for i, node in enumerate(nodes):
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(type_map[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))

        if node.token == test.mask_token and mask_pos == -1:
            mask_pos = i

    if mask_pos == -1:
        return TestResult(test.name, False, f"Token '{test.mask_token}' not found", [])

    # Determine which head to use based on masked node's position
    masked_node = nodes[mask_pos]
    parent_key: str = Vocabulary.extract_parent_key(masked_node.parent_path)
    use_kind_head: bool = (
        masked_node.depth == 1
        and parent_key not in UNIVERSAL_ROOT_KEYS
        and parent_key != ""
    )

    token_ids[mask_pos] = vocab.special_tokens["[MASK]"]

    t = lambda x: torch.tensor([x])
    with torch.no_grad():
        simple_logits, kind_logits = model(t(token_ids), t(node_types), t(depths), t(siblings))

    # Build reverse vocab for decoding
    if use_kind_head:
        logits = kind_logits
        id_to_target = {v: k for k, v in vocab.kind_target_vocab.items()}
    else:
        logits = simple_logits
        id_to_target = {v: k for k, v in vocab.simple_target_vocab.items()}

    # Add special tokens to reverse map
    for tok, tok_id in vocab.special_tokens.items():
        id_to_target[tok_id] = tok

    probs = F.softmax(logits[0, mask_pos], dim=-1)
    topk = probs.topk(10)
    predictions: list[tuple[str, float]] = []
    for i in range(10):
        idx = topk.indices[i].item()
        target_str = id_to_target.get(idx, f"[ID:{idx}]")
        key_name = _extract_key_from_target(target_str)
        predictions.append((key_name, topk.values[i].item()))

    return _check_assertions(test, predictions)


def _check_assertions(
    test: TestCase,
    predictions: list[tuple[str, float]],
) -> TestResult:
    """Check test assertions against predictions."""
    top5_keys: list[str] = [k for k, _ in predictions[:5]]
    top1_key: str = predictions[0][0]
    top1_conf: float = predictions[0][1]
    passed: bool = True
    details: list[str] = []

    if test.expect_in_top5:
        for expected in test.expect_in_top5:
            if expected in top5_keys:
                details.append(f"OK: '{expected}' in top 5")
                break
        else:
            passed = False
            details.append(f"FAIL: none of {test.expect_in_top5} in top 5 {top5_keys}")

    if test.expect_not_top1:
        for bad in test.expect_not_top1:
            if top1_key == bad:
                passed = False
                details.append(f"FAIL: '{bad}' is top prediction (should not be)")

    if test.expect_confidence_above is not None:
        if top1_conf >= test.expect_confidence_above:
            details.append(f"OK: confidence {top1_conf:.2%} >= {test.expect_confidence_above:.0%}")
        else:
            passed = False
            details.append(f"FAIL: confidence {top1_conf:.2%} < {test.expect_confidence_above:.0%}")

    if test.expect_confidence_below is not None:
        if top1_conf <= test.expect_confidence_below:
            details.append(f"OK: confidence {top1_conf:.2%} <= {test.expect_confidence_below:.0%}")
        else:
            passed = False
            details.append(f"FAIL: confidence {top1_conf:.2%} > {test.expect_confidence_below:.0%}")

    return TestResult(test.name, passed, "; ".join(details), predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v1/vocab.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    torch.manual_seed(42)
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size, value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb, simple_vocab_size=vocab.simple_target_vocab_size, kind_vocab_size=vocab.kind_target_vocab_size)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"YAML-BERT Capability Tests — Epoch {checkpoint['epoch']}")
    print(f"{'=' * 70}\n")

    capabilities: list[Capability] = build_capabilities()
    total_tests: int = 0
    total_passed: int = 0
    cap_results: list[tuple[str, int, int]] = []

    for cap in capabilities:
        cap_passed: int = 0
        cap_total: int = len(cap.tests)

        print(f"CAPABILITY: {cap.name}")
        print(f"  {cap.description}")

        for test in cap.tests:
            result: TestResult = run_test(model, vocab, test)
            total_tests += 1
            if result.passed:
                total_passed += 1
                cap_passed += 1
                status: str = "PASS"
            else:
                status = "FAIL"

            print(f"  [{status}] {result.test_name}")
            if args.verbose or not result.passed:
                print(f"         {result.details}")
                for i, (key, prob) in enumerate(result.predictions[:5]):
                    print(f"           {i+1}. '{key}' ({prob:.2%})")

        pct: float = cap_passed / cap_total * 100 if cap_total > 0 else 0
        print(f"  Result: {cap_passed}/{cap_total} ({pct:.0f}%)\n")
        cap_results.append((cap.name, cap.phase, cap_passed, cap_total))

    # Summary
    print(f"{'=' * 70}")
    print(f"CAPABILITY COVERAGE SUMMARY")
    print(f"{'=' * 70}")

    # Pre-training capabilities
    pretrain_caps = [(n, p, t) for n, ph, p, t in cap_results if ph == "pretrain"]
    finetune_caps = [(n, p, t) for n, ph, p, t in cap_results if ph == "finetune"]

    pretrain_fully_passed: int = 0
    pretrain_total_passed: int = 0
    pretrain_total_tests: int = 0
    print("\n  Pre-training capabilities:")
    for name, passed, total in pretrain_caps:
        pct = passed / total * 100 if total > 0 else 0
        status = "PASS" if passed == total else "PARTIAL" if passed > 0 else "FAIL"
        print(f"    [{status:>7}] {name}: {passed}/{total} ({pct:.0f}%)")
        if passed == total:
            pretrain_fully_passed += 1
        pretrain_total_passed += passed
        pretrain_total_tests += total

    if finetune_caps:
        finetune_fully_passed: int = 0
        finetune_total_passed: int = 0
        finetune_total_tests: int = 0
        print("\n  Fine-tuning capabilities (requires fine-tuned model):")
        for name, passed, total in finetune_caps:
            pct = passed / total * 100 if total > 0 else 0
            status = "PASS" if passed == total else "PARTIAL" if passed > 0 else "FAIL"
            print(f"    [{status:>7}] {name}: {passed}/{total} ({pct:.0f}%)")
            if passed == total:
                finetune_fully_passed += 1
            finetune_total_passed += passed
            finetune_total_tests += total

    print(f"\nPre-training: {pretrain_fully_passed}/{len(pretrain_caps)} capabilities, {pretrain_total_passed}/{pretrain_total_tests} tests")
    if finetune_caps:
        print(f"Fine-tuning:  {finetune_fully_passed}/{len(finetune_caps)} capabilities, {finetune_total_passed}/{finetune_total_tests} tests")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
