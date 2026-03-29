"""YAML-BERT Capability Tests.

Behavioral testing framework inspired by CheckList (Ribeiro et al., 2020).
Tests semantic understanding of Kubernetes YAML structure, not syntax or memorization.

Each capability represents a structural concept the model should understand.
Multiple test cases per capability. Track capability coverage.

Usage:
    python test_capabilities.py output_hf/checkpoints/yaml_bert_epoch_10.pt
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary
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

    return capabilities


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
    token_ids, node_types, depths, siblings, parent_keys = [], [], [], [], []
    mask_pos: int = -1

    for i, node in enumerate(nodes):
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(type_map[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))
        parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

        if node.token == test.mask_token and mask_pos == -1:
            mask_pos = i

    if mask_pos == -1:
        return TestResult(test.name, False, f"Token '{test.mask_token}' not found", [])

    token_ids[mask_pos] = vocab.special_tokens["[MASK]"]

    t = lambda x: torch.tensor([x])
    with torch.no_grad():
        logits = model(t(token_ids), t(node_types), t(depths), t(siblings), t(parent_keys))

    probs = F.softmax(logits[0, mask_pos], dim=-1)
    topk = probs.topk(10)
    predictions: list[tuple[str, float]] = [
        (vocab.decode_key(topk.indices[i].item()), topk.values[i].item())
        for i in range(10)
    ]

    # Check assertions
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
    parser.add_argument("--vocab", type=str, default="output_hf/vocab.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size, value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb, key_vocab_size=vocab.key_vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
        cap_results.append((cap.name, cap_passed, cap_total))

    # Summary
    print(f"{'=' * 70}")
    print(f"CAPABILITY COVERAGE SUMMARY")
    print(f"{'=' * 70}")
    caps_fully_passed: int = 0
    for name, passed, total in cap_results:
        pct = passed / total * 100 if total > 0 else 0
        status = "PASS" if passed == total else "PARTIAL" if passed > 0 else "FAIL"
        print(f"  [{status:>7}] {name}: {passed}/{total} ({pct:.0f}%)")
        if passed == total:
            caps_fully_passed += 1

    print(f"\nCapabilities: {caps_fully_passed}/{len(capabilities)} fully passing")
    print(f"Test cases: {total_passed}/{total_tests} passing")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
