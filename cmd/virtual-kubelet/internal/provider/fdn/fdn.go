package fdn

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"strings"
	"time"

	"github.com/Function-Delivery-Network/virtual-kubelet/cmd/virtual-kubelet/internal/provider/fdn/openfaas"
	"github.com/Function-Delivery-Network/virtual-kubelet/cmd/virtual-kubelet/internal/provider/fdn/openwhisk"
	"github.com/Function-Delivery-Network/virtual-kubelet/cmd/virtual-kubelet/internal/provider/fdn/gcf"
	"github.com/Function-Delivery-Network/virtual-kubelet/errdefs"
	"github.com/Function-Delivery-Network/virtual-kubelet/log"
	"github.com/Function-Delivery-Network/virtual-kubelet/node/api"
	stats "github.com/Function-Delivery-Network/virtual-kubelet/node/api/statsv1alpha1"
	"github.com/Function-Delivery-Network/virtual-kubelet/trace"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// openfaas provider constants
const (
	functionNamePrefix = "fdn-virtual-kubelet"
	// Provider configuration defaults.
	defaultCPUCapacity    = "20"
	defaultMemoryCapacity = "100Gi"

	defaultPodCapacity = "20"
	minMemoryCapacity  = "128"
	maxMemoryCapacity  = "1024"
	minPodCapacity     = "1"
	defaultTimeout     = "60000"

	// Values used in tracing as attribute keys.
	namespaceKey     = "namespace"
	nameKey          = "name"
	containerNameKey = "containerName"
)

// MockProvider implements the virtual-kubelet provider interface and stores pods in memory.
type FDNProvider struct { // nolint:golint
	nodeName                             string
	operatingSystem                      string
	internalIP                           string
	daemonEndpointPort                   int32
	serverlessPlatformName               string
	serverlessPlatformApiHost            string
	serverlessPlatformAuth               string
	serverlessPlatformConfigBucket       string
	serverlessPlatformConfigBucketObject string
	minioEndpoint                        string
	minioAccessKeyID                     string
	minioSecretAccessKey                 string
	pods                                 map[string]*corev1.Pod
	config                               FDNConfig
	startTime                            time.Time
	notifier                             func(*corev1.Pod)
}

var (
	errNotImplemented = fmt.Errorf("not implemented by FDN provider")
)

// MockConfig contains a mock virtual-kubelet's configurable parameters.
type FDNConfig struct { // nolint:golint
	CPU    string `json:"cpu,omitempty"`
	Memory string `json:"memory,omitempty"`
	Pods   string `json:"pods,omitempty"`
}

// NewFDNProviderConfig creates a new MockV0Provider. Mock legacy provider does not implement the new asynchronous podnotifier interface
func NewFDNProviderConfig(config FDNConfig, nodeName, operatingSystem string, internalIP string, daemonEndpointPort int32,
	serverlessPlatformName string, serverlessPlatformApiHost string, serverlessPlatformAuth string, serverlessPlatformConfigBucket string, serverlessPlatformConfigBucketObject string, minioEndpoint string,
	minioAccessKeyID string, minioSecretAccessKey string) (*FDNProvider, error) {
	// set defaults
	if config.CPU == "" {
		config.CPU = defaultCPUCapacity
	}
	if config.Memory == "" {
		config.Memory = defaultMemoryCapacity
	}
	if config.Pods == "" {
		config.Pods = defaultPodCapacity
	}
	provider := FDNProvider{
		nodeName:                             nodeName,
		operatingSystem:                      operatingSystem,
		internalIP:                           internalIP,
		daemonEndpointPort:                   daemonEndpointPort,
		serverlessPlatformName:               serverlessPlatformName,
		serverlessPlatformApiHost:            serverlessPlatformApiHost,
		serverlessPlatformAuth:               serverlessPlatformAuth,
		serverlessPlatformConfigBucket:       serverlessPlatformConfigBucket,
		serverlessPlatformConfigBucketObject: serverlessPlatformConfigBucketObject,
		minioEndpoint:                        minioEndpoint,
		minioAccessKeyID:                     minioAccessKeyID,
		minioSecretAccessKey:                 minioSecretAccessKey,
		pods:                                 make(map[string]*corev1.Pod),
		config:                               config,
		startTime:                            time.Now(),
	}

	return &provider, nil
}

// NewMockProvider creates a new MockProvider, which implements the PodNotifier interface
func NewFDNProvider(providerConfig, nodeName, operatingSystem string, internalIP string, daemonEndpointPort int32,
	serverlessPlatformName string, serverlessPlatformApiHost string, serverlessPlatformAuth string,  serverlessPlatformConfigBucket string, serverlessPlatformConfigBucketObject string, minioEndpoint string,
	minioAccessKeyID string, minioSecretAccessKey string) (*FDNProvider, error) {
	config, err := loadConfig(providerConfig, nodeName)
	if err != nil {
		return nil, err
	}

	return NewFDNProviderConfig(config, nodeName, operatingSystem, internalIP, daemonEndpointPort,
		serverlessPlatformName, serverlessPlatformApiHost, serverlessPlatformAuth, serverlessPlatformConfigBucket, serverlessPlatformConfigBucketObject, minioEndpoint,
		minioAccessKeyID, minioSecretAccessKey)
}

// loadConfig loads the given json configuration files.
func loadConfig(providerConfig, nodeName string) (config FDNConfig, err error) {
	data, err := ioutil.ReadFile(providerConfig)
	if err != nil {
		return config, err
	}
	configMap := map[string]FDNConfig{}
	err = json.Unmarshal(data, &configMap)
	if err != nil {
		return config, err
	}
	if _, exist := configMap[nodeName]; exist {
		config = configMap[nodeName]
		if config.CPU == "" {
			config.CPU = defaultCPUCapacity
		}
		if config.Memory == "" {
			config.Memory = defaultMemoryCapacity
		}
		if config.Pods == "" {
			config.Pods = defaultPodCapacity
		}
	}

	if _, err = resource.ParseQuantity(config.CPU); err != nil {
		return config, fmt.Errorf("Invalid CPU value %v", config.CPU)
	}
	if _, err = resource.ParseQuantity(config.Memory); err != nil {
		return config, fmt.Errorf("Invalid memory value %v", config.Memory)
	}
	if _, err = resource.ParseQuantity(config.Pods); err != nil {
		return config, fmt.Errorf("Invalid pods value %v", config.Pods)
	}
	return config, nil
}

// CreatePod takes a Kubernetes Pod and deploys it within the Fargate provider.
func (p *FDNProvider) CreatePod(ctx context.Context, pod *corev1.Pod) (err error) {

	ctx, span := trace.StartSpan(ctx, "CreatePod")
	defer span.End()

	// Add the pod's coordinates to the current span.
	ctx = addAttributes(ctx, span, namespaceKey, pod.Namespace, nameKey, pod.Name)

	log.G(ctx).Infof("receive CreatePod %q", pod.Name)

	log.G(ctx).Infof("Received CreatePod request for %+v.\n", pod)

	// Ignore daemonSet Pod
	if pod != nil && pod.OwnerReferences != nil && len(pod.OwnerReferences) != 0 && pod.OwnerReferences[0].Kind == "DaemonSet" {
		log.G(ctx).Infof("Skip to create DaemonSet pod %q\n", pod.Name)
		return nil
	}

	key, err := buildKey(pod)
	if err != nil {
		return err
	}

	now := metav1.NewTime(time.Now())
	pod.Status = corev1.PodStatus{
		Phase:     corev1.PodRunning,
		HostIP:    "1.2.3.4",
		PodIP:     "5.6.7.8",
		StartTime: &now,
		Conditions: []corev1.PodCondition{
			{
				Type:   corev1.PodInitialized,
				Status: corev1.ConditionTrue,
			},
			{
				Type:   corev1.PodReady,
				Status: corev1.ConditionTrue,
			},
			{
				Type:   corev1.PodScheduled,
				Status: corev1.ConditionTrue,
			},
		},
	}

	for _, container := range pod.Spec.Containers {
		pod.Status.ContainerStatuses = append(pod.Status.ContainerStatuses, corev1.ContainerStatus{
			Name:         container.Name,
			Image:        container.Image,
			Ready:        true,
			RestartCount: 0,
			State: corev1.ContainerState{
				Running: &corev1.ContainerStateRunning{
					StartedAt: now,
				},
			},
		})
	}

	p.pods[key] = pod

	// Initialize minio client object.
	log.G(ctx).Infof("Initialize minio client object.\n")
	minioClient, err := minio.New(p.minioEndpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(p.minioAccessKeyID, p.minioSecretAccessKey, ""),
		Secure: false,
	})
	if err != nil {
		log.G(ctx).Errorf("Initialize minio client failed: %v.\n", err)
	}
	log.G(ctx).Infof("Initialized minio client object successfully.\n")

	if p.serverlessPlatformName == "openwhisk" {
		log.G(ctx).Infof("serverless platform : %s", p.serverlessPlatformName)
		err := openwhisk.CreateServerlessFunctionOW(ctx, p.serverlessPlatformApiHost, p.serverlessPlatformAuth, pod, minioClient)
		if err != nil {
			log.G(ctx).Infof("Failed to create pod: %v.\n", err)
			return err
		}

	}
	if p.serverlessPlatformName == "openfaas" {
		log.G(ctx).Infof("serverless platform : %s", p.serverlessPlatformName)
		err := openfaas.CreateServerlessFunctionOF(ctx, p.serverlessPlatformApiHost, p.serverlessPlatformAuth, pod, minioClient)
		if err != nil {
			log.G(ctx).Infof("Failed to create pod: %v.\n", err)
			return err
		}

	}

	if p.serverlessPlatformName == "gcf" {
		log.G(ctx).Infof("serverless platform : %s", p.serverlessPlatformName)
		err := gcf.CreateServerlessFunctionGCF(ctx, p.serverlessPlatformApiHost, p.serverlessPlatformAuth, p.serverlessPlatformConfigBucket, p.serverlessPlatformConfigBucketObject, pod, minioClient)
		if err != nil {
			log.G(ctx).Infof("Failed to create pod: %v.\n", err)
			return err
		}

	}
	p.notifier(pod)
	return nil
}

// UpdatePod takes a Kubernetes Pod and updates it within the provider.
func (p *FDNProvider) UpdatePod(ctx context.Context, pod *corev1.Pod) (err error) {
	log.G(ctx).Infof("Received UpdatePod request for %s/%s.\n", pod.Namespace, pod.Name)
	return errNotImplemented
}

// DeletePod accepts a Pod definition and deletes a Nomad job.
func (p *FDNProvider) DeletePod(ctx context.Context, pod *corev1.Pod) (err error) {
	// Deregister job

	ctx, span := trace.StartSpan(ctx, "DeletePod")
	defer span.End()

	// Add the pod's coordinates to the current span.
	ctx = addAttributes(ctx, span, namespaceKey, pod.Namespace, nameKey, pod.Name)

	log.G(ctx).Infof("receive DeletePod %q", pod.Name)

	key, err := buildKey(pod)
	if err != nil {
		return err
	}

	if _, exists := p.pods[key]; !exists {
		return errdefs.NotFound("pod not found")
	}

	// Initialize minio client object.
	log.G(ctx).Infof("Initialize minio client object.\n")
	minioClient, err := minio.New(p.minioEndpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(p.minioAccessKeyID, p.minioSecretAccessKey, ""),
		Secure: false,
	})
	if err != nil {
		log.G(ctx).Errorf("Initialize minio client failed: %v.\n", err)
	}
	log.G(ctx).Infof("Initialized minio client object successfully.\n")

	if p.serverlessPlatformName == "openwhisk" {
		log.G(ctx).Infof("serverless platform : %s", p.serverlessPlatformName)
		err := openwhisk.DeleteServerlessFunctionOW(ctx, p.serverlessPlatformApiHost, p.serverlessPlatformAuth, pod)
		if err != nil {
			log.G(ctx).Infof("Failed to delete pod: %v.\n", err)
			return err
		}
	}
	if p.serverlessPlatformName == "openfaas" {
		log.G(ctx).Infof("serverless platform : %s", p.serverlessPlatformName)
		err := openfaas.DeleteServerlessFunctionOF(ctx, p.serverlessPlatformApiHost, p.serverlessPlatformAuth, pod)
		if err != nil {
			log.G(ctx).Infof("Failed to delete pod: %v.\n", err)
			return err
		}
	}

	if p.serverlessPlatformName == "gcf" {
		log.G(ctx).Infof("serverless platform : %s", p.serverlessPlatformName)
		err := gcf.DeleteServerlessFunctionGCF(ctx, p.serverlessPlatformApiHost, p.serverlessPlatformAuth, p.serverlessPlatformConfigBucket, p.serverlessPlatformConfigBucketObject, pod, minioClient)
		if err != nil {
			log.G(ctx).Infof("Failed to delete pod: %v.\n", err)
			return err
		}
	}

	log.G(ctx).Infof("deleted serverless application %q response\n", pod.Name)

	now := metav1.Now()
	delete(p.pods, key)
	pod.Status.Phase = corev1.PodSucceeded
	pod.Status.Reason = "FDNProviderPodDeleted"

	for idx := range pod.Status.ContainerStatuses {
		pod.Status.ContainerStatuses[idx].Ready = false
		pod.Status.ContainerStatuses[idx].State = corev1.ContainerState{
			Terminated: &corev1.ContainerStateTerminated{
				Message:    "FDN provider terminated container upon deletion",
				FinishedAt: now,
				Reason:     "FDNProviderPodContainerDeleted",
				StartedAt:  pod.Status.ContainerStatuses[idx].State.Running.StartedAt,
			},
		}
	}

	p.notifier(pod)

	return nil
}

// GetPod returns the pod running in the Nomad cluster. returns nil
// if pod is not found.
func (p *FDNProvider) GetPod(ctx context.Context, namespace, name string) (pod *corev1.Pod, err error) {

	ctx, span := trace.StartSpan(ctx, "GetPod")
	defer func() {
		span.SetStatus(err)
		span.End()
	}()

	// Add the pod's coordinates to the current span.
	ctx = addAttributes(ctx, span, namespaceKey, namespace, nameKey, name)

	log.G(ctx).Infof("receive GetPod %q", name)

	// key, err := buildKeyFromNames(namespace, name)
	// if err != nil {
	// 	return nil, err
	// }

	// if pod, ok := p.pods[key]; ok {
	// 	return pod, nil
	// }

	// Get serverless function
	if p.serverlessPlatformName == "openwhisk" {
		log.G(ctx).Infof("serverless platform : %s", p.serverlessPlatformName)
		function, err := openwhisk.GetServerlessFunctionOW(ctx, p.serverlessPlatformApiHost, p.serverlessPlatformAuth, name)
		if err != nil {
			log.G(ctx).Infof("Failed to get pod: %v.\n", err)
			return nil, err
		}
		// Change a serverless function into a kubernetes pod
		pod, err = openwhisk.FunctionToPod(function, p.nodeName)
		if err != nil {
			return nil, fmt.Errorf("couldn't convert a serverless function into a pod: %s", err)
		}
		log.G(ctx).Infof("send function as pod :  %s", pod.Name)
		return pod, nil
	}
	if p.serverlessPlatformName == "openfaas" {
		log.G(ctx).Infof("serverless platform : %s", p.serverlessPlatformName)
		function, err := openfaas.GetServerlessFunctionOF(ctx, p.serverlessPlatformApiHost, p.serverlessPlatformAuth, name)
		if err != nil {
			log.G(ctx).Infof("Failed to get pod: %v.\n", err)
			return nil, err
		}
		// Change a serverless function into a kubernetes pod
		pod, err = openfaas.FunctionToPod(function, p.nodeName)
		if err != nil {
			return nil, fmt.Errorf("couldn't convert a serverless function into a pod: %s", err)
		}
		log.G(ctx).Infof("send function as pod :  %s", pod.Name)
		return pod, nil
	}

	return nil, nil
}

// GetContainerLogs retrieves the logs of a container by name from the provider.
func (p *FDNProvider) GetContainerLogs(ctx context.Context, namespace, podName, containerName string, opts api.ContainerLogOpts) (io.ReadCloser, error) {
	return ioutil.NopCloser(strings.NewReader("")), nil
}

// GetPodFullName retrieves the full pod name as defined in the provider context.
func (p *FDNProvider) GetPodFullName(namespace string, pod string) string {
	return ""
}

// RunInContainer executes a command in a container in the pod, copying data
// between in/out/err and the container's stdin/stdout/stderr.
func (p *FDNProvider) RunInContainer(ctx context.Context, namespace, podName, containerName string, cmd []string, attach api.AttachIO) error {
	return errNotImplemented
}

// GetPodStatus returns the status of a pod by name that is running as a job
// in the FDN cluster returns nil if a pod by that name is not found.
func (p *FDNProvider) GetPodStatus(ctx context.Context, namespace, name string) (*corev1.PodStatus, error) {
	ctx, span := trace.StartSpan(ctx, "GetPodStatus")
	defer span.End()

	// Add namespace and name as attributes to the current span.
	ctx = addAttributes(ctx, span, namespaceKey, namespace, nameKey, name)

	log.G(ctx).Infof("receive GetPodStatus %q", name)

	pod, err := p.GetPod(ctx, namespace, name)
	if err != nil {
		return nil, err
	}

	return &pod.Status, nil
}

// GetPods returns a list of all pods known to be running in Nomad nodes.
func (p *FDNProvider) GetPods(ctx context.Context) ([]*corev1.Pod, error) {
	ctx, span := trace.StartSpan(ctx, "GetPods")
	defer span.End()

	log.G(ctx).Info("receive GetPods")

	var pods = []*corev1.Pod{}

	if p.serverlessPlatformName == "openwhisk" {
		log.G(ctx).Infof("serverless platform : %s", p.serverlessPlatformName)

		functionsList, err := openwhisk.GetServerlessFunctionsOW(ctx, p.serverlessPlatformApiHost, p.serverlessPlatformAuth)
		if err != nil {
			return nil, fmt.Errorf("couldn't get fn list from ow: %s", err)
		}
		for _, function := range functionsList {
			// Change a function into a kubernetes pod
			pod, err := openwhisk.FunctionToPod(&function, p.nodeName)
			if err != nil {
				return nil, fmt.Errorf("couldn't convert a ow function into a pod: %s", err)
			}

			pods = append(pods, pod)
		}
		return pods, nil
	}
	if p.serverlessPlatformName == "openfaas" {
		log.G(ctx).Infof("serverless platform : %s", p.serverlessPlatformName)

		functionsList, err := openfaas.GetServerlessFunctionsOF(ctx, p.serverlessPlatformApiHost, p.serverlessPlatformAuth)
		if err != nil {
			return nil, fmt.Errorf("couldn't get fn list from OF: %s", err)
		}
		for _, function := range functionsList {
			// Change a function into a kubernetes pod
			pod, err := openfaas.FunctionToPod(&function, p.nodeName)
			if err != nil {
				return nil, fmt.Errorf("couldn't convert a OF function into a pod: %s", err)
			}

			pods = append(pods, pod)
		}
		return pods, nil
	}
	return nil, nil
}

func (p *FDNProvider) ConfigureNode(ctx context.Context, n *corev1.Node) { // nolint:golint
	ctx, span := trace.StartSpan(ctx, "mock.ConfigureNode") // nolint:staticcheck,ineffassign
	defer span.End()

	n.Status.Capacity = p.capacity()
	n.Status.Allocatable = p.capacity()
	n.Status.Conditions = p.nodeConditions()
	n.Status.Addresses = p.nodeAddresses()
	n.Status.DaemonEndpoints = p.nodeDaemonEndpoints()
	os := p.operatingSystem
	if os == "" {
		os = "linux"
	}
	n.Status.NodeInfo.OperatingSystem = os
	n.Status.NodeInfo.Architecture = "amd64"
	n.ObjectMeta.Labels["alpha.service-controller.kubernetes.io/exclude-balancer"] = "true"
	n.ObjectMeta.Labels["node.kubernetes.io/exclude-from-external-load-balancers"] = "true"
}

// Capacity returns a resource list containing the capacity limits.
func (p *FDNProvider) capacity() corev1.ResourceList {
	return corev1.ResourceList{
		"cpu":    resource.MustParse(p.config.CPU),
		"memory": resource.MustParse(p.config.Memory),
		"pods":   resource.MustParse(p.config.Pods),
	}
}

// NodeConditions returns a list of conditions (Ready, OutOfDisk, etc), for updates to the node status
// within Kubernetes.
func (p *FDNProvider) nodeConditions() []corev1.NodeCondition {
	// TODO: Make these dynamic.
	return []corev1.NodeCondition{
		{
			Type:               "Ready",
			Status:             corev1.ConditionTrue,
			LastHeartbeatTime:  metav1.Now(),
			LastTransitionTime: metav1.Now(),
			Reason:             "KubeletReady",
			Message:            "kubelet is ready.",
		},
		{
			Type:               "OutOfDisk",
			Status:             corev1.ConditionFalse,
			LastHeartbeatTime:  metav1.Now(),
			LastTransitionTime: metav1.Now(),
			Reason:             "KubeletHasSufficientDisk",
			Message:            "kubelet has sufficient disk space available",
		},
		{
			Type:               "MemoryPressure",
			Status:             corev1.ConditionFalse,
			LastHeartbeatTime:  metav1.Now(),
			LastTransitionTime: metav1.Now(),
			Reason:             "KubeletHasSufficientMemory",
			Message:            "kubelet has sufficient memory available",
		},
		{
			Type:               "DiskPressure",
			Status:             corev1.ConditionFalse,
			LastHeartbeatTime:  metav1.Now(),
			LastTransitionTime: metav1.Now(),
			Reason:             "KubeletHasNoDiskPressure",
			Message:            "kubelet has no disk pressure",
		},
		{
			Type:               "NetworkUnavailable",
			Status:             corev1.ConditionFalse,
			LastHeartbeatTime:  metav1.Now(),
			LastTransitionTime: metav1.Now(),
			Reason:             "RouteCreated",
			Message:            "RouteController created a route",
		},
	}

}

// NodeAddresses returns a list of addresses for the node status
// within Kubernetes.
func (p *FDNProvider) nodeAddresses() []corev1.NodeAddress {
	return []corev1.NodeAddress{
		{
			Type:    "InternalIP",
			Address: p.internalIP,
		},
	}
}

// NodeDaemonEndpoints returns NodeDaemonEndpoints for the node status
// within Kubernetes.
func (p *FDNProvider) nodeDaemonEndpoints() corev1.NodeDaemonEndpoints {
	return corev1.NodeDaemonEndpoints{
		KubeletEndpoint: corev1.DaemonEndpoint{
			Port: p.daemonEndpointPort,
		},
	}
}

// addAttributes adds the specified attributes to the provided span.
// attrs must be an even-sized list of string arguments.
// Otherwise, the span won't be modified.
// TODO: Refactor and move to a "tracing utilities" package.
func addAttributes(ctx context.Context, span trace.Span, attrs ...string) context.Context {
	if len(attrs)%2 == 1 {
		return ctx
	}
	for i := 0; i < len(attrs); i += 2 {
		ctx = span.WithField(ctx, attrs[i], attrs[i+1])
	}
	return ctx
}

// GetStatsSummary returns dummy stats for all pods known by this provider.
func (p *FDNProvider) GetStatsSummary(ctx context.Context) (*stats.Summary, error) {
	var span trace.Span
	ctx, span = trace.StartSpan(ctx, "GetStatsSummary") //nolint: ineffassign,staticcheck
	defer span.End()

	// Grab the current timestamp so we can report it as the time the stats were generated.
	time := metav1.NewTime(time.Now())

	// Create the Summary object that will later be populated with node and pod stats.
	res := &stats.Summary{}

	// Populate the Summary object with basic node stats.
	res.Node = stats.NodeStats{
		NodeName:  p.nodeName,
		StartTime: metav1.NewTime(p.startTime),
	}

	// Populate the Summary object with dummy stats for each pod known by this provider.
	for _, pod := range p.pods {
		var (
			// totalUsageNanoCores will be populated with the sum of the values of UsageNanoCores computes across all containers in the pod.
			totalUsageNanoCores uint64
			// totalUsageBytes will be populated with the sum of the values of UsageBytes computed across all containers in the pod.
			totalUsageBytes uint64
		)

		// Create a PodStats object to populate with pod stats.
		pss := stats.PodStats{
			PodRef: stats.PodReference{
				Name:      pod.Name,
				Namespace: pod.Namespace,
				UID:       string(pod.UID),
			},
			StartTime: pod.CreationTimestamp,
		}

		// Iterate over all containers in the current pod to compute dummy stats.
		for _, container := range pod.Spec.Containers {
			// Grab a dummy value to be used as the total CPU usage.
			// The value should fit a uint32 in order to avoid overflows later on when computing pod stats.
			dummyUsageNanoCores := uint64(rand.Uint32())
			totalUsageNanoCores += dummyUsageNanoCores
			// Create a dummy value to be used as the total RAM usage.
			// The value should fit a uint32 in order to avoid overflows later on when computing pod stats.
			dummyUsageBytes := uint64(rand.Uint32())
			totalUsageBytes += dummyUsageBytes
			// Append a ContainerStats object containing the dummy stats to the PodStats object.
			pss.Containers = append(pss.Containers, stats.ContainerStats{
				Name:      container.Name,
				StartTime: pod.CreationTimestamp,
				CPU: &stats.CPUStats{
					Time:           time,
					UsageNanoCores: &dummyUsageNanoCores,
				},
				Memory: &stats.MemoryStats{
					Time:       time,
					UsageBytes: &dummyUsageBytes,
				},
			})
		}

		// Populate the CPU and RAM stats for the pod and append the PodsStats object to the Summary object to be returned.
		pss.CPU = &stats.CPUStats{
			Time:           time,
			UsageNanoCores: &totalUsageNanoCores,
		}
		pss.Memory = &stats.MemoryStats{
			Time:       time,
			UsageBytes: &totalUsageBytes,
		}
		res.Pods = append(res.Pods, pss)
	}

	// Return the dummy stats.
	return res, nil
}

// NotifyPods is called to set a pod notifier callback function. This should be called before any operations are done
// within the provider.
func (p *FDNProvider) NotifyPods(ctx context.Context, notifier func(*corev1.Pod)) {
	p.notifier = notifier
}

func buildKeyFromNames(namespace string, name string) (string, error) {
	return fmt.Sprintf("%s-%s", namespace, name), nil
}

// buildKey is a helper for building the "key" for the providers pod store.
func buildKey(pod *corev1.Pod) (string, error) {
	if pod.ObjectMeta.Namespace == "" {
		return "", fmt.Errorf("pod namespace not found")
	}

	if pod.ObjectMeta.Name == "" {
		return "", fmt.Errorf("pod name not found")
	}

	return buildKeyFromNames(pod.ObjectMeta.Namespace, pod.ObjectMeta.Name)
}
