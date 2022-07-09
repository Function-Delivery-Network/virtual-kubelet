package openwhisk

import (
	"context"
	"net/http"
	"os"
	"io"
    "io/ioutil"
	"strconv"

	"time"

	"github.com/Function-Delivery-Network/virtual-kubelet/log"
	"github.com/apache/openwhisk-client-go/whisk"
	"github.com/minio/minio-go/v7"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// openfaas provider constants
const (
	defaultFunctionMemory = 256
	defaultFunctionTimeout = 60000
	defaultFunctionConcurrency = 100
	defaultFunctionLogSize = 80
)
type openWhiskPlatformClient struct {
	apiHost     string
	auth        string
	timeout     string
	pods        string
	clusterName string
}

// createFunctions takes the containers in a kubernetes pod and creates
// a list of serverless functions from them.
func CreateServerlessFunctionOW(ctx context.Context, apiHost string, auth string, pod *v1.Pod, minioClient *minio.Client) error {
	config := &whisk.Config{
		Host:      apiHost,
		Version:   "v1",
		Verbose:   true,
		Namespace: "_",
		AuthToken: auth,
		Debug:     true,
		Insecure:  true,
	}
	client, err := whisk.NewClient(http.DefaultClient, config)

	if err != nil {
		log.G(ctx).Errorf("Initialize minio client failed: %v.\n", err)
		return err
	}
	for _, ctr := range pod.Spec.Containers {
		bucket_name := ""
		object_name := ""

		memory := defaultFunctionMemory
		timeout := defaultFunctionTimeout
		concurrency := defaultFunctionConcurrency
		logSize := defaultFunctionLogSize
		for _, s := range ctr.Env {
			if(s.Name == "BUCKET_NAME"){
				bucket_name = s.Value	
			}
			if(s.Name == "OBJECT_NAME"){
				object_name = s.Value	
			}
			if(s.Name == "FUNCTION_MEMORY"){
				number,_ := strconv.ParseUint(s.Value, 10, 32)
				memory = int(number)
			}
			if(s.Name == "FUNCTION_TIMEOUT"){
				number,_ := strconv.ParseUint(s.Value, 10, 32)
				timeout = int(number)
			}
			if(s.Name == "FUNCTION_CONCURRENCY"){
				number,_ := strconv.ParseUint(s.Value, 10, 32)
				concurrency = int(number)
			}
			if(s.Name == "FUNCTION_LOGSIZE"){
				number,_ := strconv.ParseUint(s.Value, 10, 32)
				logSize = int(number)
			}
		}
		object, err := minioClient.GetObject(context.Background(), bucket_name, object_name, minio.GetObjectOptions{})
		if err != nil {
			log.G(ctx).Errorf(" minio GetObject failed: %v.\n", err)
			return err
		}
		localFile, err := os.Create("/tmp/funcode")
		if err != nil {
			log.G(ctx).Errorf(" cannot create file failed: %v.\n", err)
			return err
		}
		if _, err = io.Copy(localFile, object); err != nil {
			log.G(ctx).Errorf("file save failed: %v.\n", err)
			return err
		}
		content, err := ioutil.ReadFile("/tmp/funcode")
		if err != nil {
			log.G(ctx).Errorf("file read failed: %v.\n", err)
			return err
		}
		

		log.G(ctx).Errorf("Environment Variables: %v.\n", ctr.Env)
		// Convert []byte to string and print to screen
		mycode := string(content)
		log.G(ctx).Errorf("Code: %v.\n", mycode)
		
		funExec := &whisk.Exec{Image: ctr.Image, Kind: "blackbox", Code: &mycode}
		funcLimits := &whisk.Limits{Memory: &memory, Timeout: &timeout, Concurrency: &concurrency, Logsize: &logSize}

		serverlessFunction := &whisk.Action{
			Name: ctr.Name,
			Exec: funExec,
			Limits: funcLimits,
		}
		_, resp, err := client.Actions.Insert(serverlessFunction, true)
		if err != nil {
			log.G(ctx).Errorf("failed to create ow function: %v.\n", err)
			return err
		}
		log.G(ctx).Infof("Returned with status: %v.\n", resp.Status)
	}

	return nil
}

func DeleteServerlessFunctionOW(ctx context.Context, apiHost string, auth string, pod *v1.Pod) error {
	config := &whisk.Config{
		Host:      apiHost,
		Version:   "v1",
		Verbose:   true,
		Namespace: "_",
		AuthToken: auth,
		Debug:     true,
		Insecure:  true,
	}
	client, err := whisk.NewClient(http.DefaultClient, config)

	if err != nil {
		log.G(ctx).Errorf("failed to connect to ow client: %v.\n", err)
		return err
	}

	for _, ctr := range pod.Spec.Containers {
		//image := ctr.Image
		resp, err := client.Actions.Delete(ctr.Name)
		if err != nil {
			log.G(ctx).Errorf("failed to  delete ow function: %v.\n", err)
			return err
		}

		log.G(ctx).Infof("Returned with status: %v.\n", resp.Status)
	}

	return nil
}

func GetServerlessFunctionOW(ctx context.Context, apiHost string, auth string, name string) (*whisk.Action, error) {
	config := &whisk.Config{
		Host:      apiHost,
		Version:   "v1",
		Verbose:   true,
		Namespace: "_",
		AuthToken: auth,
		Debug:     true,
		Insecure:  true,
	}
	client, err := whisk.NewClient(http.DefaultClient, config)

	if err != nil {
		log.G(ctx).Errorf("failed to connect to ow client: %v.\n", err)
		return nil, err
	}
	action, resp, err := client.Actions.Get(name, false)
	if err != nil {
		log.G(ctx).Errorf("failed to connect to get action: %v.\n", err)
		return nil, err
	}
	log.G(ctx).Infof("Returned with status: %v.\n", resp.Status)

	return action, nil
}

func GetServerlessFunctionsOW(ctx context.Context, apiHost string, auth string) ([]whisk.Action, error) {
	config := &whisk.Config{
		Host:      apiHost,
		Version:   "v1",
		Verbose:   true,
		Namespace: "_",
		AuthToken: auth,
		Debug:     true,
		Insecure:  true,
	}
	client, err := whisk.NewClient(http.DefaultClient, config)

	if err != nil {
		log.G(ctx).Errorf("failed to connect to ow client: %v.\n", err)
		return nil, err
	}

	options := &whisk.ActionListOptions{
		Limit: 10,
		Skip:  0,
	}

	actions, resp, err := client.Actions.List("", options)
	if err != nil {
		log.G(ctx).Errorf("failed to connect to get action: %v.\n", err)
		return nil, err
	}
	log.G(ctx).Infof("Returned with status: %v.\n", resp.Status)
	log.G(ctx).Infof("Returned actions: %v.\n", actions)
	return actions, nil
}

func FunctionToPod(action *whisk.Action, nodeName string) (*v1.Pod, error) {
	containers := []v1.Container{}
	containerStatues := []v1.ContainerStatus{}
	podCondition := convertFunctionStatusToPodCondition("running")

	var containerPorts []v1.ContainerPort
	containerPorts = append(containerPorts, v1.ContainerPort{
		Name:     "http",
		HostPort: int32(31001),
		HostIP:   "127.0.0.1",
	})

	containers = append(containers, v1.Container{
		Name:  action.Name,
		Image: action.Exec.Image,
		Ports: containerPorts,
	})

	readyFlag := true
	containerStatus := v1.ContainerStatus{
		Name:         action.Name,
		RestartCount: int32(0),
		Ready:        readyFlag,
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{
				StartedAt: metav1.NewTime(time.Unix(275, 0)),
			},
		},
	}
	containerStatus.Image = action.Exec.Image
	containerStatus.ImageID = action.Exec.Image
	containerStatues = append(containerStatues, containerStatus)

	pod := v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:              action.Name,
			Namespace:         nodeName,
			CreationTimestamp: metav1.NewTime(time.Unix(275, 0)),
		},
		Spec: v1.PodSpec{
			NodeName:   nodeName,
			Volumes:    []v1.Volume{},
			Containers: containers,
		},
		Status: v1.PodStatus{
			Phase:             jobStatusToPodPhase("running"),
			Conditions:        []v1.PodCondition{podCondition},
			Message:           "",
			Reason:            "",
			HostIP:            "", // TODO: find out the HostIP
			PodIP:             "", // TODO: find out the equalent for PodIP
			ContainerStatuses: containerStatues,
		},
	}

	return &pod, nil
}

func jobStatusToPodPhase(status string) v1.PodPhase {
	switch status {
	case "pending":
		return v1.PodPending
	case "running":
		return v1.PodRunning
	// TODO: Make sure we take PodFailed into account.
	case "dead":
		return v1.PodFailed
	}
	return v1.PodUnknown
}

func convertFunctionStatusToPodCondition(jobStatus string) v1.PodCondition {
	podCondition := v1.PodCondition{}

	switch jobStatus {
	case "pending":
		podCondition = v1.PodCondition{
			Type:   v1.PodInitialized,
			Status: v1.ConditionFalse,
		}
	case "running":
		podCondition = v1.PodCondition{
			Type:   v1.PodReady,
			Status: v1.ConditionTrue,
		}
	case "dead":
		podCondition = v1.PodCondition{
			Type:   v1.PodReasonUnschedulable,
			Status: v1.ConditionFalse,
		}
	default:
		podCondition = v1.PodCondition{
			Type:   v1.PodReasonUnschedulable,
			Status: v1.ConditionUnknown,
		}
	}

	return podCondition
}

// func convertFunctionStateToContainerState(functionState string, startedAt time.Time, finishedAt time.Time) (v1.ContainerState, bool) {
// 	containerState := v1.ContainerState{}
// 	readyFlag := false

// 	switch functionState {
// 	case "pending":
// 		containerState = v1.ContainerState{
// 			Waiting: &v1.ContainerStateWaiting{},
// 		}
// 	case "running":
// 		containerState = v1.ContainerState{
// 			Running: &v1.ContainerStateRunning{
// 				StartedAt: metav1.NewTime(startedAt),
// 			},
// 		}
// 		readyFlag = true
// 	// TODO: Make sure containers that are exiting with non-zero status codes
// 	// are accounted for using events or something similar?
// 	//case v1.PodSucceeded:
// 	//	podCondition = v1.PodCondition{
// 	//		Type:   v1.PodReasonUnschedulable,
// 	//		Status: v1.ConditionFalse,
// 	//	}
// 	//	containerState = v1.ContainerState{
// 	//		Terminated: &v1.ContainerStateTerminated{
// 	//			ExitCode:   int32(container.State.ExitCode),
// 	//			FinishedAt: metav1.NewTime(finishedAt),
// 	//		},
// 	//	}
// 	case "dead":
// 		containerState = v1.ContainerState{
// 			Terminated: &v1.ContainerStateTerminated{
// 				ExitCode:   0,
// 				FinishedAt: metav1.NewTime(finishedAt),
// 			},
// 		}
// 	default:
// 		containerState = v1.ContainerState{}
// 	}

// 	return containerState, readyFlag
// }
