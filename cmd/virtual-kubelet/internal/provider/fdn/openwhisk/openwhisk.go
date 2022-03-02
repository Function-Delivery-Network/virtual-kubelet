package openwhisk

import (
	"fmt"
	"log"
	"net/http"

	"time"

	"github.com/apache/openwhisk-client-go/whisk"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
func CreateServerlessFunctionOW(apiHost string, auth string, pod *v1.Pod) error {
	config := &whisk.Config{
		Host: apiHost,
		Version: "v1",
		Verbose: true,
		Namespace: "_",
		AuthToken: auth,
		Debug: true,
		Insecure: true,
	  }
	  client, err := whisk.NewClient(http.DefaultClient, config)

	  if err != nil {
		err = fmt.Errorf("failed to connect to ow client %v", err)
		return err
	}

	

	for _, ctr := range pod.Spec.Containers {
		//image := ctr.Image
		mycode := "function main(params) { return {payload:\"Hello \"+params.name}}"
		funExec := &whisk.Exec{Kind: "nodejs:12", Code: &mycode}

		serverlessFunction := &whisk.Action{
			Name:   ctr.Name,
			Exec: funExec,
			// Config: map[string]interface{}{
			// 	"image":    image,
			// 	"port_map": portMap,
			// 	"labels":   labels,
			// 	// TODO: Add volumes support
			// 	"command": strings.Join(command, ""),
			// 	"args":    args,
			// },
			// Resources: resources,
			// Env:       envVars,
		}
		_, resp, err := client.Actions.Insert(serverlessFunction, true)
		if err != nil {
			err = fmt.Errorf("failed to create ow function %v", err)
			return err
		}

		log.Println("Returned with status: ", resp.Status)
	}

	return nil
}

func DeleteServerlessFunctionOW(apiHost string, auth string, pod *v1.Pod) error {
	config := &whisk.Config{
		Host: apiHost,
		Version: "v1",
		Verbose: true,
		Namespace: "_",
		AuthToken: auth,
		Debug: true,
		Insecure: true,
	  }
	  client, err := whisk.NewClient(http.DefaultClient, config)

	  if err != nil {
		err = fmt.Errorf("failed to connect to ow client %v", err)
		return err
	}

	for _, ctr := range pod.Spec.Containers {
		//image := ctr.Image
		resp, err := client.Actions.Delete(ctr.Name)
		if err != nil {
			err = fmt.Errorf("failed to delete ow function %v", err)
			return nil
		}

		log.Println("Returned with status: ", resp.Status)
	}

	return nil
}

func GetServerlessFunctionOW(apiHost string, auth string, name string) (*whisk.Action, error) {
	config := &whisk.Config{
		Host: apiHost,
		Version: "v1",
		Verbose: true,
		Namespace: "_",
		AuthToken: auth,
		Debug: true,
		Insecure: true,
	  }
	  client, err := whisk.NewClient(http.DefaultClient, config)

	  if err != nil {
		err = fmt.Errorf("failed to connect to ow client %v", err)
		return nil, err
	}
	action, resp, err := client.Actions.Get(name, false)
	if err != nil {
		err = fmt.Errorf("failed to connect to get action %v", err)
		return nil, err
	}
	log.Println("Returned with status: ", resp.Status)

	return action, nil
}

func GetServerlessFunctionsOW(apiHost string, auth string) ([]whisk.Action, error) {
	config := &whisk.Config{
		Host: apiHost,
		Version: "v1",
		Verbose: true,
		Namespace: "_",
		AuthToken: auth,
		Debug: true,
		Insecure: true,
	  }
	  client, err := whisk.NewClient(http.DefaultClient, config)

	  if err != nil {
		err = fmt.Errorf("failed to connect to ow client %v", err)
		return nil, err
	}

	options := &whisk.ActionListOptions{
		Limit: 10,
		Skip:  0,
	}

	actions, resp, err := client.Actions.List("", options)
	if err != nil {
		err = fmt.Errorf("failed to get actions %v", err)
		return nil, err
	}
	log.Println("Returned with status: ", resp.Status)
	log.Printf("Returned actions: \n %+v", actions)

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
		Name:    action.Name,
		Image:   action.Exec.Image,
		Ports:   containerPorts,
	})
	
	readyFlag := true
	containerStatus := v1.ContainerStatus{
		Name:         action.Name,
		RestartCount: int32(0),
		Ready:        readyFlag,
		State:        v1.ContainerState{
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
			Namespace:         "default",
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

