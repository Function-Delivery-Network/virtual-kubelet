package openfaas

import (
	"context"
	"io"
	"io/ioutil"
	"os"
	"strconv"
	"time"

	"fmt"
	"os/exec"
	"strings"

	"github.com/Function-Delivery-Network/virtual-kubelet/log"
	"github.com/minio/minio-go/v7"
	"gopkg.in/yaml.v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// openfaas provider constants
const (
	defaultFunctionMemory      = 256
	defaultFunctionCPU         = 200
	defaultFunctionTimeout     = 60000
	defaultFunctionConcurrency = 100
	defaultFunctionLogSize     = 80
)

type openFaaSPlatformClient struct {
	apiHost     string
	auth        string
	timeout     string
	pods        string
	clusterName string
}

type FuncLimitsMem struct {
	Memory string
}
const MinScaleLabel = "com.openfaas.scale.min"
// MaxScaleLabel label indicating max scale for a function
const MaxScaleLabel = "com.openfaas.scale.max"

// ScalingFactorLabel label indicates the scaling factor for a function
const ScalingFactorLabel = "com.openfaas.scale.factor"


type FuncLabels struct {
	MinScaleLabel int `com.openfaas.scale.min`
	MaxScaleLabel int `com.openfaas.scale.max`
	ScalingFactorLabel int `com.openfaas.scale.factor`
}

type FuncMem struct {
	Lang     string
	Handler  string
	Image    string
	Limits   FuncLimitsMem
	Requests FuncLimitsMem
	Labels   FuncLabels
}

type FuncLimitsCpu struct {
	Cpu string
}
type FuncCpu struct {
	Lang     string
	Handler  string
	Image    string
	Limits   FuncLimitsCpu
	Requests FuncLimitsCpu
	Labels   FuncLabels
}

type OpenFaaSFunc struct {
	Image string
	Name  string
}
type OpenFaaSProvider struct {
	Name    string
	Gateway string
}

type OpenFaaSYamlMem struct {
	Provider  OpenFaaSProvider
	Functions map[string]FuncMem
}

type OpenFaaSYamlCpu struct {
	Provider  OpenFaaSProvider
	Functions map[string]FuncCpu
}

// createFunctions takes the containers in a kubernetes pod and creates
// a list of serverless functions from them.
func CreateServerlessFunctionOF(ctx context.Context, apiHost string, auth string, pod *v1.Pod, minioClient *minio.Client) error {

	prg := "faas-cli"
	arg1 := "--gateway"
	arg2 := "login"
	arg3 := "--username"
	arg4 := "--password"
	log.G(ctx).Infof("faas-cli --gateway %v login --username admin --password %v\n", string(apiHost), string(auth))

	out, err := exec.Command(prg, arg1, apiHost, arg2, arg3, "admin", arg4, auth).Output()

	if err != nil {
		log.G(ctx).Errorf("Executing FaaS Cli register command failed: %v.\n", err)
		return err
	} else {
		log.G(ctx).Infof("Executing FaaS Cli register command success: %v.\n", string(out))
	}
	for _, ctr := range pod.Spec.Containers {
		bucket_name := ""
		object_name := ""
		cpu_specified := false

		memory := defaultFunctionMemory
		cpu := defaultFunctionCPU
		concurrency := defaultFunctionConcurrency
		for _, s := range ctr.Env {
			if s.Name == "BUCKET_NAME" {
				bucket_name = s.Value
			}
			if s.Name == "OBJECT_NAME" {
				object_name = s.Value
			}
			if s.Name == "FUNCTION_MEMORY" {
				number, _ := strconv.ParseUint(s.Value, 10, 32)
				memory = int(number)
			}
			if s.Name == "FUNCTION_CPU" {
				number, _ := strconv.ParseUint(s.Value, 10, 32)
				cpu = int(number)
				cpu_specified = true

			}
			if s.Name == "FUNCTION_CONCURRENCY" {
				number, _ := strconv.ParseUint(s.Value, 10, 32)
				concurrency = int(number)
			}
			/* 			if(s.Name == "FUNCTION_TIMEOUT"){
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
			} */
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

		log.G(ctx).Infof("Environment Variables: %v.\n", ctr.Env)
		// Convert []byte to string and print to screen
		mycode := string(content)
		log.G(ctx).Infof("Code: %v.\n", mycode)

		s1 := OpenFaaSProvider{
			Name:    "openfaas",
			Gateway: apiHost,
		}

		if cpu_specified == false {
			f1 := FuncMem{
				Lang:     "python3",
				Handler:  "./" + ctr.Name,
				Image:    ctr.Image,
				Limits:   FuncLimitsMem{Memory: strconv.Itoa(memory) + "Mi"},
				Requests: FuncLimitsMem{Memory: strconv.Itoa(memory) + "Mi"},
				Labels: FuncLabels{MinScaleLabel: 1, MaxScaleLabel: concurrency, ScalingFactorLabel: 50},
			}
			m := make(map[string]FuncMem)
			m[ctr.Name] = f1
			yaml1 := OpenFaaSYamlMem{
				Provider:  s1,
				Functions: m,
			}

			yamlData, err := yaml.Marshal(&yaml1)
			if err != nil {
				log.G(ctx).Errorf("Error while Marshaling. %v \n", err)
			}
			err2 := ioutil.WriteFile("/tmp/func_deployment.yaml", yamlData, 0)

			if err2 != nil {

				log.G(ctx).Errorf("file write failed: %v.\n", err2)
				return err
			} else {

				log.G(ctx).Infof("File written")
			}
			fmt.Println(" --- YAML ---")
			fmt.Println(string(yamlData)) //

		} else {
			f1 := FuncCpu{
				Lang:     "python3",
				Handler:  "./" + ctr.Name,
				Image:    ctr.Image,
				Limits:   FuncLimitsCpu{Cpu: strconv.Itoa(cpu) + "m"},
				Requests: FuncLimitsCpu{Cpu: strconv.Itoa(cpu) + "m"},
				Labels: FuncLabels{MinScaleLabel: 1, MaxScaleLabel: concurrency, ScalingFactorLabel: 50},
			}
			m := make(map[string]FuncCpu)
			m[ctr.Name] = f1
			yaml1 := OpenFaaSYamlCpu{
				Provider:  s1,
				Functions: m,
			}

			yamlData, err := yaml.Marshal(&yaml1)
			if err != nil {
				log.G(ctx).Errorf("Error while Marshaling. %v \n", err)
			}
			err2 := ioutil.WriteFile("/tmp/func_deployment.yaml", yamlData, 0)

			if err2 != nil {

				log.G(ctx).Errorf("file write failed: %v.\n", err2)
				return err
			} else {

				log.G(ctx).Infof("File written")
			}
			fmt.Println(" --- YAML ---")
			fmt.Println(string(yamlData)) //
		}

		prg1 := "faas-cli"
		out1, err1 := exec.Command(prg1, "template", "pull", "-f", "/tmp/func_deployment.yaml").Output()

		if err1 != nil {
			log.G(ctx).Errorf("Executing FaaS template pull command failed: %v.\n", err1)
			return err1
		} else {
			log.G(ctx).Infof("Executing FaaS template pull command success: %v.\n", string(out1))
		}

		prg := "faas-cli"
		arg1 := "--gateway"
		arg2 := "deploy"
		arg3 := "-f"
		out, err := exec.Command(prg, arg1, apiHost, arg2, arg3, "/tmp/func_deployment.yaml").Output()
		if err != nil {
			log.G(ctx).Errorf("failed to create of function: %v.\n", err)
			return err
		} else {
			log.G(ctx).Infof("Function create Returned with status: %v.\n", out)
		}
	}
	return nil
}

func DeleteServerlessFunctionOF(ctx context.Context, apiHost string, auth string, pod *v1.Pod) error {

	prg := "faas-cli"
	arg1 := "--gateway"
	arg2 := "login"
	arg3 := "--username"
	arg4 := "--password"
	log.G(ctx).Infof("faas-cli --gateway %v login --username admin --password %v\n", string(apiHost), string(auth))

	out, err := exec.Command(prg, arg1, apiHost, arg2, arg3, "admin", arg4, auth).Output()

	if err != nil {
		log.G(ctx).Errorf("Executing FaaS Cli register command failed: %v.\n", err)
		return err
	} else {
		log.G(ctx).Infof("Executing FaaS Cli register command success: %v.\n", string(out))
	}

	for _, ctr := range pod.Spec.Containers {
		//image := ctr.Image
		arg_f2 := "remove"
		resp, err := exec.Command(prg, arg1, apiHost, arg_f2, ctr.Name).Output()

		if err != nil {
			log.G(ctx).Errorf("Executing FaaS Cli delete command failed: %v.\n", err)
			return err
		} else {
			log.G(ctx).Infof("Executing FaaS Cli delete command success: %v.\n", string(resp))
		}
	}

	return nil
}

func GetServerlessFunctionOF(ctx context.Context, apiHost string, auth string, name string) (*OpenFaaSFunc, error) {

	prg := "faas-cli"
	arg1 := "--gateway"
	out, err := exec.Command(prg, arg1, apiHost, "list", "--verbose").Output()

	if err != nil {
		log.G(ctx).Errorf("Executing FaaS Cli register command failed: %v.\n", err)
		return nil, err
	} else {
		log.G(ctx).Infof("Executing FaaS Cli register command success: %v.\n", string(out))
		s := strings.Split(string(out), "\n")
		for i, v := range s {
			if i == 0 {
				continue
			}
			s1 := strings.Fields(v)
			log.G(ctx).Infof("Returned function: %v.\n", s1)
			if len(s1) > 0 {
				if s1[0] == name {
					action := OpenFaaSFunc{
						Name:  s1[0],
						Image: s1[1],
					}
					log.G(ctx).Infof("Returned action: %v.\n", action)
					return &action, nil
				}
			}
		}
		
		log.G(ctx).Infof("No Returned action\n")
		return nil, nil
	}
}

func GetServerlessFunctionsOF(ctx context.Context, apiHost string, auth string) ([]OpenFaaSFunc, error) {

	prg := "faas-cli"
	arg1 := "--gateway"
	out, err := exec.Command(prg, arg1, apiHost, "list", "--verbose").Output()

	if err != nil {
		log.G(ctx).Errorf("Executing FaaS Cli register command failed: %v.\n", err)
		return nil, err
	} else {
		log.G(ctx).Infof("Executing FaaS Cli register command success: %v.\n", string(out))
		s := strings.Split(string(out), "\n")
		var actions []OpenFaaSFunc
		for i, v := range s {
			if i == 0 {
				continue
			}
			s1 := strings.Fields(v)
			log.G(ctx).Infof("Returned function: %v.\n", s1)
			if len(s1) > 0 {
				actions = append(actions, OpenFaaSFunc{
					Name:  s1[0],
					Image: s1[1],
				})
			}
		}
		if len(actions)>0  {
			log.G(ctx).Infof("Returned actions: %v.\n", actions)
			return actions, nil
		} else{
			log.G(ctx).Infof("No Returned actions.\n")
			return nil, nil
		}
	}
}

func FunctionToPod(action *OpenFaaSFunc, nodeName string) (*v1.Pod, error) {
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
		Image: action.Image,
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
	containerStatus.Image = action.Image
	containerStatus.ImageID = action.Image
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
