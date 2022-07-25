package gcf

import (
	"context"
	"encoding/json"
	"io"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"time"

	"os/exec"

	"github.com/Function-Delivery-Network/virtual-kubelet/log"
	"github.com/minio/minio-go/v7"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type GCFPlatformAuth struct {
	Type                        string `json:"type"`
	Project_id                  string `json:"project_id"`
	Private_key_id              string `json:"private_key_id"`
	Private_key                 string `json:"private_key"`
	Client_email                string `json:"client_email"`
	Client_id                   string `json:"client_id"`
	Auth_uri                    string `json:"auth_uri"`
	Token_uri                   string `json:"token_uri"`
	Auth_provider_x509_cert_url string `json:"auth_provider_x509_cert_url"`
	Client_x509_cert_url        string `json:"client_x509_cert_url"`
}

// openfaas provider constants
const (
	defaultFunctionMemory      = 256
	defaultFunctionCPU         = 200
	defaultFunctionTimeout     = 60
	defaultFunctionConcurrency = 100
	defaultFunctionLogSize     = 80
)

type GCFFaaSFunc struct {
	Image string
	Name  string
}

// createFunctions takes the containers in a kubernetes pod and creates
// a list of serverless functions from them.
func CreateServerlessFunctionGCF(ctx context.Context, apiHost string, auth string, auth_bucket_name string, auth_object_name string, region string, pod *v1.Pod, minioClient *minio.Client) error {

	log.G(ctx).Infof("auth_bucket_name: %v auth_object_name:%v region:%v \n", auth_bucket_name, auth_object_name, region)

	object, err := minioClient.GetObject(context.Background(), auth_bucket_name, auth_object_name, minio.GetObjectOptions{})
	if err != nil {
		log.G(ctx).Errorf(" minio auth_object_name GetObject failed: %v.\n", err)
		return err
	}
	localFile, err := os.Create("/tmp/auth.json")
	if err != nil {
		log.G(ctx).Errorf(" cannot create auth.json file failed: %v.\n", err)
		return err
	}
	if _, err = io.Copy(localFile, object); err != nil {
		log.G(ctx).Errorf("auth.json file save failed: %v.\n", err)
		return err
	}
	jsonFile, err := os.Open("/tmp/auth.json")
	byteValue, _ := ioutil.ReadAll(jsonFile)
	if err != nil {
		log.G(ctx).Errorf("Reading auth json command failed: %v.\n", err)
		return err
	} else {
		log.G(ctx).Infof("Reading authr command success: %v.\n", string(byteValue))
	}
	// we initialize our Users array
	var gcfPlatformAuth GCFPlatformAuth

	// we unmarshal our byteArray which contains our
	// jsonFile's content into 'users' which we defined above
	json.Unmarshal(byteValue, &gcfPlatformAuth)

	//url := "http://"+region+"-"+gcfPlatformAuth.Project_id+".cloudfunctions.net/"+pod.Name
	//pod.Status.PodIP = url



	out, err := exec.Command("gcloud", "auth", "activate-service-account", gcfPlatformAuth.Client_email, "--key-file=/tmp/auth.json", "--project="+string(gcfPlatformAuth.Project_id)).Output()

	if err != nil {
		log.G(ctx).Errorf("Executing gcloud Cli register command failed: %v.\n", err)
		return err
	} else {
		log.G(ctx).Infof("Executing gcloud Cli register command success: %v.\n", string(out))
	}
	for _, ctr := range pod.Spec.Containers {
		bucket_name := ""
		object_name := ""
		memory := defaultFunctionMemory
		timeout := defaultFunctionTimeout
		concurrency := defaultFunctionConcurrency
/* 		url := "http://"+region+"-"+gcfPlatformAuth.Project_id+".cloudfunctions.net/"+ctr.Name
		pod.Status.PodIP = url */
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
			if s.Name == "FUNCTION_TIMEOUT" {
				number, _ := strconv.ParseUint(s.Value, 10, 32)
				timeout = int(number) / 1000
			}
			if s.Name == "FUNCTION_CONCURRENCY" {
				number, _ := strconv.ParseUint(s.Value, 10, 32)
				concurrency = int(number)
			}
		}
		object, err := minioClient.GetObject(context.Background(), bucket_name, object_name, minio.GetObjectOptions{})
		if err != nil {
			log.G(ctx).Errorf(" minio GetObject failed: %v.\n", err)
			return err
		}

		if err := os.Mkdir("/tmp/"+ctr.Name, os.ModePerm); err != nil {
			log.G(ctx).Errorf(" create dir failed: %v.\n", err)
		}

		localFile, err := os.Create("/tmp/" + ctr.Name + "/main.py")
		if err != nil {
			log.G(ctx).Errorf(" cannot create file failed: %v.\n", err)
			return err
		}
		if _, err = io.Copy(localFile, object); err != nil {
			log.G(ctx).Errorf("file save failed: %v.\n", err)
			return err
		}
		content, err := ioutil.ReadFile("/tmp/" + ctr.Name + "/main.py")
		if err != nil {
			log.G(ctx).Errorf("file read failed: %v.\n", err)
			return err
		}

		log.G(ctx).Infof("Environment Variables: %v.\n", ctr.Env)
		// Convert []byte to string and print to screen
		mycode := string(content)
		log.G(ctx).Infof("Code: %v.\n", mycode)

		log.G(ctx).Infof("gcloud functions deploy %v --runtime=%v --region=%v --allow-unauthenticated --memory=%v --source=/tmp/%v --max-instances=%v --timeout=%v --trigger-http\n", ctr.Name, ctr.Image, region, strconv.Itoa(memory), ctr.Name, strconv.Itoa(concurrency), strconv.Itoa(timeout))
		prg1 := "gcloud"
		out, err := exec.Command(prg1, "functions", "deploy", ctr.Name,
			"--runtime="+string(ctr.Image),
			"--region="+string(region),
			"--entry-point=handle",
			"--allow-unauthenticated",
			"--memory="+strconv.Itoa(memory),
			"--source=/tmp/"+ctr.Name,
			"--max-instances="+strconv.Itoa(concurrency),
			"--timeout="+strconv.Itoa(timeout),
			"--trigger-http").Output()
		if err != nil {
			log.G(ctx).Errorf("failed to create GCF function: %v.\n", err)
			return err
		} else {
			log.G(ctx).Infof("Function create Returned with status: %v.\n", out)
		}
	}
	return nil
}

func DeleteServerlessFunctionGCF(ctx context.Context, apiHost string, auth string, auth_bucket_name string, auth_object_name string, region string, pod *v1.Pod, minioClient *minio.Client) error {

	object, err := minioClient.GetObject(context.Background(), auth_bucket_name, auth_object_name, minio.GetObjectOptions{})
	if err != nil {
		log.G(ctx).Errorf(" minio auth_object_name GetObject failed: %v.\n", err)
		return err
	}
	localFile, err := os.Create("/tmp/auth.json")
	if err != nil {
		log.G(ctx).Errorf(" cannot create auth.json file failed: %v.\n", err)
		return err
	}
	if _, err = io.Copy(localFile, object); err != nil {
		log.G(ctx).Errorf("auth.json file save failed: %v.\n", err)
		return err
	}
	jsonFile, err := os.Open("/tmp/auth.json")
	byteValue, _ := ioutil.ReadAll(jsonFile)

	if err != nil {
		log.G(ctx).Errorf("Reading auth json command failed: %v.\n", err)
		return err
	} else {
		log.G(ctx).Infof("Reading authr command success: %v.\n", string(byteValue))
	}

	// we initialize our Users array
	var gcfPlatformAuth GCFPlatformAuth

	// we unmarshal our byteArray which contains our
	// jsonFile's content into 'users' which we defined above
	json.Unmarshal(byteValue, &gcfPlatformAuth)

	out, err := exec.Command("gcloud", "auth", "activate-service-account", gcfPlatformAuth.Client_email, "--key-file=/tmp/auth.json", "--project="+string(gcfPlatformAuth.Project_id)).Output()

	if err != nil {
		log.G(ctx).Errorf("Executing gcloud Cli register command failed: %v.\n", err)
		return err
	} else {
		log.G(ctx).Infof("Executing gcloud Cli register command success: %v.\n", string(out))
	}

	for _, ctr := range pod.Spec.Containers {

		resp, err := exec.Command("gcloud", "functions", "delete", ctr.Name, "--quiet", "--region="+string(region)).Output()

		if err != nil {
			log.G(ctx).Errorf("Executing gcloud delete command failed: %v.\n", err)
			return err
		} else {
			log.G(ctx).Infof("Executing gcloud delete command success: %v.\n", string(resp))
		}
	}

	return nil
}

func GetServerlessFunctionGCF(ctx context.Context, apiHost string, auth string, auth_bucket_name string, auth_object_name string, region string, minioClient *minio.Client, name string) (*GCFFaaSFunc, error) {

	object, err := minioClient.GetObject(context.Background(), auth_bucket_name, auth_object_name, minio.GetObjectOptions{})
	if err != nil {
		log.G(ctx).Errorf(" minio auth_object_name GetObject failed: %v.\n", err)
		return nil, err
	}
	localFile, err := os.Create("/tmp/auth.json")
	if err != nil {
		log.G(ctx).Errorf(" cannot create auth.json file failed: %v.\n", err)
		return nil, err
	}
	if _, err = io.Copy(localFile, object); err != nil {
		log.G(ctx).Errorf("auth.json file save failed: %v.\n", err)
		return nil, err
	}
	jsonFile, err := os.Open("/tmp/auth.json")
	byteValue, _ := ioutil.ReadAll(jsonFile)

	if err != nil {
		log.G(ctx).Errorf("Reading auth json command failed: %v.\n", err)
		return nil, err
	} else {
		log.G(ctx).Infof("Reading authr command success: %v.\n", string(byteValue))
	}

	// we initialize our Users array
	var gcfPlatformAuth GCFPlatformAuth

	// we unmarshal our byteArray which contains our
	// jsonFile's content into 'users' which we defined above
	json.Unmarshal(byteValue, &gcfPlatformAuth)

	out, err := exec.Command("gcloud", "auth", "activate-service-account", gcfPlatformAuth.Client_email, "--key-file=/tmp/auth.json", "--project="+string(gcfPlatformAuth.Project_id)).Output()

	if err != nil {
		log.G(ctx).Errorf("Executing gcloud Cli register command failed: %v.\n", err)
		return nil, err
	} else {
		log.G(ctx).Infof("Executing gcloud Cli register command success: %v.\n", string(out))
	}

	funcs_out, err := exec.Command("gcloud", "functions", "list", "--regions="+string(region)).Output()

	if err != nil {
		log.G(ctx).Errorf("Executing gcloud functions list command failed: %v.\n", err)
		return nil, err
	} else {
		log.G(ctx).Infof("Executing gcloud functions list command success: %v.\n", string(funcs_out))
		s := strings.Split(string(funcs_out), "\n")
		for i, v := range s {
			if i == 0 {
				continue
			}
			s1 := strings.Fields(v)
			log.G(ctx).Infof("Returned function: %v.\n", s1)
			if len(s1) > 0 {
				if s1[0] == name {
					action := GCFFaaSFunc{
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

func GetServerlessFunctionsGCF(ctx context.Context, apiHost string, auth string, auth_bucket_name string, auth_object_name string, minioClient *minio.Client, region string) ([]GCFFaaSFunc, error) {

	object, err := minioClient.GetObject(context.Background(), auth_bucket_name, auth_object_name, minio.GetObjectOptions{})
	if err != nil {
		log.G(ctx).Errorf(" minio auth_object_name GetObject failed: %v.\n", err)
		return nil, err
	}
	localFile, err := os.Create("/tmp/auth.json")
	if err != nil {
		log.G(ctx).Errorf(" cannot create auth.json file failed: %v.\n", err)
		return nil, err
	}
	if _, err = io.Copy(localFile, object); err != nil {
		log.G(ctx).Errorf("auth.json file save failed: %v.\n", err)
		return nil, err
	}
	jsonFile, err := os.Open("/tmp/auth.json")
	byteValue, _ := ioutil.ReadAll(jsonFile)

	if err != nil {
		log.G(ctx).Errorf("Reading auth json command failed: %v.\n", err)
		return nil, err
	} else {
		log.G(ctx).Infof("Reading authr command success: %v.\n", string(byteValue))
	}

	// we initialize our Users array
	var gcfPlatformAuth GCFPlatformAuth

	// we unmarshal our byteArray which contains our
	// jsonFile's content into 'users' which we defined above
	json.Unmarshal(byteValue, &gcfPlatformAuth)

	out, err := exec.Command("gcloud", "auth", "activate-service-account", gcfPlatformAuth.Client_email, "--key-file=/tmp/auth.json", "--project="+string(gcfPlatformAuth.Project_id)).Output()

	if err != nil {
		log.G(ctx).Errorf("Executing gcloud Cli register command failed: %v.\n", err)
		return nil, err
	} else {
		log.G(ctx).Infof("Executing gcloud Cli register command success: %v.\n", string(out))
	}

	if err != nil {
		log.G(ctx).Errorf("Executing gcloud Cli register command failed: %v.\n", err)
		return nil, err
	} else {
		log.G(ctx).Infof("Executing gcloud Cli register command success: %v.\n", string(out))
	}

	funcs_out, err := exec.Command("gcloud", "functions", "list", "--regions="+string(region)).Output()

	if err != nil {
		log.G(ctx).Errorf("Executing gcloud functions list command failed: %v.\n", err)
		return nil, err
	} else {
		log.G(ctx).Infof("Executing gcloud functions list command success: %v.\n", string(funcs_out))
		s := strings.Split(string(funcs_out), "\n")
		var actions []GCFFaaSFunc
		for i, v := range s {
			if i == 0 {
				continue
			}
			s1 := strings.Fields(v)
			log.G(ctx).Infof("Returned function: %v Length: %v.\n", s1, strconv.Itoa(len(s1)))
			if len(s1) > 0 {
				actions = append(actions, GCFFaaSFunc{
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

func FunctionToPod(action *GCFFaaSFunc, nodeName string) (*v1.Pod, error) {
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
