package linguist_test

import (
	"github.com/GeertJohan/go.rice"
	"github.com/stretchr/testify/assert"

	_ "fmt"
	"testing"

	log "github.com/Sirupsen/logrus"
	"github.com/abduld/webgpu-grader/src/backend/code/detect"
)

var (
	cudaSources    []string
	thrustSources  []string
	openclSources  []string
	openaccSources []string
	cppSources     []string
)

func TestLanguageDetection(t *testing.T) {
	if len(cudaSources) == 0 || len(thrustSources) == 0 {
		log.Panic("cannot find language detection test code")
	}
	assert.Equal(t, "CUDA", detect.Language(cudaSources[0]))
	assert.Equal(t, "Thrust", detect.Language(thrustSources[0]))
	assert.Equal(t, "OpenCL", detect.Language(openclSources[0]))
	assert.Equal(t, "OpenACC", detect.Language(openaccSources[0]))
	assert.Equal(t, "CPP", detect.Language(cppSources[0]))
}

func init() {
	if box, err := rice.FindBox("testcases"); err == nil {
		cudaSources = []string{
			box.MustString("cuda_sample.cu"),
		}
		thrustSources = []string{
			box.MustString("thrust_sample.cu"),
		}
		openclSources = []string{
			box.MustString("opencl_sample.cpp"),
		}
		openaccSources = []string{
			box.MustString("openacc_sample.cpp"),
		}
		cppSources = []string{
			box.MustString("cpp_sample.cpp"),
		}
	}
}
