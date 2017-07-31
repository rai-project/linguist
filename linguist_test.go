package linguist

import (
	"github.com/stretchr/testify/assert"

	_ "fmt"
	"testing"

	log "github.com/sirupsen/logrus"
)

func TestLanguageDetection(t *testing.T) {
	if len(cudaSources) == 0 || len(thrustSources) == 0 {
		log.Panic("cannot find language detection test code")
	}
	assert.Equal(t, "CUDA", Detect(cudaSources[0]))
	assert.Equal(t, "Thrust", Detect(thrustSources[0]))
	assert.Equal(t, "OpenCL", Detect(openclSources[0]))
	assert.Equal(t, "OpenACC", Detect(openaccSources[0]))
	assert.Equal(t, "CUDA", Detect(`__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index < len) {
        out[index] = in1[index] + in2[index];
	} else {
		printf("%d\n", index);
	}
}`))
}
