package linguist

import (
	"github.com/stretchr/testify/assert"

	_ "fmt"
	"testing"

	log "github.com/Sirupsen/logrus"
)

func TestLanguageDetection(t *testing.T) {
	if len(cudaSources) == 0 || len(thrustSources) == 0 {
		log.Panic("cannot find language detection test code")
	}
	assert.Equal(t, "CUDA", Detect(cudaSources[0]))
	assert.Equal(t, "Thrust", Detect(thrustSources[0]))
	assert.Equal(t, "OpenCL", Detect(openclSources[0]))
	assert.Equal(t, "OpenACC", Detect(openaccSources[0]))
	assert.Equal(t, "C++", Detect(cppSources[0]))
}
