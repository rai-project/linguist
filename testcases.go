//go:generate rice embed-go
package linguist

import "github.com/GeertJohan/go.rice"

var (
	cudaSources    []string
	thrustSources  []string
	openclSources  []string
	openaccSources []string
	cppSources     []string
)

func init() {
	if box, err := rice.FindBox("_testcases"); err == nil {
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
