package linguist

import (
	"strings"

	"github.com/generaltso/linguist"
)

const KeywordThreashold = 3

var (
	ThrustKeywords = []string{
		"thrust::",
		"thrust::device_vector",
		"make_zip_iterator",
		"thrust::transform",
	}
	OpenACCKeywords = []string{
		"#pragma ",
		"pragma acc",
		"acc kernels ",
	}
	OpenCLKeywords = []string{
		"__kernel ",
		"clGetPlatformIDs",
		"clBuildProgram",
		"clCreateKernel",
	}
	CUDAKeywords = []string{
		"cudaError_t",
		"cudaDeviceProp",
		"cudaMemcpyKind",
		"__device__",
		"__global__",
		"__host__",
		"__constant__",
		"__shared__",
		"__inline__",
		"__align__",
		"__thread__",
		"__constant__",
		"__import__",
		"__export__",
		"__location__",
		"<<<",
		">>>",
	}
	// todo: figure out what are common keywords
	CPPAMPKeywords = []string{}
)

func hasKeywords(str string, keywords []string) bool {
	count := 0
	for _, keyword := range keywords {
		if strings.Contains(str, keyword) {
			count++
			if count >= KeywordThreashold {
				return true
			}
		}
	}
	return false
}

func Detect(contents string) string {
	lang := "CPP"

	lang = linguist.LanguageByContents([]byte(contents), linguist.LanguageHints(""))

	if lang == "C" {
		src := string(contents)
		if hasKeywords(src, ThrustKeywords) {
			lang = "Thrust"
		} else if hasKeywords(src, CUDAKeywords) {
			lang = "CUDA"
		} else if hasKeywords(src, OpenACCKeywords) {
			lang = "OpenACC"
		} else if hasKeywords(src, OpenCLKeywords) {
			lang = "OpenCL"
		} else if hasKeywords(src, CPPAMPKeywords) {
			lang = "CPPAMP"
		} else {
			lang = "CPP"
		}
	}

	return lang
}
