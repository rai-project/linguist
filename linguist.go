package linguist

import (
	"strings"

	"github.com/generaltso/linguist"
	"gitlab.com/abduld/wgx/pkg/codecompile/compiletarget/target"
)

const KeywordThreashold = 3

var keywords struct {
	CUDA    []string
	Thrust  []string
	CppAMP  []string
	OpenCL  []string
	OpenACC []string
}

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

func Language(contents string) target.Language {
	lang := "cpp"

	lang = linguist.LanguageByContents([]byte(contents), linguist.LanguageHints(""))

	if lang == "C" {
		src := string(contents)
		if hasKeywords(src, keywords.Thrust) {
			lang = "thrust"
		} else if hasKeywords(src, keywords.CUDA) {
			lang = "cuda"
		} else if hasKeywords(src, keywords.OpenACC) {
			lang = "openacc"
		} else if hasKeywords(src, keywords.OpenCL) {
			lang = "opencl"
		} else if hasKeywords(src, keywords.CppAMP) {
			lang = "cppamp"
		} else {
			lang = "cpp"
		}
	}

	return target.MustGetLanguage(lang)
}

func init() {
	keywords.Thrust = []string{
		"thrust::",
		"thrust::device_vector",
		"make_zip_iterator",
		"thrust::transform",
	}
	keywords.OpenACC = []string{
		"#pragma ",
		"pragma acc",
		"acc kernels ",
	}
	keywords.OpenCL = []string{
		"__kernel ",
		"clGetPlatformIDs",
		"clBuildProgram",
		"clCreateKernel",
	}
	keywords.CUDA = []string{
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
}
