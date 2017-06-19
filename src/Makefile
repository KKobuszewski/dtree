# GPU COMPUTE CAPABILITY
GPU_ARCH = -gencode arch=compute_52,code=compute_52


# C COMPILATION
CC         = gcc
CC_FLAGS   = -Wall -Wundef -m64 -march=native -mtune=native -O3 -msse4 -ftree-vectorize -ftree-vectorizer-verbose=1 -fopenmp -fPIC
CC_INC     = -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I.
CC_DEFS    = 

# C++ COMPILATION
CXX        = g++
CXX_FLAGS  = -Wall -Wundef -m64 -march=native -mtune=native -O3 -msse4 -ftree-vectorize -fopenmp -fPIC
CXX_INC    = -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I.
CXX_DEFS   = 

# CUDA COMPILATION
NVCC       = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -m64 -O3 $(GPU_ARCH) -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -Xcompiler "-fPIC -fopenmp -Wall -m64 -march=native -mtune=native -O3 -msse4 -ftree-vectorize"
NVCC_INC   = -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I.
NVCC_DEFS  = 

# LINKING
LD         = /usr/local/cuda/bin/nvcc
LD_FLAGS   = $(GPU_ARCH)
LIBS       = -L./usr/local/cuda/lib64 -lcudart -lcufft -lcublas -lgomp -lpthread -lm



partition:
	$(NVCC) -std=c++11 -o partition.exe partition.cu $(GPU_ARCH) $(NVCC_FLAGS) $(NVCC_INC)


best_split:
	$(NVCC) -std=c++11 -o best_split.exe best_split.cu $(GPU_ARCH) $(NVCC_FLAGS) $(NVCC_INC) --keep --keep-dir="keep_ptx/" -Xptxas "--warn-on-spills --warn-on-local-memory-usage"


construct:
	$(NVCC) -std=c++11 -o construct.exe construct.cu $(GPU_ARCH) $(NVCC_FLAGS) $(NVCC_INC) --keep --keep-dir="keep_ptx/" -Xptxas "--warn-on-spills --warn-on-local-memory-usage"