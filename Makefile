CUDA_PATH   = /usr/local/cuda
NVCC        = $(CUDA_PATH)/bin/nvcc
SM          = 86
CCFLAG      = -std=c++14 -O3 -gencode=arch=compute_$(SM),code=sm_$(SM)
SOFLAG      = $(CCFLAG)
CUDA_HOME   = /usr/local/cuda-10.2
INCLUDE     = -I. -I$(CUDA_HOME)/include/ -Iinclude/
LDFLAG      = -lz -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lcudnn


# SOURCE_CU   = $(shell find . -regextype posix-extended -regex '.*\.(cu|cuh)')
SOURCE_CU   = $(shell find . -name '*.cu')
SOURCE_CPP  = $(shell find . -name '*.cpp')
# SOURCE_PY   = $(shell find . -name '*.py')
CU_OBJ      = $(SOURCE_CU:.cu=.cu.o)
CPP_OBJ     = $(SOURCE_CPP:.cpp=.cpp.o)

all: $(CU_OBJ) $(CPP_OBJ)
	$(NVCC) $(SOFLAG) -o ./test_conv_relu $^ $(LDFLAG)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(CUDA_HOME)/lib64 ./test_conv_relu
	@echo "test Python code"
	python3 test.py
# $(SOURCE_CU:%.cu=%.so)

# %.so: %.cu.o
# 	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

%.cpp.o: %.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

%.cu.o: %.cu
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<
	
.PHONY: test
test:
	make clean
	make
	python $(SOURCE_PY)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.plan ./test_conv_relu ./*.bin ./*.npy ./*.npz

