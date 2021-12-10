# Tensorflow includes and defines
TF_CFLAGS = $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS = $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_CUDA = $(shell python -c 'import tensorflow as tf; print(int(tf.test.is_built_with_cuda()))')

#TF_FLAGS=-D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D_GLIBCXX_USE_CXX11_ABI=0 -g -Wall

# Dependencies
DEPDIR:=.d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS=-MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

# Define our sources, compiling CUDA code if it's enabled
ifeq ($(TF_CUDA), 1)
    SOURCES=$(wildcard *.cc *.cu)
else
    SOURCES=$(wildcard *.cc)
endif

# Define objects and shared_library
OBJECTS=$(addsuffix .o, $(basename $(SOURCES)))
LIBRARY=gelu.so

# Compiler flags
CPPFLAGS =-std=c++11 -DGOOGLE_CUDA=$(TF_CUDA) $(TF_CFLAGS) -fPIC -fopenmp \
         -O2 -march=native -mtune=native -I /usr/local/cuda/include

NVCCFLAGS =-std=c++11 -DGOOGLE_CUDA=$(TF_CUDA) $(TF_CFLAGS) $(INCLUDES) \
        -x cu --compiler-options "-fPIC"  -lineinfo --expt-relaxed-constexpr -DNDEBUG  -use_fast_math -arch=sm_70
#NVCCFLAGS += -DDOTIMING #print timing for each Kernel Execution
#NVCCFLAGS += -DCHECKCUDAERROR  #Add CUDA CHECK ERROR after each  Kernel (trouble shooting)
LDFLAGS = -fPIC -fopenmp $(TF_LFLAGS)

ifeq ($(TF_CUDA), 1)
    LDFLAGS := $(LDFLAGS) -L /usr/local/cuda/lib64
    LDFLAGS := $(LDFLAGS) -lcuda -lcudart -lcusolver
endif

# Compiler directives
COMPILE.cc = g++ $(DEPFLAGS) $(CPPFLAGS) -I include -c
COMPILE.nvcc = nvcc --compiler-options " $(DEPFLAGS)" $(NVCCFLAGS) -I include -c

all : $(LIBRARY)

%.o : %.cc
	$(COMPILE.cc) $<

%.o : %.cu
	$(COMPILE.nvcc) $<

clean :
	rm -f $(OBJECTS) $(LIBRARY)

$(LIBRARY) : $(OBJECTS)
	g++  -shared $(OBJECTS) -o $(LIBRARY) $(LDFLAGS)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))