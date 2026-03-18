.PHONY: all compile lenia

all: compile link

HEADERS_COMMON = common/headers
HEADERS_KERNELS_2D = kernels/headers/2d
HEADERS_KERNELS_3D = kernels/headers/3d

HEADERS_COMMON_FILES = $(HEADERS_COMMON)/*
HEADERS_KERNELS_2D_FILES = $(HEADERS_KERNELS_2D)/*
HEADERS_KERNELS_3D_FILES = $(HEADERS_KERNELS_3D)/*

SOURCES_COMMON = common/src/*
SOURCES_KERNELS_2D = kernels/src/2d/*
SOURCES_KERNELS_3D = kernels/src/3d/*
SOURCE_MAIN = lenia.cu

ALL_SOURCES = $(SOURCES_COMMON) $(SOURCES_KERNELS_2D) $(SOURCES_KERNELS_3D) $(SOURCE_MAIN)
ALL_HEADERS = $(HEADERS_COMMON_FILES) $(HEADERS_KERNELS_2D_FILES) $(HEADERS_KERNELS_3D_FILES)
ALL = $(ALL_HEADERS) $(ALL_SOURCES)

compile: $(ALL)
	@echo "All headers:"
	@for header in $(ALL_HEADERS); do echo $$header; done
	@echo ""

	@echo "All sources:"
	@for source in $(ALL_SOURCES); do echo $$source; done
	@echo ""

	@mkdir -p build
	@rm build/*

	nvcc -Wno-deprecated-gpu-targets \
		-Xcompiler -fopenmp \
		-dc -g -G \
		-I $(HEADERS_COMMON) \
		-I $(HEADERS_KERNELS_2D) \
		-I $(HEADERS_KERNELS_3D) \
		-odir build \
		$(ALL_SOURCES)

link: compile
	nvcc -Wno-deprecated-gpu-targets \
		-lgomp \
		-g \
		-G \
		-o \
		lenia \
		$(shell ls build/*.o)
