# System detection
UNAME_M := $(shell uname -m)
OS_NAME := $(shell uname -s)

# Defaults (Cross-compilation)
CXX = aarch64-linux-gnu-g++
CXXFLAGS = -O3 -march=armv8-a+simd -Wall -Wextra -Iinclude
LDFLAGS = -static
EMULATOR = qemu-aarch64 -L /usr/aarch64-linux-gnu

# Native ARM64 (macOS/Linux) override
ifeq ($(UNAME_M),arm64)
    CXX = g++
    ifeq ($(OS_NAME),Darwin)
        BREW_PREFIX := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew)
        CXXFLAGS = -O3 -Wall -Wextra -Iinclude -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/include
        LDFLAGS = -L$(BREW_PREFIX)/lib -lomp
    else
        CXXFLAGS = -O3 -Wall -Wextra -Iinclude -fopenmp
        LDFLAGS = -fopenmp
    endif
    EMULATOR = 
endif

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
TEST_DIR = tests

# Sources
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Targets
all: $(BUILD_DIR)/hello_neon

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: benchmarks/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/hello_neon: $(BUILD_DIR)/hello_neon.o
	$(CXX) $(LDFLAGS) $^ -o $@

# Test targets
test_hello: $(BUILD_DIR)/hello_neon
	$(EMULATOR) ./$(BUILD_DIR)/hello_neon

test_phase1: $(BUILD_DIR)/test_phase1
	$(EMULATOR) ./$(BUILD_DIR)/test_phase1

test_phase2: $(BUILD_DIR)/test_phase2
	$(EMULATOR) ./$(BUILD_DIR)/test_phase2

bench_dot: $(BUILD_DIR)/bench_dot
	$(EMULATOR) ./$(BUILD_DIR)/bench_dot

test_phase3: $(BUILD_DIR)/test_phase3
	$(EMULATOR) ./$(BUILD_DIR)/test_phase3

bench_gemm: $(BUILD_DIR)/bench_gemm
	$(EMULATOR) ./$(BUILD_DIR)/bench_gemm

$(BUILD_DIR)/test_phase1: $(BUILD_DIR)/test_phase1.o $(BUILD_DIR)/vector_math.o
	$(CXX) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/test_phase2: $(BUILD_DIR)/test_phase2.o $(BUILD_DIR)/dot_product.o
	$(CXX) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/test_phase3: $(BUILD_DIR)/test_phase3.o $(BUILD_DIR)/gemm.o
	$(CXX) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/bench_dot: $(BUILD_DIR)/bench_dot.o $(BUILD_DIR)/dot_product.o
	$(CXX) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/bench_gemm: $(BUILD_DIR)/bench_gemm.o $(BUILD_DIR)/gemm.o
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean test_hello test_phase1 test_phase2 test_phase3 bench_dot bench_gemm
