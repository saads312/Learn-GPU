# Compiler
NVCC = nvcc

# Flags
NVCC_FLAGS = -std=c++11 -O2 -arch=sm_50

# Target executable
TARGET = matrix_mult

# Source files
SOURCES = matrix_mult.cu

# Header files
HEADERS = matrix_utils.h

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)

# Clean rule
clean:
	rm -f $(TARGET)

# Run rule
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run