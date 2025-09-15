# Compiler
NVCC = nvcc

# Flags
NVCC_FLAGS = -std=c++11 -O2

# Target executable
TARGET = vector_add

# Source files
SOURCES = vector_add.cu

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)

# Clean rule
clean:
	rm -f $(TARGET)

# Run rule
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run