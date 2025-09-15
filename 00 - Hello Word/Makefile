# Compiler
NVCC = nvcc

# Flags
NVCC_FLAGS = -std=c++11

# Target executable
TARGET = hello_world

# Source files
SOURCES = hello_world.cu

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