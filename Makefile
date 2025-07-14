# Makefile for Bine Trees Collective Operations

# Compiler and flags
CC = mpicc
CFLAGS = -Wall -Wextra -O3 -std=c99
LIBS = -lm

# Directories
SRC_DIR = src
SRC_CARTESIAN_DIR = src_cartesian
TEST_DIR = test

# Source files
GENERAL_SOURCES = $(wildcard $(SRC_DIR)/*.c)
CARTESIAN_SOURCES = $(wildcard $(SRC_CARTESIAN_DIR)/*.c)
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)

# Object files
GENERAL_OBJECTS = $(GENERAL_SOURCES:.c=.o)
CARTESIAN_OBJECTS = $(CARTESIAN_SOURCES:.c=.o)

# Libraries
GENERAL_LIB = libbine_general.a
CARTESIAN_LIB = libbine_cartesian.a

# Test executable
TEST_EXEC = test.exe

.PHONY: all clean test

all: $(GENERAL_LIB) $(CARTESIAN_LIB) $(TEST_EXEC)

test:
	./bin/test.sh

# Compile general algorithm objects
$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Compile cartesian algorithm objects  
$(SRC_CARTESIAN_DIR)/%.o: $(SRC_CARTESIAN_DIR)/%.c
	$(CC) $(CFLAGS) -I$(SRC_DIR) -I$(SRC_CARTESIAN_DIR) -c $< -o $@

# Create libraries
$(GENERAL_LIB): $(GENERAL_OBJECTS)
	ar rcs lib/$@ $^

$(CARTESIAN_LIB): $(CARTESIAN_OBJECTS)
	ar rcs lib/$@ $^

# Build test executable
$(TEST_EXEC): $(TEST_DIR)/test.c $(GENERAL_LIB)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -I$(SRC_CARTESIAN_DIR) $< lib/$(GENERAL_LIB) $(LIBS) -o bin/$@

clean:
	rm -f $(SRC_DIR)/*.o $(SRC_CARTESIAN_DIR)/*.o lib/*.a bin/$(TEST_EXEC)
