# Makefile for Bine Trees Collective Operations

# Compiler and flags
CC = mpicc
CFLAGS = -Wall -Wextra -O3 -std=c99
LIBS = -lm

# Directories
SRC_DIR = src
SRC_CARTESIAN_DIR = src_cartesian

# Source files
GENERAL_SOURCES = $(wildcard $(SRC_DIR)/*.c)
CARTESIAN_SOURCES = $(wildcard $(SRC_CARTESIAN_DIR)/*.c)

# Object files
GENERAL_OBJECTS = $(GENERAL_SOURCES:.c=.o)
CARTESIAN_OBJECTS = $(CARTESIAN_SOURCES:.c=.o)

# Libraries
GENERAL_LIB = libbine_general.a
CARTESIAN_LIB = libbine_cartesian.a

.PHONY: all clean

all: $(GENERAL_LIB) $(CARTESIAN_LIB)

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

clean:
	rm -f $(SRC_DIR)/*.o $(SRC_CARTESIAN_DIR)/*.o lib/*.a
