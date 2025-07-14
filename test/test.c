#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../src/utils.h"

// Forward declarations of custom algorithms
int bine_bcast_small(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm);
int bine_bcast_large(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm);

// Test configuration structure
typedef struct {
    char collective[64];    // Name of collective operation (bcast, reduce, etc.)
    char algorithm[64];     // Name of algorithm (bine_small, etc.)
    int count;              // Number of elements
    char datatype[16];      // Datatype name (int, double, float, char)
    int root;               // Root rank for operations that need it
    int verbose;            // Verbose output flag
} test_config_t;

// Function pointer types for collective operations
typedef int (*bcast_func_t)(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm);

// Structure to hold function mappings
typedef struct {
    char name[64];
    bcast_func_t func;
} algorithm_map_t;

// Algorithm registry for broadcast
static algorithm_map_t bcast_algorithms[] = {
    {"bine_bcast_small", bine_bcast_small},
    {"bine_bcast_large", bine_bcast_large},
    {"", NULL}  // Sentinel
};

/**
 * @brief Get MPI datatype from string name
 */
MPI_Datatype get_mpi_datatype(const char* type_name, int* type_size) {
    if (strcmp(type_name, "int") == 0) {
        *type_size = sizeof(int);
        return MPI_INT;
    } else if (strcmp(type_name, "double") == 0) {
        *type_size = sizeof(double);
        return MPI_DOUBLE;
    } else if (strcmp(type_name, "float") == 0) {
        *type_size = sizeof(float);
        return MPI_FLOAT;
    } else if (strcmp(type_name, "char") == 0) {
        *type_size = sizeof(char);
        return MPI_CHAR;
    }
    return MPI_DATATYPE_NULL;
}

/**
 * @brief Initialize test data with rank-specific values
 */
void init_test_data(void* buf, int count, MPI_Datatype dtype, int rank) {
    if (dtype == MPI_INT) {
        int* data = (int*)buf;
        for (int i = 0; i < count; i++) {
            data[i] = rank * 1000 + i;
        }
    } else if (dtype == MPI_DOUBLE) {
        double* data = (double*)buf;
        for (int i = 0; i < count; i++) {
            data[i] = rank * 1000.0 + i * 0.1;
        }
    } else if (dtype == MPI_FLOAT) {
        float* data = (float*)buf;
        for (int i = 0; i < count; i++) {
            data[i] = rank * 1000.0f + i * 0.1f;
        }
    } else if (dtype == MPI_CHAR) {
        char* data = (char*)buf;
        for (int i = 0; i < count; i++) {
            data[i] = (char)((rank + i) % 128);
        }
    }
}

/**
 * @brief Compare two buffers for equality
 */
int compare_buffers(void* buf1, void* buf2, int count, MPI_Datatype dtype, double tolerance) {
    if (dtype == MPI_INT) {
        int* data1 = (int*)buf1;
        int* data2 = (int*)buf2;
        for (int i = 0; i < count; i++) {
            if (data1[i] != data2[i]) return 0;
        }
    } else if (dtype == MPI_DOUBLE) {
        double* data1 = (double*)buf1;
        double* data2 = (double*)buf2;
        for (int i = 0; i < count; i++) {
            if (fabs(data1[i] - data2[i]) > tolerance) return 0;
        }
    } else if (dtype == MPI_FLOAT) {
        float* data1 = (float*)buf1;
        float* data2 = (float*)buf2;
        for (int i = 0; i < count; i++) {
            if (fabsf(data1[i] - data2[i]) > tolerance) return 0;
        }
    } else if (dtype == MPI_CHAR) {
        char* data1 = (char*)buf1;
        char* data2 = (char*)buf2;
        for (int i = 0; i < count; i++) {
            if (data1[i] != data2[i]) return 0;
        }
    }
    return 1;
}

/**
 * @brief Print buffer contents (for debugging)
 */
void print_buffer(void* buf, int count, MPI_Datatype dtype, int rank, const char* label) {
    printf("Rank %d - %s: ", rank, label);
    if (dtype == MPI_INT) {
        int* data = (int*)buf;
        for (int i = 0; i < count && i < 10; i++) {
            printf("%d ", data[i]);
        }
    } else if (dtype == MPI_DOUBLE) {
        double* data = (double*)buf;
        for (int i = 0; i < count && i < 10; i++) {
            printf("%.2f ", data[i]);
        }
    } else if (dtype == MPI_FLOAT) {
        float* data = (float*)buf;
        for (int i = 0; i < count && i < 10; i++) {
            printf("%.2f ", data[i]);
        }
    } else if (dtype == MPI_CHAR) {
        char* data = (char*)buf;
        for (int i = 0; i < count && i < 10; i++) {
            printf("%c ", data[i]);
        }
    }
    if (count > 10) printf("...");
    printf("\n");
}

/**
 * @brief Find algorithm function by name
 */
bcast_func_t find_bcast_algorithm(const char* name) {
    for (int i = 0; bcast_algorithms[i].func != NULL; i++) {
        if (strcmp(bcast_algorithms[i].name, name) == 0) {
            return bcast_algorithms[i].func;
        }
    }
    return NULL;
}

/**
 * @brief Test broadcast correctness
 */
int test_bcast_correctness(test_config_t* config) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get datatype and size
    int type_size;
    MPI_Datatype dtype = get_mpi_datatype(config->datatype, &type_size);
    if (dtype == MPI_DATATYPE_NULL) {
        if (rank == 0) printf("Error: Unknown datatype '%s'\n", config->datatype);
        return 1;
    }
    
    // Find algorithm
    bcast_func_t custom_bcast = find_bcast_algorithm(config->algorithm);
    if (custom_bcast == NULL) {
        if (rank == 0) printf("Error: Unknown algorithm '%s'\n", config->algorithm);
        return 1;
    }
    
    // Allocate buffers
    void* ref_buf = malloc(config->count * type_size);
    void* test_buf = malloc(config->count * type_size);
    if (!ref_buf || !test_buf) {
        if (rank == 0) printf("Error: Memory allocation failed\n");
        return 1;
    }
    
    // Initialize data on root
    if (rank == config->root) {
        init_test_data(ref_buf, config->count, dtype, config->root);
        memcpy(test_buf, ref_buf, config->count * type_size);
    } else {
        // Initialize with different data on non-root ranks
        init_test_data(ref_buf, config->count, dtype, rank);
        init_test_data(test_buf, config->count, dtype, rank);
    }
    
    if (config->verbose && rank == 0) {
        printf("Testing %s with algorithm %s\n", config->collective, config->algorithm);
        printf("Count: %d, Datatype: %s, Root: %d\n", 
               config->count, config->datatype, config->root);
    }
    
    // Test reference MPI implementation
    double start_time = MPI_Wtime();
    int ref_result = MPI_Bcast(ref_buf, config->count, dtype, config->root, MPI_COMM_WORLD);
    double ref_time = MPI_Wtime() - start_time;
    
    // Test custom implementation
    start_time = MPI_Wtime();
    int test_result = custom_bcast(test_buf, config->count, dtype, config->root, MPI_COMM_WORLD);
    double test_time = MPI_Wtime() - start_time;
    
    // Check for errors
    if (ref_result != MPI_SUCCESS) {
        if (rank == 0) printf("Error: MPI_Bcast failed with code %d\n", ref_result);
        free(ref_buf);
        free(test_buf);
        return 1;
    }
    
    if (test_result != MPI_SUCCESS) {
        if (rank == 0) printf("Error: Custom algorithm failed with code %d\n", test_result);
        free(ref_buf);
        free(test_buf);
        return 1;
    }
    
    // Compare results
    double tolerance = (dtype == MPI_DOUBLE) ? 1e-12 : (dtype == MPI_FLOAT) ? 1e-6 : 0.0;
    int match = compare_buffers(ref_buf, test_buf, config->count, dtype, tolerance);
    
    // Gather results from all ranks
    int global_match;
    MPI_Allreduce(&match, &global_match, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    
    if (config->verbose) {
        print_buffer(ref_buf, config->count, dtype, rank, "MPI_Bcast");
        print_buffer(test_buf, config->count, dtype, rank, config->algorithm);
    }
    
    // Report results
    if (rank == 0) {
        printf("Test %s: %s\n", global_match ? "PASSED" : "FAILED", config->algorithm);
        if (config->verbose) {
            printf("  MPI_Bcast time: %.6f seconds\n", ref_time);
            printf("  %s time: %.6f seconds\n", config->algorithm, test_time);
            printf("  Speedup: %.2fx\n", ref_time / test_time);
        }
    }
    
    free(ref_buf);
    free(test_buf);
    return global_match ? 0 : 1;
}

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -c, --collective <name>   Collective operation (bcast, reduce, etc.) [default: bcast]\n");
    printf("  -a, --algorithm <name>    Algorithm name [default: bine_small]\n");
    printf("  -n, --count <number>      Number of elements [default: 1000]\n");
    printf("  -t, --type <datatype>     Data type (int, double, float, char) [default: int]\n");
    printf("  -r, --root <rank>         Root rank [default: 0]\n");
    printf("  -v, --verbose             Verbose output\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nAvailable algorithms:\n");
    printf("  Broadcast: bine_small\n");
    printf("\nExample:\n");
    printf("  mpirun -np 4 %s -c bcast -a bine_small -n 100 -t double -r 0 -v\n", program_name);
}

/**
 * @brief Parse command line arguments
 */
int parse_args(int argc, char* argv[], test_config_t* config) {
    // Set defaults
    strcpy(config->collective, "bcast");
    strcpy(config->algorithm, "bine_small");
    config->count = 1000;
    strcpy(config->datatype, "int");
    config->root = 0;
    config->verbose = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--collective") == 0) {
            if (++i >= argc) return 1;
            strcpy(config->collective, argv[i]);
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--algorithm") == 0) {
            if (++i >= argc) return 1;
            strcpy(config->algorithm, argv[i]);
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--count") == 0) {
            if (++i >= argc) return 1;
            config->count = atoi(argv[i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--type") == 0) {
            if (++i >= argc) return 1;
            strcpy(config->datatype, argv[i]);
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--root") == 0) {
            if (++i >= argc) return 1;
            config->root = atoi(argv[i]);
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            config->verbose = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            return 1;
        } else {
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    test_config_t config;
    if (parse_args(argc, argv, &config)) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    // Validate root rank
    if (config.root < 0 || config.root >= size) {
        if (rank == 0) printf("Error: Root rank %d is invalid (must be 0-%d)\n", 
                              config.root, size - 1);
        MPI_Finalize();
        return 1;
    }
    
    int result = 0;
    
    // Run the appropriate test
    if (strcmp(config.collective, "bcast") == 0) {
        result = test_bcast_correctness(&config);
    } else {
        if (rank == 0) printf("Error: Unsupported collective '%s'\n", config.collective);
        result = 1;
    }
    
    MPI_Finalize();
    return result;
}
