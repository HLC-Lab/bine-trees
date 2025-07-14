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
int bine_reduce_small(const void *sendbuf, void *recvbuf, size_t count, MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm);
int bine_reduce_large(const void *sendbuf, void *recvbuf, size_t count, MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm);
int bine_gather_any(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int bine_scatter_any(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int bine_alltoall_small(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int bine_allgather_block_by_block(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, MPI_Comm comm);

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
typedef int (*reduce_func_t)(const void *sendbuf, void *recvbuf, size_t count, MPI_Datatype dtype, MPI_Op op, int root, MPI_Comm comm);
typedef int (*gather_func_t)(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
typedef int (*scatter_func_t)(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
typedef int (*alltoall_func_t)(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, MPI_Comm comm);
typedef int (*allgather_func_t)(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, MPI_Comm comm);

// Generic function pointer for algorithm lookup

// Structure to hold function mappings
typedef struct {
    char name[64];
    void* func;
} algorithm_map_t;

// Algorithm registries
static algorithm_map_t bcast_algorithms[] = {
    {"small", (bcast_func_t)bine_bcast_small},
    {"large", (bcast_func_t)bine_bcast_large},
    {"", NULL}
};

static algorithm_map_t reduce_algorithms[] = {
    {"small", (reduce_func_t)bine_reduce_small},
    {"large", (reduce_func_t)bine_reduce_large},
    {"", NULL}
};

static algorithm_map_t gather_algorithms[] = {
    {"any", (gather_func_t)bine_gather_any},
    {"", NULL}
};

static algorithm_map_t scatter_algorithms[] = {
    {"any", (scatter_func_t)bine_scatter_any},
    {"", NULL}
};

static algorithm_map_t alltoall_algorithms[] = {
    {"small", (alltoall_func_t)bine_alltoall_small},
    {"", NULL}
};

static algorithm_map_t allgather_algorithms[] = {
    {"block_by_block", (allgather_func_t)bine_allgather_block_by_block},
    {"", NULL}
};

// Collective operation types
typedef enum {
    COLLECTIVE_BCAST,
    COLLECTIVE_REDUCE,
    COLLECTIVE_GATHER,
    COLLECTIVE_SCATTER,
    COLLECTIVE_ALLTOALL,
    COLLECTIVE_ALLGATHER,
    COLLECTIVE_UNKNOWN
} collective_type_t;

/**
 * @brief Get collective type from string
 */
collective_type_t get_collective_type(const char* name) {
    if (strcmp(name, "bcast") == 0) return COLLECTIVE_BCAST;
    if (strcmp(name, "reduce") == 0) return COLLECTIVE_REDUCE;
    if (strcmp(name, "gather") == 0) return COLLECTIVE_GATHER;
    if (strcmp(name, "scatter") == 0) return COLLECTIVE_SCATTER;
    if (strcmp(name, "alltoall") == 0) return COLLECTIVE_ALLTOALL;
    if (strcmp(name, "allgather") == 0) return COLLECTIVE_ALLGATHER;
    return COLLECTIVE_UNKNOWN;
}

/**
 * @brief Generic algorithm finder
 */
void* find_algorithm(const char* name, algorithm_map_t* algorithms) {
    for (int i = 0; algorithms[i].func != NULL; i++) {
        if (strcmp(algorithms[i].name, name) == 0) {
            return algorithms[i].func;
        }
    }
    return NULL;
}

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
            data[i] = rank * 1000.0 + i * 0.25;  
        }
    } else if (dtype == MPI_FLOAT) {
        float* data = (float*)buf;
        for (int i = 0; i < count; i++) {
            data[i] = rank * 1000.0f + i * 0.25f;
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
 * @brief Get MPI operation for reduce tests
 */
MPI_Op get_mpi_operation(MPI_Datatype dtype) {
    return (dtype == MPI_CHAR) ? MPI_MIN : MPI_SUM;
}

/**
 * @brief Common test setup and validation
 */
int test_setup(test_config_t* config, int* rank, int* size, int* type_size, MPI_Datatype* dtype) {
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
    
    *dtype = get_mpi_datatype(config->datatype, type_size);
    if (*dtype == MPI_DATATYPE_NULL) {
        if (*rank == 0) printf("Error: Unknown datatype '%s'\n", config->datatype);
        return 1;
    }
    
    if (config->verbose && *rank == 0) {
        printf("Testing %s with algorithm %s\n", config->collective, config->algorithm);
        printf("Count: %d, Datatype: %s, Root: %d\n", 
               config->count, config->datatype, config->root);
    }
    
    return 0;
}

/**
 * @brief Common error checking for MPI operations
 */
int check_mpi_result(int result, const char* operation, int rank) {
    if (result != MPI_SUCCESS) {
        if (rank == 0) printf("Error: %s failed with code %d\n", operation, result);
        return 1;
    }
    return 0;
}

/**
 * @brief Common result reporting
 */
void report_results(int match, const char* algorithm, double ref_time, double test_time, int verbose, int rank) {
    if (rank == 0) {
        printf("Test %s: %s\n", match ? "PASSED" : "FAILED", algorithm);
        if (verbose) {
            printf("  Reference time: %.6f seconds\n", ref_time);
            printf("  %s time: %.6f seconds\n", algorithm, test_time);
            printf("  Speedup: %.2fx\n", ref_time / test_time);
        }
    }
}

/**
 * @brief Test broadcast correctness
 */
int test_bcast_correctness(test_config_t* config) {
    int rank, size, type_size;
    MPI_Datatype dtype;
    
    if (test_setup(config, &rank, &size, &type_size, &dtype)) return 1;
    
    // Find algorithm
    bcast_func_t custom_bcast = (bcast_func_t)find_algorithm(config->algorithm, bcast_algorithms);
    if (!custom_bcast) {
        if (rank == 0) printf("Error: Unknown algorithm '%s'\n", config->algorithm);
        return 1;
    }
    
    // Allocate buffers
    void* ref_buf = malloc(config->count * type_size);
    void* test_buf = malloc(config->count * type_size);
    if (!ref_buf || !test_buf) {
        if (rank == 0) printf("Error: Memory allocation failed\n");
        free(ref_buf); free(test_buf);
        return 1;
    }
    
    // Initialize data
    if (rank == config->root) {
        init_test_data(ref_buf, config->count, dtype, config->root);
        memcpy(test_buf, ref_buf, config->count * type_size);
    } else {
        init_test_data(ref_buf, config->count, dtype, rank);
        init_test_data(test_buf, config->count, dtype, rank);
    }
    
    // Test reference implementation
    double start_time = MPI_Wtime();
    int ref_result = MPI_Bcast(ref_buf, config->count, dtype, config->root, MPI_COMM_WORLD);
    double ref_time = MPI_Wtime() - start_time;
    
    // Test custom implementation
    start_time = MPI_Wtime();
    int test_result = custom_bcast(test_buf, config->count, dtype, config->root, MPI_COMM_WORLD);
    double test_time = MPI_Wtime() - start_time;
    
    // Check for errors
    if (check_mpi_result(ref_result, "MPI_Bcast", rank) || 
        check_mpi_result(test_result, "Custom bcast", rank)) {
        free(ref_buf); free(test_buf);
        return 1;
    }
    
    // Compare and report results
    double tolerance = (dtype == MPI_DOUBLE) ? 1e-12 : (dtype == MPI_FLOAT) ? 1e-6 : 0.0;
    int match = compare_buffers(ref_buf, test_buf, config->count, dtype, tolerance);
    
    int global_match;
    MPI_Allreduce(&match, &global_match, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    
    if (config->verbose) {
        print_buffer(ref_buf, config->count, dtype, rank, "MPI_Bcast");
        print_buffer(test_buf, config->count, dtype, rank, config->algorithm);
    }
    
    report_results(global_match, config->algorithm, ref_time, test_time, config->verbose, rank);
    
    free(ref_buf); free(test_buf);
    return global_match ? 0 : 1;
}

/**
 * @brief Test reduce correctness
 */
int test_reduce_correctness(test_config_t* config) {
    int rank, size, type_size;
    MPI_Datatype dtype;
    
    if (test_setup(config, &rank, &size, &type_size, &dtype)) return 1;
    
    // Get operation and find algorithm
    MPI_Op op = get_mpi_operation(dtype);
    reduce_func_t custom_reduce = (reduce_func_t)find_algorithm(config->algorithm, reduce_algorithms);
    if (!custom_reduce) {
        if (rank == 0) printf("Error: Unknown reduce algorithm '%s'\n", config->algorithm);
        return 1;
    }
    
    // Allocate buffers
    void* sendbuf = malloc(config->count * type_size);
    void* ref_recvbuf = malloc(config->count * type_size);
    void* test_recvbuf = malloc(config->count * type_size);
    if (!sendbuf || !ref_recvbuf || !test_recvbuf) {
        if (rank == 0) printf("Error: Memory allocation failed\n");
        free(sendbuf); free(ref_recvbuf); free(test_recvbuf);
        return 1;
    }
    
    // Initialize data
    init_test_data(sendbuf, config->count, dtype, rank);
    if (rank == config->root) {
        memset(ref_recvbuf, 0, config->count * type_size);
        memset(test_recvbuf, 0, config->count * type_size);
    }
    
    if (config->verbose && rank == 0) {
        printf("Operation: %s\n", (op == MPI_SUM) ? "SUM" : "MIN");
    }
    
    // Test implementations
    double start_time = MPI_Wtime();
    int ref_result = MPI_Reduce(sendbuf, ref_recvbuf, config->count, dtype, op, config->root, MPI_COMM_WORLD);
    double ref_time = MPI_Wtime() - start_time;
    
    start_time = MPI_Wtime();
    int test_result = custom_reduce(sendbuf, test_recvbuf, config->count, dtype, op, config->root, MPI_COMM_WORLD);
    double test_time = MPI_Wtime() - start_time;
    
    // Check for errors
    if (check_mpi_result(ref_result, "MPI_Reduce", rank) || 
        check_mpi_result(test_result, "Custom reduce", rank)) {
        free(sendbuf); free(ref_recvbuf); free(test_recvbuf);
        return 1;
    }
    
    // Compare and report results
    int match = 1;
    if (rank == config->root) {
        double tolerance = (dtype == MPI_DOUBLE) ? 1e-12 : (dtype == MPI_FLOAT) ? 1e-6 : 0.0;
        match = compare_buffers(ref_recvbuf, test_recvbuf, config->count, dtype, tolerance);
        
        if (config->verbose) {
            print_buffer(ref_recvbuf, config->count, dtype, rank, "MPI_Reduce");
            print_buffer(test_recvbuf, config->count, dtype, rank, config->algorithm);
        }
    }
    
    MPI_Bcast(&match, 1, MPI_INT, config->root, MPI_COMM_WORLD);
    report_results(match, config->algorithm, ref_time, test_time, config->verbose, rank);
    
    free(sendbuf); free(ref_recvbuf); free(test_recvbuf);
    return match ? 0 : 1;
}

/**
 * @brief Test gather correctness
 */
int test_gather_correctness(test_config_t* config) {
    int rank, size, type_size;
    MPI_Datatype dtype;
    
    if (test_setup(config, &rank, &size, &type_size, &dtype)) return 1;
    
    // Find algorithm
    gather_func_t custom_gather = (gather_func_t)find_algorithm(config->algorithm, gather_algorithms);
    if (!custom_gather) {
        if (rank == 0) printf("Error: Unknown gather algorithm '%s'\n", config->algorithm);
        return 1;
    }
    
    // Allocate buffers
    void* sendbuf = malloc(config->count * type_size);
    void* ref_recvbuf = (rank == config->root) ? malloc(config->count * size * type_size) : NULL;
    void* test_recvbuf = (rank == config->root) ? malloc(config->count * size * type_size) : NULL;
    
    if (!sendbuf || (rank == config->root && (!ref_recvbuf || !test_recvbuf))) {
        if (rank == 0) printf("Error: Memory allocation failed\n");
        free(sendbuf); free(ref_recvbuf); free(test_recvbuf);
        return 1;
    }
    
    // Initialize data
    init_test_data(sendbuf, config->count, dtype, rank);
    
    if (config->verbose && rank == 0) {
        printf("Size: %d\n", size);
    }
    
    // Test implementations
    double start_time = MPI_Wtime();
    int ref_result = MPI_Gather(sendbuf, config->count, dtype, ref_recvbuf, config->count, dtype, config->root, MPI_COMM_WORLD);
    double ref_time = MPI_Wtime() - start_time;
    
    start_time = MPI_Wtime();
    int test_result = custom_gather(sendbuf, config->count, dtype, test_recvbuf, config->count, dtype, config->root, MPI_COMM_WORLD);
    double test_time = MPI_Wtime() - start_time;
    
    // Check for errors
    if (check_mpi_result(ref_result, "MPI_Gather", rank) || 
        check_mpi_result(test_result, "Custom gather", rank)) {
        free(sendbuf); free(ref_recvbuf); free(test_recvbuf);
        return 1;
    }
    
    // Compare and report results
    int match = 1;
    if (rank == config->root) {
        match = compare_buffers(ref_recvbuf, test_recvbuf, config->count * size, dtype, 0.0);
        
        if (config->verbose) {
            int display_count = (config->count * size > 10) ? 10 : config->count * size;
            printf("Gathered data comparison (showing first %d elements):\n", display_count);
            print_buffer(ref_recvbuf, display_count, dtype, rank, "MPI_Gather");
            print_buffer(test_recvbuf, display_count, dtype, rank, config->algorithm);
        }
    }
    
    MPI_Bcast(&match, 1, MPI_INT, config->root, MPI_COMM_WORLD);
    report_results(match, config->algorithm, ref_time, test_time, config->verbose, rank);
    
    free(sendbuf); free(ref_recvbuf); free(test_recvbuf);
    return match ? 0 : 1;
}

/**
 * @brief Test scatter correctness
 */
int test_scatter_correctness(test_config_t* config) {
    int rank, size, type_size;
    MPI_Datatype dtype;
    
    if (test_setup(config, &rank, &size, &type_size, &dtype)) return 1;
    
    // Find algorithm
    scatter_func_t custom_scatter = (scatter_func_t)find_algorithm(config->algorithm, scatter_algorithms);
    if (!custom_scatter) {
        if (rank == 0) printf("Error: Unknown scatter algorithm '%s'\n", config->algorithm);
        return 1;
    }
    
    // Allocate buffers
    void* ref_sendbuf = (rank == config->root) ? malloc(config->count * size * type_size) : NULL;
    void* test_sendbuf = (rank == config->root) ? malloc(config->count * size * type_size) : NULL;
    void* ref_recvbuf = malloc(config->count * type_size);
    void* test_recvbuf = malloc(config->count * type_size);
    
    if (!ref_recvbuf || !test_recvbuf || 
        (rank == config->root && (!ref_sendbuf || !test_sendbuf))) {
        if (rank == 0) printf("Error: Memory allocation failed\n");
        free(ref_sendbuf); free(test_sendbuf); free(ref_recvbuf); free(test_recvbuf);
        return 1;
    }
    
    // Initialize send data (only on root)
    if (rank == config->root) {
        // Fill send buffer with data from each "rank"
        for (int r = 0; r < size; r++) {
            void* rank_data = (char*)ref_sendbuf + r * config->count * type_size;
            init_test_data(rank_data, config->count, dtype, r);
        }
        memcpy(test_sendbuf, ref_sendbuf, config->count * size * type_size);
    }
    
    // Initialize receive buffers with different data
    init_test_data(ref_recvbuf, config->count, dtype, rank + 100);
    init_test_data(test_recvbuf, config->count, dtype, rank + 100);
    
    if (config->verbose && rank == 0) {
        printf("Size: %d\n", size);
    }
    
    // Test implementations
    double start_time = MPI_Wtime();
    int ref_result = MPI_Scatter(ref_sendbuf, config->count, dtype, 
                                ref_recvbuf, config->count, dtype, 
                                config->root, MPI_COMM_WORLD);
    double ref_time = MPI_Wtime() - start_time;
    
    start_time = MPI_Wtime();
    int test_result = custom_scatter(test_sendbuf, config->count, dtype, 
                                    test_recvbuf, config->count, dtype, 
                                    config->root, MPI_COMM_WORLD);
    double test_time = MPI_Wtime() - start_time;
    
    // Check for errors
    if (check_mpi_result(ref_result, "MPI_Scatter", rank) || 
        check_mpi_result(test_result, "Custom scatter", rank)) {
        free(ref_sendbuf); free(test_sendbuf); free(ref_recvbuf); free(test_recvbuf);
        return 1;
    }
    
    // Compare results (each rank compares its own received data)
    double tolerance = (dtype == MPI_DOUBLE) ? 1e-12 : (dtype == MPI_FLOAT) ? 1e-6 : 0.0;
    int match = compare_buffers(ref_recvbuf, test_recvbuf, config->count, dtype, tolerance);
    
    // Gather results from all ranks
    int global_match;
    MPI_Allreduce(&match, &global_match, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    
    if (config->verbose) {
        printf("Rank %d scattered data comparison:\n", rank);
        print_buffer(ref_recvbuf, config->count, dtype, rank, "MPI_Scatter");
        print_buffer(test_recvbuf, config->count, dtype, rank, config->algorithm);
    }
    
    report_results(global_match, config->algorithm, ref_time, test_time, config->verbose, rank);
    
    free(ref_sendbuf); free(test_sendbuf); free(ref_recvbuf); free(test_recvbuf);
    return global_match ? 0 : 1;
}

/**
 * @brief Test alltoall correctness
 */
int test_alltoall_correctness(test_config_t* config) {
    int rank, size, type_size;
    MPI_Datatype dtype;
    
    if (test_setup(config, &rank, &size, &type_size, &dtype)) return 1;
    
    // Find algorithm
    alltoall_func_t custom_alltoall = (alltoall_func_t)find_algorithm(config->algorithm, alltoall_algorithms);
    if (!custom_alltoall) {
        if (rank == 0) printf("Error: Unknown alltoall algorithm '%s'\n", config->algorithm);
        return 1;
    }
    
    // Allocate buffers
    void* ref_sendbuf = malloc(config->count * size * type_size);
    void* test_sendbuf = malloc(config->count * size * type_size);
    void* ref_recvbuf = malloc(config->count * size * type_size);
    void* test_recvbuf = malloc(config->count * size * type_size);
    
    if (!ref_sendbuf || !test_sendbuf || !ref_recvbuf || !test_recvbuf) {
        if (rank == 0) printf("Error: Memory allocation failed\n");
        free(ref_sendbuf); free(test_sendbuf); free(ref_recvbuf); free(test_recvbuf);
        return 1;
    }
    
    // Initialize send data - each rank creates data for all other ranks
    for (int dest = 0; dest < size; dest++) {
        void* dest_data = (char*)ref_sendbuf + dest * config->count * type_size;
        // Create unique data: rank sends (rank*1000 + dest*100 + i) to dest
        if (dtype == MPI_INT) {
            int* data = (int*)dest_data;
            for (int i = 0; i < config->count; i++) {
                data[i] = rank * 1000 + dest * 100 + i;
            }
        } else if (dtype == MPI_DOUBLE) {
            double* data = (double*)dest_data;
            for (int i = 0; i < config->count; i++) {
                data[i] = rank * 1000.0 + dest * 100.0 + i * 0.25;
            }
        } else if (dtype == MPI_FLOAT) {
            float* data = (float*)dest_data;
            for (int i = 0; i < config->count; i++) {
                data[i] = rank * 1000.0f + dest * 100.0f + i * 0.25f;
            }
        } else if (dtype == MPI_CHAR) {
            char* data = (char*)dest_data;
            for (int i = 0; i < config->count; i++) {
                data[i] = (char)((rank + dest + i) % 128);
            }
        }
    }
    memcpy(test_sendbuf, ref_sendbuf, config->count * size * type_size);
    
    if (config->verbose && rank == 0) {
        printf("Size: %d\n", size);
    }
    
    // Test implementations
    double start_time = MPI_Wtime();
    int ref_result = MPI_Alltoall(ref_sendbuf, config->count, dtype, 
                                 ref_recvbuf, config->count, dtype, MPI_COMM_WORLD);
    double ref_time = MPI_Wtime() - start_time;
    
    start_time = MPI_Wtime();
    int test_result = custom_alltoall(test_sendbuf, config->count, dtype, 
                                     test_recvbuf, config->count, dtype, MPI_COMM_WORLD);
    double test_time = MPI_Wtime() - start_time;
    
    // Check for errors
    if (check_mpi_result(ref_result, "MPI_Alltoall", rank) || 
        check_mpi_result(test_result, "Custom alltoall", rank)) {
        free(ref_sendbuf); free(test_sendbuf); free(ref_recvbuf); free(test_recvbuf);
        return 1;
    }
    
    // Compare results (each rank compares its received data)
    double tolerance = (dtype == MPI_DOUBLE) ? 1e-12 : (dtype == MPI_FLOAT) ? 1e-6 : 0.0;
    int match = compare_buffers(ref_recvbuf, test_recvbuf, config->count * size, dtype, tolerance);
    
    // Gather results from all ranks
    int global_match;
    MPI_Allreduce(&match, &global_match, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    
    if (config->verbose) {
        int display_count = (config->count * size > 10) ? 10 : config->count * size;
        printf("Rank %d alltoall data comparison (showing first %d elements):\n", rank, display_count);
        print_buffer(ref_recvbuf, display_count, dtype, rank, "MPI_Alltoall");
        print_buffer(test_recvbuf, display_count, dtype, rank, config->algorithm);
    }
    
    report_results(global_match, config->algorithm, ref_time, test_time, config->verbose, rank);
    
    free(ref_sendbuf); free(test_sendbuf); free(ref_recvbuf); free(test_recvbuf);
    return global_match ? 0 : 1;
}

/**
 * @brief Test allgather correctness
 */
int test_allgather_correctness(test_config_t* config) {
    int rank, size, type_size;
    MPI_Datatype dtype;
    
    if (test_setup(config, &rank, &size, &type_size, &dtype)) return 1;
    
    // Find algorithm
    allgather_func_t custom_allgather = (allgather_func_t)find_algorithm(config->algorithm, allgather_algorithms);
    if (!custom_allgather) {
        if (rank == 0) printf("Error: Unknown allgather algorithm '%s'\n", config->algorithm);
        return 1;
    }
    
    // Allocate buffers
    void* ref_sendbuf = malloc(config->count * type_size);
    void* test_sendbuf = malloc(config->count * type_size);
    void* ref_recvbuf = malloc(config->count * size * type_size);
    void* test_recvbuf = malloc(config->count * size * type_size);
    
    if (!ref_sendbuf || !test_sendbuf || !ref_recvbuf || !test_recvbuf) {
        if (rank == 0) printf("Error: Memory allocation failed\n");
        free(ref_sendbuf); free(test_sendbuf); free(ref_recvbuf); free(test_recvbuf);
        return 1;
    }
    
    // Initialize send data - each rank sends its own data
    init_test_data(ref_sendbuf, config->count, dtype, rank);
    memcpy(test_sendbuf, ref_sendbuf, config->count * type_size);
    
    if (config->verbose && rank == 0) {
        printf("Size: %d\n", size);
    }
    
    // Test implementations
    double start_time = MPI_Wtime();
    int ref_result = MPI_Allgather(ref_sendbuf, config->count, dtype, 
                                  ref_recvbuf, config->count, dtype, MPI_COMM_WORLD);
    double ref_time = MPI_Wtime() - start_time;
    
    start_time = MPI_Wtime();
    int test_result = custom_allgather(test_sendbuf, config->count, dtype, 
                                      test_recvbuf, config->count, dtype, MPI_COMM_WORLD);
    double test_time = MPI_Wtime() - start_time;
    
    // Check for errors
    if (check_mpi_result(ref_result, "MPI_Allgather", rank) || 
        check_mpi_result(test_result, "Custom allgather", rank)) {
        free(ref_sendbuf); free(test_sendbuf); free(ref_recvbuf); free(test_recvbuf);
        return 1;
    }
    
    // Compare results (each rank compares its received data)
    double tolerance = (dtype == MPI_DOUBLE) ? 1e-12 : (dtype == MPI_FLOAT) ? 1e-6 : 0.0;
    int match = compare_buffers(ref_recvbuf, test_recvbuf, config->count * size, dtype, tolerance);
    
    // Gather results from all ranks
    int global_match;
    MPI_Allreduce(&match, &global_match, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    
    if (config->verbose) {
        int display_count = (config->count * size > 10) ? 10 : config->count * size;
        printf("Rank %d allgather data comparison (showing first %d elements):\n", rank, display_count);
        print_buffer(ref_recvbuf, display_count, dtype, rank, "MPI_Allgather");
        print_buffer(test_recvbuf, display_count, dtype, rank, config->algorithm);
    }
    
    report_results(global_match, config->algorithm, ref_time, test_time, config->verbose, rank);
    
    free(ref_sendbuf); free(test_sendbuf); free(ref_recvbuf); free(test_recvbuf);
    return global_match ? 0 : 1;
}

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -c, --collective <name>   Collective operation (bcast, reduce, gather, scatter, alltoall, allgather) [default: bcast]\n");
    printf("  -a, --algorithm <name>    Algorithm name [default: small]\n");
    printf("  -n, --count <number>      Number of elements [default: 1000]\n");
    printf("  -t, --type <datatype>     Data type (int, double, float, char) [default: int]\n");
    printf("  -r, --root <rank>         Root rank [default: 0]\n");
    printf("  -v, --verbose             Verbose output\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nAvailable algorithms:\n");
    printf("  Broadcast: small, large\n");
    printf("  Reduce: small, large\n");
    printf("  Gather: any\n");
    printf("  Scatter: any\n");
    printf("  Alltoall: small\n");
    printf("  Allgather: small\n");
    printf("\nExamples:\n");
    printf("  mpirun -np 4 %s -c bcast -a small -n 100 -t double -r 0 -v\n", program_name);
    printf("  mpirun -np 4 %s -c reduce -a large -n 1000 -t int -r 0 -v\n", program_name);
    printf("  mpirun -np 4 %s -c gather -a any -n 50 -t float -r 0 -v\n", program_name);
    printf("  mpirun -np 4 %s -c scatter -a any -n 25 -t int -r 0 -v\n", program_name);
    printf("  mpirun -np 4 %s -c alltoall -a small -n 10 -t int -v\n", program_name);
    printf("  mpirun -np 4 %s -c allgather -a small -n 20 -t double -v\n", program_name);
}

int parse_args(int argc, char* argv[], test_config_t* config) {
    // Set defaults
    strcpy(config->collective, "bcast");
    strcpy(config->algorithm, "small");
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
    
    // Run appropriate test based on collective type
    int result = 0;
    collective_type_t type = get_collective_type(config.collective);
    
    switch (type) {
        case COLLECTIVE_BCAST:
            result = test_bcast_correctness(&config);
            break;
        case COLLECTIVE_REDUCE:
            result = test_reduce_correctness(&config);
            break;
        case COLLECTIVE_GATHER:
            result = test_gather_correctness(&config);
            break;
        case COLLECTIVE_SCATTER:
            result = test_scatter_correctness(&config);
            break;
        case COLLECTIVE_ALLTOALL:
            result = test_alltoall_correctness(&config);
            break;
        case COLLECTIVE_ALLGATHER:
            result = test_allgather_correctness(&config);
            break;
        default:
            if (rank == 0) printf("Error: Unsupported collective '%s'\n", config.collective);
            result = 1;
    }
    
    MPI_Finalize();
    return result;
}