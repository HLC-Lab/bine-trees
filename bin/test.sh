#!/bin/bash
# filepath: \home\desensi\work\bine_trees\test\run_tests.sh

# Test configuration arrays
VECTOR_SIZES=(16 100 1000 10000 100000)
DATATYPES=("int" "double" "float" "char")
RANK_COUNTS=(2 4 8 16)
COLLECTIVES=("bcast")
ALGORITHMS=("bine_bcast_small" "bine_bcast_large")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
START_TIME=$(date +%s)

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "PASS")
            echo -e "${GREEN}[PASS]${NC} $message"
            ;;
        "FAIL")
            echo -e "${RED}[FAIL]${NC} $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
        "TEST")
            echo -e "${PURPLE}[TEST]${NC} $message"
            ;;
    esac
}

# Function to print progress bar
print_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    
    printf "\r${CYAN}Progress: ["
    printf "%*s" $filled | tr ' ' '='
    printf "%*s" $((width - filled))
    printf "] %d%% (%d/%d)${NC}" $percentage $current $total
}

# Function to format time
format_time() {
    local seconds=$1
    if [ $seconds -lt 60 ]; then
        printf "%ds" $seconds
    elif [ $seconds -lt 3600 ]; then
        printf "%dm %ds" $((seconds / 60)) $((seconds % 60))
    else
        printf "%dh %dm %ds" $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
    fi
}

# Function to run a single test
run_test() {
    local ranks=$1
    local collective=$2
    local algorithm=$3
    local count=$4
    local datatype=$5
    local root=$6
    
    local test_name="${collective}_${algorithm}_${ranks}ranks_${count}${datatype}"
    
    # Run the test with timeout
    timeout 30s mpirun -np $ranks ./bin/test.exe \
        -c $collective \
        -a $algorithm \
        -n $count \
        -t $datatype \
        -r $root \
        > /tmp/test_output_$$ 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Check if test actually passed by looking for "PASSED" in output
        if grep -q "Test PASSED" /tmp/test_output_$$; then
            print_status "PASS" "$test_name"
            ((PASSED_TESTS++))
            return 0
        else
            print_status "FAIL" "$test_name - Test reported failure"
            echo "Output:"
            cat /tmp/test_output_$$
            ((FAILED_TESTS++))
            return 1
        fi
    elif [ $exit_code -eq 124 ]; then
        print_status "FAIL" "$test_name - Timeout (30s)"
        ((FAILED_TESTS++))
        return 1
    else
        print_status "FAIL" "$test_name - Exit code: $exit_code"
        echo "Output:"
        cat /tmp/test_output_$$
        ((FAILED_TESTS++))
        return 1
    fi
}

# Function to calculate total number of tests
calculate_total_tests() {
    local total=0
    for ranks in "${RANK_COUNTS[@]}"; do
        for collective in "${COLLECTIVES[@]}"; do
            for algorithm in "${ALGORITHMS[@]}"; do
                for count in "${VECTOR_SIZES[@]}"; do
                    for datatype in "${DATATYPES[@]}"; do
                        ((total++))
                    done
                done
            done
        done
    done
    echo $total
}

# Main execution
main() {
    print_status "INFO" "Starting comprehensive test suite..."
    
    # Check if test executable exists
    if [ ! -f "./bin/test.exe" ]; then
        print_status "FAIL" "test.exe not found. Please compile the test first."
        exit 1
    fi
    
    # Calculate total tests
    TOTAL_TESTS=$(calculate_total_tests)
    print_status "INFO" "Total tests to run: $TOTAL_TESTS"
    echo
    
    local current_test=0
    
    # Main test loop
    for ranks in "${RANK_COUNTS[@]}"; do
        print_status "INFO" "Testing with $ranks ranks"
        
        for collective in "${COLLECTIVES[@]}"; do
            for algorithm in "${ALGORITHMS[@]}"; do
                print_status "TEST" "Algorithm: $algorithm"
                
                for count in "${VECTOR_SIZES[@]}"; do
                    for datatype in "${DATATYPES[@]}"; do
                        ((current_test++))
                        print_progress $current_test $TOTAL_TESTS
                        
                        # Choose root rank (0 for most tests, vary occasionally)
                        local root=0
                        if [ $((current_test % 5)) -eq 0 ] && [ $ranks -gt 1 ]; then
                            root=$((ranks - 1))
                        fi
                        
                        # Run the test
                        if ! run_test $ranks $collective $algorithm $count $datatype $root; then
                            echo
                            print_status "FAIL" "Test suite stopped due to failure"
                            echo
                            print_status "INFO" "Failed test details:"
                            print_status "INFO" "  Ranks: $ranks"
                            print_status "INFO" "  Collective: $collective"
                            print_status "INFO" "  Algorithm: $algorithm"
                            print_status "INFO" "  Count: $count"
                            print_status "INFO" "  Datatype: $datatype"
                            print_status "INFO" "  Root: $root"
                            cleanup_and_exit 1
                        fi
                    done
                done
            done
        done
        echo
    done
    
    # Final progress update
    echo
    print_status "INFO" "All tests completed!"
    
    # Print summary
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    
    echo
    echo "===== TEST SUMMARY ====="
    print_status "INFO" "Total tests run: $TOTAL_TESTS"
    print_status "PASS" "Passed: $PASSED_TESTS"
    print_status "FAIL" "Failed: $FAILED_TESTS"
    print_status "INFO" "Success rate: $((PASSED_TESTS * 100 / TOTAL_TESTS))%"
    print_status "INFO" "Total time: $(format_time $duration)"
    print_status "INFO" "Average time per test: $(format_time $((duration / TOTAL_TESTS)))"
    
    # Cleanup
    cleanup_and_exit 0
}

# Cleanup function
cleanup_and_exit() {
    local exit_code=$1
    rm -f /tmp/test_output_$$
    exit $exit_code
}

# Signal handlers
trap 'print_status "WARN" "Interrupted by user"; cleanup_and_exit 130' INT TERM

# Run main function
main "$@"