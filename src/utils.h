/**
 * @file utils.h
 * @brief Utility functions.
 */
#include <stdint.h>
#include <assert.h>

#define BINE_MAX_STEPS 20

static int smallest_negabinary[BINE_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42,
          -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[BINE_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85,
          341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

 /**
  * @brief Computes the mathematical modulo of a and b.
  * @param a The dividend.
  * @param b The divisor.
  * @returns The mathematical modulo of a and b.
  *          If a is negative, it returns a positive remainder.
  */  
static inline int mod(int a, int b){
    int r = a % b;
    return r < 0 ? r + b : r;
}

/**
 * @brief Check if a value is a power of two.
 * @param value The integer value to check.
 * @returns 1 if value is a power of two, 0 otherwise.
 */
static inline int is_power_of_two(int value) {
    return (value & (value - 1)) == 0;
}

/**
 * @brief Returns log_2(value). Value must be a positive integer.
 *        If value is not a power of two, returns ceil(log2(value)).
 * @param value The **POSITIVE** integer value to return its log_2.
 * @returns The log_2 of value or -1 for negative value.
 */
static inline int log_2(int value) {
  if(value < 1) {
    return -1;
  }
  int log = sizeof(int)*8 - 1 - __builtin_clz(value); 
  if (!is_power_of_two(value)) {
    log += 1;
  }
  return log;
}

/**
 * @brief Converts a binary number to its negabinary representation.
 * @param bin The binary number to convert.
 * @returns The negabinary representation of the input binary number.
 */
static uint32_t binary_to_negabinary(int32_t bin) {
    if(bin > 0x55555555) return -1;
    const uint32_t mask = 0xAAAAAAAA;
    return (mask + bin) ^ mask;
}

/**
 * @brief Converts a negabinary number to its binary representation.
 * @param neg The negabinary number to convert.
 * @returns The binary representation of the input negabinary number.
 */
static int32_t negabinary_to_binary(uint32_t neg) {
    const uint32_t mask = 0xAAAAAAAA;
    return (mask ^ neg) - mask;
}

/**
 * @brief Reverses the bits of a 32-bit unsigned integer.
 * @param x The 32-bit unsigned integer to reverse.
 * @returns The 32-bit unsigned integer with its bits reversed.
 */
static inline uint32_t reverse(uint32_t x){
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
}

/**
 * @brief Checks if a value is in the range of valid negabinary representations
 * for a given number of bits.
 * @param x The value to check.
 * @param nbits The number of bits for the negabinary representation.
 * @returns 1 if x is in the valid range, 0 otherwise.
 */
static inline int in_range(int x, uint32_t nbits){
    return x >= smallest_negabinary[nbits] && x <= largest_negabinary[nbits];
}

static inline uint32_t nb_to_nu(uint32_t nb, uint32_t size){
  return reverse(nb ^ (nb >> 1)) >> (32 - log_2(size));
}

/**
 * @brief Computes nu(r, p) (see the paper).
 * @param rank The rank to convert (denoted as 'r' in the paper).
 * @param size The total number of ranks (denoted as 'p' in the paper).
 * @returns The remapped rank.
 */
static inline uint32_t nu(uint32_t rank, uint32_t size){
  uint32_t nba = UINT32_MAX, nbb = UINT32_MAX;
  size_t num_bits = log_2(size);
  if(rank % 2){
      if(in_range(rank, num_bits)){
          nba = binary_to_negabinary(rank);
      }
      if(in_range(rank - size, num_bits)){
          nbb = binary_to_negabinary(rank - size);
      }
  }else{
      if(in_range(-rank, num_bits)){
          nba = binary_to_negabinary(-rank);
      }
      if(in_range(-rank + size, num_bits)){
          nbb = binary_to_negabinary(-rank + size);
      }
  }
  assert(nba != UINT32_MAX || nbb != UINT32_MAX);

  if(nba == UINT32_MAX && nbb != UINT32_MAX){
      return nb_to_nu(nbb, size);
  }else if(nba != UINT32_MAX && nbb == UINT32_MAX){
      return nb_to_nu(nba, size);
  }else{ // Check MSB
      int nu_a = nb_to_nu(nba, size);
      int nu_b = nb_to_nu(nbb, size);
      if(nu_a < nu_b){
          return nu_a;
      }else{
          return nu_b;
      }
  }
}

static inline uint32_t mersenne(int n) {
    return (1UL << (n + 1)) - 1;
}

static inline int remap_distance_doubling(uint32_t num) {
    int remapped = 0;
    while (num > 0) {
        int k = 31 - __builtin_clz(num); // Find the position of the highest set bit
        remapped ^= (0x1 << k); // Set the k-th bit in the remapped number
        num ^= mersenne(k); // XOR the Mersenne number with the remaining number
    }
    return remapped;
}