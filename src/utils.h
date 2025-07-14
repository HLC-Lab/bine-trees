#include <stdint.h>
/**
 * @file utils.h
 * @brief Utility functions.
 */

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
