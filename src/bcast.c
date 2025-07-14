#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include <strings.h>
#include "utils.h"

/**
 * @brief Broadcasts a message using a binomial tree algorithm.
 *        This implementation is optimized for small vectors.
 * @param buf Pointer to the buffer containing the data to broadcast.
 * @param count Number of elements in the buffer.
 * @param dtype MPI datatype of the elements in the buffer.
 * @param root Rank of the process that is broadcasting the message.
 * @param comm MPI communicator over which to broadcast.
 * @returns MPI_SUCCESS on success, or an error code on failure.
 */
int bine_bcast_small(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
{
  int size, rank, dtsize, err = MPI_SUCCESS, btnb_vrank;
  int vrank, mask, recvd, req_count = 0, steps;
  MPI_Request *requests;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dtype, &dtsize);
 
  // TODO: Implement non-power of two support similarly to standard binomial trees
  if(!is_power_of_two(size)) return MPI_ERR_SIZE;

  vrank = mod(rank - root, size); // mod computes math modulo rather than reminder
  steps = log_2(size);
  mask = 0x1 << (int) (steps - 1);
  recvd = (root == rank);
  btnb_vrank = binary_to_negabinary(vrank);
  requests = (MPI_Request *) malloc(steps * sizeof(MPI_Request));
  if(requests == NULL) return MPI_ERR_NO_MEM;
  while(mask > 0){
    int partner = btnb_vrank ^ ((mask << 1) - 1);
    partner = mod(negabinary_to_binary(partner) + root, size);
    int mask_lsbs = (mask << 1) - 1; // Mask with num_steps - step + 1 LSBs set to 1
    int lsbs = btnb_vrank & mask_lsbs; // Extract k LSBs
    int equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

    if(recvd){
      err = MPI_Isend(buf, count, dtype, partner, 0, comm, &requests[req_count++]);
      if(MPI_SUCCESS != err) { goto err_hndl; }
    }else if(equal_lsbs){
      err = MPI_Recv(buf, count, dtype, partner, 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto err_hndl; }
      recvd = 1;
    }
    mask >>= 1;
  }

  MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

  free(requests);
  return MPI_SUCCESS;

err_hndl:
  if (NULL != requests) free(requests);
  return err;
}

/**
 * @brief Broadcasts a message using a binomial tree algorithm.
 *        This implementation is optimized for large vectors.
 * @param buf Pointer to the buffer containing the data to broadcast.
 * @param count Number of elements in the buffer.
 * @param dtype MPI datatype of the elements in the buffer.
 * @param root Rank of the process that is broadcasting the message.
 * @param comm MPI communicator over which to broadcast.
 * @returns MPI_SUCCESS on success, or an error code on failure.
 */
int bine_bcast_large(void *buffer, size_t count, MPI_Datatype dt, int root, MPI_Comm comm){
  assert(root == 0); // TODO: Generalize
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  assert(is_power_of_two(size)); // TODO: Generalize
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);

  int* displs = (int*) malloc(size*sizeof(int));
  int* recvcounts = (int*) malloc(size*sizeof(int));
  if(displs == NULL || recvcounts == NULL){
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }
  int count_per_rank = count / size;
  int rem = count % size;
  for(int i = 0; i < size; i++){
    displs[i] = count_per_rank*i + (i < rem ? i : rem);
    recvcounts[i] = count_per_rank + (i < rem ? 1 : 0);
  }

  int mask = 0x1;
  int inverse_mask = 0x1 << (int) (log_2(size) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int remapped_rank = nu(rank, size);
  int receiving_mask = inverse_mask << 1; // Root never receives. By having a large mask inverse_mask will always be < receiving_mask
  // I receive in the step corresponding to the position (starting from right)
  // of the first 1 in my remapped rank -- this indicates the step when the data reaches me
  if(rank != root){
    receiving_mask = 0x1 << (ffs(remapped_rank) - 1); // ffs starts counting from 1, thus -1
  }
  
  /***** Scatter *****/
  int recvd = (root == rank);
  while(mask < size){
    int partner;
    if(rank % 2 == 0){
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size); 
    }else{
      partner = mod(rank - negabinary_to_binary((mask << 1) - 1), size); 
    }
  
    // For sure I need to send my (remapped) partner's data
    // the actual start block however must be aligned to 
    // the power of two
    int send_block_first = nu(partner, size) & block_first_mask;
    int send_block_last = send_block_first + inverse_mask - 1;
    int send_count = displs[send_block_last] - displs[send_block_first] + recvcounts[send_block_last];
    // Something similar for the block to recv.
    // I receive my block, but aligned to the power of two
    int recv_block_first = remapped_rank & block_first_mask;
    int recv_block_last = recv_block_first + inverse_mask - 1;
    int recv_count = displs[recv_block_last] - displs[recv_block_first] + recvcounts[recv_block_last];
    
    if(recvd){
      err = MPI_Send((char*) buffer + displs[send_block_first]*dtsize, send_count, dt, partner, 0, comm);
      if(MPI_SUCCESS != err) { goto err_hndl; }
    }else if(inverse_mask == receiving_mask || partner == root){
      err = MPI_Recv((char*) buffer + displs[recv_block_first]*dtsize, recv_count, dt, partner, 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto err_hndl; }
      recvd = 1;
    }

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

  /***** Allgather *****/  
  mask >>= 1;
  inverse_mask = 0x1;
  block_first_mask = ~0x0;
  while(mask > 0){
    int spartner, rpartner;
    int send_block_first = 0, send_block_last = 0, send_count = 0, recv_block_first = 0, recv_block_last = 0, recv_count = 0;
    int partner;
    if(rank % 2 == 0){
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size); 
    }else{
      partner = mod(rank - negabinary_to_binary((mask << 1) - 1), size); 
    }

    rpartner = (inverse_mask < receiving_mask) ? MPI_PROC_NULL : partner;
    spartner = (inverse_mask == receiving_mask) ? MPI_PROC_NULL : partner;

    if(spartner != MPI_PROC_NULL){
      send_block_first = remapped_rank & block_first_mask;
      send_block_last = send_block_first + inverse_mask - 1;
      send_count = displs[send_block_last] - displs[send_block_first] + recvcounts[send_block_last];  
    }
    if(rpartner != MPI_PROC_NULL){
      recv_block_first = nu(rpartner, size) & block_first_mask;
      recv_block_last = recv_block_first + inverse_mask - 1;
      recv_count = displs[recv_block_last] - displs[recv_block_first] + recvcounts[recv_block_last];
    }
    err = MPI_Sendrecv((char*) buffer + displs[send_block_first]*dtsize, send_count, dt, spartner, 0, 
                       (char*) buffer + displs[recv_block_first]*dtsize, recv_count, dt, rpartner, 0, comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { goto err_hndl; }

    mask >>= 1;
    inverse_mask <<= 1;
    block_first_mask <<= 1;
  }

  free(displs);
  free(recvcounts);
  return MPI_SUCCESS;

err_hndl:
  if(NULL!= displs)     free(displs);
  if(NULL!= recvcounts) free(recvcounts);
  return err;
}