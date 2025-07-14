#include <mpi.h>
#include <stdlib.h>
#include "utils.h"

/**
 * @brief Broadcasts a message using a binomial tree algorithm.
 *        This implementation is optimized for small messages.
 * @param buf Pointer to the buffer containing the data to broadcast.
 * @param count Number of elements in the buffer.
 * @param dtype MPI datatype of the elements in the buffer.
 * @param root Rank of the process that is broadcasting the message.
 * @param comm MPI communicator over which to broadcast.
 * @returns MPI_SUCCESS on success, or an error code on failure.
 */
int bcast_bine_small(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
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