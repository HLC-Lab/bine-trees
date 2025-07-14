#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include "utils.h"

/**
 * @brief Bine-tree based allgather algorithm that works for any even number of ranks.
 * 
 * This function implements a block-by-block allgather algorithm using a bine butterfly structure.
 * It assumes that the number of ranks is even and that the send and receive counts are equal.
 * This algorithm does not require any data permutation since it sends each block individually.
 * 
 * @param sendbuf Pointer to the buffer containing data to be sent.
 * @param sendcount Number of elements to be sent from each process.
 * @param sendtype MPI datatype of the elements in the send buffer.
 * @param recvbuf Pointer to the buffer where received data will be stored.
 * @param recvcount Number of elements to be received by each process.
 * @param recvtype MPI datatype of the elements in the receive buffer.
 * @param comm MPI communicator.
 * @return int MPI_SUCCESS on success, or an error code on failure.
 */
int bine_allgather_block_by_block(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype,
                                  void* recvbuf, size_t recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
  assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
  assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  assert(size % 2 == 0); // This algorithm works only for even number of ranks
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(recvtype, &dtsize);
  MPI_Request *requests = NULL;
  memcpy((char*) recvbuf + sendcount * rank * dtsize, sendbuf, sendcount * dtsize);

  int inverse_mask = 0x1 << (int) (log_2(size) - 1);
  int step = 0;

  requests = (MPI_Request *) malloc(2 * size * sizeof(MPI_Request));
  while(inverse_mask > 0){
    int partner, req_count = 0;
    if(rank % 2 == 0){
      partner = mod(rank + negabinary_to_binary((inverse_mask << 1) - 1), size); 
    }else{
      partner = mod(rank - negabinary_to_binary((inverse_mask << 1) - 1), size); 
    }
    // We start from 1 because 0 never sends block 0
    for(int block = 1; block < size; block++){
      // Get the position of the highest set bit using clz
      // That gives us the first at which block departs from 0
      int k = 31 - __builtin_clz(nu(block, size));
      // Check if this must be sent (recvd in allgather)
      if(k == step || block == 0){
        // 0 would send this block
        int block_to_send, block_to_recv;
        // I invert what to send and what to receive wrt reduce-scatter
        if(rank % 2 == 0){
          // I am even, thus I need to shift by rank position to the right
          block_to_recv = mod(block + rank, size);
          // What to receive? What my partner is sending
          // Since I am even, my partner is odd, thus I need to mirror it and then shift
          block_to_send = mod(partner - block, size);
        }else{
          // I am odd, thus I need to mirror it
          block_to_recv = mod(rank - block, size);
          // What to receive? What my partner is sending
          // Since I am odd, my partner is even, thus I need to mirror it and then shift   
          block_to_send = mod(block + partner, size);
        }

        int partner_send = (block_to_send != partner) ? partner : MPI_PROC_NULL;
        int partner_recv = (block_to_recv != rank)  ? partner : MPI_PROC_NULL;

        err = MPI_Isend((char*) recvbuf + block_to_send*sendcount*dtsize, sendcount, sendtype, partner_send, 0, comm, &requests[req_count++]);
        if(MPI_SUCCESS != err) { goto err_hndl; }

        err = MPI_Irecv((char*) recvbuf + block_to_recv*recvcount*dtsize, recvcount, recvtype, partner_recv, 0, comm, &requests[req_count++]);
        if(MPI_SUCCESS != err) { goto err_hndl; }
      }
    }
    err = MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    inverse_mask >>= 1;
    step++;
  }

  free(requests);
  return MPI_SUCCESS;

err_hndl:
  if (requests != NULL) free(requests);
  return err;
}

