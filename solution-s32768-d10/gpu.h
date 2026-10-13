// The device side of the answer, behind a C++ face.
//
// Everything that includes a FIDESlib header is CUDA and lives in gpu.cu, so
// solve.cpp and the platform's envelope stay plain C++. These two functions
// are the whole boundary: the device context made once per point from the
// envelope's own, and one multiplication of two ciphertexts on the card.
#pragma once

#include "openfhe.h"

// The GPU context adapted from `cc`, with its relinearisation key on the
// card. Not measured: called from solve_init.
void* gpu_init(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc);

// a ⊙ b: to the card, tensor, relinearise, rescale, back. Measured.
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_mult(
    void* state,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& a,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& b);

void gpu_free(void* state);
