// One CKKS ciphertext multiplication, computed on the GPU with FIDESlib.
//
// The parameters are the CPU answer's, to the number: the same config.jsonc
// builds the context, so the ring, the scale, the key-switch digits and the
// security level are identical and what separates the two measurements is
// the machine and the library. This file is the plain C++ half — how the
// vectors are packed and read back. The multiplication itself is in gpu.cu,
// compiled by nvcc.
#include "solve.h"

#include "gpu.h"

void* solve_init(const fherma::Point& p, CryptoContext<DCRTPoly> cc) {
    // The device context and the relinearisation key, before the clock.
    return gpu_init(cc);
}

std::vector<Plaintext> solve_encoding(CryptoContext<DCRTPoly> cc,
                                      const fherma::Inputs& inp) {
    // Slot i holds element i, as in the CPU answer.
    return {cc->MakeCKKSPackedPlaintext(inp.a.data),
            cc->MakeCKKSPackedPlaintext(inp.b.data)};
}

std::vector<Ciphertext<DCRTPoly>> solve_run(
    void* state,
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& cts) {
    return {gpu_mult(state, cc, cts[0], cts[1])};
}

fherma::Outputs solve_decoding(const fherma::Point& p,
                               CryptoContext<DCRTPoly> cc,
                               const std::vector<Plaintext>& pts) {
    auto packing = pts.at(0);
    packing->SetLength(p.slots);
    fherma::Outputs out{};
    out.c.shape = {static_cast<int64_t>(p.slots)};
    out.c.data = packing->GetRealPackedValue();
    return out;
}

void solve_free(void* state) { gpu_free(state); }
