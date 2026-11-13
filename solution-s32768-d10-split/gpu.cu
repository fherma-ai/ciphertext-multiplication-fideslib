// One CKKS multiplication on the card, through FIDESlib's OpenFHE interop,
// with the copies to and from the card outside the measured call.
//
// The envelope makes the context, the keys and the two ciphertexts with
// OpenFHE before the clock starts. Once per point (gpu_init, not measured) the
// device context is adapted from the envelope's own, the relinearisation key
// is moved over, and the two device ciphertexts are allocated. Per case the
// loop calls three functions, each on its own clock:
//
//   gpu_h2d   both operands as FIDESlib raw ciphertexts, loaded onto the card    h2d_s
//   gpu_mult  tensor, relinearise, rescale, on the card, until it has finished   seconds, the score
//   gpu_d2h   the answer back, as an OpenFHE ciphertext the envelope decrypts    d2h_s
//
// The arithmetic is the -reused answer's, call for call; what moves is where
// the clock is. The difference against -reused is what the copies cost.
//
// `mult(b, rescale)` with rescale set does the tensor, the key switch and,
// under FIXEDMANUAL, the rescale — the three stages the specification names,
// in one call, from the same config.jsonc the CPU answer is built from.
#include "gpu.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "openfhe.h"
#ifdef duration
#undef duration  // OpenFHE defines the word; CUDA's headers use it.
#endif
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

namespace {

using GpuCiphertext = FIDESlib::CKKS::Ciphertext;

struct Device {
    FIDESlib::CKKS::Context gpu;
    // Allocated once, loaded per case. A ciphertext built on a context holds
    // its limbs; constructing one per call means allocating 2 × (level + 1)
    // residue polynomials on the device every time.
    std::unique_ptr<GpuCiphertext> a;
    std::unique_ptr<GpuCiphertext> b;
    // The first operand as the envelope made it: the answer is a copy of it
    // with its elements, level and scale overwritten, as before.
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> shape;
};

}  // namespace

void* gpu_init(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
    // The device context, adapted from the envelope's: its primes, its
    // digits, its scaling technique. FIDESlib's own parameters only add the
    // batch it processes limbs in.
    FIDESlib::CKKS::RawParams raw = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Parameters seed{};
    seed.batch = 100;
    FIDESlib::CKKS::Context gpu = FIDESlib::CKKS::GenCryptoContextGPU(seed.adaptTo(raw), {0});

    // The relinearisation key, read out of the context's own key store. The
    // envelope made one key pair, so the store holds one entry; the key is
    // the first of its vector, as it is in OpenFHE's own EvalMult.
    auto& store = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetAllEvalMultKeys();
    if (store.empty()) throw std::runtime_error("gpu_init: the context holds no evaluation key");
    auto relin = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
        store.begin()->second.at(0));
    if (!relin) throw std::runtime_error("gpu_init: the evaluation key is not a relinearisation key");

    FIDESlib::CKKS::RawKeySwitchKey rawKey = FIDESlib::CKKS::GetKeySwitchKey(relin);
    FIDESlib::CKKS::KeySwitchingKey key(gpu);
    key.Initialize(rawKey);
    gpu->AddEvalKey(std::move(key));

    auto* device = new Device{std::move(gpu), nullptr, nullptr, nullptr};
    device->a = std::make_unique<GpuCiphertext>(device->gpu);
    device->b = std::make_unique<GpuCiphertext>(device->gpu);
    return device;
}

void gpu_h2d(
    void* state,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& a,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& b) {
    Device& device = *static_cast<Device*>(state);
    // Both operands to the card, into the buffers gpu_init allocated.
    FIDESlib::CKKS::RawCipherText rawA = FIDESlib::CKKS::GetRawCipherText(cc, a);
    FIDESlib::CKKS::RawCipherText rawB = FIDESlib::CKKS::GetRawCipherText(cc, b);
    device.a->load(rawA);
    device.b->load(rawB);
    device.shape = a;
    cudaDeviceSynchronize();
}

void gpu_mult(void* state) {
    Device& device = *static_cast<Device*>(state);
    // Tensor, relinearise, rescale — and returned once the card has finished.
    device.a->mult(*device.b, /*rescale=*/true);
    cudaDeviceSynchronize();
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_d2h(void* state) {
    Device& device = *static_cast<Device*>(state);
    // Back, as an OpenFHE ciphertext the envelope can decrypt.
    FIDESlib::CKKS::RawCipherText rawOut;
    device.a->store(rawOut);
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> answer = device.shape->Clone();
    FIDESlib::CKKS::GetOpenFHECipherText(answer, rawOut);
    return answer;
}

void gpu_free(void* state) { delete static_cast<Device*>(state); }
