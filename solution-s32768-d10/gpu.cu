// One CKKS multiplication on the card, through FIDESlib's OpenFHE interop.
//
// The envelope makes the context, the keys and the two ciphertexts with
// OpenFHE before the clock starts; this file takes them as they are. Once per
// point (gpu_init, not measured) the device context is adapted from the
// envelope's own and the relinearisation key is moved over. Per case
// (gpu_mult, which is what the clock sees) both ciphertexts go to the card,
// the multiplication runs there, and the answer comes back as an OpenFHE
// ciphertext the envelope can decrypt.
//
// Moving the data is inside the measurement on purpose: it is part of what
// answering on a GPU costs, and for a single multiplication it is a large
// part. The same convention as the other FIDESlib answers on the platform.
//
// `mult(b, rescale)` with rescale set does the tensor, the key switch and,
// under FIXEDMANUAL, the rescale — the three stages the specification names,
// in one call, from the same config.jsonc the CPU answer is built from.
#include "gpu.h"

#include <stdexcept>
#include <utility>

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

    return new Device{std::move(gpu)};
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_mult(
    void* state,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& a,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& b) {
    Device& device = *static_cast<Device*>(state);
    FIDESlib::CKKS::Context& gpu = device.gpu;

    // Both operands to the card.
    FIDESlib::CKKS::RawCipherText rawA = FIDESlib::CKKS::GetRawCipherText(cc, a);
    FIDESlib::CKKS::RawCipherText rawB = FIDESlib::CKKS::GetRawCipherText(cc, b);
    GpuCiphertext x(gpu, rawA);
    GpuCiphertext y(gpu, rawB);

    // Tensor, relinearise, rescale.
    x.mult(y, /*rescale=*/true);

    // Back, as an OpenFHE ciphertext the envelope can decrypt: a copy of an
    // input carries the context and the key tag; the elements, the level and
    // the scale are overwritten with the answer's.
    FIDESlib::CKKS::RawCipherText rawOut;
    x.store(rawOut);
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> answer = a->Clone();
    FIDESlib::CKKS::GetOpenFHECipherText(answer, rawOut);
    return answer;
}

void gpu_free(void* state) { delete static_cast<Device*>(state); }
