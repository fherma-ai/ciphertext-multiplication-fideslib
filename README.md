# Ciphertext multiplication on FIDESlib

An answer to `ciphertext-multiplication/ckks@1.0.0` on FHERMA that finishes on
the card: the envelope makes the context, the keys and the ciphertexts with
OpenFHE, and the multiplication runs on the GPU through FIDESlib's interop
layer.

The point is the same, and so is everything the author chooses: these
directories carry the **same `config.jsonc`** as
[ciphertext-multiplication-openfhe](https://github.com/fherma-ai/ciphertext-multiplication-openfhe),
byte for byte. The ring, the scale, the first modulus, the key-switch digit
count, the scaling technique and the security level are therefore identical on
both sides, and what separates the two measurements is the machine and the
library.

| | `solution-s32768-d10` | `solution-s4096-d20` |
| --- | --- | --- |
| point | slots 32768, depth 10, precision 15 | slots 4096, depth 20, precision 20 |
| `ring_dimension` | 65536 | 65536 |
| `batch_size` / `mult_depth` | 32768 / 10 | 4096 / 20 |
| `scale_mod_size` | 50 | 48 |
| residue polynomials per ciphertext | 11 | 21 |

## What is measured

`solve_run` calls `gpu_mult`, and that is one FIDESlib call plus the two
transfers:

```cpp
GpuCiphertext x(gpu, rawA);
GpuCiphertext y(gpu, rawB);
x.mult(y, /*rescale=*/true);   // tensor, key switch, rescale
```

`mult` with `rescale` set performs the rescale when the technique is
`FIXEDMANUAL`, which is what the config declares — so the measured call holds
the same three stages the CPU answer's `EvalMult` + `RescaleInPlace` does.

Both transfers are inside the window: the operands go to the card and the
answer comes back as an OpenFHE ciphertext the envelope can decrypt. That is
the convention the other FIDESlib answers on the platform follow, and for a
single multiplication it is a large share of the measurement rather than a
rounding error. A measurement of the arithmetic alone would need a native
FIDESlib envelope, which would also hold the keys on the card.

## The one parameter that had to move

The image carries FIDESlib's own patched OpenFHE 1.5.1, and it refuses the
deep set at a 50-bit scale exactly as the stock 1.5.1 does: the first
measurement aborted with *EstimateLogP() failure: expected sizeP [7], but got
sizeP [6]* before a single case ran. 1.6.1, which the CPU rows are built
with, accepts that scale. So the deep point is measured at a 48-bit scale on
both sides rather than on one, and the pair keeps differing in the device
alone.

## Building it

The first build happens on the fleet: the image needs CUDA and the device link
is an LTO link against FIDESlib's static library, which carries its own
OpenFHE.

```sh
docker run --rm --gpus all -v "$PWD/solution-s32768-d10":/src:ro --entrypoint sh \
  fherma/fideslib:2.1.3 -c "cmake -S /src -B /tmp/b && cmake --build /tmp/b -j"
```
