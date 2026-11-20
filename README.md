# Sub-Generic Exact Waring Decompositions of the $E_6$ Cartan Cubic, Peirce-Split Octonion Rank, $E_8$ Born Decoherence, and `0-ULP` Silicon

**Author:** David Tom Foss (Wirkung · `david@wirku.ng`)  
**Links:** [GitHub](https://github.com/DT-Foss) · [Hugging Face](https://huggingface.co/davidfoss) · [X](https://x.com/fossdt)  
**Preprint (Part II of II):** [DOI 10.5281/zenodo.24405793](https://doi.org/10.5281/zenodo.24405793)  
**Companion Flagship Paper (Part I of II):** *Quotient Silicon and the Non-Associative Organism* — [DOI 10.5281/zenodo.24405791](https://doi.org/10.5281/zenodo.24405791)  
**GitHub Mirror:** https://github.com/DT-Foss/octonion-e6-quotient-rank  
**License:** PolyForm Noncommercial License 1.0.0 (`PolyForm-Noncommercial-1.0.0`)

## Headline Certified Results (Cold-Verified From Textbook Axioms in One Command)
- **Octonion & Split-Octonion Bilinear Rank:** $R_{\mathbb{F}_2}(T_{\mathrm{Zorn}}) \le 21$, universal odd-prime $R_{\mathbb{F}_p}(T_{\mathbb{O}}) = R_{\mathbb{Z}[i,1/4]}(T_{\mathbb{O}}) = R_{\mathbb{Z}[1/2]}(T_{\mathrm{Zorn}}) \le 22$ (all 12 odd primes $p = 3..41$, plus $\mathbb{Q}_p$ and $\mathbb{Q}(\sqrt{-d})$), pure-$\mathbb{Z}$ $\{\pm 1,0\}$ $R_{\mathbb{Z}}(T_{\mathrm{Zorn}}) \le 23$, formally-real $165$-digit Rump–Neumann interval Kantorovich $R_{\mathbb{R}\cap\overline{\mathbb{Q}}}(T_{\mathbb{O}}) \le 25$ ($128\times 128$ $V_4\le G_2$), and exact rational $R_{\mathbb{Z}[1/4]}(T_{\mathbb{O}}) \le 26$.
- **$E_6$ Cartan Cubic & Albert Algebra $J_3(\mathbb{O})$:** First strictly sub-generic ($< 136$) exact rational decompositions $W_{\mathbb{Q}}(\det X), W_{\mathbb{Q}}(\operatorname{Tr} X^3) \le 132 < 136$ ($0/3{,}654$ monomial and $0/19{,}683$ tensor `Fraction` mismatches) and $W_{\mathbb{Q}}(\det_{J_3(\mathbb{H})}) \le 55$; exact $O(8)$ norm-distance obstruction on standalone $R = 22$; unconditional square $53\times 53$ ($0$ dropped equations) $165$-digit interval Kantorovich $W_{\mathbb{R}\cap\overline{\mathbb{Q}}} \le 122$; and unconditional 3-prime (`304/304` Terracini) sub-100 complex border rank $\underline{W}_{\mathbb{C}} \le 95 < 100$.
- **Post-Koszul Barriers & $E_8$ Born Decoherence:** Certified $K_5^W \ge 49$, analytic substitution–Koszul ceiling $\le 51$, live Borel dual-prolongation cactus onset $r = 46$ on $E_6$ ($\mathbf{351}'$), exact $240 = 16 \oplus 96 \oplus 128$ $E_8$ Born decoherence ($\theta^* = 69.2952^\circ$), and sharp $k^* = 3$ Moufang $A_5$ ignition ($7/20 = 35.0\%$).
- **Division-Free `0-ULP` C++17 Silicon Kernels (`certificates/fast_octonion_kernels.hpp`):** 13 exact `int64` kernels (`0/512` basis, `0/100,000` random, `0/8,192` $D=32$ Block-GEMM errors) — `ButterflyZorn_R23` wins `#1` scalar at $D=1$ (`27.55 ns`) and multi-head `o1` carriers (`2.534x` M4, `2.444x` Zen-4), while `ButterflyZorn_R22` achieves `2.38x–2.49x` vs Naive64 (`1.32x–1.36x` vs Cariow30) at $D=32$.

## One-Command Cold Verification (Zero External CAS, Zero NPU Required)
```bash
cd certificates
OMP_NUM_THREADS=1 python3 verify_all_certificates.py
```
Runs all **10 independent axiom-up gates** (exact `Fraction` checks, `mpmath.iv` 165-digit Rump–Neumann interval Kantorovich, 3-prime Terracini modular rank, live Borel dual prolongation, $E_8$ Born decoherence, and live `clang++ -O3 -march=native` C++17 `0-ULP` compilation & benchmarking) in ~38 s on Apple M4 (`10/10 PASS`).

## Citation
```bibtex
@misc{foss2026octonione6,
  author       = {Foss, David Tom},
  title        = {Sub-Generic Exact Waring Decompositions of the E\_6 Cartan Cubic, Peirce-Split Octonion Rank, E\_8 Born Decoherence, and 0-ULP Silicon},
  year         = {2026},
  month        = nov,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24405793},
  url          = {https://doi.org/10.5281/zenodo.24405793}
}
```
