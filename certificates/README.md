# Standalone Reproducible Certificate & `0-ULP` Silicon Verification Suite

**Companion Artifact to:**  
*Sub-Generic Exact Waring Decompositions of the $E_6$ Cartan Cubic, Peirce-Split Octonion Tensor Rank Across Number Fields, $E_8$ Born Decoherence, and `0-ULP` Silicon Kernels*  
**Author:** David Tom Foss (`DT-Foss`) · **License:** PolyForm Noncommercial License 1.0.0 (`polyform-noncommercial-1.0.0`)

## Quick Start (One-Command Axiom-Up Cold Verification)

```bash
OMP_NUM_THREADS=1 python3 verify_all_certificates.py
```

Requires only Python 3 (`numpy`, `mpmath`) and a C++17 compiler (`clang++` or `g++`).  
Every algebraic tensor ($T_{\mathbb{O}}$ from oriented Fano triples $(b, b+1, b+3)\pmod 7$, $T_{\mathrm{Zorn}}$, $J_3(\mathbb{O})$ $\det X$ and $\operatorname{Tr} X^3$, Coxeter $2\mathbf{O}^\times$ / $M^*(2)$, and $J_3(\mathbb{H})$) is reconstructed **from textbook axioms at runtime** with zero reliance on pre-saved summary fields.

## The 10 Cold-Verification Gates

1. **Gate 1 — Axioms:** Alternativity, Hurwitz norm multiplicativity, Zorn `32`-nonzero tensor, and exact coordinate bijection between contiguous Peirce (`W=132`) and grouped `m0v` (`W=122`, `W=95`) orderings of $J_3(\mathbb{O})$.
2. **Gate 2 — Exact Octonion & Split-Octonion Bilinear Ranks:**
   - $R_{\mathbb{F}_2}(T_{\mathrm{Zorn}}) \le 21$ (`256/256` schemes in `data/zorn_F2.json`)
   - $R_{\mathbb{Z}[1/2]}(T_{\mathrm{Zorn}}) \le 22$ (`0/512` exact `Fraction`, denoms $\{1,2\}$ in `data/exact_Zorn_R22_CERTIFIED.json`)
   - $R_{\mathbb{Z}[i,1/4]}(T_{\mathbb{O}}) \le 22$ (`0/512` exact Gaussian `Fraction`, denoms $\{1,2,4\}$ in `data/exact_Qi_Octonion_R22_CERTIFIED.json`)
   - $R_{\mathbb{Z}}(T_{\mathrm{Zorn}}) \le 23$ (`128/128` pure-$\mathbb{Z}$ schemes in `data/zorn_Z_B1.json`; `#20` strictly $\{-1,0,+1\}$)
   - Universal Odd-Prime Theorem $R_{\mathbb{F}_p}(T_{\mathbb{O}}) \le 22$ constructively built and verified `0/512` mod $p$ across **all 12 odd primes** $p \in \{3,5,7,11,13,17,19,23,29,31,37,41\}$
   - Formally-real $R_{\mathbb{Z}[1/4]}(T_{\mathbb{O}}) \le 26$ (`data/exact_Q_R26_from_F3.json`) & $R_{\mathbb{Z}[1/2]}(T_{\mathbb{O}}) \le 27$ (`data/octo_Zhalf_B2.json`)
   - Exact modular Koszul flattenings $K_1, K_2, K_3$ on $T_{\mathbb{O}}$ ($\ge 10, 11, 13$)
3. **Gate 3 — 160-Digit Outward-Rounded Interval Kantorovich ($R_{\mathbb{R}\cap\overline{\mathbb{Q}}}(T_{\mathbb{O}}) \le 25$):** Square `128×128` Rump–Neumann interval Kantorovich proof on the $V_4 \le G_2$ `[6×4 + 1 = 25]` manifold (`data/v4_R25_kantorovich_150digit_CERTIFIED.json`), checking all `512/512` entries and `384` identically vanishing non-invariant entries.
4. **Gate 4 — Exact Rational Sub-Generic $E_6$ Waring Decompositions ($W_{\mathbb{Q}} \le 132 < 136$ & $J_3(\mathbb{H}) \le 55$):**
   - `data/r5_exact_Q_det_W132.json` and `data/r5_exact_Q_J3O_W132.json`: `0 / 3,654` exact `Fraction` monomial and `0 / 19,683` exact tensor mismatches directly against textbook Fano $J_3(\mathbb{O})$
   - Stepping stones `W=140` (`data/carbon_exact_Q_*_W140.json`) and `W=156` (`data/albert_*_exact_Q_R156.json`)
   - Hurwitz-$H_8$ quaternionic cubic $W_{\mathbb{Q}}(\det_{J_3(\mathbb{H})}) \le 55$ (`0 / 680` exact `Fraction`)
5. **Gate 5 — $O(8)$-Invariant Norm-Distance Obstruction:** Proves $\mathrm{dist}_F(I_8, \operatorname{span}\{u_r u_r^\top\}) > 0$ on `Qi_R22`, `Zorn_R22` (vs $Q_{\mathrm{Zorn}}$), and `V4_R25`.
6. **Gate 6 — Square `53×53` 160-Digit Interval Kantorovich ($W_{\mathbb{R}\cap\overline{\mathbb{Q}}} \le 122$):** Proves `0` dropped equations over $\mathbb{F}_{1000003}$ (`rk = 53/53 = dim Sym^3(V^*)^{G_{126}}`), checks all `3,654` monomials in 165-digit `mpmath.iv`, and certifies $h \ll 0.5$ on `data/r5_certified_real_true53_G126_{det,tr3}_W122.json`.
7. **Gate 7 — Sub-100 Complex Border Waring Rank ($\underline{W}_{\mathbb{C}} \le 95 < 100$):** Rebuilds $G_{21} = \mathbb{Z}_7 \rtimes \mathbb{Z}_3$ (`dim Sym^3(V^*)^{G_{21}} = 304`) and verifies exact `304 / 304` Terracini Jacobian rank of the `3,654 × 309` map across **3 independent primes** (`1000003`, `1000033`, `999983`).
8. **Gate 8 — Post-Koszul Barriers & Live $E_6$ Borel Dual Prolongation:** Verifies the universal Koszul ceiling $\le 51$, Substitution–Koszul strict-loss barrier, and executes the exact $E_6$ `351'` weight-poset Borel dual prolongation scan proving `0` survivors for $r \le 45$ and exact `3` non-smoothable cactus ideals at $r = 46$.
9. **Gate 9 — $E_8$ `240`-Root Born Decoherence & $M^*(2)$ Moufang Ignition:** Rebuilds Coxeter $2\mathbf{O}^\times$ (`240` roots, `0/57,600` closure errors) and $M^*(2)$ (`120`), verifies the exact `240 = 16 + 96 + 128` Born trichotomy, magic angle $\theta^* = \arccos(1/\sqrt{8}) = 69.2952^\circ$, sharp $k^* = 3$ Moufang ignition (`7/20 = 35.0%`), and Strahler-$k=2$ Hankel rank `64/64`.
10. **Gate 10 — Live C++17 `0-ULP` Silicon Benchmark:** Compiles `fast_octonion_0ulp_bench.cpp` + `fast_octonion_kernels.hpp` with `clang++ -O3 -march=native -std=c++17` and verifies `0/512` basis, `0/100,000` random `int64`, and `0/8,192` $D=32$ Block-GEMM errors live on the host CPU.

Verification output is written to `verification_report.json`.
