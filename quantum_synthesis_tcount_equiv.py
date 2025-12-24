# sat_cnot_t_synthesis_clean.py
"""
Cleaned-up implementation of a SAT-based {CNOT, T} quantum circuit synthesis
workflow, inspired by "SAT-based {CNOT, T} Quantum Circuit Synthesis".

This file provides:
  - GF(2) helpers and phase-polynomial extraction
  - Z3-based SAT encoding of the HasCNOT(G, F_i, c_i, K) problem
  - Exact CNOT-count minimization for small instances
  - Utilities to:
      * generate random {CNOT, PHASE} circuits
      * generate random Qiskit Clifford+T circuits
      * optimize Clifford+T blocks in a Qiskit circuit
      * run experiments on random circuits and on QASM testbench files
"""

from __future__ import annotations

import glob
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from z3 import (
    Solver,
    Bool,
    BoolVal,
    And,
    Or,
    Xor,
    Implies,
    Not,
    sat,
)

from qiskit import QuantumCircuit


# ---------------------------------------------------------------------------
# 1. GF(2) helpers
# ---------------------------------------------------------------------------


def gf2_rank(M: np.ndarray) -> int:
    """Rank of binary matrix M over GF(2) via Gaussian elimination."""
    M = M.copy().astype(int)
    n_rows, n_cols = M.shape
    rank = 0
    col = 0
    for r in range(n_rows):
        if col >= n_cols:
            break
        # Find pivot
        pivot = None
        for i in range(r, n_rows):
            if M[i, col] == 1:
                pivot = i
                break
        if pivot is None:
            col += 1
            r -= 1
            continue
        # Swap rows
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        # Eliminate
        for i in range(n_rows):
            if i != r and M[i, col] == 1:
                M[i, :] ^= M[r, :]
        rank += 1
        col += 1
    return rank


def random_invertible_matrix(n: int, rng: random.Random | None = None) -> np.ndarray:
    """Generate a random n x n invertible matrix over GF(2)."""
    if rng is None:
        rng = random.Random()

    while True:
        M = np.array([[rng.randint(0, 1) for _ in range(n)] for _ in range(n)], dtype=int)
        if gf2_rank(M) == n:
            return M


def random_phase_functions(
    n: int,
    m: int,
    rng: random.Random | None = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate m random non-zero linear Boolean functions f_i : {0,1}^n -> {0,1}
    as row vectors in {0,1}^n, plus random coefficients c_i in {1,...,7}.
    """
    if rng is None:
        rng = random.Random()
    F_list: List[np.ndarray] = []
    c_list: List[int] = []
    while len(F_list) < m:
        v = np.array([rng.randint(0, 1) for _ in range(n)], dtype=int)
        if np.all(v == 0):
            continue
        F_list.append(v)
        c_list.append(rng.randint(1, 7))  # T (1), S (2), T† (7), etc.
    return F_list, c_list


# ---------------------------------------------------------------------------
# 2. {CNOT, PHASE} gate-list utilities
# ---------------------------------------------------------------------------


def random_cnot_phase_circuit(
    n: int,
    num_cx: int = 5,
    num_phase: int = 3,
    rng: random.Random | None = None,
) -> List[Tuple[str, int, int]]:
    """
    Generate a random {CNOT, PHASE} circuit on n qubits.

    Returns:
        gates: list of ("CX", ctrl, targ) or ("PHASE", wire, coeff)
    """
    if rng is None:
        rng = random.Random()

    gates = []
    types = ["CX"] * num_cx + ["PHASE"] * num_phase
    rng.shuffle(types)

    for gate_type in types:
        if gate_type == "CX":
            ctrl = rng.randrange(n)
            targ = rng.randrange(n)
            while targ == ctrl:
                targ = rng.randrange(n)
            gates.append(("CX", ctrl, targ))
        else:
            wire = rng.randrange(n)
            coeff = rng.randint(1, 7)  # T^1..T^7
            gates.append(("PHASE", wire, coeff))

    return gates


def pretty_print_gate_list(gates: List[Tuple[str, int, int]]) -> None:
    """Print a human-readable description of a {CNOT, PHASE} gate list."""
    print("Original circuit:")
    for idx, g in enumerate(gates):
        if g[0] == "CX":
            _, c, t = g
            print(f"  {idx:2d}: CX(q{c}, q{t})")
        else:
            _, w, c = g
            print(f"  {idx:2d}: PHASE coeff={c} on q{w}")


# ---------------------------------------------------------------------------
# 3. Phase-polynomial extraction from gate lists
# ---------------------------------------------------------------------------


def phase_polynomial_from_gate_list(
    gates: List[Tuple[str, int, int]],
    n: int,
) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    """
    Given a {CNOT, PHASE} circuit as a gate list, extract:

        - G      : n x n GF(2) matrix (final linear reversible map)
        - F_list : list of row vectors in {0,1}^n
        - c_list : list of corresponding coefficients in {0,...,7}

    Algorithm:
        - Maintain A as n x n identity (tracks linear map).
        - For each gate:
            - If CX(c,t): A[t,:] ^= A[c,:]
            - If PHASE(w, coeff): record F = A[w,:] and coefficient.
    """
    A = np.eye(n, dtype=int)
    F_list: List[np.ndarray] = []
    c_list: List[int] = []

    for g in gates:
        if g[0] == "CX":
            _, c, t = g
            A[t, :] ^= A[c, :]
        else:
            _, w, coeff = g
            F = A[w, :].copy()
            F_list.append(F)
            c_list.append(coeff)

    G = A
    return G, F_list, c_list


# ---------------------------------------------------------------------------
# 4. Qiskit-based Clifford+T benchmarks
# ---------------------------------------------------------------------------


def random_clifford_t_circuit(
    n: int,
    num_cx: int = 6,
    num_t: int = 4,
    num_s: int = 2,
    seed: int | None = None,
) -> QuantumCircuit:
    """
    Generate a random {CX, T, Tdg, S, Sdg} circuit on n qubits.

    We *avoid* H, Rx, Rz, etc., so the whole circuit is one {CNOT, phase} block.
    """
    rng = random.Random(seed)
    qc = QuantumCircuit(n)

    types = (["cx"] * num_cx +
             ["t_or_tdg"] * num_t +
             ["s_or_sdg"] * num_s)
    rng.shuffle(types)

    for gate_type in types:
        if gate_type == "cx":
            ctrl = rng.randrange(n)
            targ = rng.randrange(n)
            while targ == ctrl:
                targ = rng.randrange(n)
            qc.cx(ctrl, targ)
        elif gate_type == "t_or_tdg":
            q = rng.randrange(n)
            if rng.random() < 0.5:
                qc.t(q)
            else:
                qc.tdg(q)
        else:
            q = rng.randrange(n)
            if rng.random() < 0.5:
                qc.s(q)
            else:
                qc.sdg(q)

    return qc


def clifford_t_qiskit_to_gate_list(qc: QuantumCircuit) -> List[Tuple[str, int, int]]:
    """
    Convert a Qiskit QuantumCircuit that only uses
    {cx, t, tdg, s, sdg} into our internal gate list:

        ("CX", ctrl, targ)
        ("PHASE", wire, coeff)    where coeff in {1,...,7}

    Coeff mapping (mod 8, exponent of π/4):
        T      -> +1
        Tdg    -> +7
        S      -> +2
        Sdg    -> +6
    """
    gates: List[Tuple[str, int, int]] = []

    for inst in qc.data:
        op = inst.operation
        qargs = inst.qubits
        name = op.name

        if name == "cx":
            c = qc.find_bit(qargs[0]).index
            t = qc.find_bit(qargs[1]).index
            gates.append(("CX", c, t))
        elif name == "t":
            w = qc.find_bit(qargs[0]).index
            gates.append(("PHASE", w, 1))
        elif name == "tdg":
            w = qc.find_bit(qargs[0]).index
            gates.append(("PHASE", w, 7))
        elif name == "s":
            w = qc.find_bit(qargs[0]).index
            gates.append(("PHASE", w, 2))
        elif name == "sdg":
            w = qc.find_bit(qargs[0]).index
            gates.append(("PHASE", w, 6))
        else:
            raise ValueError(f"Unsupported gate in Clifford+T block: {name}")

    return gates


# ---------------------------------------------------------------------------
# 5. SAT encoding of HasCNOT(G, F_i, c_i, K)
# ---------------------------------------------------------------------------


class HasCNOTEncoding:
    """
    Holds Z3 variables and solver for the HasCNOT(G, F_i, c_i, K) instance.

    Variables:
        - A[k][i][j] : matrix A_k entry (k=0..K , i=0..n-1, j=0..n-1)
        - q[k][i]    : control selection for gate k (k=0..K-1, i=0..n-1)
        - t[k][i]    : target selection for gate k (k=0..K-1, i=0..n-1)
        - h[k][j]    : helper for column j at step k (k=0..K-1, j=0..n-1)
        - use[r][k][i]: phase function r placed as row i of A_k (r=0..m-1, k=0..K, i=0..n-1)
    """

    def __init__(self, G: np.ndarray, F_list: List[np.ndarray], K: int):
        self.G = np.array(G, dtype=int)
        self.F_list = [np.array(F, dtype=int) for F in F_list]
        self.K = K
        self.n = self.G.shape[0]
        self.m = len(self.F_list)

        self.solver = Solver()
        self._make_variables()
        self._add_constraints()

    # ------------- variable creation -------------

    def _make_variables(self) -> None:
        n, K, m = self.n, self.K, self.m

        self.A = [
            [[Bool(f"A_{k}_{i}_{j}") for j in range(n)] for i in range(n)]
            for k in range(K + 1)
        ]
        self.q = [
            [Bool(f"q_{k}_{i}") for i in range(n)]
            for k in range(K)
        ]
        self.t = [
            [Bool(f"t_{k}_{i}") for i in range(n)]
            for k in range(K)
        ]
        self.h = [
            [Bool(f"h_{k}_{j}") for j in range(n)]
            for k in range(K)
        ]
        self.use = [
            [
                [Bool(f"use_{r}_{k}_{i}") for i in range(n)]
                for k in range(K + 1)
            ]
            for r in range(m)
        ]

    # ------------- constraints -------------

    def _add_constraints(self) -> None:
        self._init_matrix_constraints()
        self._gate_selection_constraints()
        self._matrix_transition_constraints()
        self._final_matrix_constraints()
        self._phase_placement_constraints()

    def _init_matrix_constraints(self) -> None:
        """A_0 = Identity."""
        n = self.n
        A0 = self.A[0]
        for i in range(n):
            for j in range(n):
                self.solver.add(A0[i][j] == BoolVal(1 if i == j else 0))

    def _gate_selection_constraints(self) -> None:
        """
        Exactly one control and one target for each gate k, and control != target.
        """
        n, K = self.n, self.K

        for k in range(K):
            qk = self.q[k]
            tk = self.t[k]

            # at least one control
            self.solver.add(Or(qk))
            # at most one control
            for i in range(n):
                for j in range(i + 1, n):
                    self.solver.add(Or(Not(qk[i]), Not(qk[j])))

            # at least one target
            self.solver.add(Or(tk))
            # at most one target
            for i in range(n):
                for j in range(i + 1, n):
                    self.solver.add(Or(Not(tk[i]), Not(tk[j])))

            # control != target (no gate with same wire as control and target)
            for i in range(n):
                self.solver.add(Or(Not(qk[i]), Not(tk[i])))

    def _matrix_transition_constraints(self) -> None:
        """
        A_{k+1} encodes the effect of the k-th CNOT on A_k.

        h_{k,j} = OR_i (A_k[i,j] & q_{k,i})
        A_{k+1}[i,j] = A_k[i,j] XOR (t_{k,i} & h_{k,j})
        """
        n, K = self.n, self.K

        for k in range(K):
            Ak = self.A[k]
            Ak1 = self.A[k + 1]
            qk = self.q[k]
            tk = self.t[k]
            hk = self.h[k]

            for j in range(n):
                self.solver.add(
                    hk[j] == Or(
                        [And(Ak[i][j], qk[i]) for i in range(n)]
                    )
                )

            for i in range(n):
                for j in range(n):
                    self.solver.add(
                        Ak1[i][j] == Xor(
                            Ak[i][j],
                            And(tk[i], hk[j]),
                        )
                    )

    def _final_matrix_constraints(self) -> None:
        """A_K must equal the target G."""
        AK = self.A[self.K]
        n = self.n
        for i in range(n):
            for j in range(n):
                self.solver.add(
                    AK[i][j] == BoolVal(1 if self.G[i, j] == 1 else 0)
                )

    def _phase_placement_constraints(self) -> None:
        """
        For each phase function F_r, ensure it appears as some row of some A_k.
        For each r:
            OR_{k,i} use[r][k][i]
        and
            use[r][k][i] -> (row i of A_k equals F_r)
        """
        n, K, m = self.n, self.K, self.m

        for r in range(m):
            F = self.F_list[r]

            # At least one placement.
            placements = []
            for k in range(K + 1):
                for i in range(n):
                    placements.append(self.use[r][k][i])
            self.solver.add(Or(placements))

            # Row equality when used.
            for k in range(K + 1):
                Ak = self.A[k]
                for i in range(n):
                    u = self.use[r][k][i]
                    row_equal = []
                    for j in range(n):
                        row_equal.append(
                            Ak[i][j] == BoolVal(1 if F[j] == 1 else 0)
                        )
                    self.solver.add(
                        Implies(u, And(row_equal))
                    )


# ---------------------------------------------------------------------------
# 6. Synthesis and model decoding
# ---------------------------------------------------------------------------


@dataclass
class SynthesizedCircuit:
    """
    Container for a synthesized {CNOT, T} circuit:

    - n            : number of qubits
    - K            : number of CNOTs
    - cnot_gates   : list of (control, target)
    - phase_locs   : list of (r, k, i) where phase function r is realized
    - F_list       : phase function row vectors
    - c_list       : coefficients for each phase term
    """

    n: int
    K: int
    cnot_gates: List[Tuple[int, int]]
    phase_locs: List[Tuple[int, int, int]]
    F_list: List[np.ndarray]
    c_list: List[int]

    def __str__(self) -> str:
        lines = []
        lines.append(f"Synthesized {len(self.cnot_gates)}-CNOT circuit on {self.n} qubits")
        lines.append("CNOT sequence (in order):")
        for idx, (c, t) in enumerate(self.cnot_gates):
            lines.append(f"  step {idx}: CX(q{c}, q{t})")
        lines.append("Phase placements:")
        for r, k, i in self.phase_locs:
            coeff = self.c_list[r]
            lines.append(
                f"  phase term r={r} (coeff {coeff}) placed at A_{k} row {i} (wire {i})"
            )
        return "\n".join(lines)


def decode_model_to_circuit(model, enc: HasCNOTEncoding, c_list: List[int]) -> SynthesizedCircuit:
    """
    Given a SAT model and the encoding, extract the sequence of CNOTs
    and the phase placements.
    """
    n, K, m = enc.n, enc.K, enc.m

    cnot_gates: List[Tuple[int, int]] = []
    for k in range(K):
        control = None
        target = None
        for i in range(n):
            if model.evaluate(enc.q[k][i], model_completion=True):
                control = i
            if model.evaluate(enc.t[k][i], model_completion=True):
                target = i
        assert control is not None and target is not None
        cnot_gates.append((control, target))

    phase_locs: List[Tuple[int, int, int]] = []
    for r in range(m):
        found = False
        for k in range(K + 1):
            for i in range(n):
                if model.evaluate(enc.use[r][k][i], model_completion=True):
                    phase_locs.append((r, k, i))
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError(f"No placement found for phase function r={r}")

    return SynthesizedCircuit(
        n=n,
        K=K,
        cnot_gates=cnot_gates,
        phase_locs=phase_locs,
        F_list=enc.F_list,
        c_list=c_list,
    )


def synthesize_cnot_t(
    G: np.ndarray,
    F_list: List[np.ndarray],
    c_list: List[int],
    K_max: int = 6,
    verbose: bool = True,
) -> SynthesizedCircuit | None:
    """
    Try K = 0..K_max, find smallest K such that HasCNOT(G, F_i, c_i, K) is satisfiable.
    """
    G = np.array(G, dtype=int)
    F_list = [np.array(F, dtype=int) for F in F_list]

    for K in range(K_max + 1):
        if verbose:
            print(f"[SAT] Trying K = {K} CNOTs...")
        enc = HasCNOTEncoding(G, F_list, K)
        result = enc.solver.check()
        if result == sat:
            if verbose:
                print(f"  SAT for K = {K}!")
            model = enc.solver.model()
            return decode_model_to_circuit(model, enc, c_list)
        else:
            if verbose:
                print(f"  UNSAT for K = {K}.")
    if verbose:
        print("[SAT] No solution up to K_max.")
    return None


# ---------------------------------------------------------------------------
# 7. Verification / simulation
# ---------------------------------------------------------------------------


def int_to_bits(x: int, n: int) -> np.ndarray:
    """Convert integer x to length-n numpy array of bits (LSB at index 0)."""
    return np.array([(x >> i) & 1 for i in range(n)], dtype=int)


def bits_to_int(bits: np.ndarray) -> int:
    """Convert numpy array of bits (LSB at index 0) to integer."""
    return int(sum((int(b) << i) for i, b in enumerate(bits)))


def apply_linear_G(G: np.ndarray, x_bits: np.ndarray) -> np.ndarray:
    """Compute y = G * x (mod 2) for n x n binary matrix G and bitvector x_bits."""
    return (G @ x_bits) % 2


def eval_phase_poly(
    F_list: List[np.ndarray],
    c_list: List[int],
    x_bits: np.ndarray,
) -> int:
    """
    Evaluate the phase polynomial:
        e(x) = sum_r c_r * F_r(x) (mod 8),
    where F_r(x) = dot(F_r, x_bits) mod 2.
    Return exponent e(x) in {0,...,7}.
    """
    e = 0
    for F, c in zip(F_list, c_list):
        val = int(np.dot(F, x_bits) % 2)
        e += c * val
    return e % 8


def simulate_synthesized_circuit(
    circuit: SynthesizedCircuit,
    x_bits: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Simulate the synthesized {CNOT, phase} circuit on input bitstring x_bits.

    Model:
      - CNOTs act as classical reversible gates on bits.
      - A phase term r with coefficient c_r placed at (k, i) adds
        c_r * z_i to the phase exponent, where z is the current bitstring
        after k-1 CNOTs (if k>0) or initially (if k=0).
    """
    n = circuit.n
    K = circuit.K
    z = x_bits.copy()
    e_syn = 0

    phase_at_step: dict[int, List[Tuple[int, int]]] = {k: [] for k in range(K + 1)}
    for (r, k, i) in circuit.phase_locs:
        phase_at_step[k].append((r, i))

    for k in range(K + 1):
        # apply phases at this step
        for (r, i) in phase_at_step[k]:
            c_r = circuit.c_list[r]
            e_syn = (e_syn + c_r * int(z[i])) % 8

        # then the k-th CNOT
        if k < K:
            ctrl, targ = circuit.cnot_gates[k]
            if z[ctrl] == 1:
                z[targ] ^= 1

    return z, e_syn


def verify_circuit(
    G: np.ndarray,
    F_list: List[np.ndarray],
    c_list: List[int],
    circuit: SynthesizedCircuit,
    verbose: bool = True,
) -> bool:
    """
    Verify that the synthesized circuit implements the same (G, F, c)
    specification on all 2^n basis inputs.
    """
    n = G.shape[0]
    F_list_np = [np.array(F, dtype=int) for F in F_list]

    for x in range(1 << n):
        x_bits = int_to_bits(x, n)
        y_target = apply_linear_G(G, x_bits)
        e_target = eval_phase_poly(F_list_np, c_list, x_bits)
        y_syn, e_syn = simulate_synthesized_circuit(circuit, x_bits)
        if not np.array_equal(y_target, y_syn) or e_target != e_syn:
            if verbose:
                print("Verification FAILED for input x =", x)
                print("  x_bits   =", x_bits)
                print("  y_target =", y_target, "e_target =", e_target)
                print("  y_syn    =", y_syn, "e_syn    =", e_syn)
            return False

    if verbose:
        print("Verification PASSED for all", 1 << n, "inputs.")
    return True


# ---------------------------------------------------------------------------
# 8. Mapping synthesized circuits back to Qiskit Clifford+T
# ---------------------------------------------------------------------------


def emit_phase_for_coeff(qc: QuantumCircuit, qubit: int, coeff: int) -> None:
    """
    Emit Clifford+T gates on 'qubit' that implement exp(i π/4 * coeff)
    using gates in {T, Tdg, S}.

    Mapping (up to global phase):
      k=0: I
      k=1: T
      k=2: S
      k=3: S; T
      k=4: S; S    (Z)
      k=5: S; S; T
      k=6: S; S; S
      k=7: Tdg
    """
    k = coeff % 8
    if k == 0:
        return
    elif k == 1:
        qc.t(qubit)
    elif k == 2:
        qc.s(qubit)
    elif k == 3:
        qc.s(qubit)
        qc.t(qubit)
    elif k == 4:
        qc.s(qubit)
        qc.s(qubit)
    elif k == 5:
        qc.s(qubit)
        qc.s(qubit)
        qc.t(qubit)
    elif k == 6:
        qc.s(qubit)
        qc.s(qubit)
        qc.s(qubit)
    elif k == 7:
        qc.tdg(qubit)


def synthesized_to_qiskit_block(
    synth: SynthesizedCircuit,
    n_qubits: int,
) -> QuantumCircuit:
    """
    Convert a SynthesizedCircuit (CNOT + phase placements) into
    a Qiskit QuantumCircuit on 'n_qubits' qubits (using CX + T/Tdg/S).
    """
    qc = QuantumCircuit(n_qubits)
    K = synth.K

    phase_at_step: dict[int, List[Tuple[int, int]]] = {k: [] for k in range(K + 1)}
    for (r, k, i) in synth.phase_locs:
        phase_at_step[k].append((r, i))

    for k in range(K + 1):
        for (r, wire) in phase_at_step[k]:
            coeff = synth.c_list[r]
            emit_phase_for_coeff(qc, wire, coeff)
        if k < K:
            ctrl, targ = synth.cnot_gates[k]
            qc.cx(ctrl, targ)

    return qc


# ---------------------------------------------------------------------------
# 9. Optimize Qiskit circuits by SAT-synthesizing Clifford+T blocks
# ---------------------------------------------------------------------------


CT_NAMES = {"cx", "t", "tdg"}


def count_t_tdg_in_qc(qc: QuantumCircuit) -> tuple[int, int]:
    """Return (num_t, num_tdg) in a Qiskit circuit."""
    num_t = 0
    num_tdg = 0
    for inst in qc.data:
        name = inst.operation.name
        if name == "t":
            num_t += 1
        elif name == "tdg":
            num_tdg += 1
    return num_t, num_tdg


def _qiskit_qubit_index(qc: QuantumCircuit, q) -> int:
    """Get integer index of a Qiskit qubit in qc."""
    return qc.find_bit(q).index


def simulate_ct_block_on_basis(
    qc: QuantumCircuit,
    x_bits: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Classically simulate a circuit containing only {cx, t, tdg, s, sdg} on a computational basis input.

    Returns:
      (output_bits, phase_exponent_mod8)
    where the state transforms as:
      |x> -> exp(i*pi/4 * e(x)) |y>
    for some y in {0,1}^n and e(x) in Z_8.
    """
    n = qc.num_qubits
    z = np.array(x_bits, dtype=int).copy()
    if z.shape != (n,):
        raise ValueError(f"x_bits must have shape (n,), got {z.shape} for n={n}")

    e = 0
    for inst in qc.data:
        name = inst.operation.name
        qargs = inst.qubits

        if name == "cx":
            c = _qiskit_qubit_index(qc, qargs[0])
            t = _qiskit_qubit_index(qc, qargs[1])
            if z[c] == 1:
                z[t] ^= 1

        elif name == "t":
            q = _qiskit_qubit_index(qc, qargs[0])
            e = (e + 1 * int(z[q])) % 8

        elif name == "tdg":
            q = _qiskit_qubit_index(qc, qargs[0])
            e = (e + 7 * int(z[q])) % 8

        elif name == "s":
            q = _qiskit_qubit_index(qc, qargs[0])
            e = (e + 2 * int(z[q])) % 8

        elif name == "sdg":
            q = _qiskit_qubit_index(qc, qargs[0])
            e = (e + 6 * int(z[q])) % 8

        elif name in ("barrier", "id"):
            # ignore
            continue

        else:
            raise ValueError(
                f"simulate_ct_block_on_basis: unsupported gate '{name}'. "
                "This checker only supports {cx,t,tdg,s,sdg} (plus barrier/id)."
            )

    return z, e


def ct_blocks_equivalent(
    qc_a: QuantumCircuit,
    qc_b: QuantumCircuit,
    *,
    exhaustive_n: int = 10,
    num_random: int = 256,
    seed: int = 0,
    verbose: bool = False,
) -> bool:
    """Check equivalence of two {CNOT, phase} circuits on computational basis states.

    We compare:
      - output bitstrings
      - phase exponent mod 8

    We allow an OPTIONAL global phase offset delta in Z_8 (independent of x),
    i.e., e_b(x) == e_a(x) + delta (mod 8) for all tested x.
    """
    if qc_a.num_qubits != qc_b.num_qubits:
        if verbose:
            print(f"[EQUIV] qubit mismatch: {qc_a.num_qubits} vs {qc_b.num_qubits}")
        return False

    n = qc_a.num_qubits
    rng = random.Random(seed)

    def iter_tests():
        if n <= exhaustive_n:
            for x in range(1 << n):
                bits = np.array([(x >> i) & 1 for i in range(n)], dtype=int)
                yield bits
        else:
            for _ in range(num_random):
                bits = np.array([rng.randint(0, 1) for _ in range(n)], dtype=int)
                yield bits

    delta = None
    for t_idx, x_bits in enumerate(iter_tests()):
        ya, ea = simulate_ct_block_on_basis(qc_a, x_bits)
        yb, eb = simulate_ct_block_on_basis(qc_b, x_bits)

        if not np.array_equal(ya, yb):
            if verbose:
                print(f"[EQUIV] output bits differ at test {t_idx}")
                print("  x =", x_bits)
                print("  y_a=", ya, "e_a=", ea)
                print("  y_b=", yb, "e_b=", eb)
            return False

        d = (eb - ea) % 8
        if delta is None:
            delta = d
        elif d != delta:
            if verbose:
                print(f"[EQUIV] phase mismatch not explainable by global offset at test {t_idx}")
                print("  x =", x_bits)
                print("  y =", ya)
                print("  e_a=", ea, "e_b=", eb, "delta_expected=", delta, "delta_got=", d)
            return False

    return True



def find_ct_blocks(qc: QuantumCircuit) -> List[Tuple[int, int]]:
    """
    Find maximal contiguous blocks of gates limited to
    {cx, t, tdg} (to match the notebook’s CT-block definition).

    Returns:
        list of (start, end) indices, meaning block is qc.data[start:end]
    """
    blocks: List[Tuple[int, int]] = []
    start = None
    data = qc.data

    for idx, inst in enumerate(data):
        name = inst.operation.name
        if name in CT_NAMES:
            if start is None:
                start = idx
        else:
            if start is not None:
                blocks.append((start, idx))
                start = None
    if start is not None:
        blocks.append((start, len(data)))

    return blocks


def optimize_qiskit_with_sat(
    qc: QuantumCircuit,
    K_max: int | None = 10,
    verbose: bool = True,
    preserve_t_count: bool = True,
    equiv_check: bool = True,
    equiv_exhaustive_n: int = 10,
    equiv_num_random: int = 256,
    equiv_seed: int = 0,
) -> QuantumCircuit:
    """
    Optimize a Qiskit circuit by replacing each {cx, t, tdg} block
    with a SAT-synthesized {CNOT, T} circuit of minimal CNOT count (up to K_max).
    """
    n = qc.num_qubits
    new_qc = QuantumCircuit(n)
    data = qc.data

    idx = 0
    while idx < len(data):
        inst = data[idx]
        name = inst.operation.name

        if name in CT_NAMES:
            start = idx
            j = idx
            while j < len(data) and data[j].operation.name in CT_NAMES:
                j += 1
            end = j

            if verbose:
                print(f"Found CT-block from instr {start} to {end} (len={end-start})")

            block_qc = QuantumCircuit(n)
            for k in range(start, end):
                op = data[k].operation
                qargs = data[k].qubits
                clargs = data[k].clbits
                block_qc.append(op, qargs, clargs)

            # Count T/Tdg in the original block (post-preprocessing).
            orig_t, orig_tdg = count_t_tdg_in_qc(block_qc)

            gates = clifford_t_qiskit_to_gate_list(block_qc)
            G, F_list, c_list = phase_polynomial_from_gate_list(gates, n)
            block_cx = sum(1 for g in gates if g[0] == "CX")

            if verbose:
                print("  Extracted G:")
                print(G)
                print("  #Phase terms:", len(F_list))
                print("  Block CNOTs:", block_cx)

            local_K_max = block_cx if K_max is None else min(K_max, block_cx)

            synth = synthesize_cnot_t(G, F_list, c_list, K_max=local_K_max, verbose=verbose)

            if synth is None:
                if verbose:
                    print("  SAT failed up to local_K_max, copying original block.")
                new_qc.compose(block_qc, qubits=range(n), inplace=True)
            else:
                opt_block = synthesized_to_qiskit_block(synth, n)
                if preserve_t_count:
                    new_t, new_tdg = count_t_tdg_in_qc(opt_block)
                    if (new_t != orig_t) or (new_tdg != orig_tdg):
                        if verbose:
                            print(
                                "  [WARN] T/Tdg count changed by SAT replacement "
                                f"(T {orig_t}->{new_t}, Tdg {orig_tdg}->{new_tdg}). "
                                "Keeping original block to preserve T-count."
                            )
                        new_qc.compose(block_qc, qubits=range(n), inplace=True)
                        idx = end
                        continue

                if equiv_check:
                    ok_eq = ct_blocks_equivalent(
                        block_qc,
                        opt_block,
                        exhaustive_n=equiv_exhaustive_n,
                        num_random=equiv_num_random,
                        seed=equiv_seed,
                        verbose=verbose,
                    )
                    if not ok_eq:
                        if verbose:
                            print(
                                "  [WARN] Equivalence check FAILED for SAT replacement. "
                                "Keeping original block."
                            )
                        new_qc.compose(block_qc, qubits=range(n), inplace=True)
                        idx = end
                        continue
                if verbose:
                    print(
                        "  Replaced block with optimized circuit: "
                        f"{block_cx} CX -> {len(synth.cnot_gates)} CX"
                    )
                new_qc.compose(opt_block, qubits=range(n), inplace=True)

            idx = end
        else:
            new_qc.append(inst.operation, inst.qubits, inst.clbits)
            idx += 1

    return new_qc


# ---------------------------------------------------------------------------
# 10. Benchmark drivers
# ---------------------------------------------------------------------------


def run_random_benchmarks(
    num_bench: int = 5,
    n: int = 3,
    m: int = 3,
    K_max: int = 6,
    seed: int = 0,
) -> None:
    """
    Run random benchmarks on (G, F, c) directly.
    """
    rng = random.Random(seed)
    for b in range(num_bench):
        print("=" * 60)
        print(f"Benchmark {b+1}/{num_bench}, n={n}, m={m}")
        G = random_invertible_matrix(n, rng)
        F_list, c_list = random_phase_functions(n, m, rng)

        print("Target G (matrix over GF(2)):")
        print(G)
        print("Phase functions F_i and coefficients c_i:")
        for i, (F, c) in enumerate(zip(F_list, c_list)):
            print(f"  F_{i} = {F}, c_{i} = {c}")

        circuit = synthesize_cnot_t(G, F_list, c_list, K_max=K_max, verbose=True)
        if circuit is None:
            print("No circuit found up to K_max.")
        else:
            print("\n--- Synthesized circuit ---")
            print(circuit)
            print("\n--- Verifying synthesized circuit ---")
            ok = verify_circuit(G, F_list, c_list, circuit, verbose=True)
            print("Verification result:", "OK" if ok else "FAIL")
        print()


def run_gate_list_benchmarks(
    num_bench: int = 5,
    n: int = 3,
    num_cx: int = 5,
    num_phase: int = 3,
    K_max: int = 10,
    seed: int = 0,
) -> None:
    """
    Benchmarks where we:
      - generate random {CNOT, PHASE} gate lists
      - extract (G, F, c)
      - synthesize optimal {CNOT, T} circuits
    """
    rng = random.Random(seed)

    for b in range(num_bench):
        print("=" * 60)
        print(f"Gate-list Benchmark {b+1}/{num_bench}, n={n}, CX={num_cx}, PHASE={num_phase}")

        gates = random_cnot_phase_circuit(n, num_cx=num_cx, num_phase=num_phase, rng=rng)
        pretty_print_gate_list(gates)
        original_cx = sum(1 for g in gates if g[0] == "CX")
        print(f"Original CNOT count: {original_cx}")

        G, F_list, c_list = phase_polynomial_from_gate_list(gates, n)
        print("\nExtracted G (linear map):")
        print(G)
        print("Extracted phase terms (F_i, c_i):")
        for i, (F, c) in enumerate(zip(F_list, c_list)):
            print(f"  F_{i} = {F}, c_{i} = {c}")

        circuit = synthesize_cnot_t(G, F_list, c_list, K_max=K_max, verbose=True)

        if circuit is None:
            print("No optimized circuit found up to K_max.")
        else:
            print("\n--- Synthesized (optimized) circuit ---")
            print(circuit)
            print(f"\nOptimized CNOT count: {len(circuit.cnot_gates)}")
            print(f"CNOT reduction: {original_cx} -> {len(circuit.cnot_gates)}")

            print("\n--- Verifying synthesized circuit ---")
            ok = verify_circuit(G, F_list, c_list, circuit, verbose=True)
            print("Verification result:", "OK" if ok else "FAIL")

        print()


def run_qiskit_clifford_t_benchmarks(
    num_bench: int = 5,
    n: int = 3,
    num_cx: int = 6,
    num_t: int = 4,
    num_s: int = 2,
    K_max: int = 10,
    seed: int = 0,
) -> None:
    """
    For each benchmark:
      1. Generate a random Qiskit circuit with {CX, T, Tdg, S, Sdg}.
      2. Convert it to our gate-list.
      3. Extract (G, F_list, c_list).
      4. Run SAT-based synthesis to minimize CNOT count.
      5. Verify equivalence on all 2^n inputs.
    """
    rng = random.Random(seed)

    for b in range(num_bench):
        print("=" * 60)
        print(
            f"Qiskit Clifford+T Benchmark {b+1}/{num_bench}, "
            f"n={n}, CX={num_cx}, T~{num_t}, S~{num_s}"
        )

        qc = random_clifford_t_circuit(
            n,
            num_cx=num_cx,
            num_t=num_t,
            num_s=num_s,
            seed=rng.randint(0, 10**9),
        )
        print("Original Qiskit circuit:")
        print(qc.draw(fold=-1))

        gates = clifford_t_qiskit_to_gate_list(qc)
        original_cx = sum(1 for g in gates if g[0] == "CX")
        print(f"Original CNOT count: {original_cx}")

        G, F_list, c_list = phase_polynomial_from_gate_list(gates, n)
        print("\nExtracted G (linear map):")
        print(G)
        print("Extracted phase terms (F_i, c_i):")
        for i, (F, c) in enumerate(zip(F_list, c_list)):
            print(f"  F_{i} = {F}, c_{i} = {c}")

        circuit = synthesize_cnot_t(G, F_list, c_list, K_max=K_max, verbose=True)

        if circuit is None:
            print("No optimized circuit found up to K_max.")
        else:
            print("\n--- Synthesized (optimized) circuit ---")
            print(circuit)
            print(f"\nOptimized CNOT count: {len(circuit.cnot_gates)}")
            print(f"CNOT reduction: {original_cx} -> {len(circuit.cnot_gates)}")

            print("\n--- Verifying synthesized circuit ---")
            ok = verify_circuit(G, F_list, c_list, circuit, verbose=True)
            print("Verification result:", "OK" if ok else "FAIL")

        print()


def demo_qiskit_end_to_end() -> None:
    """
    Demo: generate one random Clifford+T circuit, run SAT optimizer on it,
    and compare CNOT counts.
    """
    n = 3
    original = random_clifford_t_circuit(n, num_cx=6, num_t=4, num_s=2, seed=1234)

    print("Original Clifford+T circuit:")
    print(original.draw(fold=-1))

    orig_cx = sum(1 for inst in original.data if inst.operation.name == "cx")
    print("Original CNOT count:", orig_cx)

    opt = optimize_qiskit_with_sat(original, K_max=10, verbose=True)

    print("\nOptimized circuit:")
    print(opt.draw(fold=-1))
    opt_cx = sum(1 for inst in opt.data if inst.operation.name == "cx")
    print("Optimized CNOT count:", opt_cx)
    print("CNOT reduction:", orig_cx, "->", opt_cx)


# ---------------------------------------------------------------------------
# 11. Random Clifford+T experiment (stats)
# ---------------------------------------------------------------------------


def count_cx_in_qc(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.name == "cx")


def experiment_random_clifford_t(
    num_bench: int = 20,
    n: int = 3,
    num_cx: int = 6,
    num_t: int = 4,
    num_s: int = 2,
    K_max: int = 10,
    seed: int = 0,
) -> None:
    """
    Run an experiment on random Clifford+T circuits and gather statistics.
    """
    rng = random.Random(seed)
    results = []

    print(f"=== Random Clifford+T experiment (n={n}, CX={num_cx}, T={num_t}, S={num_s}) ===")
    for b in range(num_bench):
        print("=" * 60)
        print(f"[Exp] Circuit {b+1}/{num_bench}: n={n}, CX={num_cx}, T~{num_t}, S~{num_s}")

        qc = random_clifford_t_circuit(
            n,
            num_cx=num_cx,
            num_t=num_t,
            num_s=num_s,
            seed=rng.randint(0, 10**9),
        )

        gates = clifford_t_qiskit_to_gate_list(qc)
        orig_cx = sum(1 for g in gates if g[0] == "CX")
        G, F_list, c_list = phase_polynomial_from_gate_list(gates, n)

        print("  Original CNOT count:", orig_cx)
        print("  #Phase terms:", len(F_list))

        t0 = time.perf_counter()
        synth = synthesize_cnot_t(G, F_list, c_list, K_max=K_max, verbose=False)
        t1 = time.perf_counter()

        if synth is None:
            print("  [WARN] No solution found up to K_max; skipping this instance.")
            continue

        opt_cx = len(synth.cnot_gates)
        dt = t1 - t0

        ok = verify_circuit(G, F_list, c_list, synth, verbose=False)
        print(f"  Optimized CNOT count: {opt_cx}")
        print(f"  CNOT reduction: {orig_cx} -> {opt_cx}")
        print(f"  SAT solve time: {dt:.4f} s")
        print(f"  Verification: {'OK' if ok else 'FAIL'}")

        results.append((orig_cx, opt_cx, dt, ok))

    if not results:
        print("No successful instances to summarize.")
        return

    total_orig = sum(r[0] for r in results)
    total_opt = sum(r[1] for r in results)
    total_time = sum(r[2] for r in results)
    num_ok = sum(1 for r in results if r[3])

    n_inst = len(results)
    avg_orig = total_orig / n_inst
    avg_opt = total_opt / n_inst
    avg_red = avg_orig - avg_opt
    avg_rel = (avg_red / avg_orig) if avg_orig > 0 else 0.0
    avg_time = total_time / n_inst

    print("=" * 60)
    print("SUMMARY over", n_inst, "instances")
    print(f"  Avg CNOT before:  {avg_orig:.2f}")
    print(f"  Avg CNOT after:   {avg_opt:.2f}")
    print(f"  Avg absolute red: {avg_red:.2f}")
    print(f"  Avg relative red: {avg_rel*100:.1f}%")
    print(f"  Avg SAT time:     {avg_time:.4f} s")
    print(f"  Verified OK on:   {num_ok}/{n_inst}")


# ---------------------------------------------------------------------------
# 12. QASM testbench experiment helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Notebook-compatible preprocessing (PyTKET) to match "initial CNOT count"
# ---------------------------------------------------------------------------

def preprocess_like_notebook_T_op(qc: QuantumCircuit):
    """Mimic the notebook's `T_op` preprocessing using PyTKET.

    Notebook pipeline:
      DecomposeBoxes -> AutoRebase(allowed_gates) -> RemoveRedundancies

    This function tries to use `pytket-qiskit` if installed. If not, it falls back to a
    QASM round-trip (Qiskit -> QASM -> PyTKET -> QASM -> Qiskit), which avoids the
    `pytket.extensions.qiskit` dependency.

    Returns:
      (tk_circ, qiskit_circ_after_passes)
    """
    from pytket.passes import DecomposeBoxes, AutoRebase
    try:
        from pytket.passes import RemoveRedundancies
    except Exception:
        RemoveRedundancies = None
    from pytket.circuit import OpType

    # -------- Qiskit -> PyTKET (prefer direct bridge; fallback via QASM) --------
    tk = None
    try:
        # Requires: pip install pytket-qiskit
        from pytket.extensions.qiskit import qiskit_to_tk
        tk = qiskit_to_tk(qc)
    except Exception:
        # Fallback: round-trip through OpenQASM 2
        try:
            from qiskit import qasm2
            qasm_str = qasm2.dumps(qc)
        except Exception:
            # Older Qiskit
            qasm_str = qc.qasm()

        try:
            # Some pytket versions provide circuit_from_qasm_str
            from pytket.qasm import circuit_from_qasm_str
            tk = circuit_from_qasm_str(qasm_str)
        except Exception:
            # Portable fallback: write a temp file then circuit_from_qasm
            import tempfile
            from pytket.qasm import circuit_from_qasm
            with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as f:
                f.write(qasm_str)
                tmp_path = f.name
            try:
                tk = circuit_from_qasm(tmp_path)
            finally:
                try:
                    import os
                    os.remove(tmp_path)
                except Exception:
                    pass

    # -------- Apply notebook-like passes --------
    DecomposeBoxes().apply(tk)

    # IMPORTANT: to match your notebook exactly, copy its allowed_gates here.
    allowed_gates = {
        OpType.CX,
        OpType.H, OpType.S, OpType.T, OpType.Tdg,
        OpType.Rz, OpType.X, OpType.Z, OpType.Y,
    }
    AutoRebase(allowed_gates).apply(tk)
    if RemoveRedundancies is not None:
        RemoveRedundancies().apply(tk)

    # -------- PyTKET -> Qiskit (prefer direct bridge; fallback via QASM) --------
    qc_after = None
    try:
        from pytket.extensions.qiskit import tk_to_qiskit
        qc_after = tk_to_qiskit(tk)
    except Exception:
        # Fallback via QASM export
        try:
            from pytket.qasm import circuit_to_qasm_str
            out_qasm = circuit_to_qasm_str(tk)
            qc_after = QuantumCircuit.from_qasm_str(out_qasm)
        except Exception:
            import tempfile
            from pytket.qasm import circuit_to_qasm
            with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as f:
                tmp_out = f.name
            try:
                circuit_to_qasm(tk, tmp_out)
                qc_after = QuantumCircuit.from_qasm_file(tmp_out)
            finally:
                try:
                    import os
                    os.remove(tmp_out)
                except Exception:
                    pass

    return tk, qc_after
def count_cx_in_tk(tk_circ) -> int:
    """Count CX in a PyTKET circuit (same rule as the notebook)."""
    from pytket.circuit import OpType
    return sum(1 for cmd in tk_circ.get_commands() if cmd.op.type == OpType.CX)

def load_qasm_circuits(pattern: str = "testbench/*.qasm", max_files: int | None = None):
    paths = sorted(glob.glob(pattern))
    if max_files is not None:
        paths = paths[:max_files]
    circuits = []
    for p in paths:
        qc = QuantumCircuit.from_qasm_file(p)
        circuits.append((p, qc))
    return circuits


def experiment_on_testbench_qasm(
    pattern: str = "testbench/*.qasm",
    max_files: int = 10,
    decompose_reps: int = 3,
    K_max: int | None = None,
    verbose_blocks: bool = False,
) -> None:
    """
    Run SAT-based optimization on Toffoli+CNOT QASM benchmarks.

    For each file:
      - Load QASM (cx + ccx).
      - Preprocess with PyTKET passes (same as notebook T_op).
      - Count CNOTs.
      - Run optimize_qiskit_with_sat.
      - Report CNOT before/after.
    """
    circs = load_qasm_circuits(pattern, max_files=max_files)
    if not circs:
        print("No QASM files found for pattern:", pattern)
        return

    summary = []

    print(f"=== Testbench QASM experiment (first {max_files} files) ===")
    for path, qc in circs:
        print("=" * 60)
        print("Benchmark:", path)
        print("  Qubits:", qc.num_qubits)
        print("  Depth (raw):", qc.depth())

        # --- Match notebook's "initial CNOT" definition ---
        # Notebook runs: DecomposeBoxes -> AutoRebase(allowed_gates) -> RemoveRedundancies
        # and then counts CX on the *PyTKET* circuit.
        tk_circ, qc_pre = preprocess_like_notebook_T_op(qc)

        orig_cx = count_cx_in_tk(tk_circ)
        print("  CNOT count after notebook-style T_op:", orig_cx)

        # Run the SAT optimizer on the Qiskit circuit after the same preprocessing.
        opt = optimize_qiskit_with_sat(qc_pre, K_max=K_max, verbose=verbose_blocks)
        opt_cx = count_cx_in_qc(opt)

        print("  CNOT after SAT optimization:", opt_cx)
        rel = (orig_cx - opt_cx) / orig_cx * 100 if orig_cx > 0 else 0.0
        print(f"  CNOT reduction: {orig_cx} -> {opt_cx} ({rel:.1f}%)")

        summary.append((path, qc_pre.num_qubits, orig_cx, opt_cx))

    print("=" * 60)
    print("SUMMARY over", len(summary), "QASM benchmarks")
    total_orig = sum(s[2] for s in summary)
    total_opt = sum(s[3] for s in summary)
    avg_orig = total_orig / len(summary)
    avg_opt = total_opt / len(summary)
    avg_red = avg_orig - avg_opt
    avg_rel = (avg_red / avg_orig) * 100 if avg_orig > 0 else 0.0

    print(f"  Avg CNOT before: {avg_orig:.2f}")
    print(f"  Avg CNOT after:  {avg_opt:.2f}")
    print(f"  Avg abs red:     {avg_red:.2f}")
    print(f"  Avg rel red:     {avg_rel:.1f}%")


# ---------------------------------------------------------------------------
# 13. Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Default entry point: run a small random Clifford+T experiment
    and the QASM testbench experiment (if QASM files are present).
    """
    experiment_random_clifford_t(
        num_bench=20,
        n=3,
        num_cx=6,
        num_t=4,
        num_s=2,
        K_max=10,
        seed=123,
    )

    experiment_on_testbench_qasm(
        pattern="testbench/*.qasm",
        max_files=10,
        decompose_reps=3,
        K_max=None,
        verbose_blocks=False,
    )


if __name__ == "__main__":
    main()