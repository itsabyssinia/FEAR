#  egd (chargenet) -> gd(analytic) -> egd
import os, time, math, csv
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.geometry.analysis import Analysis
from ase.optimize import FIRE

USE_CHGNET = True
try:
    from chgnet.model import CHGNet  
    from helper import CHGNetCalculator
except Exception:
    USE_CHGNET = False
    from ase.calculators.emt import EMT

import gemmi  


inputs = {
    "crystal": "SiO2_mp-6930_computed.cif",
    "glass": "FinalGlass.data",
    "target_format": "lammps-data",
    "supercell": (2, 2, 2),
    "out_dir": "outputs",

    "cycles": 1,

    # egd
    "egd_steps": 50,
    "egd_fmax": 0.05,
    "egd_twice_per_cycle": True,  

    # rdf-gd
    "rdf_steps_per_cycle": 40,
    "rdf_lr": 0.010,
    "rdf_step_clip": 0.08,  

    # scattering / rdf model
    "mode": "geometric",     
    "s_eff": 0.10,          
    "force_weight": 0.0,     # not using this anymore. egd energy term
    "rdf_rmax": 4.0,
    "rdf_bins": 150,
    "sigma_factor": 1.0,    

    "plot_every_cycles": 5,
    "save_every_cycles": 5,

    # this is to stop if loss isn't going down
    "ew": 60,
    "et": 1e-7,
}

def ensure_dirs(root):
    for d in ["plots", "checkpoints", "logs"]:
        os.makedirs(os.path.join(root, d), exist_ok=True)

def safe_norm(vec):
    n = np.linalg.norm(vec)
    return max(n, 1e-12)


class grSolver:
    def __init__(self, initial, target, cfg, calculator=None):
        self.cfg = cfg
        self.atoms = initial.copy()
        self.natoms = len(initial)
        self.loss_calls = 0

        self.mode = cfg["mode"]
        self.s_eff = cfg["s_eff"]
        self.force_weight = cfg["force_weight"]

        # emt is just to run on my laptop
        self.calculator = calculator if calculator is not None else (CHGNetCalculator() if USE_CHGNET else EMT())

        if isinstance(target, (list, tuple)) and len(target) == 2:
            self.target = (np.array(target[0]), np.array(target[1]))
        else:
            self.target = self.ASEGr(target)

        self._wbar = self._compute_avg_pair_weight(self.atoms) 
        self.r = self.target[0]

        self.dr = np.mean(np.diff(self.r)) / 2.0
        self.sigma = self.cfg["sigma_factor"] * np.mean(np.diff(self.r))
        self.shell_volumes = (4.0/3.0)*np.pi * (((self.r + self.dr)**3) - ((self.r - self.dr)**3))

#weighting modes 
    def neutron_b(self, sym):
        n92 = gemmi.Element[sym].neutron92
        b = getattr(n92, "b_coh", None)
        if b is None:  
            b = getattr(n92, "b", 1.0)
        return float(b)

    def xray_f(self, sym, s):

        it = gemmi.Element[sym].it92.get_coefs()
        return float(it.calculate_sf(stol2=s*s))

    def pair_weight(self, sym_i, sym_j):
        if self.mode == "geometric":
            return 1.0
        elif self.mode == "neutron":
            return self.neutron_b(sym_i) * self.neutron_b(sym_j)
        elif self.mode == "xray":
            return self.xray_f(sym_i, self.s_eff) * self.xray_f(sym_j, self.s_eff)
        return 1.0

    def _compute_avg_pair_weight(self, atoms):
        syms = atoms.get_chemical_symbols()
        if len(syms) < 2:
            return 1.0
        s, c = 0.0, 0
        for i in range(len(syms)):
            for j in range(i+1, len(syms)):
                s += self.pair_weight(syms[i], syms[j])
                c += 1
        return s / c if c else 1.0

#ase rdf
    def ASEGr(self, atoms):
        ana = Analysis(atoms)
        rdf, r = ana.get_rdf(self.cfg["rdf_rmax"], self.cfg["rdf_bins"], return_dists=True)[0]
        return np.array(r), np.array(rdf)

#analytic g(r) 
    def calculate_gr_and_grad(self, atoms):
        r = self.r
        sigma = self.sigma
        N = len(atoms)
        rho = N / atoms.get_volume()
        wbar = self._wbar if self.mode != "geometric" else 1.0

        g_r = np.zeros(len(r))
        grad = np.zeros((N, 3, len(r)))

        rij_vec = atoms.get_all_distances(mic=True, vector=True)
        symbols = atoms.get_chemical_symbols()
        cutoff = self.cfg["rdf_rmax"] + 3*sigma 

        for i in range(N):
            for j in range(i+1, N):
                vec = rij_vec[i, j]
                rij = safe_norm(vec)
                if rij > cutoff:
                    continue

                w_ij = self.pair_weight(symbols[i], symbols[j])

                # gaussian
                weight = np.exp(-0.5*((r - rij)**2)/(sigma*sigma)) / math.sqrt(2.0*math.pi*sigma*sigma)
                g_r += 2.0 * w_ij * weight

                # this is derivative wrt distance and then projecting along vector
                dweight_drij = ((r - rij)/(sigma**2)) * weight * w_ij
                force_like = np.outer(dweight_drij / rij, vec)  # (len(r),3)

                grad[i] -= 2.0 * force_like.T
                grad[j] += 2.0 * force_like.T

        # normalization
        denom = (4.0*np.pi * r * r * N * rho)
        g_r /= (denom * wbar + 1e-12)
        grad /= ((rho * self.shell_volumes[np.newaxis, np.newaxis, :] * N * wbar) + 1e-12)

        return r, g_r, grad

    # this is only rdf loss. im doing energy in egd
    def loss(self, atom_positions):
        self.loss_calls += 1
        atoms = self.atoms.copy()
        atoms.set_positions(atom_positions.reshape((self.natoms, 3)))
        _, g_sim, _ = self.calculate_gr_and_grad(atoms)
        return np.sum((g_sim - self.target[1])**2)

    # analytic derivative
    def grad(self, atom_positions):
        atoms = self.atoms.copy()
        atoms.set_positions(atom_positions.reshape((self.natoms, 3)))
        _, g_sim, grad = self.calculate_gr_and_grad(atoms)
        delta = (g_sim - self.target[1])[np.newaxis, np.newaxis, :]
        g = 2.0 * np.sum(delta * grad, axis=2) 
        return g.flatten()

    # fire
    def energyMinimization(self, atoms):
        a = atoms.copy()
        a.calc = self.calculator
        opt = FIRE(a, logfile=None)
        opt.run(fmax=self.cfg["egd_fmax"], steps=self.cfg["egd_steps"])
        return a

    # this is what i replaced rmc with. it's deterministic rdf gd
    def rdf_gradient_updates(self, atoms):
        a = atoms.copy()
        pos = a.get_positions().copy()
        lr = self.cfg["rdf_lr"]
        clip = self.cfg["rdf_step_clip"]

        for _ in range(self.cfg["rdf_steps_per_cycle"]):
            g = self.grad(pos).reshape((self.natoms, 3))
            step = -lr * g

           
            if clip and clip > 0.0:
                norms = np.linalg.norm(step, axis=1) + 1e-12
                factors = np.minimum(1.0, clip / norms)
                step = step * factors[:, None]

            pos = pos + step
            a.set_positions(pos)
            a.wrap()

        return a

    def save_rdf_plot(self, r, rdf_init, rdf_target, rdf_curr, path_png):
        plt.figure()
        plt.plot(r, rdf_init, label="initial")
        plt.plot(r, rdf_target, label="target", linestyle="--")
        plt.plot(r, rdf_curr, label="current", linestyle="-.")
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_png, dpi=150)
        plt.close()

    # this is my main loop: EGD -> RDF-GD -> EGD
    def solve(self):
        cfg = self.cfg

        a = self.atoms.copy()
        r, rdf_init, _ = self.calculate_gr_and_grad(a)
        loss_trace = [self.loss(a.get_positions())]
        t_start = time.time()

        for cycle in range(1, cfg["cycles"] + 1):
            t0 = time.time()

            if cfg.get("egd_twice_per_cycle", True):
                a = self.energyMinimization(a)

            a = self.rdf_gradient_updates(a)
            a = self.energyMinimization(a)

            L = self.loss(a.get_positions())
            loss_trace.append(L)

            print(f"cycle {cycle}/{cfg['cycles']} | loss={L:.8e} | dt={time.time()-t0:.1f}s")

            if len(loss_trace) >= cfg["ew"]:
                recent = np.array(loss_trace[-cfg["ew"]:])
                if np.std(recent) < cfg["et"]:
                    print(f"early stop at cycle {cycle}: loss plateau (std < {cfg['et']}).")
                    break

        return a, loss_trace


if __name__ == "__main__":
    crystal = read(inputs["crystal"]) * inputs["supercell"]
    glass = read(inputs["glass"], format=inputs["target_format"])

    calc = CHGNetCalculator() if USE_CHGNET else EMT()

    solver = grSolver(crystal, glass, inputs, calc)
    final_atoms, loss_trace = solver.solve()

    r, rdf_init, _ = solver.calculate_gr_and_grad(read(inputs["crystal"]) * inputs["supercell"])
    _, rdf_final, _ = solver.calculate_gr_and_grad(final_atoms)
    final_plot = "rdf_final.png"
    solver.save_rdf_plot(r, rdf_init, solver.target[1], rdf_final, final_plot)

    print("heloooooo")
    print(f"saved: {final_plot}")
    r1, g1 = solver.ASEGr(crystal)
    r2, g2 = solver.ASEGr(glass)
    print(np.mean(np.abs(g1 - g2)))
