import os, time, math, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from ase.io import read, write
from ase.geometry.analysis import Analysis
from ase.optimize import FIRE
from ase import Atoms

USE_CHGNET = True
try:
    from chgnet.model import CHGNet  
    from helper import CHGNetCalculator
except Exception:
    USE_CHGNET = False
    from ase.calculators.emt import EMT  

import gemmi  


CFG = {
 
    "crystal_path": "SiO2_mp-6930_computed.cif",
    "target_path":  "FinalGlass.data",     
    "target_format": "lammps-data",
    "supercell": (2, 2, 2),                
    "randomize_initial": True,             
    "out_dir": "outputs",                  
    "plot_every_cycles": 5,                
    "save_every_cycles": 5,                

    "cycles": 400,                         
    "rmc_steps_per_cycle": 200,            
    "rmc_disp": 0.18,                      
    "rmc_temp": 0.6,                      
    "cool": 0.98,                          
    "adapt_temp": True,                    
    "target_acc_band": (0.20, 0.40),    
    "temp_up": 1.15, "temp_dn": 0.90,      
    "adapt_disp": True,                    
    "disp_up": 1.05, "disp_dn": 0.95,      
    "overlap_cut": 0.85,                   

    "FEAR": True,
    "fear_every": 3,                      
    "fear_steps": 25,                      
    "fear_fmax": 0.05,                     


    "mode": "geometric",                 
    "s_eff": 0.10,                         
    "force_weight": 0.005,                 
    "rdf_rmax": 4.0,                       
    "rdf_bins": 150,                     
    "sigma_factor": 1.0,                  
    "smooth_plots": False,                 
    "smooth_sigma": 0.15,

   
    "early_window": 60,                    
    "early_tol": 1e-7,                     
}


def ensure_dirs(root):
    sub = {}
    for name in ["plots", "checkpoints", "logs"]:
        path = os.path.join(root, name)
        os.makedirs(path, exist_ok=True)
        sub[name] = path
    return sub

def safe_norm(vec):
    n = np.linalg.norm(vec)
    return max(n, 1e-12)

def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())



class grSolver:
   
    def __init__(self, atoms, target, cfg, calculator=None):
        self.cfg = cfg
        self.atoms = atoms.copy()
        self.volume = atoms.get_volume()
        self.natoms = len(atoms)
        self.loss_calls = 0

        self.mode = cfg["mode"]
        self.s_eff = cfg["s_eff"]
        self.force_weight = cfg["force_weight"]


        if calculator is not None:
            self.calculator = calculator
        else:
            if USE_CHGNET:
                self.calculator = CHGNetCalculator()
            else:
                self.calculator = EMT()

  
        if isinstance(target, (list, tuple)) and len(target) == 2:
            self.target = (np.array(target[0]), np.array(target[1]))
        else:
            self.target = self.ASEGr(target) 

        self._wbar = self._compute_avg_pair_weight(self.atoms)


        self.r = self.target[0]
        self.dr = np.mean(np.diff(self.r)) / 2.0
        self.sigma = self.cfg["sigma_factor"] * np.mean(np.diff(self.r))
        self.shell_volumes = (4.0/3.0)*np.pi * (((self.r + self.dr)**3) - ((self.r - self.dr)**3))

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
        s = 0.0
        c = 0
        for i in range(len(syms)):
            for j in range(i+1, len(syms)):
                s += self.pair_weight(syms[i], syms[j])
                c += 1
        return s / c if c else 1.0


    def ASEGr(self, atoms):
        ana = Analysis(atoms)
        rdf, r = ana.get_rdf(self.cfg["rdf_rmax"], self.cfg["rdf_bins"], return_dists=True)[0]
        return np.array(r), np.array(rdf)

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

                weight = np.exp(-0.5*((r - rij)**2)/(sigma*sigma)) / math.sqrt(2.0*math.pi*sigma*sigma)
                g_r += 2.0 * w_ij * weight

                dweight_drij = ((r - rij)/(sigma**2)) * weight * w_ij
                force_like = np.outer(dweight_drij / rij, vec)  

                grad[i] -= 2.0 * force_like.T
                grad[j] += 2.0 * force_like.T

        denom = (4.0*np.pi * r * r * N * rho)
        g_r /= (denom * wbar + 1e-12)
        grad /= ((rho * self.shell_volumes[np.newaxis, np.newaxis, :] * N * wbar) + 1e-12)

        return r, g_r, grad

   
    def loss(self, atom_positions):
        self.loss_calls += 1
        atoms = self.atoms.copy()
        atoms.set_positions(atom_positions.reshape((self.natoms, 3)))
        _, g_sim, _ = self.calculate_gr_and_grad(atoms)
        rdf_loss = np.sum((g_sim - self.target[1])**2)

        if self.force_weight and self.force_weight > 0.0:
            atoms.calc = self.calculator
            try:
                f = atoms.get_forces()
                f_term = np.sum(f**2) / len(atoms)
            except Exception:
                f_term = 0.0
            return rdf_loss + self.force_weight * f_term

        return rdf_loss

    def grad(self, atom_positions):
        atoms = self.atoms.copy()
        atoms.set_positions(atom_positions.reshape((self.natoms, 3)))
        _, g_sim, grad = self.calculate_gr_and_grad(atoms)
        delta = (g_sim - self.target[1])[np.newaxis, np.newaxis, :] 
        g = 2.0 * np.sum(delta * grad, axis=2)  
        return g.flatten()

    def energyMinimization(self, atoms, steps, fmax):
        atoms = atoms.copy()
        atoms.calc = self.calculator
        e0 = atoms.get_potential_energy()
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=fmax, steps=steps)
        e1 = atoms.get_potential_energy()
        return atoms, e0, e1

 
    def save_rdf_plot(self, r, rdf_init, rdf_target, rdf_curr, path_png, smooth=False, sigma=0.1):
        if smooth:
            rdf_init   = gaussian_filter1d(rdf_init,   sigma=sigma)
            rdf_target = gaussian_filter1d(rdf_target, sigma=sigma)
            rdf_curr   = gaussian_filter1d(rdf_curr,   sigma=sigma)

        plt.figure()
        plt.plot(r, rdf_init,   label="initial")
        plt.plot(r, rdf_target, label="target", linestyle="--")
        plt.plot(r, rdf_curr,   label="current", linestyle="-.")
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.title("RDF")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path_png, dpi=150)
        plt.close()


    def solve(self, out_dirs):
        cfg = self.cfg
        cycles = cfg["cycles"]
        rmc_steps = cfg["rmc_steps_per_cycle"]
        rmc_disp = cfg["rmc_disp"]
        rmc_temp = cfg["rmc_temp"]


        curr_atoms = self.atoms.copy()
        r, rdf_init, _ = self.calculate_gr_and_grad(curr_atoms)
        _, rdf_target = self.target
        curr_loss = self.loss(curr_atoms.get_positions())
        accepted = 0
        loss_trace = [curr_loss]
        t_start = time.time()

 
        log_csv = os.path.join(out_dirs["logs"], "run_log.csv")
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "cycle", "acc_rate", "temp", "disp", "loss", "e_before", "e_after", "accepted"])

        def log_row(cycle, acc_rate, temp, disp, loss, e0, e1, accepted_total):
            with open(log_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([timestamp(), cycle, f"{acc_rate:.4f}", f"{temp:.4f}", f"{disp:.4f}",
                            f"{loss:.8f}", f"{e0 if e0 is not None else ''}",
                            f"{e1 if e1 is not None else ''}", accepted_total])

        print(f"[{timestamp()}] Initial loss: {curr_loss:.8f} | CHGNet={USE_CHGNET} | mode={self.mode}")

  
        _, rdf_curr, _ = self.calculate_gr_and_grad(curr_atoms)

        for cycle in range(1, cycles+1):
            t0 = time.time()
            accepted_this_cycle = 0


            for _ in range(rmc_steps):
                t_atoms = curr_atoms.copy()
                t_pos = t_atoms.get_positions()
                idx = np.random.randint(len(t_pos))
                disp = (np.random.rand(3) - 0.5) * 2.0 * rmc_disp
                t_pos[idx] += disp
                t_atoms.set_positions(t_pos)
                t_atoms.wrap()

                dists_idx = t_atoms.get_distances(idx, np.arange(len(t_atoms)), mic=True)
                if np.min(dists_idx[np.nonzero(dists_idx)]) < cfg["overlap_cut"]:
                    continue 

                t_loss = self.loss(t_atoms.get_positions())
                dL = t_loss - curr_loss
                accept = (dL < 0) or (np.exp(-dL / max(rmc_temp, 1e-12)) > np.random.rand())
                if accept:
                    curr_atoms = t_atoms
                    curr_loss = t_loss
                    accepted += 1
                    accepted_this_cycle += 1
                    loss_trace.append(curr_loss)

            acc_rate = accepted_this_cycle / max(1, rmc_steps)
            if cfg["adapt_temp"]:
                lo, hi = cfg["target_acc_band"]
                if acc_rate < lo:
                    rmc_temp *= cfg["temp_up"]
                elif acc_rate > hi:
                    rmc_temp *= cfg["temp_dn"]
            if cfg["adapt_disp"]:
                lo, hi = cfg["target_acc_band"]
                if acc_rate < lo:
                    rmc_disp *= cfg["disp_dn"]
                elif acc_rate > hi:
                    rmc_disp *= cfg["disp_up"]

         
            rmc_temp *= cfg["cool"]


            e0 = e1 = None
            if cfg["FEAR"] and (cycle % cfg["fear_every"] == 0):
                curr_atoms, e0, e1 = self.energyMinimization(curr_atoms, cfg["fear_steps"], cfg["fear_fmax"])
                curr_loss = self.loss(curr_atoms.get_positions())
                loss_trace.append(curr_loss)


            if (cycle % cfg["plot_every_cycles"] == 0) or (cycle == 1):
                _, rdf_curr, _ = self.calculate_gr_and_grad(curr_atoms)
                plot_path = os.path.join(out_dirs["plots"], f"rdf_cycle_{cycle:04d}.png")
                self.save_rdf_plot(r, rdf_init, rdf_target, rdf_curr, plot_path,
                                   smooth=self.cfg["smooth_plots"], sigma=self.cfg["smooth_sigma"])

            if (cycle % cfg["save_every_cycles"] == 0) or (cycle == cycles):
                chk_path = os.path.join(out_dirs["checkpoints"], f"FEAR_cycle_{cycle:04d}.xyz")
                write(chk_path, curr_atoms)


            log_row(cycle, acc_rate, rmc_temp, rmc_disp, curr_loss, e0, e1, accepted)


            print(f"Cycle {cycle:4d}/{cycles} | loss={curr_loss:.8e} | acc={acc_rate:5.2%} "
                  f"| T={rmc_temp:.3f} | disp={rmc_disp:.3f} | dt={time.time()-t0:.1f}s")

 
            if len(loss_trace) >= self.cfg["early_window"]:
                recent = np.array(loss_trace[-self.cfg["early_window"]:])
                if np.std(recent) < self.cfg["early_tol"]:
                    print(f"Early stopping at cycle {cycle}: loss plateau (std<{self.cfg['early_tol']}).")
                    break

        self.atoms = curr_atoms.copy()
        try:
            np.savetxt(os.path.join(out_dirs["logs"], "loss_trace.csv"),
                       np.array(loss_trace), delimiter=",")
        except Exception:
            pass

        print(f"[{timestamp()}] Done. Total time: {(time.time()-t_start)/3600:.2f} h")
        return curr_atoms, loss_trace


if __name__ == "__main__":
 
    OUT = ensure_dirs(CFG["out_dir"])

 
    crystal = read(CFG["crystal_path"]) * CFG["supercell"]
    if CFG["randomize_initial"]:
     
        crystal.positions = np.random.rand(len(crystal), 3) @ crystal.cell

    target_atoms = read(CFG["target_path"], format=CFG["target_format"])


    calc = CHGNetCalculator() if USE_CHGNET else EMT()


    solver = grSolver(
        atoms=crystal,
        target=target_atoms,     
        cfg=CFG,
        calculator=calc
    )


    final_atoms, loss_trace = solver.solve(OUT)


    r, rdf_init, _ = solver.calculate_gr_and_grad(read(CFG["crystal_path"]) * CFG["supercell"])
    _, rdf_target = solver.target
    _, rdf_final, _ = solver.calculate_gr_and_grad(final_atoms)
    final_plot = os.path.join(OUT["plots"], "rdf_final.png")
    solver.save_rdf_plot(r, rdf_init, rdf_target, rdf_final, final_plot,
                         smooth=CFG["smooth_plots"], sigma=CFG["smooth_sigma"])

  
    write(os.path.join(OUT["checkpoints"], "final_structure.xyz"), final_atoms)

    print("Outputs:")
    print(f"  - Loss trace:     {os.path.join(OUT['logs'], 'loss_trace.csv')}")
    print(f"  - Run log:        {os.path.join(OUT['logs'], 'run_log.csv')}")
    print(f"  - Final RDF plot: {final_plot}")
    print(f"  - Final struct:   {os.path.join(OUT['checkpoints'], 'final_structure.xyz')}")
