# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
# *
# **************************************************************************

import time
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool
import copy
import pickle
import numpy as np

import continuousflex.protocols.utilities.src as src
from continuousflex.protocols.utilities.src.constants import FIT_VAR_LOCAL,FIT_VAR_GLOBAL, FIT_VAR_ROTATION, FIT_VAR_SHIFT, \
                                KCAL_TO_JOULE, AVOGADRO_CONST, ATOMIC_MASS_UNIT,K_BOLTZMANN, ANGSTROM_TO_METER
from continuousflex.protocols.utilities.src.viewers import fit_viewer, chimera_fit_viewer, fit_potentials_viewer,\
    ramachandran_viewer

#################################################################################################################
#                          Flexible Fitting
#################################################################################################################


class FlexibleFitting:
    """
    Perform flexible fitting between initial atomic structure and target Density using HMC
    """

    def __init__(self, init, target, vars, n_chain, params, verbose=0, prefix=None):
        """
        Constructor
        :param init: inititial atomic structure Molecule
        :param target: target Density
        :param vars: list of fitted variables
        :param n_chain: number of chain
        :param params: fit parameters
        :param verbose: verbose level
        :param prefix: prefix path of outputs
        """
        self.init = init
        self.target = target
        self.n_chain = n_chain
        self.verbose = verbose
        self.vars = vars
        self.prefix = prefix
        if self.prefix is not None:
            with open(self.prefix +"_log.txt", "w") as f:
                f.write("")
        self._set_init_fit_params(params)

    # =============================================== HMC ======================================================

    def HMC(self):
        """
        Run HMC fitting with the specified number of chain in parallel
        :return: FlexibleFitting
        """
        print("> Bayesian Flexible Fitting method ...")

        with Pool(self.n_chain) as p:
            # launch n_chain times HMC_chain()
            fits = p.starmap(FlexibleFitting.HMC_chain, [(self,i) for i in range(self.n_chain)])
            p.close()
            p.join()

        # Regroup the chains results
        self.res = {"mol": self.init.copy()}
        self.res["mol"].coords = np.mean([i.res["mol"].coords for i in fits], axis=0)
        for v in self.vars:
            self.res[v] = np.mean([i.res[v] for i in fits], axis=0)
        self.fit = [i.fit for i in fits]
        if self.prefix is not None:
            self.res["mol"].save_pdb(file=self.prefix+"_output.pdb")
            self.save(file=self.prefix+"_output.pkl")

        print("\t Done \n")
        return self

    def HMC_chain(self, chain_id=0):
        """
        Run one HMC chain fitting
        :param chain_id: chain index
        :return: FlexibleFitting
        """

        # set the random seed of numpy for parallel computation
        np.random.seed()

        t = time.time()
        self.chain_id = chain_id

        # initialize fit variables
        self.fit= {"coord":[copy.deepcopy(self.init.coords)],"accept":[0]}
        for i in self.vars:
            self._set(i ,[self.params[i+"_init"]])

        # initialize trajectory
        if self.prefix is not None:
            self.init.save_pdb(self.prefix+"_chain"+str(self.chain_id)+".pdb")
            src.io.append_dcd(pdb_file=self.prefix+"_chain"+str(self.chain_id)+".pdb",
                              dcd_file=self.prefix+"_chain"+str(self.chain_id)+".dcd",
                              first_frame=True)

        # HMC Loop
        for i in range(self.params["n_iter"]):
            self._set("Iter", i)
            # try:
            self.HMC_step()
            # except RuntimeError as rte:
            #     self._write("Warning : Trajectory rejected because of the following error : "+str(rte.args[0]))
            #     self._acceptation(1,1,True)

        # Generate results
        self.res = {"mol": self.init.copy()}
        self.res["mol"].coords = np.mean(np.array(self.fit["coord"][self.params["n_warmup"] + 1:]), axis=0)
        for i in self.vars:
            self.res[i] = np.mean(np.array(self.fit[i][self.params["n_warmup"]+1:]), axis=0)

        # End
        if self.verbose >0 :
            self._write("############### HMC FINISHED ##########################")
            self._write("### Total execution time : "+str(time.time()-t)+" s")
            self._write("### Initial CC value : "+str(self.fit["CC"][0]))
            self._write("### Mean CC value : "+str(np.mean(self.fit["CC"][self.params["n_warmup"]:])))
            self._write("#######################################################")

        # Cleaning
        # for i in range(len(self.fit["coord"])):
        #     if i%10 != 0:
        #         del (self.fit["coord"])[i]
        # del self.fit["coord_t"]
        del self.fit["psim"]
        del self.fit["expnt"]
        for i in self.vars:
            del self.fit[i]
            del self.fit[i+"_v"]
            del self.fit[i+"_t"]
            #EDIT del self.fit[i+"_Ft"]

        return self

    def HMC_step(self):
        """
        Run HMC iteration
        """
    # Init vars
        self._add("C", 0)
        self._add("L", 0)
    # Initial coordinates
        self._initialize()
    # Compute Forward model
        self._forward_model()
    # initial density
        self._set_density()
    # Check pairlist
        self._set_pairlist()
    # Initial Potential Energy
        self._set_energy()
    # Initial gradient
        self._set_gradient()
    # Initial Kinetic Energy
        self._set_kinetic()
    # Temperature update
        self._set_instant_temp()
    # Initial Hamiltonian
        H_init = self._get_hamiltonian()
    # MD loop
        while (self._get("C") >= 0 and self._get("L")< self.params["n_step"]):
            tt = time.time()
        # velocities update
            self._update_velocities()
        # Coordinate update
            self._update_positions()
        # Compute Forward model
            self._forward_model()
        # Density update
            self._set_density()
        # CC update
            self._add("CC", src.density.get_CC(self._get("psim"), self.target.data))
            if "target" in self.params:
                self._add("RMSD", src.functions.get_RMSD_coords(
                    self._get("coord_t")[self.params["target_idx"][:,0]],
                    self.params["target"].coords[self.params["target_idx"][:,1]]))
        # Check pairlist
            self._set_pairlist()
        # Potential energy update
            self._set_energy()
        # Gradient Update
            self._set_gradient()
        #EDIT
        # Kinetic update
            self._set_kinetic()
        # Temperature update
            self._set_instant_temp()
            # if FIT_VAR_LOCAL in self.vars:
            #     self._set("local_v", self._get("local_v") * (self.params["temperature"]/self._get("T")))
        # criterion update
            self._set_criterion()
            self.fit["L"][-1] +=1
        # Prints
            self._add("Time", time.time() -tt)
            if self.verbose > 1:
                self._print_step()
        # save step
            if self._get("L") % self.params["output_update"] ==0 :
                self._save_step()
        if self.verbose == 1:
            self._print_step()
    # Hamiltonian update
        H = self._get_hamiltonian()
    # Metropolis acceptation
        self._acceptation(H, H_init)
    # save pdb step
        self._save_step()

    # ==========================================   FlexibleFitting  IO        ===============================================

    def show(self,save=None):
        """
        Show fitting statistics
        """
        fit_viewer(self,save=save)

    def show_3D(self):
        """
        Show fitting results in 3D
        """
        chimera_fit_viewer(self.res["mol"], self.target)

    def show_forcefield(self, save=None):
        fit_potentials_viewer(self, save)

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(file=f, obj=self)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            fit = pickle.load(file=f)
            return fit

    # ==========================================     FlexibleFitting controls        ===============================================

    def get_all(self, key):
        if isinstance(self.fit, list):
            fits = self.fit
        else:
            fits = [self.fit]
        val = []
        for i in fits:
            val.append(i[key])
        return np.mean(val, axis=0)

    def _get(self, key):
        if isinstance(self.fit[key], list):
            return self.fit[key][-1]
        else:
            return self.fit[key]

    def _add(self, key, value):
        if key in self.fit:
            self.fit[key].append(value)
        else:
            self.fit[key] = [value]

    def _set(self, key, value):
        self.fit[key] = value

    def _remove(self, key):
        if key in self.fit:
            del self.fit[key]

    def _write(self, s):
        """
        Write string to console/log file
        """
        if self.prefix is not None:
            with open(self.prefix + "_log.txt", "a") as f :
                f.write(s+"\n")
        print(s)

    # ==========================================  Initialization        ===============================================

    def _set_init_fit_params(self, params):
        """
        Set initial parameters of the fitting
        :param params: dict of parameters
        """
        self.params = copy.deepcopy(src.constants.DEFAULT_FIT_PARAMS)
        if FIT_VAR_LOCAL in self.vars:
            self.params[FIT_VAR_LOCAL+"_init"] = np.zeros(self.init.coords.shape)
        if FIT_VAR_GLOBAL in self.vars:
            self.params[FIT_VAR_GLOBAL+"_init"] = np.zeros(self.init.normalModeVec.shape[1])
        if FIT_VAR_ROTATION in self.vars:
            self.params[FIT_VAR_ROTATION+"_init"] = np.zeros(3)
        if FIT_VAR_SHIFT in self.vars:
            self.params[FIT_VAR_SHIFT+"_init"] = np.zeros(3)

        self.params.update(params)

        self.params["local_sigma"] = (np.ones((3, self.init.n_atoms)) * np.sqrt((K_BOLTZMANN * self.params["temperature"]) /
                        (self.init.forcefield.mass * ATOMIC_MASS_UNIT)) * ANGSTROM_TO_METER**-1).T

        if FIT_VAR_GLOBAL in self.vars:
            self.params["global_mass"] = np.zeros(self.init.normalModeVec.shape[1])
            for i in range(self.init.n_atoms):
                self.params["global_mass"] += np.linalg.norm(self.init.normalModeVec[i], axis=1) * self.init.forcefield.mass[i]

        if "initial_biasing_factor" in self.params:
            self.params["biasing_factor"] = self._set_factor(self.params["initial_biasing_factor"], potentials=self.params["potentials"])

        if "target" in self.params:
            self.params["target_idx"] = src.functions.get_mol_conv(self.init, self.params["target"])


    def _initialize(self):
        """
        Initialize all variables positions and velocities
        """
        for i in self.vars:
            self._set(i + "_t", self._get(i))
            if i == FIT_VAR_GLOBAL:
                vq = np.zeros(self._get("global").shape)
                vx = np.random.normal(0, self.params["local_sigma"], self.init.coords.shape)
                for i in range(self.init.n_atoms):
                    vq += np.dot(self.init.normalModeVec[i], vx[i])
                self._set("global_v", vq)
                # self._set("local_v", vx)
            else:
                self._set(i + "_v", np.random.normal(0, self.params[i + "_sigma"], self._get(i).shape))


    # ==========================================   Forcefield       ===============================================

    def _set_energy(self):
        """
        Compute the total energy from the simulated Density
        """
        U = 0
        t = time.time()

        # Biased Potential
        if self.params["gradient"]=="LS":
            U_biased = src.density.get_LS(map1=self._get("psim"), map2=self.target.data)
        else:
            U_biased = 1 - src.density.get_CC(map1=self._get("psim"), map2=self.target.data)
        U_biased *= self.params["biasing_factor"]
        self._add("U_biased", U_biased)
        U+= U_biased

        # Energy Potential
        if self.verbose>3: verbose =True
        else: verbose = False
        U_potential = src.forcefield.get_energy(coords=self._get("coord_t"), forcefield=self.init.forcefield,
                potentials=self.params["potentials"], pairlist=self._get("pairlist"), verbose=verbose)
        for i in U_potential:
            if i == "total":
                self._add("U_potential", U_potential["total"]* self.params["potential_factor"])
            else:
                self._add("U_"+i, U_potential[i])
        U += self._get("U_potential")

        # Additional Priors on parameters
        for i in self.vars:
            if i+"_factor" in self.params:
                U_var = np.sum(np.square(self._get(i+"_t"))) * self.params[i+"_factor"]
                self._add("U_"+i, U_var)
                U += U_var

        # Total energy
        self._add("U", U)
        if self.verbose>=3: self._write("Energy="+str(time.time()-t))

    def _set_gradient(self):
        """
        Compute the gradient of the total energy from the simulated Density
        """
        t = time.time()
        vals={}
        for i in self.vars:
            vals[i] = self._get(i+"_t")

        if isinstance(self.target, src.density.Image):
            dU_biased = src.density.get_gradient_LS_img(mol=self.init, psim=self._get("psim"), pexp =self.target,
                                params=vals, expnt = self._get("expnt"), normalModeVec=self.init.normalModeVec)
        else:
            if self.params["gradient"] == "LS":
                dU_biased = src.density.get_gradient_LS(mol=self.init, psim=self._get("psim"), pexp=self.target,
                                    params=vals, expnt=self._get("expnt"), normalModeVec=self.init.normalModeVec)
            else:
                dU_biased = src.density.get_gradient_CC(mol=self.init, psim=self._get("psim"), pexp=self.target,
                                    params=vals, expnt=self._get("expnt"), normalModeVec=self.init.normalModeVec)

        dU_potential = src.forcefield.get_autograd(params=vals, mol = self.init, normalModeVec=self.init.normalModeVec,
                                               potentials=self.params["potentials"], pairlist=self._get("pairlist"),
                                                      limit=self.params["limit"])

        for i in self.vars:
            F = -((self.params["biasing_factor"] * dU_biased[i]) + (self.params["potential_factor"] *  dU_potential[i]))
            if i == FIT_VAR_LOCAL:
                F = (F.T * (1 / (self.init.forcefield.mass * ATOMIC_MASS_UNIT))).T  # Force -> acceleration
                F *= (KCAL_TO_JOULE / AVOGADRO_CONST)  # kcal/mol -> Joule
                F *= ANGSTROM_TO_METER**-2  # kg * m2 * s-2 -> kg * A2 * s-2

            elif i == FIT_VAR_GLOBAL:
                F = (F.T * (1 / (self.params["global_mass"] * ATOMIC_MASS_UNIT))).T  # Force -> acceleration
                F *= (KCAL_TO_JOULE / AVOGADRO_CONST)  # kcal/mol -> Joule
                F *= ANGSTROM_TO_METER**-2  # kg * m2 * s-2 -> kg * A2 * s-2

            if i+"_factor" in self.params:
                F+= - 2* self._get(i+"_t") * self.params[i+"_factor"]

            self._set(i + "_F", F)

            #EDIT
            # if not i+"_F" in self.fit: self._set(i+"_F", F)
            # else: self._set(i+"_Ft", F)

        if self.verbose >= 3: self._write("Gradient=" + str(time.time() - t))

    def _set_kinetic(self):
        """
        Compute the Kinetic energy
        """
        K = 0
        if FIT_VAR_LOCAL in self.vars:
            K_local=  1 / 2 * np.sum((self.init.forcefield.mass*ATOMIC_MASS_UNIT)*np.square(self._get(FIT_VAR_LOCAL+"_v")).T)
            K+= K_local
            self._add("K_local", K_local)

        if FIT_VAR_GLOBAL in self.vars:
            K_global = 1 / 2 * np.sum((self.init.forcefield.mass * ATOMIC_MASS_UNIT) * np.square(
                np.dot(self._get(FIT_VAR_GLOBAL + "_v"), self.init.normalModeVec)).T)
            K+= K_global
            self._add("K_global", K_global)

        K *= ANGSTROM_TO_METER**2 *(AVOGADRO_CONST /KCAL_TO_JOULE)# kg * A2 * s-2 -> kcal * mol-1
        self._add("K", K)

    def _set_instant_temp(self):
        """
        Compute instant temperature
        """
        T = 2 * self._get("K")*(KCAL_TO_JOULE/AVOGADRO_CONST ) / (K_BOLTZMANN * 3 * self.init.n_atoms)
        self._add("T", T)

    def _set_density(self):
        """
        Compute the density (Image or Volume)
        :return: Density object (Image or Volume)
        """
        t = time.time()
        if isinstance(self.target, src.density.Image):
            psim, expnt = src.density.pdb2img(coord=self._get("coord_t"), size=self.target.size,
                                                 voxel_size=self.target.voxel_size,
                                                 sigma=self.target.sigma, cutoff=self.target.cutoff)
        else:
            psim, expnt = src.density.pdb2vol(coord=self._get("coord_t"), size=self.target.size,
                                  voxel_size=self.target.voxel_size,
                                  sigma=self.target.sigma, cutoff=self.target.cutoff)
        self._set("psim",psim)
        self._set("expnt",expnt)
        if self.verbose >= 3: self._write("Density=" + str(time.time() - t))

    def _get_hamiltonian(self):
        return self._get("U") +  self._get("K")

    # ==========================================   HMC update       ===============================================

    def _update_positions(self):
        """
        Update all variables positions
        """
        for i in self.vars:
            self._set(i+"_t", self._update_pstep(self._get(i + "_t"), self._get(i + "_v"), self.params[i + "_dt"],
                                               self._get(i + "_F")))

    def _update_pstep(self, x, v, dx, F):
        return x+ dx*v #+ dx**2 *(F/2)

    def _update_velocities(self):
        """
        Update all variables velocities
        """
        for i in self.vars:
            self._set(i+"_v", self._update_vstep(self._get(i+"_v") , self.params[i+"_dt"] , self._get(i+"_F") ))#EDIT,self._get(i+"_Ft")))
            #EDIT self._set(i+"_F" ,self._get(i+"_Ft"))

    def _update_vstep(self, v, dx, F): #EDIT, Ft):
        return v + 0.5*dx*F

    def _forward_model(self):
        """
        Compute the forward model
        """
        coord = copy.deepcopy(self.init.coords)
        if FIT_VAR_LOCAL in self.vars:
            coord += self._get(FIT_VAR_LOCAL+"_t")
        if FIT_VAR_GLOBAL in self.vars:
            coord += np.dot(self._get(FIT_VAR_GLOBAL+"_t"), self.init.normalModeVec)
        if FIT_VAR_ROTATION in self.vars:
            coord = np.dot(src.functions.generate_euler_matrix(self._get(FIT_VAR_ROTATION+"_t")),  coord.T).T
        if FIT_VAR_SHIFT  in self.vars:
            coord += self._get(FIT_VAR_SHIFT+"_t")
        self._set("coord_t", coord)

    # ==========================================   HMC IO       ===============================================

    def _print_step(self):
        """
        Print step information
        """
        s=[]
        if "RMSD" in self.fit:
            s.append("RMSD")
        s+=["CC", "Time", "K", "T", "U_biased", "U_potential",]
        for i in self.vars :
            if i+"_factor" in self.params:
                s.append("U_"+i)
        for i in self.params["potentials"]:
            s.append("U_"+i)
        if self.params["criterion"]:
            s.append("C")

        print_values = [self.chain_id, self._get("Iter"),self._get("L")] + [self._get(i) for i in s]
        print_values_str = " ".join(["%6i"%i if isinstance(i,int) else "%12.2f"%i for i in print_values])
        print_header_str = " Chain   Iter   Step "+" ".join(["%12s" % i for i in s])

        self._write(print_header_str +"\n"+print_values_str)

    def _save_step(self):
        if self.prefix is not None:
            cp = self.init.copy()
            cp.coords = self._get("coord_t")
            cp.save_pdb(file=self.prefix+"_chain"+str(self.chain_id)+".pdb")
            del cp
            self.show(save=self.prefix+"_chain"+str(self.chain_id)+".png")
            self.save(file=self.prefix+"_chain"+str(self.chain_id)+".pkl")
            self.show_forcefield(save=self.prefix+"_chain"+str(self.chain_id)+"_forcefield.png")
            src.io.append_dcd(pdb_file=self.prefix+"_chain"+str(self.chain_id)+".pdb",
                              dcd_file=self.prefix+"_chain"+str(self.chain_id)+".dcd")
            ramachandran_viewer(self.prefix+"_chain"+str(self.chain_id)+".pdb", save=self.prefix+"_chain"+str(self.chain_id)+"_rama.png")

    # ==========================================   HMC Others       ===============================================

    def _set_pairlist(self):
        """
        Generate Non-bonded pairlist based on cutoff parameters
        """
        if "vdw" in self.params["potentials"] or "elec" in self.params["potentials"]:
            if self.params["nb_update"] is not None:
                if self._get("L") % self.params["nb_update"] == 0 and self._get("L") != 0:
                    return
            if not "coord_pl" in self.fit:
                self._set("coord_pl", self._get("coord_t"))
            dx_max =np.max(np.linalg.norm(self._get("coord_pl")-self._get("coord_t"), axis=1))/2
            if (dx_max > (self.params["cutoffpl"] - self.params["cutoffnb"])) or (not "pairlist" in self.fit):
                if self.verbose >2 : self._write("Computing pairlist ...")
                t=time.time()
                self._set("pairlist", src.forcefield.get_pairlist(self._get("coord_t"),
                        excluded_pairs= self.init.forcefield.excluded_pairs,cutoff=self.params["cutoffpl"]))
                self._set("coord_pl",self._get("coord_t"))
                if self.verbose > 2: self._write("Done "+str(time.time()-t)+" s")
        else:
            self._set("pairlist",None)

    def _set_factor(self, init_factor=100, **kwargs):
        psim = src.density.Volume.from_coords(coord=self.init.coords, size=self.target.size,
                                                 voxel_size=self.target.voxel_size,
                                                 sigma=self.target.sigma, cutoff=self.target.cutoff)
        if self.params["gradient"] == "LS":
            U_biased = src.density.get_LS(map1=psim.data, map2=self.target.data)
        else:
            U_biased= 1 - src.density.get_CC(map1=psim.data, map2=self.target.data)
        factor = np.abs(init_factor/U_biased)
        self._write("Optimal initial factor : "+str(factor))
        return factor

    def _set_criterion(self):
        """
        Compute NUTS criterion (optional)
        """
        C = 0
        if self.params["criterion"]:
            for i in self.vars:
                C += self.params[i+"_dt"]*np.dot((self._get(i+"_t").flatten() - self._get(i).flatten()), self._get(i+"_v").flatten())
            self._add("C",C)
        else:
            self._add("C", 0)

    def _acceptation(self,H, H_init, force_reject=False):
        """
        Perform Metropolis Acceptation step
        :param H: Current Hamiltonian
        :param H_init: Initial Hamiltonian
        """
        # Set acceptance value
        self._add("accept",  np.min([1, H_init/H]) )

        # Update variables
        if self._get("accept") > np.random.rand() or force_reject:
            suffix = "_t"
            if self.verbose > 2 : self._write("ACCEPTED " + str(self._get("accept")))
        else:
            suffix = ""
            if self.verbose > 2 : self._write("REJECTED " + str(self._get("accept")))
        for i in self.vars:
            self._add(i, self._get(i+suffix))
        self._add("coord", self._get("coord"+suffix))

        # clean forces
        for i in self.vars:
            self._remove(i + "_F")

def multiple_fitting(models, n_chain, n_proc):
    class NoDaemonProcess(multiprocessing.Process):
        @property
        def daemon(self):
            return False

        @daemon.setter
        def daemon(self, value):
            pass

    class NoDaemonContext(type(multiprocessing.get_context())):
        Process = NoDaemonProcess

    class NestablePool(multiprocessing.pool.Pool):
        def __init__(self, *args, **kwargs):
            kwargs['context'] = NoDaemonContext()
            super(NestablePool, self).__init__(*args, **kwargs)

    models = np.array(models)

    N = len(models)
    n_loop = (N * n_chain) // n_proc
    n_last_process = ((N * n_chain) % n_proc)//n_chain
    n_process = n_proc//n_chain
    process = [np.arange(i*n_process, (i+1)*n_process) for i in range(n_loop)]
    process.append(np.arange(n_loop*n_process, n_loop*n_process + n_last_process))
    fits=[]
    print("Number of loops : "+str(n_loop))
    for i in process:
        t = time.time()
        print("\t fitting models # "+str(i))
        try :
            with NestablePool(n_process) as p:
                fits += p.map(FlexibleFitting.HMC, models[i])
                p.close()
                p.join()
        except RuntimeError as rte:
            print("Failed to run multiple fitting : " + str(rte.args))

        print("\t\t done : "+str(time.time()-t))
    return fits






