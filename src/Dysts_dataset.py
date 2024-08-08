import os
# from dysts.base import make_trajectory_ensemble
# from dysts.base import get_attractor_list
import dysts.flows as flows
# import dysts.datasets as datasets
import numpy as np
import inspect
from sympy import *
from dysts.equation_utils import *
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import multiprocessing
import itertools as it
from sklearn.preprocessing import PolynomialFeatures


def sample_initial_conditions(model, points_to_sample, random_state, traj_length=1000, pts_per_period=30):
    
    initial_sol = model.make_trajectory(
        traj_length, resample=True, pts_per_period=pts_per_period, postprocess=False
    )
    rand = np.random.default_rng(seed=random_state)
    sample_inds = rand.choice(
        np.arange(initial_sol.shape[0]), points_to_sample, replace=False
    )
    sample_pts = initial_sol[sample_inds]
    return sample_pts


def _make_data(equation_name, dt, n, seed_list, noise_list):
    data_each_noise = {noise_ratio: None for noise_ratio in noise_list}
    for noise_ratio in noise_list:
        data_each_seed = {seed: None for seed in seed_list}
        data_no_noise = {seed: None for seed in seed_list}
        for seed in seed_list:
            eq = getattr(flows, equation_name)()
            eq.dt = dt
            eq.ic, _ = sample_initial_conditions(eq, 2, random_state=seed, traj_length=n, pts_per_period=30)
            _, sol = eq.make_trajectory(n, method="Radau", resample=True, return_times=True, standardize=False)
            rand = np.random.default_rng(seed=seed)
            data_norm = np.linalg.norm(sol, ord='fro')
            noise = rand.normal(0, 1.0, size=(sol.shape[0], sol.shape[1]))
            noise_norm = np.linalg.norm(noise, ord='fro')
            data_each_seed[seed] = sol + noise * (data_norm * noise_ratio) / noise_norm
            data_no_noise[seed] = sol
        data_each_noise[noise_ratio] = data_each_seed
        data_each_noise[0] = data_no_noise
    return data_each_noise

def make_dysts_true_coefficients(systems_list, dimension_list, param_list):
    true_coefficients = {}
    for i, system in enumerate(systems_list):
        # print(i, system)
        if dimension_list[i] == 3:
            feature_names = ['1',
                             'x',
                             'y',
                             'z',
                             'x^2',
                             'x y',
                             'x z',
                             'y^2',
                             'y z',
                             'z^2',
                             'x^3',
                             'x^2 y',
                             'x^2 z',
                             'x y^2',
                             'x y z',
                             'x z^2',
                             'y^3',
                             'y^2 z',
                             'y z^2',
                             'z^3',
                             'x^4',
                             'x^3 y',
                             'x^3 z',
                             'x^2 y^2',
                             'x^2 y z',
                             'x^2 z^2',
                             'x y^3',
                             'x y^2 z',
                             'x y z^2',
                             'x z^3',
                             'y^4',
                             'y^3 z',
                             'y^2 z^2',
                             'y z^3',
                             'z^4']
        else:
            feature_names = ['1',
                             'x',
                             'y',
                             'z',
                             'w',
                             'x^2',
                             'x y',
                             'x z',
                             'x w',
                             'y^2',
                             'y z',
                             'y w',
                             'z^2',
                             'z w',
                             'w^2',
                             'x^3',
                             'x^2 y',
                             'x^2 z',
                             'x^2 w',
                             'x y^2',
                             'x y z',
                             'x y w',
                             'x z^2',
                             'x z w',
                             'x w^2',
                             'y^3',
                             'y^2 z',
                             'y^2 w',
                             'y z^2',
                             'y z w',
                             'y w^2',
                             'z^3',
                             'z^2 w',
                             'z w^2',
                             'w^3',
                             'x^4',
                             'x^3 y',
                             'x^3 z',
                             'x^3 w',
                             'x^2 y^2',
                             'x^2 y z',
                             'x^2 y w',
                             'x^2 z^2',
                             'x^2 z w',
                             'x^2 w^2',
                             'x y^3',
                             'x y^2 z',
                             'x y^2 w',
                             'x y z^2',
                             'x y z w',
                             'x y w^2',
                             'x z^3',
                             'x z^2 w',
                             'x z w^2',
                             'x w^3',
                             'y^4',
                             'y^3 z',
                             'y^3 w',
                             'y^2 z^2',
                             'y^2 z w',
                             'y^2 w^2',
                             'y z^3',
                             'y z^2 w',
                             'y z w^2',
                             'y w^3',
                             'z^4',
                             'z^3 w',
                             'z^2 w^2',
                             'z w^3',
                             'w^4']
        for k, feature in enumerate(feature_names):
            feature = feature.replace(" ", "", 10)
            feature = feature.replace("y^3z", "zy^3", 10)
            feature = feature.replace("x^3z", "zx^3", 10)
            feature = feature.replace("x^3y", "yx^3", 10)
            feature = feature.replace("z^3y", "yz^3", 10)
            feature = feature.replace("y^3x", "xy^3", 10)
            feature = feature.replace("z^3x", "xz^3", 10)
            feature_names[k] = feature
        # print(feature_names)
        num_poly = len(feature_names)
        coef_matrix_i = np.zeros((dimension_list[i], num_poly))
        system_str = inspect.getsource(getattr(flows, system))
        cut1 = system_str.find("return")
        system_str = system_str[: cut1 - 1]
        cut2 = system_str.rfind("):")
        system_str = system_str[cut2 + 5 :]
        chunks = system_str.split("\n")[:-1]
        params = param_list[i]
        # print(system, chunks)
        for j, chunk in enumerate(chunks):
            cind = chunk.rfind("=")
            chunk = chunk[cind + 1 :]
            for key in params.keys():
                if "Lorenz" in system and "rho" in params.keys():
                    chunk = chunk.replace("rho", str(params["rho"]), 10)
                if "Bouali2" in system:
                    chunk = chunk.replace("bb", "0", 10)
                chunk = chunk.replace(key, str(params[key]), 10)
            # print(chunk)
            chunk = chunk.replace("--", "", 10)
            chunk = chunk.replace("- -", "+ ", 10)
            # get all variables into (x, y, z, w) form
            chunk = chunk.replace("q1", "x", 10)
            chunk = chunk.replace("q2", "y", 10)
            chunk = chunk.replace("p1", "z", 10)
            chunk = chunk.replace("p2", "w", 10)
            chunk = chunk.replace("px", "z", 10)
            chunk = chunk.replace("py", "w", 10)
            # change notation of squared and cubed terms
            chunk = chunk.replace(" ** 2", "^2", 10)
            chunk = chunk.replace(" ** 3", "^3", 10)
            # reorder cubic terms
            chunk = chunk.replace("y * x^2", "x^2y", 10)
            chunk = chunk.replace("z * x^2", "x^2z", 10)
            chunk = chunk.replace("z * y^2", "y^2z", 10)
            # reorder quartic terms
            chunk = chunk.replace("y * x^3", "yx^3", 10)
            chunk = chunk.replace("z * x^3", "zx^3", 10)
            chunk = chunk.replace("z * y^2", "zy^3", 10)
            # Reorder quadratics
            chunk = chunk.replace("x * y", "xy", 10)
            chunk = chunk.replace("x * z", "xz", 10)
            chunk = chunk.replace("y * x", "xy", 10)
            chunk = chunk.replace("z * x", "xz", 10)
            chunk = chunk.replace("y * z", "yz", 10)
            chunk = chunk.replace("z * y", "yz", 10)
            chunk = chunk.replace("x * w", "xw", 10)
            chunk = chunk.replace("w * x", "xw", 10)
            chunk = chunk.replace("y * w", "yw", 10)
            chunk = chunk.replace("w * y", "yw", 10)
            chunk = chunk.replace("z * w", "zw", 10)
            chunk = chunk.replace("w * z", "zw", 10)

            # Do any unique ones
            chunk = chunk.replace("1 / 0.03", "33.3333333333", 10)
            chunk = chunk.replace("1.0 / 0.03", "33.3333333333", 10)
            chunk = chunk.replace("1 / 0.8", "1.25", 10)
            chunk = chunk.replace("1.0 / 0.8", "1.25", 10)
            chunk = chunk.replace("0.0322 / 0.8", "0.04025", 10)
            chunk = chunk.replace("0.49 / 0.03", "16.3333333333", 10)
            chunk = chunk.replace("(-10 + -4)", "-14", 10)
            chunk = chunk.replace("(-10 * -4)", "40", 10)
            chunk = chunk.replace("3.0 * 1.0", "3", 10)
            chunk = chunk.replace(" - 0 * z", "", 10)
            chunk = chunk.replace("(28 - 35)", "-7", 10)
            chunk = chunk.replace("(1 / 0.2 - 0.001)", "4.999", 10)
            chunk = chunk.replace("- (1.0 - 1.0) * x^2 ", "", 10)
            chunk = chunk.replace("(26 - 37)", "-11", 10)
            chunk = chunk.replace("64^2", "4096", 10)
            chunk = chunk.replace("64**2", "4096", 10)
            chunk = chunk.replace("3 / np.sqrt(2) * 0.55", "1.166726189", 10)
            chunk = chunk.replace("3 * np.sqrt(2) * 0.55", "2.333452378", 10)
            chunk = chunk.replace("+ -", "- ", 10)
            chunk = chunk.replace("-1.5 * -0.0026667", "0.00400005", 10)

            for num_str in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                for x_str in ["x", "y", "z", "w"]:
                    chunk = chunk.replace(num_str + " * " + x_str, num_str + x_str, 20)

            chunk = chunk.replace("- 0.0026667 * 0xz", "", 10)
            chunk = chunk.replace("1/4096", "0.000244140625", 10)
            chunk = chunk.replace("10/4096", "0.00244140625", 10)
            chunk = chunk.replace("28/4096", "0.0068359375", 10)
            chunk = chunk.replace("2.667/4096", "0.000651123046875", 10)
            chunk = chunk.replace("0.2 * 9", "1.8", 10)
            chunk = chunk.replace(" - 3 * 0", "", 10)
            chunk = chunk.replace("2 * 1", "2", 10)
            chunk = chunk.replace("3 * 2.1 * 0.49", "3.087", 10)
            chunk = chunk.replace("2 * 2.1", "4.2", 10)
            chunk = chunk.replace("-40 / -14", "2.85714285714", 10)
            # change notation of squared and cubed terms
            chunk = chunk.replace(" 1x", " x", 10)
            chunk = chunk.replace(" 1y", " y", 10)
            chunk = chunk.replace(" 1z", " z", 10)
            chunk = chunk.replace(" 1w", " w", 10)
            chunks[j] = chunk
            chunk = chunk.replace(" ", "", 400)
            chunk = chunk.replace("-x", "-1x", 10)
            chunk = chunk.replace("-y", "-1y", 10)
            chunk = chunk.replace("-z", "-1z", 10)
            chunk = chunk.replace("-w", "-1w", 10)
            chunk = chunk.replace("--", "-", 20)


            for k, feature in enumerate(np.flip(feature_names[1:])):
                # print(k, feature)
                feature_ind = (chunk + " ").find(feature)
                if feature_ind != -1:
                    feature_chunk = chunk[: feature_ind + len(feature)]
                    find = max(feature_chunk.rfind("+"), feature_chunk.rfind("-"))
                    # print('find = ', find, feature_chunk)
                    if find == -1 or find == 0:
                        feature_chunk = feature_chunk[0:] + " "
                    else:
                        feature_chunk = feature_chunk[find:] + " "
                    # print(feature_chunk)
                    if feature_chunk != chunk:
                        feature_chunk_compact = feature_chunk.replace("+", "")
                        # print(feature, feature_chunk_compact[:-len(feature) - 1])
                        if (
                            len(
                                feature_chunk_compact[: -len(feature) - 1].replace(
                                    " ", ""
                                )
                            )
                            == 0
                        ):
                            coef_matrix_i[j, len(feature_names) - k - 1] = 1
                        else:
                            coef_matrix_i[j, len(feature_names) - k - 1] = float(
                                feature_chunk_compact[: -len(feature) - 1]
                            )
                        # print(feature_chunk, chunk)
                        chunk = chunk.replace(feature_chunk.replace(" ", ""), "")
                        
            if len(chunk.replace(" ", "")) != 0:
                coef_matrix_i[j, 0] = chunk.replace(" ", "")

        true_coefficients[system] = coef_matrix_i
    return true_coefficients


class Data:
    def __init__(self):
        systems_list = ["Aizawa", "Arneodo", "Bouali2", 
                        "GenesioTesi", "HyperBao", "HyperCai", "HyperJha", 
                        "HyperLorenz", "HyperLu", "HyperPang", "Laser",
                        "Lorenz", "LorenzBounded", 
                        "MooreSpiegel", "Rossler", "ShimizuMorioka",
                        "HenonHeiles", "GuckenheimerHolmes", "Halvorsen", "KawczynskiStrizhak",
                        "VallisElNino", "RabinovichFabrikant", "NoseHoover", "Dadras", "RikitakeDynamo",
                        "NuclearQuadrupole", "PehlivanWei", "SprottTorus", "SprottJerk", "SprottA", "SprottB",
                        "SprottC", "SprottD", "SprottE", "SprottF", "SprottG", "SprottH", "SprottI", "SprottJ",
                        "SprottK", "SprottL", "SprottM", "SprottN", "SprottO", "SprottP", "SprottQ", "SprottR",
                        "SprottS", "Rucklidge", "Sakarya", "RayleighBenard", "Finance", "LuChenCheng",
                        "LuChen", "QiChen", "ZhouChen", "BurkeShaw", "Chen", "ChenLee", "WangSun", "DequanLi",
                        "NewtonLiepnik", "HyperRossler", "HyperQi", "Qi", "LorenzStenflo", "HyperYangChen", 
                        "HyperYan", "HyperXu", "HyperWang", "Hadley",
                        ]
        
        alphabetical_sort = np.argsort(systems_list)
        self.systems_list = np.array(systems_list)[alphabetical_sort]
        
        # # simplify attributes
        dimension_list = []
        self.param_list = []
        self.dimensions = {}
        for i, equation_name in enumerate(systems_list):
            eq = getattr(flows, equation_name)()
            dim = getattr(eq, 'embedding_dimension', None)
            dimension_list.append(dim)
            self.dimensions[equation_name] = dim
            self.param_list.append(getattr(eq, 'parameters', None))

            
        self.true_coefficients = make_dysts_true_coefficients(systems_list, 
                                                        dimension_list, 
                                                        self.param_list)
        
        self.poly_degree = {}
        self.gt = {}
        for i, equation_name in enumerate(systems_list):
            coef = self.true_coefficients[equation_name]
            dim = coef.shape[0]
            num_quad = int(comb(2 + dim - 1, dim - 1))
            num_cubic = int(comb(3 + dim - 1, dim - 1))
            coeff_index = dim + 1 + num_quad + num_cubic
            
            if np.count_nonzero(coef[:, coeff_index:]) > 0:
                self.poly_degree[equation_name] = 4
                self.gt[equation_name] = coef[:, 1:].copy()
                continue
            
            coeff_index = dim + 1 + num_quad
            if np.count_nonzero(coef[:, coeff_index : coeff_index + num_cubic]) > 0:
                self.poly_degree[equation_name] = 3
                self.gt[equation_name] = coef[:, 1: coeff_index + num_cubic].copy()
            else:
                self.poly_degree[equation_name] = 2
                self.gt[equation_name] = coef[:, 1: coeff_index].copy()
            
        
    
    def make_all_data(self, noise_list, seed_list, dt=0.01, n=500):
        self.dt =dt
        self.n = n
        num_works = int(os.cpu_count()/3*2)
        with ProcessPoolExecutor(max_workers=num_works, mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = {executor.submit(_make_data, equation_name, dt, n, seed_list, noise_list): equation_name for equation_name in self.systems_list}
            i = 0
            data_list={}
            for future in concurrent.futures.as_completed(futures):
                equation_name = futures[future]
                data_list[equation_name] = future.result()
                i += 1
                
        self.syn_data = data_list
        
        
    def make_each_data(self, equation_name, noise_list, seed_list, dt=0.01, n=500):
        self.dt = dt
        self.n = n
        if equation_name not in self.systems_list:
            print('Name is nonexistent in system list')
            return 
        return _make_data(equation_name, dt, n, seed_list, noise_list)
        
        
    def get_len(self):
        return len(self.systems_list)
    
    
    def get_dataset_name(self, id):
        return self.systems_list[id]
    
    
    def get_true_coefficients(self, equation_name):
        return self.gt[equation_name]
    
    def make_data(self, equation_name, dt, n, seed):
        eq = getattr(flows, equation_name)()
        eq.dt = dt
        eq.ic, _ = sample_initial_conditions(eq, 2, random_state=seed, traj_length=n, pts_per_period=30)
        _, sol = eq.make_trajectory(n, method="Radau", resample=True, return_times=True, standardize=False)
        return sol
        
    def get_data(self, equation_name, seed, noise_ratio):
        return self.syn_data[equation_name][noise_ratio][seed], self.dt
    
    
    
    def make_setting(self, equation_name):
        if equation_name not in self.systems_list:
            print('Name is nonexistent in system list')
            return
        
        setting={'data_name':equation_name, 'num_step':self.n, 'dt':self.dt}
        setting['xticklabels'] = self.make_feature_names(equation_name, self.poly_degree[equation_name])
        setting['yticklabels'] = setting['xticklabels'][:3]
        setting['gt'] = self.gt[equation_name]
        return setting
        
    def make_feature_names(self, equation_name, dim):
        poly = PolynomialFeatures(degree=dim, include_bias=False)
        if self.dimensions[equation_name] == 3:
            features = ['x', 'y', 'z']
            poly.fit(np.zeros((1,3)))
        elif self.dimensions[equation_name] == 4:
            features = ['x_0', 'x_1', 'x_2', 'x_3']
            poly.fit(np.zeros((1,4)))
        names = poly.get_feature_names_out(features) 
        return ['$' + s + '$' for s in names]
        