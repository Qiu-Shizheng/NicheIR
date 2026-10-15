"""Conditional finite-volume inversion of visible spatial cell counts.

Each query has its own 4x16 grid, fixed exposure and convex energy. Small
networks generate coefficients from visible data; all extrapolation is the
result of an unrolled nonnegative variational solve. There is no query pooling,
hidden/reference input, cohort identity, or direct missing-count decoder.

This is a finite-iteration estimate, not an assertion of exact MAP convergence.
Optional inverse-Hessian diagnostics are local curvature approximations; they
are not calibrated posterior/clinical intervals. Units are pi*(72 um)^2.
"""
from __future__ import annotations
from collections.abc import Mapping
import math
import torch
from torch import Tensor, nn

INPUT_KEYS = frozenset(('polar_counts', 'crop_coverage', 'visible_hull_support',
                       'local_observed_counts', 'query_features', 'valid'))
RING_EDGES = (0., .5, 1., 2., 3.)
NODE_FEATURES = 23
EDGE_FEATURES = 49
GAMMA_KAPPA = .01  # Fixed positive-support prior, not tuned or learned.


def validate_inputs(x: Mapping[str, Tensor], numerical=True):
    if set(x) != INPUT_KEYS:
        raise ValueError(f'Exactly six visible keys required; extra={set(x)-INPUT_KEYS}; missing={INPUT_KEYS-set(x)}')
    c = x['polar_counts']
    if c.ndim != 5 or c.shape[-3:] != (4, 16, 4) or not c.is_floating_point():
        raise ValueError('polar_counts must be floating [B,A,4,16,4]')
    leading = c.shape[:2]
    expected = {'crop_coverage': (*leading, 4, 16), 'visible_hull_support': (*leading, 4, 16),
                'local_observed_counts': (*leading, 4), 'query_features': (*leading, 7), 'valid': leading}
    for key, shape in expected.items():
        if tuple(x[key].shape) != tuple(shape) or x[key].device != c.device:
            raise ValueError(f'Invalid shape/device for {key}')
    if x['valid'].dtype != torch.bool:
        raise ValueError('valid must be bool')
    if numerical:
        for key in INPUT_KEYS - {'valid'}:
            if not torch.isfinite(x[key]).all(): raise ValueError(f'Nonfinite {key}')
        if (c < 0).any() or (x['local_observed_counts'] < 0).any():
            raise ValueError('Negative visible count')
        for key in ('crop_coverage', 'visible_hull_support'):
            if ((x[key] < 0) | (x[key] > 1)).any(): raise ValueError(f'{key} outside [0,1]')
        if not torch.equal(c[..., :2, :, :].sum((-3, -2)), x['local_observed_counts']):
            raise ValueError('Local counts must equal visible inner polar counts')
        if ((x['crop_coverage'] == 0) & (c.sum(-1) > 0)).any():
            raise ValueError('A zero-exposure bin cannot contain observed cells')


def reflect_inputs(x):
    y = dict(x); y['polar_counts'] = x['polar_counts'].flip(-2)
    for key in ('crop_coverage', 'visible_hull_support'): y[key] = x[key].flip(-1)
    y['query_features'] = x['query_features'].clone(); y['query_features'][..., 2] *= -1
    return y


def observed_composition(c):
    total = c.sum(-1, keepdim=True)
    return torch.where(total > 0, c / total.clamp_min(1), torch.full_like(c, .25))


def _logit(v): return math.log(v / (1-v))


class NicheIR(nn.Module):
    """Poisson exposure + learned reaction + symmetric anisotropic diffusion.

    For each class, E(u) = sum_i[e_i*u_i - n_i*log(u_i)
        + kappa*a_i*(u_i-mu_i*log(u_i))
        + tau_i*a_i/2*(u_i-mu_i)^2] + sum_edges[g_ij/2*(u_i-u_j)^2].
    e=a*q, n=c/s, and lambda=s*u. The actual likelihood/energy is s*E;
    dividing by the visible density scale improves conditioning without
    changing the optimizer. g,tau,mu depend only on X and are fixed in the solve.
    The fixed kappa=.01 Gamma-type prior has mode mu and prevents a sampled
    zero observed count from forcing a zero hidden-event rate. Its effective
    exposure/count are kappa*a and kappa*a*mu.
    tau*a > 0 makes E strictly convex. Diffusion is an undirected M-matrix
    Laplacian. A diagonal smooth-energy majorizer followed by the exact
    scalar Poisson prox preserves nonnegativity and descends E for relax<=1.
    """
    def __init__(self, *, strict_inputs: bool = True):
        width, steps = 32, 48
        super().__init__()
        if width < 8 or not isinstance(steps, int) or steps < 1: raise ValueError('width>=8, steps>=1 required')
        self.steps, self.strict_inputs = steps, strict_inputs
        self._register_geometry()
        self.potential_net = nn.Sequential(nn.Linear(NODE_FEATURES, width), nn.SiLU(), nn.Linear(width, 4))
        nn.init.normal_(self.potential_net[-1].weight, std=.015)
        nn.init.zeros_(self.potential_net[-1].bias)
        self.reaction_net = nn.Linear(width, 4)
        nn.init.normal_(self.reaction_net.weight, std=.015)
        nn.init.zeros_(self.reaction_net.bias)
        self.reaction_bias = nn.Parameter(torch.full((4,), _logit((.12-.03)/(.5-.03))))
        self.diffusion_bias = nn.Parameter(torch.tensor([_logit((.12-.015)/(.5-.015)), _logit((.06-.015)/(.5-.015))]).expand(4, 2).clone())
        self.edge_net = nn.Sequential(nn.Linear(EDGE_FEATURES, width), nn.SiLU(), nn.Linear(width, 8))
        nn.init.normal_(self.edge_net[-1].weight, std=.015)
        nn.init.zeros_(self.edge_net[-1].bias)
        self.interface_strength = nn.Parameter(torch.full((4,), _logit(2./12.)))
        self.density_strength = nn.Parameter(torch.full((4,), _logit(.4/4.)))
        self.relaxation_logit = nn.Parameter(torch.tensor(_logit(.9)))

    def _register_geometry(self):
        edges = torch.tensor(RING_EDGES, dtype=torch.float64)
        area = ((edges[1:]**2-edges[:-1]**2)/16)[:, None].expand(4, 16).clone()
        # Sector-area centroids in radius; exact left/right reflection tables
        # avoid trigonometric rounding making mirrored learned features unequal.
        radius = (2./3)*(edges[1:]**3-edges[:-1]**3)/(edges[1:]**2-edges[:-1]**2)
        theta = (torch.arange(16, dtype=torch.float64)+.5)*(2*math.pi/16)
        cosine = theta.cos(); cosine = (cosine+cosine.flip(0))*.5
        self.register_buffer('bin_area', area.float())
        self.register_buffer('radius', radius[:, None].expand(4, 16).reshape(64).float()/3.)
        self.register_buffer('cosine', cosine.expand(4, 16).reshape(64).float())
        self.register_buffer('radial_weight', torch.tensor([1., 1., .25, .1])[:, None].expand(4, 16).reshape(64))
        source, target, conductance, normal_squared = [], [], [], []
        width = 2*math.pi/16
        # Internal angular faces; each undirected edge occurs exactly once.
        for r in range(4):
            for j in range(16):
                face = (j+1)*width
                source.append(r*16+j); target.append(r*16+(j+1)%16)
                conductance.append(float((edges[r+1]-edges[r])/(radius[r]*width)))
                normal_squared.append([math.sin(face)**2, math.cos(face)**2])
        # Radial interfaces; no wrap or edge leaving the outer 3R ring.
        for r in range(3):
            for j in range(16):
                source.append(r*16+j); target.append((r+1)*16+j)
                conductance.append(float(edges[r+1]*width/(radius[r+1]-radius[r])))
                normal_squared.append([float(cosine[j]**2), float(1-cosine[j]**2)])
        source, target = torch.tensor(source), torch.tensor(target)
        incidence = torch.zeros(len(source), 64, dtype=torch.float64)
        incidence[torch.arange(len(source)), source] = 1
        incidence[torch.arange(len(source)), target] = -1
        conductance = torch.tensor(conductance, dtype=torch.float64)
        degree = incidence.abs().T @ conductance
        # One fixed geometry-only rescaling, identical across queries/classes.
        conductance /= degree.max()
        self.register_buffer('edge_source', source)
        self.register_buffer('edge_target', target)
        self.register_buffer('geometric_conductance', conductance.float())
        self.register_buffer('face_normal_squared', torch.tensor(normal_squared, dtype=torch.float32))
        self.register_buffer('incidence', incidence.float())

    def coefficients(self, x):
        validate_inputs(x, self.strict_inputs)
        c = x['polar_counts'].reshape(-1, 64, 4)
        q = x['crop_coverage'].reshape(-1, 64)
        hull = x['visible_hull_support'].reshape(-1, 64)
        area = self.bin_area.reshape(1, 64)
        exposure = area*q
        wc = (c*self.radial_weight[None, :, None]).sum(1)
        wa = (exposure*self.radial_weight).sum(1, keepdim=True)
        density = (wc+.125)/(wa+.25)
        scale = density.sum(-1, keepdim=True)
        pi = density/scale
        local_total = c.sum(-1)
        composition = (c+.25*pi[:, None, :])/(local_total[..., None]+.25)
        # Only evidence at both endpoints can activate an interface barrier.
        reliability = local_total/(local_total+4.)*(hull/(q+.1)).clamp(0, 1)
        relative = (c/scale[:, None, :])/(exposure[..., None]+.05*area[..., None])
        obs = x['local_observed_counts'].reshape(-1, 4)
        inner_pi = (obs+.25*pi)/(obs.sum(-1, keepdim=True)+.25)
        distance = x['query_features'].reshape(-1, 7)[:, :1].clamp(0, 3)/3
        n = len(c)
        features = torch.cat((torch.log1p(relative)/3., composition, pi[:, None].expand(-1, 64, -1),
            q[..., None], hull[..., None], self.radius[None, :, None].expand(n, -1, -1),
            self.cosine[None, :, None].expand(n, -1, -1), distance[:, None].expand(-1, 64, -1),
            inner_pi[:, None].expand(-1, 64, -1), torch.log1p(scale)[:, None].expand(-1, 64, -1)/5., reliability[..., None]), -1)
        if features.shape[-1] != NODE_FEATURES: raise RuntimeError('Node feature contract')
        hidden = self.potential_net[:2](features)
        potential = self.potential_net[-1](hidden)
        mu = pi[:, None, :]*torch.exp(2*torch.tanh(potential/2))
        data_exposure = exposure[..., None].expand_as(c)
        data_count = c/scale[:, None, :]
        gamma_exposure = (GAMMA_KAPPA*area[..., None]).expand_as(c)
        gamma_count = gamma_exposure*mu
        tau = .03+.47*torch.sigmoid(self.reaction_bias+self.reaction_net(hidden))
        reaction = tau*area[..., None]
        i, j = self.edge_source, self.edge_target
        geometry = torch.cat((self.face_normal_squared, self.geometric_conductance[:, None]), -1)
        edge_features = torch.cat(((features[:, i]+features[:, j])*.5,
                                   (features[:, i]-features[:, j]).abs(), geometry[None].expand(n, -1, -1)), -1)
        modulation = self.edge_net(edge_features).reshape(n, -1, 4, 2)
        rates = .015+.485*torch.sigmoid(self.diffusion_bias[None, None]+modulation)
        projected = (rates*self.face_normal_squared[None, :, None, :]).sum(-1)
        contrast = (composition[:, i].sqrt()-composition[:, j].sqrt()).square().sum(-1)
        dens = torch.log1p(relative.sum(-1))
        density_contrast = (dens[:, i]-dens[:, j]).square()
        observed_pair = reliability[:, i]*reliability[:, j]
        penalty = observed_pair[..., None]*(contrast[..., None]*(12*torch.sigmoid(self.interface_strength))
                    + density_contrast[..., None]*(4*torch.sigmoid(self.density_strength)))
        interface = .02+.98*torch.exp(-penalty)
        conductance = self.geometric_conductance[None, :, None]*projected*interface
        degree = self._edge_sum(conductance, len(c))
        return dict(mu=mu, reaction=reaction, conductance=conductance, degree=degree,
                    exposure=data_exposure+gamma_exposure, normalized_count=data_count+gamma_count,
                    data_exposure=data_exposure, data_normalized_count=data_count,
                    gamma_exposure=gamma_exposure, gamma_normalized_count=gamma_count,
                    scale=scale, coverage=q, interface=interface)

    def _edge_sum(self, edge_value, batch):
        output = edge_value.new_zeros((batch, 64, edge_value.shape[-1]))
        return output.index_add(1, self.edge_source, edge_value).index_add(1, self.edge_target, edge_value)

    def laplacian(self, u, conductance):
        flux = conductance*(u[:, self.edge_source]-u[:, self.edge_target])
        output = torch.zeros_like(u)
        return output.index_add(1, self.edge_source, flux).index_add(1, self.edge_target, -flux)

    @staticmethod
    def poisson_prox(v, step, exposure, normalized_count):
        t = v-step*exposure
        root = torch.sqrt((t.square()+4*step*normalized_count).clamp_min(torch.finfo(t.dtype).tiny))
        # Stable root of z^2-t*z-step*n=0. n=0 correctly yields max(t,0).
        positive_count = torch.where(t >= 0, .5*(t+root),
                           2*step*normalized_count/(root-t).clamp_min(torch.finfo(t.dtype).tiny))
        return torch.where(normalized_count > 0, positive_count, t.clamp_min(0))

    def energy(self, u, coef):
        c = coef['normalized_count']
        # Exact extended-real energy: positive pseudo-count implies infinite
        # energy at zero; a neutral log argument only avoids evaluating log(0).
        log_u = torch.where(u > 0, u, torch.ones_like(u)).log()
        observation_and_gamma = coef['exposure']*u-c*log_u
        observation_and_gamma = torch.where((u < 0) | ((u == 0) & (c > 0)),
                                            torch.full_like(u, float('inf')), observation_and_gamma)
        reaction = .5*coef['reaction']*(u-coef['mu']).square()
        diffusion = .5*coef['conductance']*(u[:, self.edge_source]-u[:, self.edge_target]).square()
        return (observation_and_gamma+reaction).sum(1)+diffusion.sum(1)

    def solve(self, coef, steps=None, trace=False):
        u = coef['mu']
        # 2D-L is PSD for any nonnegative undirected graph. Consequently this
        # diagonal is a global majorizer of the smooth-energy Hessian.
        majorizer = coef['reaction']+2*coef['degree']
        step = (.1+.9*torch.sigmoid(self.relaxation_logit))/majorizer
        energies = [self.energy(u, coef)] if trace else None
        for _ in range(self.steps if steps is None else steps):
            grad = coef['reaction']*(u-coef['mu'])+self.laplacian(u, coef['conductance'])
            u = self.poisson_prox(u-step*grad, step, coef['exposure'], coef['normalized_count'])
            if trace: energies.append(self.energy(u, coef))
        return u, torch.stack(energies) if trace else None

    def forward(self, x):
        coef = self.coefficients(x); u, _ = self.solve(coef)
        leading = x['polar_counts'].shape[:2]
        field = (u*coef['scale'][:, None]).reshape(-1, 4, 16, 4)
        q = coef['coverage'].reshape(-1, 4, 16)
        missing_area = self.bin_area[None, :2]*(1-q[:, :2])
        bins = missing_area[..., None]*field[:, :2]
        obs = x['local_observed_counts'].reshape(-1, 4)
        missing = bins.sum((1, 2)); complete = obs+missing
        total = complete.sum(-1, keepdim=True)
        p = torch.where(total > 0, complete/total.clamp_min(torch.finfo(total.dtype).tiny), torch.full_like(complete, .25))
        full = (q[:, :2] == 1).all((1, 2))
        p = torch.where(full[:, None], observed_composition(obs), p)
        valid = x['valid'].reshape(-1)
        p = torch.where(valid[:, None], p, torch.full_like(p, .25))
        missing = torch.where(valid[:, None], missing, torch.zeros_like(missing))
        bins = torch.where(valid[:, None, None, None], bins, torch.zeros_like(bins))
        field = torch.where(valid[:, None, None, None], field, torch.zeros_like(field))
        return dict(local=p.reshape(*leading, 4), missing_rate=missing.reshape(*leading, 4),
                    missing_bins=bins.reshape(*leading, 2, 16, 4), missing_area=missing_area.reshape(*leading, 2, 16),
                    field_density=field.reshape(*leading, 4, 16, 4))

    def operator_diagnostics(self, x, curvature=False, steps=None):
        """Expensive opt-in diagnostics, no normal forward/inference overhead.

        Covariance is inverse local curvature of s*E at the finite iterate,
        conditional on learned X-dependent coefficients; it excludes coefficient
        uncertainty, model misspecification and boundary truncation. Positive
        off-diagonal covariance reflects spatial coupling; no coverage claim.
        """
        coef = self.coefficients(x); u, energy = self.solve(coef, steps=steps, trace=True)
        grad = (coef['exposure']-coef['normalized_count']/u
                + coef['reaction']*(u-coef['mu'])+self.laplacian(u, coef['conductance']))
        # Gamma barrier gives an interior solution: do not conceal residuals
        # by declaring a small positive iterate an active zero boundary.
        kkt = grad.abs()
        out = dict(energy_trace=energy, normalized_density=u, stationarity_linf=kkt.amax(1),
                   conductance=coef['conductance'], interface=coef['interface'],
                   density_scale=coef['scale'], interior=u.amin(1) > 0,
                   numerically_near_boundary=u.amin(1) < 1e-7,
                   data_exposure=coef['data_exposure'], data_normalized_count=coef['data_normalized_count'],
                   gamma_exposure=coef['gamma_exposure'], gamma_normalized_count=coef['gamma_normalized_count'])
        if curvature:
            # Small positive intensities have genuine high curvature. Evaluate
            # this optional diagnostic in float64 instead of flooring u.
            b = self.incidence.double()
            laplace = torch.einsum('ev,nec,ew->ncvw', b, coef['conductance'].double(), b)
            diagonal = coef['reaction'].double()+coef['normalized_count'].double()/u.double().square()
            hessian = laplace+torch.diag_embed(diagonal.transpose(1, 2))
            inverse = torch.linalg.inv(hessian)
            scale = coef['scale'].double()[:, :, None, None]
            covariance = scale*inverse  # Cov(lambda) = s * H_E^{-1}.
            missing_weight = self.bin_area.double().reshape(1, 64)*(1-coef['coverage'].double())
            missing_weight = torch.cat((missing_weight[:, :32], torch.zeros_like(missing_weight[:, 32:])), 1)
            mean = (missing_weight[..., None]*u*coef['scale'][:, None]).sum(1)
            out.update(energy_hessian=hessian, density_covariance=covariance,
                       missing_count_variance=mean+torch.einsum('nv,ncvw,nw->nc', missing_weight, covariance, missing_weight))
        return out

