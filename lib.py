import torch
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_smooth_style
from utils import *
from generator import *

from torch import nn 

class FullMixtureModel(nn.Module):
    def __init__(
        self,
        features_dim,
        n_flavours,
        norm_factors,
        scores_model,
        kin_model,
        fit_conditional_pdf=True,
        lnN_constraints=None,
        num_nuisances=1,
        norm_nuisance_limit=5.0,
        lnN_mix_matrix=None,
        norm_nuisance_profile_mask=None,
        m_vector_profile_mask=None,
    ):
        super().__init__()
        self.fit_conditional_pdf = fit_conditional_pdf
        self.features_dim = features_dim
        self.n_flavours = n_flavours

        self.log_norm_factors = nn.Parameter(norm_factors.log(), requires_grad=False)
        # Z = 1
        self.m_vector  = nn.Parameter(torch.zeros(num_nuisances,
                                                                    device=self.log_norm_factors.device,
                                                                    dtype=torch.float32),
                                                                requires_grad=True)

        # lnN nuisances are independent of flavours: there is one entry per lnN
        # constraint (= number of columns of lnN_mix_matrix), NOT one per flavour.
        # In v10/v11 these happened to coincide (n_flavours == n_lnN == 2); v12 has
        # n_flavours=2 but a single global lnN nuisance, so size by lnN_constraints.
        if lnN_constraints is None:
            lnN_constraints = torch.zeros_like(self.log_norm_factors)
        lnN_constraints = torch.as_tensor(lnN_constraints, dtype=self.log_norm_factors.dtype)
        n_lnN = lnN_constraints.shape[0]

        self.norm_nuisance = nn.Parameter(
            torch.zeros(n_lnN, device=self.log_norm_factors.device, dtype=torch.float32),
            requires_grad=True,
        )
        self.norm_nuisance_limit = float(norm_nuisance_limit)

        self.lnN_constraints = nn.Parameter(torch.as_tensor(lnN_constraints, dtype=self.log_norm_factors.dtype), requires_grad=False)
        self.lnN_constraints_mask = nn.Parameter(torch.as_tensor((lnN_constraints > 0.), dtype=torch.bool), requires_grad=False)
        self.lnN_constraints_factor_log = nn.Parameter(
            torch.as_tensor((1.0 + torch.as_tensor(lnN_constraints)).log(), dtype=self.log_norm_factors.dtype),
            requires_grad=False
        )

        # Mixing matrix A [n_flavours, n_lnN_nuisances]:
        #   log N_eff = log N_nom + A @ (f(theta) * log(1+sigma))
        # Default: identity (independent per-flavour lnN).
        # For global+ratio with 2 flavours: [[1,+1],[1,-1]]
        if lnN_mix_matrix is None:
            lnN_mix_matrix = torch.eye(n_flavours)
        self.lnN_mix_matrix = nn.Parameter(
            torch.as_tensor(lnN_mix_matrix, dtype=self.log_norm_factors.dtype,
                            device=self.log_norm_factors.device),
            requires_grad=False,
        )

        # Per-lnN-nuisance mask: True = profiled, False = frozen at nominal (factor = 1.0).
        # Frozen entries get no gradient and contribute no prior term.
        if norm_nuisance_profile_mask is None:
            norm_nuisance_profile_mask = torch.ones(n_lnN, dtype=torch.bool)
        self.norm_nuisance_profile_mask = nn.Parameter(
            torch.as_tensor(norm_nuisance_profile_mask, dtype=torch.bool,
                            device=self.log_norm_factors.device),
            requires_grad=False,
        )

        # Per-shape-nuisance mask on m_vector: True = profiled, False = frozen at nominal (0).
        # Used to freeze e.g. ν_y during the mixture stage so it doesn't compete with T.
        # `log_prob` zeros frozen entries before they reach the score/kin/T models, blocking
        # gradient flow back into the corresponding m_vector slot.
        if m_vector_profile_mask is None:
            m_vector_profile_mask = torch.ones(num_nuisances, dtype=torch.bool)
        self.m_vector_profile_mask = nn.Parameter(
            torch.as_tensor(m_vector_profile_mask, dtype=torch.bool,
                            device=self.log_norm_factors.device),
            requires_grad=False,
        )

        self.scores_model = scores_model
        self.kin_model    = kin_model

        for p in self.scores_model.parameters():
            p.requires_grad = False
        for p in self.kin_model.parameters():
            p.requires_grad = False


    @property
    def norm_nuisance_factor(self):
        # smooth interval between the limit
        raw = self.norm_nuisance / (1.0 + torch.abs(self.norm_nuisance) / self.norm_nuisance_limit)
        # Pin frozen entries to 0 (multiplicative factor 1.0). torch.where blocks gradient flow
        # to self.norm_nuisance on the False branch, so frozen entries stay at their init value.
        return torch.where(self.norm_nuisance_profile_mask, raw, torch.zeros_like(raw))

    @property
    def modified_log_normalization(self):
        scaled = self.norm_nuisance_factor * self.lnN_constraints_factor_log
        return self.log_norm_factors + self.lnN_mix_matrix @ scaled

    def get_lnN_likelihood_term(self):
        """
        Log-likelihood of Gaussian prior (up to additive constant),
        returned with the same sign convention as in the notebook.
        """
        active_mask = self.lnN_constraints_mask & self.norm_nuisance_profile_mask
        nuis = torch.where(
            active_mask,
            -self.lnN_constraints.log() - 0.5 * (self.norm_nuisance_factor / self.lnN_constraints)**2,
            torch.zeros_like(self.norm_nuisance_factor, device=self.norm_nuisance_factor.device),
        )
        return nuis.sum()


    def logprob_scores_by_flavour(self, s, z, m=None):
        """ 
        Compute log probability of scores by flavour.
        
        Args:
            s: Scores tensor [B, 3] or [Nflav, B, 3]
            z: Kinematic features [B, Z]
            nu_vector: Nuisance vector [B, n_nuisances]
        
        Returns:
            Log probabilities [B, F]
        """
        B = z.shape[0]
        # --- Refactoring Step 1: s_input ---
        if s.ndim == 2:
            # Original: s.repeat(self.n_flavours, 1)
            # New: Expand [B, 3] -> [F, B, 3] -> Flatten
            s_input = s.unsqueeze(0).expand(self.n_flavours, -1, -1).reshape(-1, s.shape[-1])
        elif s.ndim == 3:
            # batch dim is the second axis
            s_input = s.reshape(-1, self.features_dim)
        else:
            raise ValueError("Score tensor must have 2 or 3 dimensions")

        # --- Refactoring Step 2: all_context ---
        # 1. Create One-Hot efficiently: [F, F] -> [F, 1, F] -> [F, B, F]
        fl_eye = torch.eye(self.n_flavours, device=s.device, dtype=s.dtype)
        fl_expanded = fl_eye.unsqueeze(1).expand(-1, B, -1)

        # 2. Expand z: [B, Z] -> [1, B, Z] -> [F, B, Z]
        z_expanded = z.unsqueeze(0).expand(self.n_flavours, -1, -1)

        # 3. Concatenate and Flatten
        # Cat [F, B, F] and [F, B, Z] -> [F, B, F+Z] -> [F*B, F+Z]
        all_context = torch.cat([fl_expanded, z_expanded], dim=-1).reshape(-1, fl_expanded.shape[-1] + z.shape[-1])

        # Feed the score residual its nuisances: the FIRST n_score_nuis entries of m.
        # This matches logprob_kin_by_flavour (m[:, :n_kin_nuis]), rsample
        # (scores_model.sample gets the full m), and how the residual is trained
        # (train_systematics passes the full m-vector, so nuisance-net i ↔ m[:, i]).
        # NB: this previously hardcoded m[:, 1:2] — a single score nuisance at index 1
        # (v11 legacy). For a >1-nuisance score residual that [B,1] slice silently
        # broadcasts against the per-nuisance central/scale, feeding the wrong values
        # (e.g. net 0, trained on ν_shift, would receive ν_squeeze) — so log_prob and
        # rsample disagreed and the mixture would not close.
        n_score_nuis = self.scores_model.num_nuisances
        if n_score_nuis > 0:
            m_all = m[:, :n_score_nuis].unsqueeze(0).expand(self.n_flavours, -1, -1).reshape(-1, n_score_nuis)
        else:
            m_all = torch.zeros(self.n_flavours * B, 0, device=m.device, dtype=m.dtype)

        _, logprob = self.scores_model(s_input, all_context, m_all)
        return logprob.reshape(self.n_flavours, -1).T # [B,F]

    def logprob_kin_by_flavour(self, z, m=None):
        """
        Compute log probability of kinematics by flavour.
        
        Args:
            z: Kinematic features [B, Z]
            nu_vector: Nuisance vector [B, n_nuisances]
            
        Returns:
            Log probabilities [B, F]
        """
        B = z.shape[0]
        # --- Refactoring Step 1: Input x (z expanded) ---
        # Original: z.repeat(self.n_flavours, 1)
        # New: [B, Z] -> [1, B, Z] -> [F, B, Z] -> [F*B, Z]
        z_input = z.unsqueeze(0).expand(self.n_flavours, -1, -1).reshape(-1, z.shape[-1])

        # --- Refactoring Step 2: Context (One-Hot expanded) ---
        # Original: complex arange/repeat/flatten/one_hot
        # New: [F, F] (Identity) -> [F, 1, F] -> [F, B, F] -> [F*B, F]
        fl_eye = torch.eye(self.n_flavours, device=z.device, dtype=z.dtype)
        all_fl = fl_eye.unsqueeze(1).expand(-1, B, -1).reshape(-1, self.n_flavours)

        # Pass exactly num_nuisances slots to the kin residual model.
        # v11: 1 (ν_x).  v12: 2 (ν_shift, ν_rot).  v10 score-as-kin: anything.
        n_kin_nuis = self.kin_model.num_nuisances
        if n_kin_nuis > 0:
            m_all = m[:, :n_kin_nuis].unsqueeze(0).expand(self.n_flavours, -1, -1).reshape(-1, n_kin_nuis)
        else:
            m_all = torch.zeros(self.n_flavours * B, 0, device=m.device, dtype=m.dtype)

        # Call model
        _, logprob = self.kin_model(z_input, all_fl, m_all)
        
        # Reshape result
        return logprob.reshape(self.n_flavours, -1).T

    def get_fractions(self, logprob_kin, log_norm):
        if self.fit_conditional_pdf:
            tot = torch.logsumexp((log_norm + logprob_kin), dim=1).unsqueeze(1)  
        else:
            tot = torch.logsumexp((log_norm ), dim=0).unsqueeze(-1)  
        return log_norm + logprob_kin - tot  

    def log_prob(self, s, z, T=None, m=None, m_transfer=None, m_score=None):
        """
        Compute log probability of the mixture model.

        Args:
            s: Score features [B, 3]
            z: Kinematic features [B, Z]
            T: Optional transform model
            m: nuisance params vector [B, N] — drives the X-space path (residual_kin).
            m_transfer: optional separate nuisance vector for the transfer T
                (residual_transfer, step-2). Defaults to `m`.
            m_score: optional separate nuisance vector for the score-density residual
                (residual_score, step-1 p(y|x,ν)). Defaults to `m_transfer`.

                Supplying distinct values routes the three residual responses
                (kin / transfer / score) to different nuisance points. This is used
                by the holdout-closure 3-way decomposition to attribute the
                m-response to each component independently:
                  - `m`          → residual_kin       (X-space, p(x|c,ν))
                  - `m_transfer` → residual_transfer  (transfer T, step-2 profiling)
                  - `m_score`    → residual_score     (score density, step-1 p(y|x,ν))
                With all three equal (the default chain) the behaviour is unchanged.

        Returns:
            Log probabilities [B], and optionally transformed scores
        """
        # Freeze masked-out nuisances to 0 before they reach the residual flows / T.
        # torch.where blocks gradient flow on the False branch back to m_vector / external m.
        if m is not None:
            m = torch.where(
                self.m_vector_profile_mask.view(1, -1),
                m,
                torch.zeros_like(m),
            )
        if m_transfer is None:
            m_transfer = m
        else:
            m_transfer = torch.where(
                self.m_vector_profile_mask.view(1, -1),
                m_transfer,
                torch.zeros_like(m_transfer),
            )
        if m_score is None:
            m_score = m_transfer
        else:
            m_score = torch.where(
                self.m_vector_profile_mask.view(1, -1),
                m_score,
                torch.zeros_like(m_score),
            )

        # Get the kinematic density (X-space response uses `m`)
        logp_kin = self.logprob_kin_by_flavour(z, m=m)  # [B, F]
        # Get the fractions
        log_Af = self.get_fractions(logp_kin, self.modified_log_normalization)
        # Score path: transfer T uses `m_transfer`, score density uses `m_score`.
        if T is not None:
            s_T, log_jac_T = self.get_transformed_scores_data(
                    s, z, n_flavours=self.n_flavours, T=T, m=m_transfer, inverse=False
                )
            logp_s = self.logprob_scores_by_flavour(s_T, z, m=m_score)
            return torch.logsumexp(log_Af + logp_s + log_jac_T, dim=-1), s_T
        else:
            logp_s = self.logprob_scores_by_flavour(s, z, m=m_score)
            return torch.logsumexp(log_Af + logp_s, dim=-1)
    
    def get_transformed_scores_data(self, s, z, n_flavours, T, m,  inverse=False):
        """
        Transform scores with memory-efficient logic.
        
        Args:
            s: Scores [B, 3] or [F, B, 3]
            z: Kinematics [B, Z]
            m: nuisance vector[B, N]
            n_flavours: Number of flavours
            inverse: Apply inverse transformation
            
        Returns:
            Transformed scores [F, B, 3] and log jacobian [B, F]
        """
        B = z.shape[0]
        
        if s.ndim == 2: 
            # 1. Expand 's' (no copy)
            # [B, 3] -> [1, B, 3] -> [F, B, 3] -> [F*B, 3]
            ss_view = s.unsqueeze(0).expand(n_flavours, -1, -1)
            ss_flat = ss_view.reshape(-1, self.features_dim)
        elif s.ndim == 3:
            ss_flat = s.reshape(-1, self.features_dim)

        # 2. Create One-Hot (efficiently)
        # [F, F] -> [F, 1, F] -> [F, B, F]
        fl_eye = torch.eye(n_flavours, device=s.device, dtype=s.dtype)
        fl_expanded = fl_eye.unsqueeze(1).expand(-1, B, -1)

        # 3. Expand 'z' (no copy)
        # [B, Z] -> [1, B, Z] -> [F, B, Z]
        z_expanded = z.unsqueeze(0).expand(n_flavours, -1, -1)

        # 4. Concatenate Contexts
        # This is where the [F, B, F+Z] shape happens before flattening --> [F*B, F+Z]
        all_context_flat = torch.cat([fl_expanded, z_expanded], dim=-1).reshape(-1, fl_expanded.shape[-1] + z.shape[-1])

        # T is class-aware in v12: conditions on the flavour one-hot + x. This matches
        # the truth structure of the data distortion (class-dependent rotation in y-space).
        t_context_flat = all_context_flat

        # m is [B, N]; replicate across flavours to [F*B, N]. unsqueeze+expand+reshape is
        # safe when m is non-contiguous (e.g. from m_vector.expand(B, -1)).
        m_all = m.unsqueeze(0).expand(self.n_flavours, -1, -1).reshape(-1, m.shape[-1])

        # 5. Transform
        if inverse:
            ss_transf, ladj = T.inverse(ss_flat, t_context_flat, m_all)
        else:
            ss_transf, ladj = T(ss_flat, t_context_flat, m_all)

        # 6. Reshape Output to match original
        ss_transf = ss_transf.view(n_flavours, B, self.features_dim)  # --> [F, B, 3]
        ladj = ladj.view(n_flavours, B).T ## --> [B, F] for the logsumexps by flavout
        return ss_transf, ladj

    def rsample(self, N, T=None, m=None):
        """
        Sample from the mixture model.
        
        Args:
            N: Number of samples
            T: Optional transform model
            m: Optional nuisance parameters
            
        Returns:
            Tuple of (scores, shifted_kinematics, flavour_labels)
        """
        probs = torch.softmax(self.modified_log_normalization, dim=0)
        fl_samples = torch.multinomial(probs, num_samples=int(N), replacement=True)
        fl_1h = torch.nn.functional.one_hot(
            fl_samples, num_classes=self.n_flavours
        ).to(self.modified_log_normalization.dtype)
        
        kin_samples = self.kin_model.sample(fl_1h.shape[0], fl_1h, m).squeeze(0)
        
        score_ctx = torch.cat([fl_1h, kin_samples], dim=-1)
        scores_samples = self.scores_model.sample(N, score_ctx, m).squeeze(0)
      
        if T is not None:
            # T is class-aware in v12: condition on the flavour one-hot + x.
            t_ctx = torch.cat([fl_1h, kin_samples], dim=-1)
            scores_samples_T, _ = T.inverse(scores_samples, t_ctx, m)
            return scores_samples_T, kin_samples, fl_samples
        else:
            return scores_samples, kin_samples, fl_samples


##########################

from zuko.lazy import LazyComposedTransform
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.transforms import MonotonicRQSTransform

#Helper function to initialize weights to near zero
def init_weights_to_identity(m):
    if isinstance(m, nn.Linear):
        # Initialize weights to small values
        nn.init.uniform_(m.weight, -1e-4, 1e-4) 
        # Initialize bias to zero
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class TransferModel(nn.Module):

    def __init__(self,features_dim, context_dim, n_transforms, nbins, 
                         hidden_net, add_rotation=False):
        super().__init__()
        self.features_dim = features_dim
        self.context_dim = context_dim
        transforms = []
        spline_shapes = ([nbins], [nbins], [nbins + 1])

        for i in range(n_transforms):
            order = None
            if not add_rotation:
                order = torch.randperm(features_dim)

            t=  MaskedAutoregressiveTransform(
                    features=features_dim,
                    context=context_dim,
                    univariate=MonotonicRQSTransform,
                    shapes=spline_shapes,
                    hidden_features=hidden_net,
                    activation=nn.ReLU,
                    order=order
                )

            # Start from
            t.apply(init_weights_to_identity)
            
            transforms.append(t )
            
            if (i != n_transforms-1) :
                if add_rotation:
                    transforms.append(           
                        # 2) learnable rotation
                        UnconditionalTransform(
                            RotationTransform,
                            torch.rand(features_dim,features_dim),
                        )
                    )
            # the permutation is to be improved

        self.T =  LazyComposedTransform(*transforms)

    def forward(self, x, context, m=None):
        # ignoring the nuisance context m in this base model
        z, ladj = self.T(context).call_and_ladj(x)
        return z, ladj

    def inverse(self, z, context, m=None):
        x, ladj = self.T(context).inv.call_and_ladj(z)
        return x, ladj

    

#############################
