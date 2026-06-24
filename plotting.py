import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
import torch
from residual_flow import IndependentPolynomialResidualTransform


# ---------------------------------------------------------------------------
# Shared quiver styling: |Δ| heatmap background + readable direction arrows.
# Used by plot_results.py (transfer map) and visualize_hessian_distortion.py.
# ---------------------------------------------------------------------------

def truncate_cmap(name, floor):
    """Lift a colormap's dark low end off black: map values to the [floor, 1] slice of
    `name`, so even |Δ|≈0 reads as a medium colour (avoids the near-black viridis floor)."""
    base = plt.get_cmap(name)
    if floor <= 0:
        return base
    return ListedColormap(base(np.linspace(min(max(floor, 0.0), 0.9), 1.0, 256)))


def field_quiver_panel(ax, g0, g1, YY1, YY2, U, V, norm, cmap, *,
                       background=True, arrows="unit", arrow_len=0.0):
    """One vector-field panel: a |Δ| heatmap background + direction arrows on top.

    background : filled `pcolormesh` of |Δ| (the colourful intensity map).
    arrows     : 'unit'   -> every arrow the same length `arrow_len` (direction reads in
                            EVERY cell; magnitude is carried by the background);
                 'scaled' -> length ∝ |Δ| (longest = `arrow_len`).
    With a background the arrows are white with a thin black outline (legible on both ends
    of the colormap); without one they are coloured by |Δ| instead (the bare-quiver look).
    `g0`, `g1` are the 1-D grid axes; `YY1`, `YY2`, `U`, `V` the meshgrid + components."""
    spd = np.sqrt(U ** 2 + V ** 2)
    smax = float(spd.max()) or 1.0
    if background:
        ax.pcolormesh(g0, g1, spd, norm=norm, cmap=cmap, shading="gouraud", zorder=0)
    if arrows == "unit":
        eps = smax * 1e-6
        f = np.where(spd > eps, arrow_len / (spd + eps), 0.0)   # unit length, 0 where flat
        Ud, Vd = U * f, V * f
    else:
        Ud, Vd = U * (arrow_len / smax), V * (arrow_len / smax)
    if background:
        ax.quiver(YY1, YY2, Ud, Vd, angles="xy", scale_units="xy", scale=1.0, pivot="mid",
                  color="white", edgecolor="black", linewidth=0.35, width=0.006, zorder=2)
    else:
        ax.quiver(YY1, YY2, Ud, Vd, spd, cmap=cmap, norm=norm, angles="xy",
                  scale_units="xy", scale=1.0, pivot="mid", width=0.006, zorder=2)
    ax.set_xlim(float(np.min(g0)), float(np.max(g0)))
    ax.set_ylim(float(np.min(g1)), float(np.max(g1)))

def plot_smooth_style(ax, points_orig, points_trans, d1_idx, d2_idx, 
                      label1, label2, xlim, ylim, grid_res=100, arrow_skip=5, title=""):
    """
    Replicates the ATLAS-style plot: Smooth heatmap background with 
    sparse, uniformly spaced arrows on top.
    
    Parameters:
    - grid_res: Resolution of the background heatmap (e.g., 100x100)
    - arrow_skip: Plot an arrow every 'n' grid points (controls arrow density)
    """
    
    # 1. Extract Coordinates and Calculate Displacement
    x = points_orig[:, d1_idx]
    y = points_orig[:, d2_idx]
    
    # Vectors (u, v)
    u = points_trans[:, d1_idx] - x
    v = points_trans[:, d2_idx] - y
    magnitude = np.sqrt(u**2 + v**2)
    
    # 2. Create a Dense Grid (for the smooth background)
    # We need a fine grid to make the colors look smooth
    xi = np.linspace(xlim[0], xlim[1], grid_res)
    yi = np.linspace(ylim[0], ylim[1], grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # 3. Interpolate Data onto the Dense Grid
    # We interpolate U, V, and Magnitude separately
    # 'linear' is robust; use 'cubic' for very smooth curves but it can overshoot
    zi_mag = griddata((x, y), magnitude, (Xi, Yi), method='linear')
    zi_u   = griddata((x, y), u,         (Xi, Yi), method='linear')
    zi_v   = griddata((x, y), v,         (Xi, Yi), method='linear')
    
    # 4. Plot the Background (The Smooth Gradient)
    # extent defines the [xmin, xmax, ymin, ymax] of the image
    im = ax.imshow(zi_mag, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
                   origin='lower', cmap='YlOrRd', aspect='auto')
    
    # 5. Plot the Arrows (The Sparse Vector Field)
    # KEY STEP: We slice the arrays [::arrow_skip] to pick only every 5th (or nth) point.
    # This gives us the "clean" look of the reference image.
    
    # Create slice objects
    s = slice(None, None, arrow_skip)
    
    ax.quiver(Xi[s, s], Yi[s, s], zi_u[s, s], zi_v[s, s], 
              color='black',      # Simple black arrows
              pivot='mid',        # Arrows rotate around their center
              scale_units='xy',   # Scale arrows relative to axis units (optional)
              angles='xy',
              width=0.004,        # Make them thin like the reference
              headwidth=4, 
              headlength=5)

    # 6. Styling
    ax.set_xlabel(label1, fontsize=12)
    ax.set_ylabel(label2, fontsize=12)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Add Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Arrow length', rotation=90, labelpad=15, fontsize=12)
    
    return im

# --- Usage Example ---
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # Adjust 'grid_res' for smoothness and 'arrow_skip' for arrow density
# plot_smooth_style(axes[0], points_orig, points_trans, 0, 1, 'logit p_b', 'logit p_c', (-1, 1), (-1, 1), grid_res=100, arrow_skip=8)
# plot_smooth_style(axes[1], points_orig, points_trans, 0, 2, 'logit p_b', 'logit p_u', (-1, 1), (-1, 1), grid_res=100, arrow_skip=8)
# plot_smooth_style(axes[2], points_orig, points_trans, 2, 1, 'logit p_u', 'logit p_c', (-1, 1), (-1, 1), grid_res=100, arrow_skip=8)

# plt.tight_layout()
# plt.show()



def get_residual_transform_output(model, x, context, m):
    """
    Computes the output of the residual transformation stack for a given input x and nuisance m.
    It replicates the part of SystematicCorrectedModel.forward that applies the residual stack.
    """
    x_curr = x.clone()
    # Iterate through the transforms in the order they are applied in forward()
    for transform in model.transforms:
        x_curr, _ = transform(x_curr, context=context, m=m)
    return x_curr


def visualize_residual_map(model, delta, nuisance_idx=0, context=None, x_range=(-3, 3), y_range=(-3, 3), 
                nbins=20, ax=None, quiver_scale=1, title="Transformation"):
    """
    Visualizes the residual shift (difference between transformed x at m vs x at central).
    
    Args:
        model: SystematicCorrectedModel instance
        delta: float. The shift magnitude for the specific nuisance parameter.
        nuisance_idx: int. The index of the nuisance parameter to vary.
        context: Tensor (optional), context for the transformation. 
        x_range, y_range: tuples, bounds for the visualization grid.
        nbins: int, grid resolution (per dimension).
        ax: matplotlib axis (optional).
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Construct delta_m vector targeting the specific nuisance index
    delta_m_vec = torch.zeros(model.num_nuisances, device=device)
    if nuisance_idx >= model.num_nuisances:
        raise ValueError(f"nuisance_idx {nuisance_idx} out of bounds (model has {model.num_nuisances} nuisances)")
    delta_m_vec[nuisance_idx] = delta
        
    central_val = model.central_nuisance_values
    
    # 2D Visualization (Arrow Field)
    if model.feat_dim == 2:
        # Create grid
        x = torch.linspace(x_range[0], x_range[1], nbins, device=device)
        y = torch.linspace(y_range[0], y_range[1], nbins, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='xy') 
        inputs = torch.stack([xx.flatten(), yy.flatten()], dim=1) # [N*N, 2]
        batch_size = inputs.shape[0]
        
        # Expand m and context
        m_shifted = (central_val + delta_m_vec).unsqueeze(0).expand(batch_size, -1)
        m_central = central_val.unsqueeze(0).expand(batch_size, -1)
        
        if context is None:
            # Default to zeros if not provided
            context = torch.zeros(batch_size, model.ctx_dim, device=device)
        else:
             if context.ndim == 1:
                  context = context.unsqueeze(0)
             if context.shape[0] != batch_size:
                 context = context[0:1].expand(batch_size, -1)

        with torch.no_grad():
            x_nom_shifted = get_residual_transform_output(model, inputs, context, m_shifted)
            x_nom_central = get_residual_transform_output(model, inputs, context, m_central)

        # Difference vector
        diff = x_nom_shifted - x_nom_central
        u = diff[:, 0].reshape(nbins, nbins).cpu().numpy()
        v = diff[:, 1].reshape(nbins, nbins).cpu().numpy()
        speed = np.sqrt(u**2 + v**2)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        
        # Quiver plot
        q = ax.quiver(xx.cpu().numpy(), yy.cpu().numpy(), u, v, speed, cmap='viridis', pivot='mid', scale=quiver_scale)
        # ax.quiverkey(q, X=0.9, Y=1.05, U=0.1, label='Shift vector', labelpos='E')
        
        try:
            plt.colorbar(q, ax=ax, label='Shift magnitude')
        except:
            pass 

        ax.set_title(f"{title}, $\Delta$ = {delta}")
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True, alpha=0.3)
    
    # 1D Visualization (Line Plot)
    elif model.feat_dim == 1:
        x = torch.linspace(x_range[0], x_range[1], nbins * 5, device=device)
        inputs = x.unsqueeze(1) # [B, 1]
        batch_size = inputs.shape[0]

        m_shifted = (central_val + delta_m_vec).unsqueeze(0).expand(batch_size, -1)
        m_central = central_val.unsqueeze(0).expand(batch_size, -1)

        if context is None:
            context = torch.zeros(batch_size, model.ctx_dim, device=device)
        else:
             if context.ndim == 1:
                  context = context.unsqueeze(0)
             if context.shape[0] != batch_size:
                 context = context[0:1].expand(batch_size, -1)

        with torch.no_grad():
            x_nom_shifted = get_residual_transform_output(model, inputs, context, m_shifted)
            x_nom_central = get_residual_transform_output(model, inputs, context, m_central)
            
        diff = x_nom_shifted - x_nom_central # [B, 1]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        ax.plot(x.cpu().numpy(), diff.cpu().numpy(), label=f"Nuisance {nuisance_idx}, $\Delta$={delta}", linewidth=2)
        ax.set_title("Residual Map Shift (1D)")
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Shift ($x_{corr} - x_{nom}$)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    return ax



def get_residual_coeffs(model, x_features, context, nuisance_idx=0):
    """
    Extracts the polynomial coefficients (linear and quadratic terms) 
    for the scaling (s) and shifting (t) parameters of the residual flow.
    """
    # Find the IndependentPolynomialResidualTransform layer
    res_layer = None
    for transform in model.transforms:
        if isinstance(transform, IndependentPolynomialResidualTransform):
            res_layer = transform
            break
            
    if res_layer is None:
        raise ValueError("No IndependentPolynomialResidualTransform found in the model")
        
    # Select the net for the specific nuisance
    if nuisance_idx >= len(res_layer.nuisance_nets):
        raise ValueError(f"Nuisance index {nuisance_idx} out of range")
        
    net = res_layer.nuisance_nets[nuisance_idx]
    
    # Prepare input: [Batch, Features + Context]
    # Ensure inputs are on the correct device
    device = next(net.parameters()).device
    x_features = x_features.to(device)
    context = context.to(device)
    
    input_tensor = torch.cat([x_features, context], dim=-1)
    
    # Forward pass (no gradient needed for visualization)
    with torch.no_grad():
        raw_out = net(input_tensor)
        
    # Reshape [Batch, Features * 4] -> [Batch, Features, 4]
    coeffs = raw_out.view(x_features.shape[0], res_layer.features, 4)
    # coeffs[..., 0] = Linear s
    # coeffs[..., 1] = Quadratic s
    # coeffs[..., 2] = Linear t
    # coeffs[..., 3] = Quadratic t
    return coeffs, res_layer.central_nuisance_values[nuisance_idx], res_layer.nuisance_scales[nuisance_idx]


def get_residual_cross_coeffs(model, x_features, context, pair_idx=0):
    """
    Extracts cross-term coefficients (C_s, C_t) for one selected nuisance pair.

    Returns:
        cross_coeffs: Tensor [batch, features, 2]
        pair: Tensor [2] with nuisance indices (i, j)
        damping: scalar cross-term damping factor
    """
    # Find the IndependentPolynomialResidualTransform layer
    res_layer = None
    for transform in model.transforms:
        if isinstance(transform, IndependentPolynomialResidualTransform):
            res_layer = transform
            break

    if res_layer is None:
        raise ValueError("No IndependentPolynomialResidualTransform found in the model")

    if len(res_layer.cross_nets) == 0:
        raise ValueError("No cross-term networks found in the residual layer")

    if pair_idx >= len(res_layer.cross_nets):
        raise ValueError(f"pair_idx {pair_idx} out of range (found {len(res_layer.cross_nets)} cross pairs)")

    cross_net = res_layer.cross_nets[pair_idx]

    # Prepare input: [Batch, Features + Context]
    device = next(cross_net.parameters()).device
    x_features = x_features.to(device)
    context = context.to(device)

    input_tensor = torch.cat([x_features, context], dim=-1)

    with torch.no_grad():
        cross_raw_out = cross_net(input_tensor)

    # [Batch, Features * 2] -> [Batch, Features, 2]
    cross_coeffs = cross_raw_out.view(x_features.shape[0], res_layer.features, 2)
    pair = res_layer.cross_term_pairs[pair_idx]
    return cross_coeffs, pair, res_layer.cross_term_damping

def plot_residual_components(model, nuisance_idx, x_range=(-4, 4), n_points=100, feature_val=0.0):
    """
    Plots the linear and quadratic components of the residual connection 
    as a function of the kinematic context variable 'x', for each flavour.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # scan context 'x'
    x_vals = torch.linspace(x_range[0], x_range[1], n_points, device=device).unsqueeze(1)
    
    # Prepare inputs for both flavours
    # Flavour 0: [1, 0]
    c0 = torch.tensor([1.0, 0.0], device=device).expand(n_points, -1)
    ctx_0 = torch.cat([c0, x_vals], dim=-1)
    
    # Flavour 1: [0, 1]
    c1 = torch.tensor([0.0, 1.0], device=device).expand(n_points, -1)
    ctx_1 = torch.cat([c1, x_vals], dim=-1)
    
    # Prepare dummy features (fixed to feature_val)
    # Assuming model features dim is known or can be inferred. 
    # For residual_T features_dim=2
    # We set features to zeros to see the intercepts (or a specific value)
    features_input = torch.full((n_points, model.feat_dim), feature_val, device=device)
    
    coeffs_0, _, _ = get_residual_coeffs(model, features_input, ctx_0, nuisance_idx)
    coeffs_1, _, _ = get_residual_coeffs(model, features_input, ctx_1, nuisance_idx)
    
    # Components names
    comp_names = ["Linear s (Scale)", "Quad s (Scale)", "Linear t (Shift)", "Quad t (Shift)"]
    
    features_dim = model.feat_dim
    
    fig, axes = plt.subplots(features_dim, 4, figsize=(20, 5 * features_dim), sharex=True)
    if features_dim == 1: axes = axes.reshape(1, -1)
    
    x_np = x_vals.cpu().numpy().flatten()
    
    for f_idx in range(features_dim):
        for comp_idx in range(4):
            ax = axes[f_idx, comp_idx]
            
            # Plot Flavour 0
            ax.plot(x_np, coeffs_0[:, f_idx, comp_idx].cpu().numpy(), label="Flav 0", color="C0")
            # Plot Flavour 1
            ax.plot(x_np, coeffs_1[:, f_idx, comp_idx].cpu().numpy(), label="Flav 1", color="C1", linestyle="--")
            
            ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
            
            if f_idx == 0:
                ax.set_title(comp_names[comp_idx])
            if comp_idx == 0:
                ax.set_ylabel(f"Feature {f_idx}")
            if f_idx == features_dim - 1:
                ax.set_xlabel("Context x")
                
            ax.legend()
            
    plt.suptitle(f"Residual Components vs Context - Nuisance {nuisance_idx} (Feature inputs fixed at {feature_val})", fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


def plot_residual_cross_components(model, pair_idx=0, x_range=(-4, 4), n_points=100, feature_val=0.0):
    """
    Plots the cross-term coefficients C_s and C_t for one selected nuisance pair
    as a function of the kinematic context variable 'x', for each flavour.

    The effective contribution in the model is:
        C_* * (m_i * m_j) * cross_term_damping
    where C_* is either C_s or C_t.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # scan context 'x'
    x_vals = torch.linspace(x_range[0], x_range[1], n_points, device=device).unsqueeze(1)

    # Prepare inputs for both flavours
    c0 = torch.tensor([1.0, 0.0], device=device).expand(n_points, -1)
    ctx_0 = torch.cat([c0, x_vals], dim=-1)

    c1 = torch.tensor([0.0, 1.0], device=device).expand(n_points, -1)
    ctx_1 = torch.cat([c1, x_vals], dim=-1)

    features_input = torch.full((n_points, model.feat_dim), feature_val, device=device)

    cross_coeffs_0, pair, damping = get_residual_cross_coeffs(
        model, features_input, ctx_0, pair_idx=pair_idx
    )
    cross_coeffs_1, _, _ = get_residual_cross_coeffs(
        model, features_input, ctx_1, pair_idx=pair_idx
    )

    comp_names = ["Cross C_s (Scale)", "Cross C_t (Shift)"]
    features_dim = model.feat_dim

    fig, axes = plt.subplots(features_dim, 2, figsize=(12, 5 * features_dim), sharex=True)
    if features_dim == 1:
        axes = axes.reshape(1, -1)

    x_np = x_vals.cpu().numpy().flatten()

    for f_idx in range(features_dim):
        for comp_idx in range(2):
            ax = axes[f_idx, comp_idx]

            ax.plot(x_np, cross_coeffs_0[:, f_idx, comp_idx].cpu().numpy(), label="Flav 0", color="C0")
            ax.plot(
                x_np,
                cross_coeffs_1[:, f_idx, comp_idx].cpu().numpy(),
                label="Flav 1",
                color="C1",
                linestyle="--",
            )

            ax.axhline(0, color="gray", lw=0.5, alpha=0.5)

            if f_idx == 0:
                ax.set_title(comp_names[comp_idx])
            if comp_idx == 0:
                ax.set_ylabel(f"Feature {f_idx}")
            if f_idx == features_dim - 1:
                ax.set_xlabel("Context x")

            ax.legend()

    pair_i = int(pair[0].item())
    pair_j = int(pair[1].item())
    plt.suptitle(
        (
            f"Residual Cross-Term Components vs Context - Pair ({pair_i}, {pair_j}) "
            f"[pair_idx={pair_idx}, damping={damping:.4g}] "
            f"(Feature inputs fixed at {feature_val})"
        ),
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()



def plot_transfer_difference(model, m1, m2, context=None, x_range=(-3, 3), y_range=(-3, 3), 
                                                                nbins=20, ax=None, title=None, colorbar=True, label="Delta Magnitude"):
    """
    Visualizes the vector field difference between the model's residual transformation 
    evaluated at two different nuisance points m1 and m2.
    
    Vector Field V(x) = ResidualStack(x, m1) - ResidualStack(x, m2)
    
    Args:
        model: SystematicCorrectedModel instance.
        m1: Tensor, first nuisance point (target).
        m2: Tensor, second nuisance point (reference).
        context: Tensor (optional), context for the transformation.
        x_range, y_range: Grid bounds.
        nbins: Grid resolution.
        ax: matplotlib axis.
        title: Plot title.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 2D Visualization only for now
    if model.feat_dim != 2:
        raise NotImplementedError("Only 2D features supported for vector field difference")

    # Grid setup
    x = torch.linspace(x_range[0], x_range[1], nbins, device=device)
    y = torch.linspace(y_range[0], y_range[1], nbins, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy') 
    inputs = torch.stack([xx.flatten(), yy.flatten()], dim=1) # [N*N, 2]
    batch_size = inputs.shape[0]

    # Prepare m1, m2
    def prepare_m(m):
        if not isinstance(m, torch.Tensor):
            m = torch.tensor(m, device=device, dtype=torch.float32)
        else:
            m = m.to(device)
        
        if m.ndim == 1: 
            return m.unsqueeze(0).expand(batch_size, -1)
        elif m.ndim == 2:
            if m.shape[0] == 1:
                return m.expand(batch_size, -1)
            elif m.shape[0] != batch_size:
                raise ValueError(f"m batch dimensions mismatch: {m.shape[0]} vs {batch_size}")
            return m
        return m

    m1_batch = prepare_m(m1)
    m2_batch = prepare_m(m2)

    # Context
    if context is None:
        context = torch.zeros(batch_size, model.ctx_dim, device=device)
    else:
        context = context.to(device)
        if context.ndim == 1: context = context.unsqueeze(0)
        if context.shape[0] != batch_size:
             context = context[0:1].expand(batch_size, -1)

    with torch.no_grad():
        out1 = get_residual_transform_output(model, inputs, context, m1_batch)
        out2 = get_residual_transform_output(model, inputs, context, m2_batch)

    # Difference
    diff = out1 - out2
    u = diff[:, 0].reshape(nbins, nbins).cpu().numpy()
    v = diff[:, 1].reshape(nbins, nbins).cpu().numpy()
    speed = np.sqrt(u**2 + v**2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    # Plot
    q = ax.quiver(xx.cpu().numpy(), yy.cpu().numpy(), u, v, speed, cmap='viridis', pivot='mid')
    
    if colorbar:
        try:
            plt.colorbar(q, ax=ax, label=label)
        except:
            pass

    if title:
        ax.set_title(title)
        
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, alpha=0.3)
    
    return ax
