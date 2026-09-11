"""Shared presentation helpers without global Matplotlib configuration changes."""


def label_axes(ax, *, xlabel, ylabel, title):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def add_dual_unit_axes(ax, *, lambda_scale, w_0, label_lower=r'\lambda', label_upper=r'w_0'):
    """Add secondary top and right axes for dual unit display (lower/left: lambda, upper/right: w_0)."""
    if lambda_scale is not None and w_0 is not None:
        try:
            lam_val = float(lambda_scale)
            w0_val = float(w_0)
            if lam_val > 0 and w0_val > 0:
                ratio = lam_val / w0_val
                secax_x = ax.secondary_xaxis('top', functions=(lambda v: v * ratio, lambda v: v / ratio))
                secax_x.set_xlabel(rf'$x/{label_upper}$')
                secax_y = ax.secondary_yaxis('right', functions=(lambda v: v * ratio, lambda v: v / ratio))
                secax_y.set_ylabel(rf'$y/{label_upper}$')
                return secax_x, secax_y
        except (ValueError, TypeError):
            pass
    return None, None


