
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function for fitting"""
    x, y = xy
    x0, y0 = float(x0), float(y0)
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    g = offset + amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    return g.ravel()


def load_parquet(file_path):
    """Load a parquet file with optional row-group filters."""
    try:
        return ak.from_parquet(file_path)
    except FileNotFoundError:
        print(f"  Warning: File not found - {file_path}")
    except Exception as e:
        print(f"  Warning: Failed to load parquet file '{file_path}': {e}")
    return None


def fit_t5(file_path):
    """Fit a 2D Gaussian to T5 position distributions and return fit parameters."""
    T5_data = load_parquet(file_path)
    if T5_data is None:
        return None

    fields = ak.fields(T5_data)
    if 'T5_hit_pos_x' not in fields or 'T5_hit_pos_y' not in fields:
        print(f"  Warning: T5_hit_pos_x or T5_hit_pos_y columns not found in {file_path}")
        return None

    x_data = ak.to_numpy(ak.flatten(T5_data['T5_hit_pos_x'], axis=None))
    y_data = ak.to_numpy(ak.flatten(T5_data['T5_hit_pos_y'], axis=None))
    if len(x_data) == 0 or len(y_data) == 0:
        print(f"  Warning: No valid T5 position data in {file_path}")
        return None

    h, xedges, yedges = np.histogram2d(x_data, y_data, bins=[50, 8])
    X, Y = np.meshgrid(xedges[:-1] + np.diff(xedges) / 2, yedges[:-1] + np.diff(yedges) / 2)
    xy = np.vstack([X.ravel(), Y.ravel()])
    z = h.T.ravel()

    amplitude_guess = np.max(h)
    if np.isnan(amplitude_guess) or amplitude_guess <= 0:
        print(f"  Warning: Invalid histogram amplitude for {file_path}")
        return None

    x0_guess = np.mean(x_data)
    y0_guess = np.mean(y_data)
    sigma_x_guess = np.std(x_data)
    sigma_y_guess = np.std(y_data)
    p0 = [amplitude_guess, x0_guess, y0_guess, sigma_x_guess, sigma_y_guess, 0, 0]

    try:
        popt, _ = curve_fit(gaussian_2d, xy, z, p0=p0, maxfev=5000)
        amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt
        return {
            'amplitude': float(amplitude),
            'x0': float(x0),
            'y0': float(y0),
            'sigma_x': float(abs(sigma_x)),
            'sigma_y': float(abs(sigma_y)),
            'theta': float(theta),
            'offset': float(offset),
        }
    except Exception:
        return None


def fit_and_plot_t5(particle_name, file_path, run_number=None):
    """Load T5 data, create 2D histogram, and fit 2D Gaussian"""
    print(f"\nProcessing {particle_name}...")

    T5_data = load_parquet(file_path)
    if T5_data is None:
        return

    fields = ak.fields(T5_data)
    # Extract T5 position coordinates
    if 'T5_hit_pos_x' not in fields or 'T5_hit_pos_y' not in fields:
        print(f"  Warning: T5_hit_pos_x or T5_hit_pos_y columns not found")
        return
    
    print("Loading T5 position data...")
    
    x_data = ak.to_numpy(ak.flatten(T5_data['T5_hit_pos_x'], axis=None))
    y_data = ak.to_numpy(ak.flatten(T5_data['T5_hit_pos_y'], axis=None))
    
    print("loaded T5 position data")

    if len(x_data) == 0:
        print(f"  Warning: No valid T5 data")
        return
    
    print(f"  Loaded {len(x_data)} events")
    
    # Create 2D histogram
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot 1: 2D histogram
    pos_hist_bins = [50, 8]
    h, xedges, yedges = np.histogram2d(x_data, y_data, bins=pos_hist_bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax1.set_xlim(-120, 120)
    im1 = ax1.imshow(h.T, extent=extent, origin='lower', cmap='plasma', aspect='auto', norm='log')
    ax1.set_xlabel('T5 X Position (mm)')
    ax1.set_ylabel('T5 Y Position (mm)')
    ax1.set_title(f'{particle_name} - T5 Position Distribution')
    plt.colorbar(im1, ax=ax1, label='Count')
    
    # Prepare data for 2D Gaussian fit
    X, Y = np.meshgrid(xedges[:-1] + np.diff(xedges)/2, yedges[:-1] + np.diff(yedges)/2)
    xy = np.vstack([X.ravel(), Y.ravel()])
    z = h.T.ravel()
    
    # Initial guess for parameters
    amplitude_guess = np.max(h)
    x0_guess = np.average(x_data, weights=None)
    y0_guess = np.average(y_data, weights=None)
    sigma_x_guess = np.std(x_data)
    sigma_y_guess = np.std(y_data)
    
    p0 = [amplitude_guess, x0_guess, y0_guess, sigma_x_guess, sigma_y_guess, 0, 0]
    
    try:
        # Fit 2D Gaussian
        popt, pcov = curve_fit(gaussian_2d, xy, z, p0=p0, maxfev=5000)
        amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt

        # After popt is computed, inside the try block:
        amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt

        # Compute sigma contour levels
        sigma_levels = [amplitude * np.exp(-0.5 * n**2) for n in [3, 2, 1]]  # order low→high for contour

        z_fit = gaussian_2d(xy, *popt).reshape(X.shape)

        contours = ax1.contour(X, Y, z_fit, levels=sigma_levels,
                            colors='white', linewidths=1.5, linestyles=['dashed', 'dashdot', 'solid'])

        # Label the contour lines as 1σ, 2σ, 3σ
        contour_labels = {level: label for level, label in
                        zip(sigma_levels, ['3σ', '2σ', '1σ'])}
        ax1.clabel(contours, inline=True, fontsize=9,
                fmt=lambda x: contour_labels[x])

        # Add fit results as a text box in the plot
        fit_text = (
            f"Amplitude: {amplitude:.2f}\n"
            f"$x_0$: {x0:.2f} mm\n"
            f"$y_0$: {y0:.2f} mm\n"
            f"$\\sigma_x$: {abs(sigma_x):.2f} mm\n"
            f"$\\sigma_y$: {abs(sigma_y):.2f} mm\n"
            f"$\\theta$: {np.degrees(theta):.2f}°"
        )
        ax1.text(0.97, 0.97, fit_text,
                transform=ax1.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, edgecolor='white'),
                color='white')
        
        print(f"  Fit Results:")
        print(f"    Amplitude: {amplitude:.2f}")
        print(f"    Center: ({x0:.2f}, {y0:.2f})")
        print(f"    Sigma X: {sigma_x:.2f} mm")
        print(f"    Sigma Y: {sigma_y:.2f} mm")
        print(f"    Rotation: {np.degrees(theta):.2f}°")
                        
    except RuntimeError:
        print(f"  Warning: Gaussian fit failed")
    
    if run_number is None:
        run_number = "unknown"
    plot_folder = os.path.join(os.getcwd(), "plots", particle_name)
    os.makedirs(plot_folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f'Run_{run_number}_T5_{particle_name}_profile.png'), dpi=100, bbox_inches='tight')
    print(f"  Saved: Run_{run_number}_T5_{particle_name}_profile.png")


if __name__ == "__main__":
    
    # Define file paths and particle names
    run_number = 1610
    particles = {
        'electrons': f'run{run_number}_T5_electrons.parquet',
        'muons': f'run{run_number}_T5_muons.parquet',
        'pions': f'run{run_number}_T5_pions.parquet',
        'protons': f'run{run_number}_T5_protons.parquet'
    }

    # Process each particle type
    for particle_name, file_name in particles.items():
        fit_and_plot_t5(particle_name, file_name, run_number)

    print("\nAnalysis complete!")
