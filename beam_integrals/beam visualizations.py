import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
from sympy import Float, Abs
import mpmath
import time
import os
from pathlib import Path

# This script assumes it's placed in the same directory as the beam_integrals package
# or that the package is installed in your Python environment

try:
    from beam_integrals import beam_types
    from beam_integrals.characteristic_equation_solvers import BaseRootfinder, find_best_root
except ImportError:
    print("Warning: beam_integrals package not found. Running in demonstration mode.")
    # Create mock classes for demonstration
    class BaseRootfinder:
        plugins = {"anderson": "AndersonRootfinder", "illinois": "IllinoisRootfinder", 
                  "pegasus": "PegasusRootfinder", "secant": "SecantRootfinder"}
        
    class beam_types:
        class BaseBeamType:
            @staticmethod
            def get_all_beam_types():
                return ["Clamped-Free", "Clamped-Clamped", "Free-Free", "Clamped-Pinned", "Pinned-Pinned"]

# Create output directory for saving plots
output_dir = Path("beam_visualizations")
output_dir.mkdir(exist_ok=True)

# Color scheme
COLORS = plt.cm.viridis(np.linspace(0, 1, 10))
BEAM_COLORS = {
    "Clamped-Free": COLORS[0],
    "Clamped-Clamped": COLORS[2],
    "Free-Free": COLORS[4],
    "Clamped-Pinned": COLORS[6],
    "Pinned-Pinned": COLORS[8]
}

def characteristic_function(beam_type, mu):
    """
    Evaluate the characteristic function for a given beam type and mu value.
    This is a simplified version for demonstration.
    """
    if beam_type == "Clamped-Free":
        # cos(mu)*cosh(mu) + 1 = 0
        return np.cos(mu) * np.cosh(mu) + 1
    elif beam_type == "Clamped-Clamped":
        # cos(mu)*cosh(mu) - 1 = 0
        return np.cos(mu) * np.cosh(mu) - 1
    elif beam_type == "Free-Free":
        # cos(mu)*cosh(mu) - 1 = 0 (same as clamped-clamped)
        return np.cos(mu) * np.cosh(mu) - 1
    elif beam_type == "Clamped-Pinned":
        # tan(mu) - tanh(mu) = 0
        return np.tan(mu) - np.tanh(mu)
    elif beam_type == "Pinned-Pinned":
        # sin(mu) = 0
        return np.sin(mu)
    else:
        raise ValueError(f"Unknown beam type: {beam_type}")

def plot_characteristic_function(beam_type, mu_range=(0, 15), num_points=1000, show_roots=True):
    """
    Plot the characteristic function for a given beam type.
    
    Parameters:
    -----------
    beam_type : str
        The type of beam to plot
    mu_range : tuple
        The range of mu values to plot
    num_points : int
        Number of points to evaluate
    show_roots : bool
        Whether to mark the roots on the plot
    """
    mu_values = np.linspace(mu_range[0], mu_range[1], num_points)
    
    # Skip values that would cause numerical issues
    valid_indices = []
    for i, mu in enumerate(mu_values):
        if beam_type == "Clamped-Pinned" and abs(np.cos(mu)) < 1e-10:
            continue
        valid_indices.append(i)
    
    valid_mu = mu_values[valid_indices]
    
    # Calculate characteristic function values
    try:
        char_values = np.array([characteristic_function(beam_type, mu) for mu in valid_mu])
    except:
        # Handle any numerical errors
        print(f"Error calculating characteristic function for {beam_type}")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(valid_mu, char_values, label=f"{beam_type} Characteristic Function", 
             color=BEAM_COLORS.get(beam_type, 'blue'), linewidth=2)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Mark the roots if requested
    if show_roots:
        # Find approximate roots
        roots = []
        for i in range(1, len(valid_mu)-1):
            if char_values[i-1] * char_values[i+1] <= 0:
                # There's a root in this interval
                roots.append(valid_mu[i])
        
        # Plot the roots
        for i, root in enumerate(roots):
            if i < 10:  # Limit to first 10 roots for clarity
                plt.scatter([root], [0], color='red', s=50, zorder=5)
                plt.annotate(f"μ{i+1}={root:.2f}", 
                             xy=(root, 0), 
                             xytext=(root, 0.2 if i % 2 == 0 else -0.2),
                             textcoords='data',
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Set plot labels and title
    plt.title(f"Characteristic Function for {beam_type} Beam", fontsize=16)
    plt.xlabel("μ", fontsize=14)
    plt.ylabel("f(μ)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / f"{beam_type.replace('-', '_')}_characteristic_function.png", dpi=300)
    plt.close()
    
    return roots

def beam_mode_shape(beam_type, mu, x):
    """
    Calculate the mode shape for a given beam type, mu value, and position x.
    This is a simplified version for demonstration.
    
    Parameters:
    -----------
    beam_type : str
        The type of beam
    mu : float
        The mu value (root of characteristic equation)
    x : array
        Normalized positions along the beam (0 to 1)
    
    Returns:
    --------
    array : The mode shape values
    """
    if beam_type == "Clamped-Free":
        # For a cantilever beam (clamped-free)
        A = np.cosh(mu * x) - np.cos(mu * x)
        B = (np.cosh(mu) + np.cos(mu)) / (np.sinh(mu) + np.sin(mu))
        C = np.sinh(mu * x) - np.sin(mu * x)
        return A - B * C
    
    elif beam_type == "Clamped-Clamped":
        # For a clamped-clamped beam
        return (np.cosh(mu * x) - np.cos(mu * x)) - \
               ((np.cosh(mu) - np.cos(mu)) / (np.sinh(mu) - np.sin(mu))) * \
               (np.sinh(mu * x) - np.sin(mu * x))
    
    elif beam_type == "Pinned-Pinned":
        # For a simply supported beam (pinned-pinned)
        return np.sin(mu * x)
    
    elif beam_type == "Free-Free":
        # For a free-free beam
        A = np.cosh(mu * x) + np.cos(mu * x)
        B = (np.cosh(mu) - np.cos(mu)) / (np.sinh(mu) - np.sin(mu))
        C = np.sinh(mu * x) + np.sin(mu * x)
        return A - B * C
    
    elif beam_type == "Clamped-Pinned":
        # For a clamped-pinned beam
        return (np.cosh(mu * x) - np.cos(mu * x)) - \
               ((np.cosh(mu) - np.cos(mu)) / (np.sinh(mu) - np.sin(mu))) * \
               (np.sinh(mu * x) - np.sin(mu * x))
    
    else:
        raise ValueError(f"Unknown beam type: {beam_type}")

def plot_mode_shapes(beam_type, roots=None, num_modes=4):
    """
    Plot the mode shapes for a given beam type.
    
    Parameters:
    -----------
    beam_type : str
        The type of beam
    roots : list
        List of mu values (roots of characteristic equation)
    num_modes : int
        Number of modes to plot
    """
    if roots is None or len(roots) < num_modes:
        # If roots aren't provided or not enough, calculate them
        mu_range = (0, 20)  # Wider range to ensure we find enough roots
        roots = plot_characteristic_function(beam_type, mu_range, show_roots=False)
        
        if roots is None or len(roots) < num_modes:
            print(f"Could not find enough roots for {beam_type}")
            return
    
    # Create x values along the beam
    x = np.linspace(0, 1, 1000)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each mode shape
    for i in range(min(num_modes, len(roots))):
        mu = roots[i]
        mode = beam_mode_shape(beam_type, mu, x)
        
        # Normalize the mode shape
        mode = mode / np.max(np.abs(mode))
        
        # Plot with offset for clarity
        offset = -i * 2.5
        plt.plot(x, mode + offset, label=f"Mode {i+1} (μ={mu:.2f})", 
                 color=COLORS[i % len(COLORS)], linewidth=2)
        
        # Add horizontal line at the offset
        plt.axhline(y=offset, color='gray', linestyle='--', alpha=0.3)
    
    # Add beam support indicators
    support_indicators(beam_type)
    
    # Set plot labels and title
    plt.title(f"Mode Shapes for {beam_type} Beam", fontsize=16)
    plt.xlabel("Normalized Position (x/L)", fontsize=14)
    plt.ylabel("Normalized Amplitude", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Remove y-axis ticks as they're not meaningful with the offsets
    plt.yticks([])
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / f"{beam_type.replace('-', '_')}_mode_shapes.png", dpi=300)
    plt.close()

def support_indicators(beam_type):
    """Add visual indicators for beam supports"""
    if "Clamped" in beam_type.split("-")[0]:
        # Left end clamped
        plt.plot([0, 0], [-10, 0], 'k-', linewidth=6)
    elif "Pinned" in beam_type.split("-")[0]:
        # Left end pinned
        plt.plot([0], [0], 'ko', markersize=10)
        plt.plot([0, 0], [-0.5, 0], 'k-', linewidth=3)
    
    if "Clamped" in beam_type.split("-")[1]:
        # Right end clamped
        plt.plot([1, 1], [-10, 0], 'k-', linewidth=6)
    elif "Pinned" in beam_type.split("-")[1]:
        # Right end pinned
        plt.plot([1], [0], 'ko', markersize=10)
        plt.plot([1, 1], [-0.5, 0], 'k-', linewidth=3)

def animate_mode_shape(beam_type, mode_number=1):
    """
    Create an animation of a vibrating beam for a specific mode.
    
    Parameters:
    -----------
    beam_type : str
        The type of beam
    mode_number : int
        The mode number to animate (1-based)
    """
    # Calculate roots
    roots = plot_characteristic_function(beam_type, (0, 20), show_roots=False)
    
    if roots is None or len(roots) < mode_number:
        print(f"Could not find enough roots for {beam_type}")
        return
    
    mu = roots[mode_number - 1]
    x = np.linspace(0, 1, 100)
    mode_shape = beam_mode_shape(beam_type, mu, x)
    
    # Normalize the mode shape
    mode_shape = mode_shape / np.max(np.abs(mode_shape))
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x, mode_shape, 'b-', linewidth=2)
    
    # Add beam support indicators
    if "Clamped" in beam_type.split("-")[0]:
        ax.plot([0, 0], [-1.5, 1.5], 'k-', linewidth=6)
    elif "Pinned" in beam_type.split("-")[0]:
        ax.plot([0], [0], 'ko', markersize=10)
    
    if "Clamped" in beam_type.split("-")[1]:
        ax.plot([1, 1], [-1.5, 1.5], 'k-', linewidth=6)
    elif "Pinned" in beam_type.split("-")[1]:
        ax.plot([1], [0], 'ko', markersize=10)
    
    # Set plot properties
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"{beam_type} Beam - Mode {mode_number} (μ={mu:.2f})", fontsize=16)
    ax.set_xlabel("Normalized Position (x/L)", fontsize=14)
    ax.set_ylabel("Normalized Amplitude", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Animation function
    def animate(i):
        t = i / 50 * 2 * np.pi
        line.set_ydata(mode_shape * np.cos(t))
        return line,
    
    # Create the animation
    anim = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
    
    # Save the animation
    anim.save(output_dir / f"{beam_type.replace('-', '_')}_mode_{mode_number}_animation.gif", 
              writer='pillow', fps=20, dpi=100)
    plt.close()

def compare_rootfinders(beam_type="Clamped-Free", mode=3, decimal_precision=30):
    """
    Compare different root-finding algorithms for a specific beam type and mode.
    This is a simplified version that simulates the behavior of the actual algorithms.
    
    Parameters:
    -----------
    beam_type : str
        The type of beam
    mode : int
        The mode number (1-based)
    decimal_precision : int
        Decimal precision for calculations
    """
    # Define the rootfinders to compare
    rootfinders = {
        "Anderson": {"color": "blue", "marker": "o"},
        "Illinois": {"color": "green", "marker": "s"},
        "Pegasus": {"color": "red", "marker": "^"},
        "Secant": {"color": "purple", "marker": "d"}
    }
    
    # Get an approximate root value for the specified mode
    roots = plot_characteristic_function(beam_type, (0, 20), show_roots=False)
    if roots is None or len(roots) < mode:
        print(f"Could not find enough roots for {beam_type}")
        return
    
    target_root = roots[mode - 1]
    
    # Simulate convergence data for each rootfinder
    # In reality, this would come from the actual algorithms
    convergence_data = {}
    
    for name in rootfinders:
        # Simulate different convergence rates
        if name == "Anderson":
            iterations = 8
            rate = 1.6
        elif name == "Illinois":
            iterations = 10
            rate = 1.4
        elif name == "Pegasus":
            iterations = 7
            rate = 1.7
        else:  # Secant
            iterations = 12
            rate = 1.3
        
        # Generate simulated error values
        errors = []
        x_values = []
        
        # Start with a poor initial guess
        current_x = target_root * 0.8
        
        for i in range(iterations):
            # Calculate error (distance from actual root)
            error = abs(characteristic_function(beam_type, current_x))
            errors.append(error)
            x_values.append(current_x)
            
            # Improve the guess (simulate convergence)
            current_x = target_root - (target_root - current_x) / rate
        
        convergence_data[name] = {
            "errors": errors,
            "x_values": x_values,
            "iterations": iterations
        }
    
    # Create the convergence plot
    plt.figure(figsize=(12, 8))
    
    # Plot error vs iteration for each method
    for name, data in convergence_data.items():
        plt.semilogy(range(1, data["iterations"] + 1), data["errors"], 
                    label=f"{name} ({data['iterations']} iterations)",
                    color=rootfinders[name]["color"], 
                    marker=rootfinders[name]["marker"],
                    linewidth=2, markersize=8)
    
    # Set plot labels and title
    plt.title(f"Root-Finding Convergence Comparison for {beam_type} Beam (Mode {mode})", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Error (log scale)", fontsize=14)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / f"{beam_type.replace('-', '_')}_rootfinder_comparison_mode_{mode}.png", dpi=300)
    plt.close()
    
    # Create the root approximation plot
    plt.figure(figsize=(12, 8))
    
    # Plot x values vs iteration for each method
    for name, data in convergence_data.items():
        plt.plot(range(1, data["iterations"] + 1), data["x_values"], 
                label=f"{name}",
                color=rootfinders[name]["color"], 
                marker=rootfinders[name]["marker"],
                linewidth=2, markersize=8)
    
    # Add horizontal line for the actual root
    plt.axhline(y=target_root, color='black', linestyle='--', 
                label=f"Actual Root (μ={target_root:.6f})")
    
    # Set plot labels and title
    plt.title(f"Root Approximation Convergence for {beam_type} Beam (Mode {mode})", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("μ Value", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / f"{beam_type.replace('-', '_')}_root_approximation_mode_{mode}.png", dpi=300)
    plt.close()

def plot_all_beam_types_comparison():
    """
    Create a comparison plot of characteristic functions for all beam types.
    """
    beam_types_list = ["Clamped-Free", "Clamped-Clamped", "Free-Free", "Clamped-Pinned", "Pinned-Pinned"]
    mu_values = np.linspace(0, 10, 1000)
    
    plt.figure(figsize=(12, 8))
    
    for beam_type in beam_types_list:
        # Calculate characteristic function values
        try:
            # Skip values that would cause numerical issues
            valid_indices = []
            for i, mu in enumerate(mu_values):
                if beam_type == "Clamped-Pinned" and abs(np.cos(mu)) < 1e-10:
                    continue
                valid_indices.append(i)
            
            valid_mu = mu_values[valid_indices]
            char_values = np.array([characteristic_function(beam_type, mu) for mu in valid_mu])
            
            # Plot the characteristic function
            plt.plot(valid_mu, char_values, label=beam_type, 
                     color=BEAM_COLORS.get(beam_type, 'blue'), linewidth=2)
        except:
            print(f"Error calculating characteristic function for {beam_type}")
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Set plot labels and title
    plt.title("Comparison of Characteristic Functions for Different Beam Types", fontsize=16)
    plt.xlabel("μ", fontsize=14)
    plt.ylabel("f(μ)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / "beam_types_comparison.png", dpi=300)
    plt.close()

def create_natural_frequency_table():
    """
    Create a table of natural frequencies for different beam types.
    """
    beam_types_list = ["Clamped-Free", "Clamped-Clamped", "Free-Free", "Clamped-Pinned", "Pinned-Pinned"]
    
    # Dictionary to store the roots for each beam type
    all_roots = {}
    
    # Calculate roots for each beam type
    for beam_type in beam_types_list:
        roots = plot_characteristic_function(beam_type, (0, 20), show_roots=False)
        if roots is not None:
            all_roots[beam_type] = roots[:5]  # Store first 5 roots
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for beam_type in beam_types_list:
        if beam_type in all_roots:
            # Calculate frequency factors (mu^2)
            freq_factors = [round(mu**2, 2) for mu in all_roots[beam_type][:5]]
            table_data.append([beam_type] + freq_factors)
    
    # Create the table
    table = ax.table(cellText=table_data,
                    colLabels=["Beam Type", "Mode 1", "Mode 2", "Mode 3", "Mode 4", "Mode 5"],
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Add a title
    plt.title("Natural Frequency Factors (μ²) for Different Beam Types", fontsize=16, pad=20)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "natural_frequency_table.png", dpi=300)
    plt.close()

def main():
    """
    Main function to generate all visualizations.
    """
    print("Generating beam vibration visualizations...")
    print(f"Output directory: {output_dir}")
    
    # Get list of beam types
    try:
        beam_types_list = beam_types.BaseBeamType.get_all_beam_types()
    except:
        beam_types_list = ["Clamped-Free", "Clamped-Clamped", "Free-Free", "Clamped-Pinned", "Pinned-Pinned"]
    
    # Generate characteristic function plots for each beam type
    print("Generating characteristic function plots...")
    for beam_type in beam_types_list:
        print(f"  Processing {beam_type}...")
        roots = plot_characteristic_function(beam_type)
        
        # Generate mode shape plots
        print(f"  Generating mode shapes for {beam_type}...")
        plot_mode_shapes(beam_type, roots)
        
        # Generate animation for first mode
        print(f"  Generating animation for {beam_type} mode 1...")
        animate_mode_shape(beam_type, 1)
    
    # Compare root-finding algorithms
    print("Comparing root-finding algorithms...")
    compare_rootfinders("Clamped-Free", 3)
    
    # Create comparison of all beam types
    print("Creating beam types comparison...")
    plot_all_beam_types_comparison()
    
    # Create natural frequency table
    print("Creating natural frequency table...")
    create_natural_frequency_table()
    
    print(f"All visualizations have been saved to {output_dir}")

if __name__ == "__main__":
    main()
