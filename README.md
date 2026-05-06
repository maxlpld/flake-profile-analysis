# AFM Flake Profile Analysis

Small Python script to analyze AFM/Gwyddion line profiles across an island or flake on a surface.

The expected profile geometry is approximately:

```text
terrace -> left flake edge -> flake top -> right flake edge -> terrace
```

The script measures the apparent height separately at the left and right edges, analyzes stripe-like oscillations on the flake top, and saves publication-style plots using the University of Basel color palette.

## What the script creates

For each analysis, the script saves:

- a mean-profile plot showing all line profiles, the mean profile, the detected left/right edge markers, and the fitted left/right heights,
- either a separate apparent-height plot or the same height plot as an inset in the mean-profile plot,
- a stripe-spacing plot,
- a CSV summary file with the numerical values.

## Input data

Use one or two text files exported from a Gwyddion graph/profile window. The file should contain line profiles in pairs of columns, for example:

```text
x1, z1, x2, z2, x3, z3, ...
```

The standard Gwyddion graph export format is also supported. The script accepts distances and heights in `m`, `nm`, `µm`, or `Å` and converts everything to nm.

You can provide:

- one forward profile file only, or
- one forward and one backward profile file.

When both forward and backward files are provided, the script computes the left-edge and right-edge heights for each scan direction and averages the corresponding values for each profile index. The error bars in the height plot are the standard deviation between the available scan-direction values.

## Installation

```bash
pip install numpy matplotlib
```

The other modules used by the script are part of the Python standard library.

## Usage

Interactive mode:

```bash
python3 flake_analysis.py
```

Command-line mode with one file:

```bash
python3 flake_analysis.py --forward-data /path/to/forward_profiles.txt
```

Command-line mode with forward and backward files:

```bash
python3 flake_analysis.py \
  --forward-data /path/to/forward_profiles.txt \
  --backward-data /path/to/backward_profiles.txt
```

Put the apparent-height plot as an inset in the mean-profile plot:

```bash
python3 flake_analysis.py \
  --forward-data /path/to/forward_profiles.txt \
  --backward-data /path/to/backward_profiles.txt \
  --height-inset yes
```

Use a transparent background:

```bash
python3 flake_analysis.py --forward-data /path/to/forward_profiles.txt --transparent yes
```

Use external LaTeX rendering if LaTeX is installed:

```bash
python3 flake_analysis.py --forward-data /path/to/forward_profiles.txt --usetex
```

## How the edge heights are computed

The current height detection does **not** use a threshold-based plateau average. Instead, it follows the edge directly from the profile endpoint.

For the **left edge**:

1. Start from the first available point of the profile.
2. Move point by point to the right.
3. Track the last point where the height was still increasing.
4. When the next `EDGE_DECREASE_RUN` points decrease, the last increasing point is selected as the left-edge maximum.
5. The left height is then computed as

   ```text
   h_left = |z(left maximum) - z(first point)|
   ```

For the **right edge**:

1. Start from the last available point of the profile.
2. Move point by point to the left.
3. Apply the same rule in the reversed direction.
4. The selected maximum is marked with a triangle.
5. The right height is computed as

   ```text
   h_right = |z(right maximum) - z(last point)|
   ```

On the mean-profile plot, the left-edge maxima are shown with circle markers and the right-edge maxima with triangle markers. These are the same points used to compute the values shown in the apparent-height plot.

## Adjustable parameter

At the top of `flake_analysis.py`, the only user-adjustable height-detection parameter is:

```python
EDGE_DECREASE_RUN = 3
```

This controls how many consecutive decreasing points are required after the edge maximum before the code accepts that maximum.

- Increase it, for example to `4` or `5`, if the data are noisy and the edge is detected too early.
- Decrease it, for example to `2`, if the edge is very sharp or if the code misses the local edge maximum.

A value of `3` is a good default for moderately noisy AFM line profiles.

## Stripe-spacing analysis

The stripe-spacing plot is still computed from peaks detected on the flake top. The script first detects the flake region, subtracts a slow background from the top region, finds local maxima, and computes the peak-to-peak distances. Forward and backward spacings are shown with different markers to avoid confusion with the left/right height markers.

## Output files

For an input file named:

```text
my_profiles.txt
```

the script saves files such as:

```text
my_profiles_profiles_mean_height.png
my_profiles_left_right_heights.png
my_profiles_stripe_spacing.png
my_profiles_height_summary.csv
```

If both forward and backward files are provided, the output base name contains `_forward_backward`.

If `--height-inset yes` is used, the apparent-height plot is included inside the mean-profile plot and the separate `_left_right_heights.png` plot is not saved.

## Example output

Place example images in the `examples/` folder:

```text
examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_profiles_mean_height.png
examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_left_right_heights.png
examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_stripe_spacing.png
```

Mean profile with height inset:

![Mean profile and height inset](examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_profiles_mean_height.png)

Left and right apparent heights:

![Left and right edge heights](examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_left_right_heights.png)

Stripe-spacing analysis:

![Stripe-spacing analysis](examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_stripe_spacing.png)
