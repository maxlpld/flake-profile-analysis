# AFM Flake Profile Analysis

Small Python script to analyze AFM line profiles exported from Gwyddion for an island or flake on a surface.

The expected profile geometry is:

```text
terrace -> left island edge -> flake top -> right island edge -> terrace
```

The script detects the two island edges, estimates the apparent flake height from the left and right sides of each profile, and analyzes stripe-like oscillations on top of the flake. It can use one profile file only, or a pair of forward/backward scan files.

## What the script does

It creates:

- a plot of all line profiles plus the mean profile,
- a plot of the left-edge and right-edge apparent heights for each profile, including a linear fit,
- a plot of the stripe spacing measured on top of the flake, including a linear fit,
- a CSV summary file with the numerical results.

The plots use the University of Basel color palette.

## Input data

Provide one or two text/CSV files exported from the Gwyddion profile graph.

The file should contain several line profiles, ideally 5 or more, exported with columns such as:

```text
x1, z1, x2, z2, x3, z3, ...
```

or the standard Gwyddion graph export format. Distances and heights can be in `m`, `nm`, `µm`, or `Å`; the script converts them to nm.

Each profile should cross the same type of feature in this order:

```text
bare terrace -> first island edge -> flake top -> second island edge -> bare terrace
```

When both forward and backward files are provided, the script calculates the left-edge height from both scan directions and averages them. The same is done for the right-edge height. This is useful when one scan direction gives a cleaner edge response than the other.

## Installation

```bash
pip install numpy matplotlib
```

The other imported modules are part of the Python standard library.

## Usage

Interactive mode:

```bash
python flake_analysis.py
```

The script will ask for:

- the path to the forward Gwyddion profile data file,
- optionally, the path to the backward Gwyddion profile data file,
- whether to use a transparent background,
- whether to add titles to the plots,
- whether to show profile legends for each plot.

Command-line mode with one file:

```bash
python flake_analysis.py --forward-data /path/to/forward_profiles.txt
```

or, using the backward-compatible alias:

```bash
python flake_analysis.py --data /path/to/forward_profiles.txt
```

Command-line mode with forward and backward files:

```bash
python flake_analysis.py \
  --forward-data /path/to/forward_profiles.txt \
  --backward-data /path/to/backward_profiles.txt
```

Transparent background:

```bash
python flake_analysis.py --forward-data /path/to/forward_profiles.txt --transparent yes
```

Use external LaTeX rendering if LaTeX is installed:

```bash
python flake_analysis.py --forward-data /path/to/forward_profiles.txt --usetex
```

## Output files

For an input file named:

```text
my_profiles.txt
```

the script saves:

```text
my_profiles_profiles_mean_height.png
my_profiles_left_right_heights.png
my_profiles_stripe_spacing.png
my_profiles_height_summary.csv
```

If both forward and backward files are provided, the output base name contains `_forward_backward`.

## Height and error calculation

For each line profile, the apparent left-edge height is calculated as the difference between the mean height of the flake plateau and the mean height of the terrace near the left edge:

$$
h_i^\mathrm{L} = \langle z_\mathrm{flake,L} \rangle - \langle z_\mathrm{terrace,L} \rangle .
$$

The right-edge height is calculated analogously:

$$
h_i^\mathrm{R} = \langle z_\mathrm{flake,R} \rangle - \langle z_\mathrm{terrace,R} \rangle .
$$

If both forward and backward scans are provided, the displayed value for each edge is the mean of the available scan-direction values:

$$
\bar{h}_i = \frac{1}{n_i}\sum_{k=1}^{n_i} h_{i,k} .
$$

The error bar is the sample standard deviation:

$$
s_i = \sqrt{\frac{1}{n_i-1}\sum_{k=1}^{n_i}\left(h_{i,k}-\bar{h}_i\right)^2} .
$$

If only one value is available, the error bar is set to zero. The same convention is used for stripe-spacing error bars, where the values are the peak-to-peak distances detected on the flake top.

The linear fits in the height and stripe-spacing plots use:

$$
y(i)=a i+b,
$$

where `i` is the profile index. The uncertainties displayed for `a` and `b` are the standard errors obtained from the least-squares covariance matrix.

## Example output

Place example images in an `examples/` folder and reference them from the README like this:

```text
examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_profiles_mean_height.png
examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_left_right_heights.png
examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_stripe_spacing.png
```

Mean profile and apparent height:

![Mean profile and apparent height](examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_profiles_mean_height.png)

Left-edge and right-edge apparent heights:

![Left and right edge heights](examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_left_right_heights.png)

Stripe-spacing analysis:

![Stripe-spacing analysis](examples/20260428_MOS2-Ag(111)0005_data_height_forward_backward_stripe_spacing.png)
