# AFM Flake Profile Analysis

Small Python script to analyze line profiles exported from Gwyddion for an island/flake measured by AFM.

The expected profile geometry is:

```text
Ag terrace -> island edge -> flake top -> island edge -> Ag terrace
```

The script automatically detects the two island edges, estimates the apparent flake height from the left and right sides of each profile, and analyzes stripe-like oscillations on top of the flake.

## What the script does

It creates:

- a plot of all line profiles plus the mean profile,
- a plot of the left-edge and right-edge apparent heights for each profile,
- a plot of the stripe spacing measured on top of the flake, including a linear fit,
- a CSV summary file with the numerical results.

The plots use the University of Basel color palette.

## Input data

Provide a text/CSV file exported from the Gwyddion profile graph.

The file should contain several line profiles, ideally 5 or more, exported with columns such as:

```text
x1, z1, x2, z2, x3, z3, ...
```

or the standard Gwyddion graph export format. Distances and heights can be in `m`, `nm`, `µm`, or `Å`; the script converts them to nm.

Each profile should cross the same type of feature in this order:

```text
bare terrace -> first island edge -> flake top -> second island edge -> bare terrace
```

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

- the path to the Gwyddion profile data file,
- whether to use a transparent background,
- whether to add titles to the plots,
- whether to show profile legends for each plot.

Command-line option for the data path:

```bash
python flake_analysis.py --data /path/to/profile_data.txt
```

Transparent background:

```bash
python flake_analysis.py --data /path/to/profile_data.txt --transparent yes
```

Use external LaTeX rendering if LaTeX is installed:

```bash
python flake_analysis.py --data /path/to/profile_data.txt --usetex
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

in the same folder as the input data file.

## Notes

The height obtained from AFM line profiles is an apparent height. For constant-frequency-shift AFM measurements, it should be interpreted carefully and compared with literature values only as a compatibility check.
