# AutoModelR (Python Edition)

AutoModelR is a data analysis toolkit that automates several common
steps in regression-based mediation and moderation analysis.  It
originated as an R script, but this project re‑implements the
workflow entirely in Python and provides a graphical user interface
for end users as well as a one‑click packaging script for Windows.

## Features

* **Data preprocessing** – converts numeric‑like columns, fills
  missing values with means, performs z‑score standardisation and
  drops constant columns.
* **Automated predictor/response selection** – uses LASSO with
  cross‑validation to rank combinations of dependent and independent
  variables by in‑sample R² and displays the top N results.
* **Role identification** – classifies variables as
  mediators, moderators or neither by comparing p‑values of
  mediation and moderation paths.
* **Mediator ordering and moderator placement** – explores all
  permutations of mediators (when there are two or three) and
  possible moderator locations to find the highest scoring structure.
* **Model fitting** – fits a chain mediation model via ordinary
  least squares or, optionally, structural equation modelling if
  [`semopy`](https://github.com/semopy/semopy) is installed.  Results
  include parameter estimates and basic fit indices.
* **Graphical user interface** – built with PySimpleGUI,
  supporting file selection, progress indication, display of tables
  and model summaries, and exporting of results to CSV/JSON/TXT.
* **Packaging script for Windows** – a batch file (`build.bat`)
  that creates a virtual environment and compiles a standalone
  executable using PyInstaller.  The resulting `AutoModelR.exe`
  includes all dependencies so that end users do not need Python
  installed.

## Installation and Usage

The easiest way to run the application on Windows is to build a
standalone executable:

1. Clone or extract this repository to a directory without spaces or
   non‑ASCII characters (e.g. `E:\autoModelR_py`).
2. Open a Command Prompt in that directory and run:

   ```cmd
   build.bat
   ```

   This script creates a virtual environment, installs the required
   packages listed in `requirements.txt`, installs PyInstaller and
   compiles the application.  If successful, you will find
   `AutoModelR.exe` in the `dist` subdirectory.
3. Double‑click `AutoModelR.exe` to launch the GUI.  Choose an
   Excel/CSV data file, adjust the options if needed, and click
   “Start Analysis”.  Once completed, you can export the results via
   the “Export Results” button.

Alternatively, if you prefer to run the application without
packaging, you can do so on any platform by installing the
dependencies and running `app.py` directly:

```bash
python -m venv venv
source venv/bin/activate  # use venv\Scripts\activate.bat on Windows
pip install -r requirements.txt
python app.py
```

## Optional SEM Support

Structural equation modelling is disabled by default to keep the
dependency footprint small.  If you want to enable SEM support,
uncomment the `semopy` line in `requirements.txt`, install it, and
check the “Use SEM” box in the application.  Note that `semopy`
may require a compiler toolchain on Windows (e.g. Visual C++).

## Limitations and Notes

* The LASSO selection uses all numeric variables to estimate
  in‑sample R².  In small samples or highly collinear data, the
  results may be unstable.
* Role identification follows a simple heuristic based on p‑values.
  It does not test all possible combinations of IVs; only the first
  provided IV is used for mediation/moderation tests.
* When multiple mediators exist, the number of permutations grows
  quickly.  The search is reasonable for up to three mediators.  For
  larger numbers, consider specifying a fixed mediator order.
* If no mediators are found, the application reduces to a simple
  linear regression of the selected DV on the selected IV(s).

## License

This software is released under the MIT License.  See the LICENSE
file for details.
