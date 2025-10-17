"""
AutoModelR Python GUI Application
================================

This script provides a desktop graphical user interface for the
AutoModelR tool using the PySimpleGUI library. Users can load an
Excel or CSV dataset, specify how many top combinations of response
and predictor variables to explore (Top N), optionally override the
automatically selected independent and dependent variables, and
choose whether to fit a structural equation model (SEM) if the
`semopy` package is available. The application then walks through
data preprocessing, variable selection via LASSO, role identification
for mediator and moderator variables, optimisation of mediator
ordering and moderator placement, and path model fitting. Progress
is shown via a progress bar and log messages, and results can be
exported to a chosen directory.

Usage
-----
Run this script with Python 3.8+ after installing the required
packages from ``requirements.txt``. To build a standalone Windows
executable, see ``build.bat``.

Author: ChatGPT (OpenAI)
"""

import json
import os
import threading
import traceback
from typing import List, Optional

import pandas as pd
import PySimpleGUI as sg

# ===== Numeric and SEM imports =====
import numpy as np
try:
    import semopy
except ImportError:
    semopy = None

# ===== Window state and DPI awareness helper functions =====
# Enable DPI awareness to prevent oversized windows on high DPI displays
sg.set_options(dpi_awareness=True)

# Settings file for storing window position and size
SETTINGS_FILE = 'user_window_pos.json'

def load_window_pos():
    """Load the last window position and size from disk."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_window_pos(win):
    """Save the current window position and size to disk."""
    try:
        x, y = win.current_location()
        w, h = win.size
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'x': x, 'y': y, 'w': w, 'h': h}, f)
    except Exception:
        pass

def create_window(app_name, layout, default_size=(1100, 720), default_loc=(80, 60)):
    """
    Create a PySimpleGUI window using saved size/position if available.

    Parameters
    ----------
    app_name : str
        The window title.
    layout : list
        The window layout.
    default_size : tuple
        Fallback width and height if no saved values are found.
    default_loc : tuple
        Fallback x/y position if no saved values are found.

    Returns
    -------
    sg.Window
        The constructed PySimpleGUI window.
    """
    pos = load_window_pos() or {}
    size = (pos.get('w', default_size[0]), pos.get('h', default_size[1]))
    loc  = (pos.get('x', default_loc[0]),  pos.get('y', default_loc[1]))
    win = sg.Window(app_name, layout, size=size, resizable=True, location=loc, finalize=True)
    try:
        win.move(*loc)
    except Exception:
        pass
    return win

from core import (
    auto_preprocess,
    auto_find_main_xy_topn,
    auto_identify_roles,
    auto_mediator_order,
    auto_moderator_position,
    auto_path_model_fit,
)


class AutoModelRApp:
    """Encapsulates the GUI and workflow for the AutoModelR application."""

    def __init__(self):
        self.topn_df: Optional[pd.DataFrame] = None
        self.roles_df: Optional[pd.DataFrame] = None
        self.mediator_order: List[str] = []
        self.mediator_order_score: float = float('-inf')
        self.mod_result: Optional[tuple] = None
        self.model_summary: str = ''
        self.model_obj: Optional[object] = None
        self.data_clean: Optional[pd.DataFrame] = None
        # Build GUI layout
        sg.theme('SystemDefault')
        # Build the layout with a scrollable column for result frames
        # Header: file selection
        header_row = [
            sg.Text('Data file:'),
            sg.InputText(key='-FILE-', enable_events=True, visible=False),
            sg.FileBrowse('Browse', key='-BROWSE-', file_types=(('Excel and CSV Files', '*.xlsx;*.xls;*.csv'),), target='-FILE-'),
            sg.Text('', key='-FILEPATH-')
        ]

        # Parameters: TopN, manual IVs/DV, SEM option
        param_row = [
            sg.Text('Top N combinations:'),
            sg.Spin([1, 2, 3, 5, 10, 20], initial_value=3, key='-TOPN-'),
            sg.Text('Manual IVs (comma separated):'), sg.InputText(key='-IVS-', size=(30, 1)),
            sg.Text('Manual DV:'), sg.InputText(key='-DV-', size=(15, 1)),
            sg.Checkbox('Use SEM (requires semopy)', key='-SEM-')
        ]

        # Control row: start button and progress bar
        control_row = [sg.Button('Start Analysis', key='-START-'), sg.ProgressBar(100, orientation='h', size=(30, 20), key='-PROG-')]

        # Log row
        log_row = [sg.Text('Log:'), sg.Multiline('', size=(80, 10), key='-LOG-', autoscroll=True, disabled=True)]

        # Scrollable column with the frames for results
        result_column = sg.Column([
            [sg.Frame('Top N combinations', [[sg.Table(values=[], headings=['y', 'x', 'r2'], key='-TOPTABLE-', auto_size_columns=True, display_row_numbers=False, num_rows=5)]])],
            [sg.Frame('Role identification', [[sg.Table(values=[], headings=['var', 'type', 'p_mediation', 'p_moderation'], key='-ROLETABLE-', auto_size_columns=True, display_row_numbers=False, num_rows=10)]])],
            [sg.Frame('Mediator order & Moderator position', [[sg.Multiline('', size=(80, 5), key='-MEDMOD-', disabled=True)]])],
            [sg.Frame('Model summary', [[sg.Multiline('', size=(80, 15), key='-SUMMARY-', disabled=True)]])]
        ], scrollable=True, vertical_scroll_only=True, size=(None, 330), key='-SCROLL-')

        # Footer row: export and exit buttons
        footer_row = [sg.Button('Export Results', key='-EXPORT-', disabled=True), sg.Button('Exit')]

        self.layout = [
            header_row,
            param_row,
            control_row,
            log_row,
            [result_column],
            footer_row
        ]
        # Create window using helper defined at top of this file.  It restores the last
        # window size and position and prevents oversized windows.
        self.window = create_window('AutoModelR (Python)', self.layout)

    def log(self, message: str):
        """Append a message to the log window."""
        current = self.window['-LOG-'].get()
        self.window['-LOG-'].update(current + message + '\n')

    def update_progress(self, percent: int):
        """Update the progress bar."""
        self.window['-PROG-'].update_bar(percent)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from an Excel or CSV file."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif ext in ['.csv']:
            df = pd.read_csv(filepath)
        else:
            raise ValueError('Unsupported file format. Please use Excel or CSV.')
        return df

    def run(self):
        """Main event loop."""
        while True:
            event, values = self.window.read(timeout=100)
            # Handle window close or exit event
            if event in (sg.WINDOW_CLOSED, 'Exit'):
                # Persist the window position and size before closing
                save_window_pos(self.window)
                break
            # Start analysis in a background thread to keep UI responsive
            if event == '-START-':
                threading.Thread(target=self._run_analysis, args=(values,), daemon=True).start()
            # Export results button
            if event == '-EXPORT-':
                self._export_results()
            # Update file path display when file is chosen
            if event == '-FILE-':
                path = values['-FILE-']
                self.window['-FILEPATH-'].update(path)
        # Close the window after loop ends
        self.window.close()

    def _run_analysis(self, values):
        """Perform the full analysis pipeline.

        This runs in a background thread; UI updates must be called via
        the main thread (which PySimpleGUI handles for us). Exceptions
        are caught and logged.
        """
        try:
            filepath = values['-FILE-']
            if not filepath:
                self.log('Please select a data file.')
                return
            self.update_progress(0)
            self.log(f'Loading data from {filepath}')
            df_raw = self.load_data(filepath)
            self.update_progress(5)
            self.log('Preprocessing data...')
            df_clean = auto_preprocess(df_raw)
            self.data_clean = df_clean
            self.update_progress(20)
            self.log('Selecting top combinations via LASSO...')
            topn = int(values['-TOPN-'])
            top_df = auto_find_main_xy_topn(df_clean.select_dtypes(include=[np.number]), topn)
            self.topn_df = top_df
            self.window['-TOPTABLE-'].update(values=top_df.values.tolist())
            self.update_progress(40)
            # Determine IVs and DV
            manual_ivs = [c.strip() for c in values['-IVS-'].split(',') if c.strip()]
            manual_dv = values['-DV-'].strip()
            if manual_ivs and manual_dv:
                IVs = [iv for iv in manual_ivs if iv in df_clean.columns]
                DV = manual_dv
                self.log(f'Using manual IVs: {IVs} and DV: {DV}')
            else:
                if top_df.empty:
                    self.log('Top combination selection yielded no results.')
                    return
                DV = top_df.iloc[0]['y']
                IVs = [x.strip() for x in top_df.iloc[0]['x'].split(',') if x.strip() != '(none)']
                if not IVs:
                    # Use all other variables except DV
                    IVs = [c for c in df_clean.columns if c != DV]
                self.log(f'Automatically selected IVs: {IVs} and DV: {DV}')
            self.update_progress(50)
            # Identify roles
            self.log('Identifying mediator and moderator roles...')
            roles_df = auto_identify_roles(df_clean, IVs, DV)
            self.roles_df = roles_df
            # Convert p-values to strings with 4 dp for display
            display_roles = roles_df.copy()
            if not display_roles.empty:
                display_roles['p_mediation'] = display_roles['p_mediation'].map(lambda x: f"{x:.4f}")
                display_roles['p_moderation'] = display_roles['p_moderation'].map(lambda x: f"{x:.4f}")
            self.window['-ROLETABLE-'].update(values=display_roles.values.tolist())
            self.update_progress(60)
            # Extract mediators and moderators
            mediators = roles_df.loc[roles_df['type'] == 'mediator', 'var'].tolist()
            moderators = roles_df.loc[roles_df['type'] == 'moderator', 'var'].tolist()
            self.log(f'Found mediators: {mediators}')
            self.log(f'Found moderators: {moderators}')
            # Determine mediator order
            self.log('Optimising mediator ordering...')
            if mediators:
                order, score = auto_mediator_order(df_clean, IVs, DV, mediators)
                self.mediator_order, self.mediator_order_score = order, score
            else:
                order, score = [], float('-inf')
                self.mediator_order, self.mediator_order_score = order, score
            self.log(f'Mediator order: {self.mediator_order}, score: {self.mediator_order_score:.4f}')
            self.update_progress(70)
            # Determine moderator position
            self.log('Determining moderator position...')
            if moderators:
                mod, score, path_desc = auto_moderator_position(df_clean, IVs, DV, self.mediator_order, moderators)
                self.mod_result = (mod, score, path_desc)
                self.log(f'Best moderator: {mod}, score: {score:.4f}, path: {path_desc}')
            else:
                self.mod_result = (None, float('nan'), '')
                self.log('No moderators found or provided.')
            self.update_progress(80)
            # Fit model
            use_sem = bool(values['-SEM-'])
            if use_sem and semopy is None:
                self.log('SEM requested but semopy is not installed. Falling back to OLS.')
                use_sem = False
            moderator_name = self.mod_result[0] if self.mod_result else None
            self.log('Fitting path model...')
            summary, model_obj = auto_path_model_fit(df_clean, IVs, DV, self.mediator_order, moderator_name, use_sem=use_sem)
            self.model_summary = summary
            self.model_obj = model_obj
            self.window['-SUMMARY-'].update(value=summary)
            self.update_progress(100)
            # Show mediator/moderator summary text
            medmod_txt = []
            medmod_txt.append(f"Mediator order: {self.mediator_order}\nScore: {self.mediator_order_score:.4f}\n")
            if self.mod_result and self.mod_result[0] is not None:
                mod, score, path_desc = self.mod_result
                medmod_txt.append(f"Best moderator: {mod}\nScore: {score:.4f}\nPath: {path_desc}\n")
            self.window['-MEDMOD-'].update(value=''.join(medmod_txt))
            self.log('Analysis complete.')
            self.window['-EXPORT-'].update(disabled=False)
        except Exception as e:
            err = traceback.format_exc()
            self.log('An error occurred during analysis:\n' + err)
            self.update_progress(0)

    def _export_results(self):
        """Allow user to export results to a chosen folder."""
        folder = sg.popup_get_folder('Select folder to save results', default_path=os.getcwd())
        if not folder:
            return
        try:
            if self.topn_df is not None:
                self.topn_df.to_csv(os.path.join(folder, 'xy_top.csv'), index=False)
            if self.roles_df is not None:
                self.roles_df.to_csv(os.path.join(folder, 'roles.csv'), index=False)
            # Save mediator order and moderator result
            med_info = {
                'mediator_order': self.mediator_order,
                'mediator_score': self.mediator_order_score,
                'moderator': self.mod_result[0] if self.mod_result else None,
                'moderator_score': self.mod_result[1] if self.mod_result else None,
                'moderator_path': self.mod_result[2] if self.mod_result else None,
            }
            with open(os.path.join(folder, 'med_mod.json'), 'w', encoding='utf-8') as f:
                json.dump(med_info, f, ensure_ascii=False, indent=4)
            # Save model summary
            if self.model_summary:
                with open(os.path.join(folder, 'model_summary.txt'), 'w', encoding='utf-8') as f:
                    f.write(self.model_summary)
            sg.popup('Results saved successfully.')
        except Exception as e:
            sg.popup_error(f'Error saving results: {e}')


def main():
    app = AutoModelRApp()
    app.run()


if __name__ == '__main__':
    main()