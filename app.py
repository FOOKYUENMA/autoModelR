import os
import json
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

from core import (
    auto_preprocess,
    auto_find_main_xy_topn,
    auto_identify_roles,
    auto_mediator_order,
    auto_moderator_position,
    auto_path_model_fit,
)


APP_TITLE = "AutoModelR (Tk GUI)"


class App(tk.Tk):
    """
    A Tkinter-based GUI for the AutoModelR pipeline.

    This class provides a simple desktop interface that allows users to:
      1. Select an Excel or CSV file.
      2. Choose the number of top variable combinations to consider.
      3. Optionally specify independent (IVs) and dependent (DV) variables manually.
      4. Run preprocessing, variable selection (via LASSO), role identification,
         mediator ordering, moderator position search, and path model fitting.
      5. View logs and progress, and export results to files.

    It relies on the functions provided by the ``core`` module for all
    statistical computations.  Only numeric columns are considered in
    the modelling pipeline.
    """

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        # Set a reasonable default size; window is resizable
        self.geometry("960x640")
        self.file_path = tk.StringVar()
        self.topn_var = tk.IntVar(value=3)
        self.ivs_var = tk.StringVar()
        self.dv_var = tk.StringVar()
        self._build_ui()
        # Cache to hold intermediate and final results for export
        self.results_cache = {}

    def _build_ui(self) -> None:
        """Construct the Tkinter widgets for the application."""
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        # Row 1: file selection
        row = ttk.Frame(frm)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="数据文件：").pack(side="left")
        ttk.Entry(row, textvariable=self.file_path, width=80).pack(side="left", padx=5)
        ttk.Button(row, text="浏览…", command=self.choose_file).pack(side="left")
        # Row 2: parameters
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="TopN：").pack(side="left")
        ttk.Spinbox(row2, from_=1, to=20, textvariable=self.topn_var, width=6).pack(side="left", padx=5)
        ttk.Label(row2, text="手动自变量(逗号分隔，可空)：").pack(side="left", padx=(20, 2))
        ttk.Entry(row2, textvariable=self.ivs_var, width=30).pack(side="left", padx=5)
        ttk.Label(row2, text="手动因变量(可空)：").pack(side="left", padx=(20, 2))
        ttk.Entry(row2, textvariable=self.dv_var, width=20).pack(side="left", padx=5)
        # Row 3: control buttons
        row3 = ttk.Frame(frm)
        row3.pack(fill="x", pady=8)
        ttk.Button(row3, text="开始分析", command=self.run_async).pack(side="left")
        self.btn_export = ttk.Button(row3, text="导出结果", command=self.export_results, state="disabled")
        self.btn_export.pack(side="left", padx=10)
        # Progress bar
        self.prog = ttk.Progressbar(frm, orient="horizontal", length=400, mode="determinate", maximum=100)
        self.prog.pack(fill="x", pady=8)
        # Log text box
        self.txt = tk.Text(frm, height=24)
        self.txt.pack(fill="both", expand=True)
        self._log("准备就绪。")

    def _log(self, s: str) -> None:
        """Append a message to the log text box and scroll to the end."""
        self.txt.insert("end", str(s) + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def choose_file(self) -> None:
        """Open a file dialog for selecting input data."""
        f = filedialog.askopenfilename(filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("All", "*.*")])
        if f:
            self.file_path.set(f)

    def run_async(self) -> None:
        """Run the pipeline in a background thread to keep the UI responsive."""
        t = threading.Thread(target=self.run_pipeline, daemon=True)
        t.start()

    def set_prog(self, v: int) -> None:
        """Update the progress bar."""
        self.prog["value"] = v
        self.update_idletasks()

    def read_data(self, path: str) -> pd.DataFrame:
        """Read the Excel or CSV file at the given path."""
        ext = os.path.splitext(path)[1].lower()
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif ext == ".csv":
            return pd.read_csv(path)
        else:
            raise ValueError("不支持的文件类型。请使用 .xlsx/.xls 或 .csv")

    def run_pipeline(self) -> None:
        """
        Execute the full modelling pipeline: preprocessing, LASSO selection,
        role identification, mediator ordering, moderator placement, and
        model fitting. Progress and results are logged in the GUI.
        """
        try:
            self.set_prog(0)
            fpath = self.file_path.get().strip()
            if not fpath or not os.path.exists(fpath):
                messagebox.showerror("错误", "请先选择数据文件")
                return
            # 1: load data
            self._log(f"[1/6] 读取数据：{fpath}")
            df_raw = self.read_data(fpath)
            self._log(f"原始维度：{df_raw.shape}")
            self.set_prog(10)
            # 2: preprocess
            self._log("[2/6] 预处理…")
            df = auto_preprocess(df_raw)
            self._log(f"预处理后维度（仅数值列）：{df.shape}")
            self.set_prog(25)
            # 3: LASSO selection
            topn = int(self.topn_var.get() or 3)
            ivs_input = self.ivs_var.get().strip()
            dv_input = self.dv_var.get().strip()
            if ivs_input and dv_input:
                xy = pd.DataFrame([{"y": dv_input, "x": ivs_input, "r2": None}])
                self._log(f"使用手动指定 IV/DV：DV={dv_input}; IVs={ivs_input}")
            else:
                self._log("[3/6] LASSO TopN 组合搜索…")
                xy = auto_find_main_xy_topn(df, topn=topn)
                self._log("TopN 变量组合：\n" + xy.to_string(index=False))
            self.set_prog(45)
            # Extract IVs and DV
            IVs = [v.strip() for v in (xy.iloc[0]["x"] or "").split(",") if v.strip()]
            DV = xy.iloc[0]["y"]
            if not DV:
                raise ValueError("无法确定因变量（DV）。")
            if not IVs:
                self._log("LASSO 未选出自变量；默认取除 DV 外的第一列为自变量。")
                candidates = [c for c in df.columns if c != DV]
                if not candidates:
                    raise ValueError("没有可用自变量。")
                IVs = [candidates[0]]
            # 4: role identification
            self._log("[4/6] 中介/调节判别…")
            roles = auto_identify_roles(df, IVs, DV)
            self._log(roles.to_string(index=False))
            self.set_prog(60)
            # Extract mediators and moderators using english labels
            mediators = roles.loc[roles["type"] == "mediator", "var"].tolist()
            moderators = roles.loc[roles["type"] == "moderator", "var"].tolist()
            # 5: search mediator order and moderator position
            self._log("[5/6] 多中介顺序与调节位置搜索…")
            order, order_score = auto_mediator_order(df, IVs, DV, mediators)
            mod_name, mod_score, mod_path = auto_moderator_position(df, IVs, DV, order, moderators)
            self._log(f"最佳中介顺序：{' -> '.join(order) if order else '(无)'}")
            self._log(f"调节插入：{mod_path or '(无)'}； 调节变量：{mod_name or '(无)'}")
            self.set_prog(80)
            # 6: fit model (SEM disabled by default)
            self._log("[6/6] 拟合模型与报告…")
            summary, model_obj = auto_path_model_fit(df, IVs, DV, order, mod_name, use_sem=False)
            # Log first 4000 characters of summary to avoid flooding the UI
            self._log(f"模型报告（截断 4000 字符）：\n{summary[:4000]}")
            # Store results
            self.results_cache = {
                "xy_top": xy,
                "roles": roles,
                "med_order": {"best_order": order, "best_score": order_score},
                "mod_pos": {"best_mod": mod_name, "best_score": mod_score, "best_path": mod_path},
                "fit": {"summary": summary, "model": str(model_obj)},
            }
            self.btn_export.config(state="normal")
            self.set_prog(100)
            messagebox.showinfo("完成", "分析完成！")
        except Exception:
            err = traceback.format_exc()
            self._log("发生错误：\n" + err)
            messagebox.showerror("错误", f"发生错误：\n{err}")

    def export_results(self) -> None:
        """Prompt the user to choose a directory and export result files."""
        outdir = filedialog.askdirectory(title="选择导出文件夹")
        if not outdir:
            return
        try:
            # Export top combinations
            if "xy_top" in self.results_cache and isinstance(self.results_cache["xy_top"], pd.DataFrame):
                self.results_cache["xy_top"].to_csv(os.path.join(outdir, "xy_top.csv"), index=False, encoding="utf-8-sig")
            # Export role identification table
            if "roles" in self.results_cache and isinstance(self.results_cache["roles"], pd.DataFrame):
                self.results_cache["roles"].to_csv(os.path.join(outdir, "roles.csv"), index=False, encoding="utf-8-sig")
            # Export mediator order
            with open(os.path.join(outdir, "med_order.json"), "w", encoding="utf-8") as f:
                json.dump(self.results_cache.get("med_order", {}), f, ensure_ascii=False, indent=2)
            # Export moderator position
            with open(os.path.join(outdir, "mod_pos.json"), "w", encoding="utf-8") as f:
                json.dump(self.results_cache.get("mod_pos", {}), f, ensure_ascii=False, indent=2)
            # Export model summary
            with open(os.path.join(outdir, "fit.txt"), "w", encoding="utf-8") as f:
                f.write(f"Summary:\n{self.results_cache['fit'].get('summary')}\n\n")
                f.write(f"Model:\n{self.results_cache['fit'].get('model')}\n")
            messagebox.showinfo("成功", "导出完成！")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败：{e}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()