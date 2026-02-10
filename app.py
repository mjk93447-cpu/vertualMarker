import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QGroupBox,
    QGridLayout,
)

from vertualmarker.strategy2 import (
    Strategy2Config,
    Strategy2Error,
    run_strategy2_on_file,
    save_result_points_txt,
    parse_txt_points,
)
from vertualmarker.visualization import visualize_result
from vertualmarker.data_generator import (
    SyntheticParams,
    generate_turtle_and_partner,
    save_points_txt,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Virtual Marker v7 - Strategy 2 Workbench")
        self.resize(1200, 820)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        self._apply_professional_theme()

        title = QLabel("Virtual Marker v7 - Strategy 2 (Turtle Head)")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Analyze edge-map TXT points, detect turtle-line geometry, and export "
            "indexed bending trajectory for downstream motion analytics."
        )
        subtitle.setWordWrap(True)
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # Input files group
        file_group = QGroupBox("Input TXT Files (max 500)")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        self.file_list = QListWidget()
        file_layout.addWidget(self.file_list)

        file_btn_layout = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Files...")
        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_clear_files = QPushButton("Clear All")
        file_btn_layout.addWidget(self.btn_add_files)
        file_btn_layout.addWidget(self.btn_remove_selected)
        file_btn_layout.addWidget(self.btn_clear_files)
        file_layout.addLayout(file_btn_layout)

        main_layout.addWidget(file_group)

        # Output directory group
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout()
        output_group.setLayout(output_layout)
        self.edit_output_dir = QLineEdit()
        self.edit_output_dir.setPlaceholderText(
            "Leave empty to save next to each input file..."
        )
        self.btn_select_output = QPushButton("Select Folder...")
        output_layout.addWidget(self.edit_output_dir)
        output_layout.addWidget(self.btn_select_output)
        main_layout.addWidget(output_group)

        # Parameters group
        param_group = QGroupBox("Strategy 2 Parameters")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)

        # FH: first vertical segment threshold
        fh_label = QLabel("FH")
        fh_label.setToolTip("Minimum vertical run length to detect the forehead segment.")
        self.spin_fh = QDoubleSpinBox()
        self.spin_fh.setRange(1.0, 1e6)
        self.spin_fh.setDecimals(2)
        self.spin_fh.setValue(50.0)
        self.spin_fh.setToolTip("Larger FH requires a longer vertical straight segment.")
        param_layout.addWidget(fh_label, 0, 0)
        param_layout.addWidget(self.spin_fh, 0, 1)

        # UH: first horizontal segment threshold
        uh_label = QLabel("UH")
        uh_label.setToolTip("Minimum horizontal run length after FH segment detection.")
        self.spin_uh = QDoubleSpinBox()
        self.spin_uh.setRange(1.0, 1e6)
        self.spin_uh.setDecimals(2)
        self.spin_uh.setValue(50.0)
        self.spin_uh.setToolTip("Larger UH requires a longer horizontal straight segment.")
        param_layout.addWidget(uh_label, 0, 2)
        param_layout.addWidget(self.spin_uh, 0, 3)

        # SX: virtual marker x-shift
        sx_label = QLabel("SX")
        sx_label.setToolTip("X offset applied to Mv before nearest-point BSP search.")
        self.spin_sx = QDoubleSpinBox()
        self.spin_sx.setRange(-1e6, 1e6)
        self.spin_sx.setDecimals(2)
        self.spin_sx.setValue(0.0)
        self.spin_sx.setToolTip("Positive moves right, negative moves left.")
        param_layout.addWidget(sx_label, 1, 0)
        param_layout.addWidget(self.spin_sx, 1, 1)

        # SY: virtual marker y-shift
        sy_label = QLabel("SY")
        sy_label.setToolTip("Y offset applied to Mv before nearest-point BSP search.")
        self.spin_sy = QDoubleSpinBox()
        self.spin_sy.setRange(-1e6, 1e6)
        self.spin_sy.setDecimals(2)
        self.spin_sy.setValue(0.0)
        self.spin_sy.setToolTip("Image coordinates: positive value moves downward.")
        param_layout.addWidget(sy_label, 1, 2)
        param_layout.addWidget(self.spin_sy, 1, 3)

        # PBL: number of output points
        pbl_label = QLabel("PBL")
        pbl_label.setToolTip("Number of indexed bending points to export.")
        self.spin_pbl = QSpinBox()
        self.spin_pbl.setRange(10, 100000)
        self.spin_pbl.setValue(500)
        self.spin_pbl.setToolTip("Output trajectory length (index 1..PBL).")
        param_layout.addWidget(pbl_label, 2, 0)
        param_layout.addWidget(self.spin_pbl, 2, 1)

        # Sampling step along the path
        step_label = QLabel("Sample Step (pixel)")
        step_label.setToolTip("Spatial interval when sampling along turtle-line path.")
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.01, 100.0)
        self.spin_step.setDecimals(2)
        self.spin_step.setValue(1.0)
        self.spin_step.setToolTip("Lower value gives denser sampling.")
        param_layout.addWidget(step_label, 2, 2)
        param_layout.addWidget(self.spin_step, 2, 3)

        main_layout.addWidget(param_group)

        # Algorithm guide
        guide_group = QGroupBox("How Strategy 2 Works")
        guide_layout = QVBoxLayout()
        guide_group.setLayout(guide_layout)
        self.text_guide = QTextEdit()
        self.text_guide.setReadOnly(True)
        self.text_guide.setPlainText(
            "1) Build connected components from edge points (8-neighborhood).\n"
            "2) Pick two longest components and choose the lower one as turtle line.\n"
            "3) Find TLSP as the lower endpoint of the turtle component.\n"
            "4) Traverse path and detect first vertical run >= FH (Front Head).\n"
            "5) Continue and detect first horizontal run >= UH (Upper Head).\n"
            "6) Compute Mv from FH/UH line intersection, then shift by SX/SY.\n"
            "7) Find BSP as nearest turtle point to shifted marker.\n"
            "8) Sample opposite-to-TLSP direction from BSP to export indexed points.\n\n"
            "Output TXT format:\n"
            "  # x,y,index\n"
            "  x,y,1\n"
            "  x,y,2\n"
            "  ...\n"
            "Use the index column as a stable key for frame-to-frame motion tracking."
        )
        guide_layout.addWidget(self.text_guide)
        main_layout.addWidget(guide_group)

        # Example data generator group
        example_group = QGroupBox("Example Data Generator")
        example_layout = QHBoxLayout()
        example_group.setLayout(example_layout)
        self.btn_generate_example = QPushButton("Generate Example TXT...")
        example_layout.addWidget(self.btn_generate_example)
        example_layout.addStretch(1)
        main_layout.addWidget(example_group)

        # Run button
        run_layout = QHBoxLayout()
        run_layout.addStretch(1)
        self.btn_run = QPushButton("Run Processing")
        run_layout.addWidget(self.btn_run)
        main_layout.addLayout(run_layout)

        # Log output
        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        log_layout.addWidget(self.text_log)
        main_layout.addWidget(log_group)

        # Connections
        self.btn_add_files.clicked.connect(self.on_add_files)
        self.btn_remove_selected.clicked.connect(self.on_remove_selected)
        self.btn_clear_files.clicked.connect(self.file_list.clear)
        self.btn_select_output.clicked.connect(self.on_select_output_dir)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_generate_example.clicked.connect(self.on_generate_example)

    def _apply_professional_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1E1F24;
                color: #E7EAF0;
                font-size: 11pt;
            }
            QGroupBox {
                border: 1px solid #3B3F4A;
                border-radius: 8px;
                margin-top: 12px;
                padding: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px 0 6px;
                color: #C5D1E8;
            }
            QTextEdit, QListWidget, QLineEdit, QDoubleSpinBox, QSpinBox {
                background-color: #252932;
                border: 1px solid #434A5A;
                border-radius: 6px;
                padding: 4px;
                selection-background-color: #365A9C;
            }
            QPushButton {
                background-color: #2B66C3;
                border: 1px solid #4A82D8;
                border-radius: 6px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3571D0;
            }
            QPushButton:pressed {
                background-color: #265CB3;
            }
            QLabel#TitleLabel {
                font-size: 18pt;
                font-weight: 700;
                color: #F4F7FF;
                padding-bottom: 2px;
            }
            """
        )

    # File list helpers
    def on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select TXT Files",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not files:
            return

        current_paths = {self.file_list.item(i).text() for i in range(self.file_list.count())}
        for path in files:
            if path not in current_paths:
                if self.file_list.count() >= 500:
                    QMessageBox.warning(
                        self,
                        "Limit Exceeded",
                        "You can select up to 500 files.",
                    )
                    break
                self.file_list.addItem(path)

        self.log(f"Selected file count: {self.file_list.count()}")

    def on_remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
        self.log(f"Selected file count: {self.file_list.count()}")

    def on_select_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self.edit_output_dir.text().strip() or "",
        )
        if not path:
            return
        self.edit_output_dir.setText(path)
        self.log(f"Output folder set to: {path}")

    def log(self, msg: str) -> None:
        self.text_log.append(msg)
        cursor = self.text_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_log.setTextCursor(cursor)

    # Run processing
    def on_run(self) -> None:
        count = self.file_list.count()
        if count == 0:
            QMessageBox.information(
                self,
                "No Input Files",
                "Please add at least one TXT file before running.",
            )
            return

        config = Strategy2Config(
            FH=self.spin_fh.value(),
            UH=self.spin_uh.value(),
            SX=self.spin_sx.value(),
            SY=self.spin_sy.value(),
            PBL=self.spin_pbl.value(),
            sample_step=self.spin_step.value(),
        )

        output_dir = self.edit_output_dir.text().strip()
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                QMessageBox.critical(
                    self,
                    "Invalid Output Folder",
                    f"Cannot create or access output folder:\n{output_dir}\n\n{e}",
                )
                return

        self.log(
            f"Run started: files={count}, "
            f"FH={config.FH}, UH={config.UH}, SX={config.SX}, SY={config.SY}, PBL={config.PBL}, step={config.sample_step}"
        )
        if output_dir:
            self.log(f"Output folder: {output_dir}")
        else:
            self.log("Output folder: same directory as each input file")
        self.log("=" * 60)

        num_success = 0
        num_fail = 0

        for i in range(count):
            path = self.file_list.item(i).text()
            input_stem = os.path.splitext(os.path.basename(path))[0]
            if output_dir:
                out_txt = os.path.join(output_dir, input_stem + "_bending_points.txt")
                out_img = os.path.join(output_dir, input_stem + "_visualization.png")
            else:
                base, _ext = os.path.splitext(path)
                out_txt = base + "_bending_points.txt"
                out_img = base + "_visualization.png"

            self.log(f"\n[{i+1}/{count}] Processing: {os.path.basename(path)}")

            try:
                self.log("  - Reading input points...")
                original_points = parse_txt_points(path)
                self.log(f"  - Loaded points: {len(original_points)}")

                self.log("  - Running Strategy 2...")
                result = run_strategy2_on_file(path, config)
                if result.longest_two_lines_info:
                    for idx, (lowest, length) in enumerate(
                        result.longest_two_lines_info, start=1
                    ):
                        self.log(
                            f"  - Longest line #{idx}: lowest=({lowest[0]}, {lowest[1]}), length={length}"
                        )
                self.log(
                    "  - Turtle line detected: "
                    f"lowest=({result.turtle_lowest_point[0]}, {result.turtle_lowest_point[1]}), "
                    f"length={result.turtle_line_length}"
                )
                self.log(f"  - Turtle path length: {len(result.turtle_line_path)}")
                self.log(f"  - Front-head run: {len(result.front_head_run)} points")
                self.log(f"  - Upper-head run: {len(result.upper_head_run)} points")
                self.log(f"  - Mv: ({result.mv[0]}, {result.mv[1]})")
                self.log(f"  - BSP: ({result.bsp[0]}, {result.bsp[1]})")

                self.log("  - Saving TXT output...")
                save_result_points_txt(out_txt, result)
                self.log(f"  - Bending points exported: {len(result.bending_points)}")

                self.log("  - Rendering visualization...")
                visualize_result(original_points, result, out_img)
                self.log(f"  - Visualization saved: {os.path.basename(out_img)}")

                self.log(f"[SUCCESS] {os.path.basename(path)}")
                self.log(f"  -> {os.path.basename(out_txt)} ({len(result.bending_points)} points)")
                self.log(f"  -> {os.path.basename(out_img)}")
                num_success += 1
            except Strategy2Error as e:
                self.log(f"[FAILED] {os.path.basename(path)}")
                self.log(f"  Error: {e}")
                import traceback
                self.log(f"  Details: {traceback.format_exc()}")
                num_fail += 1
            except Exception as e:  # noqa: BLE001
                self.log(f"[ERROR] {os.path.basename(path)}")
                self.log(f"  Exception: {type(e).__name__}: {e}")
                import traceback
                self.log(f"  Details:\n{traceback.format_exc()}")
                num_fail += 1

        QMessageBox.information(
            self,
            "Completed",
            f"Processing complete.\nSuccess: {num_success}\nFailed: {num_fail}",
        )
        self.log("=== Processing complete ===")

    def on_generate_example(self) -> None:
        """Generate a synthetic example TXT close to real edge shape."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Save Path for Example TXT",
            "example_turtle.txt",
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return

        params = SyntheticParams()
        points = generate_turtle_and_partner(params)
        try:
            save_points_txt(path, points)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Example Generation Failed", str(e))
            return

        self.log(f"Example TXT generated: {path}")
        # Auto-append generated sample to input file list
        self.file_list.addItem(path)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

