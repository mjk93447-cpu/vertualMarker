import html
import os
import sys
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vertualmarker.data_generator import SyntheticParams, generate_turtle_and_partner, save_points_txt
from vertualmarker.strategy2 import (
    Strategy2Config,
    Strategy2Error,
    parse_txt_points,
    run_strategy2_on_file,
    save_result_points_txt,
)
from vertualmarker.visualization import visualize_result


class ImagePreviewDialog(QDialog):
    def __init__(self, image_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Preview - {os.path.basename(image_path)}")
        self.resize(1200, 900)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = QPixmap(image_path)
        if not pix.isNull():
            image_label.setPixmap(
                pix.scaled(
                    1800,
                    1400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        holder_layout.addWidget(image_label)
        scroll.setWidget(holder)
        layout.addWidget(scroll)


class ClickableImageLabel(QLabel):
    def __init__(self, image_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            dlg = ImagePreviewDialog(self.image_path, self)
            dlg.exec()
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Virtual Marker v11 - Final Release Workbench")
        self.resize(1500, 980)
        self.preview_paths: List[str] = []

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        self._apply_professional_theme()

        title = QLabel("Virtual Marker v11 - Final Release")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Production-ready workstation for Strategy 2: diagnostics, trajectory export, "
            "visual issue highlighting, and high-density visualization preview."
        )
        subtitle.setWordWrap(True)
        self.status_badge = QLabel("UI/UX Quality Status: Ready")
        self.status_badge.setObjectName("StatusBadge")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)
        root_layout.addWidget(self.status_badge)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([670, 830])
        root_layout.addWidget(main_splitter)

        self.btn_add_files.clicked.connect(self.on_add_files)
        self.btn_remove_selected.clicked.connect(self.on_remove_selected)
        self.btn_clear_files.clicked.connect(self.file_list.clear)
        self.btn_select_output.clicked.connect(self.on_select_output_dir)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_generate_example.clicked.connect(self.on_generate_example)

        default_output = self._default_output_dir()
        self.edit_output_dir.setText(default_output)
        os.makedirs(default_output, exist_ok=True)
        self.log(f"Default output folder initialized: {default_output}", "info")

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.addWidget(self._create_file_group())
        left_splitter.addWidget(self._create_output_group())
        left_splitter.addWidget(self._create_params_group())
        left_splitter.addWidget(self._create_guide_group())
        left_splitter.addWidget(self._create_example_group())
        left_splitter.addWidget(self._create_run_group())
        left_splitter.setSizes([240, 80, 210, 280, 70, 70])
        layout.addWidget(left_splitter)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self._create_preview_group())
        right_splitter.addWidget(self._create_log_group())
        right_splitter.setSizes([600, 320])
        layout.addWidget(right_splitter)
        return panel

    def _create_file_group(self) -> QGroupBox:
        group = QGroupBox("Input TXT Files (max 500)")
        layout = QVBoxLayout(group)
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        btns = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Files...")
        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_clear_files = QPushButton("Clear All")
        btns.addWidget(self.btn_add_files)
        btns.addWidget(self.btn_remove_selected)
        btns.addWidget(self.btn_clear_files)
        layout.addLayout(btns)
        return group

    def _create_output_group(self) -> QGroupBox:
        group = QGroupBox("Output Directory")
        layout = QHBoxLayout(group)
        self.edit_output_dir = QLineEdit()
        self.edit_output_dir.setPlaceholderText("Defaults to <exe_folder>/output")
        self.btn_select_output = QPushButton("Select Folder...")
        layout.addWidget(self.edit_output_dir)
        layout.addWidget(self.btn_select_output)
        return group

    def _create_params_group(self) -> QGroupBox:
        group = QGroupBox("Strategy 2 Parameters")
        grid = QGridLayout(group)

        fh_label = QLabel("FH")
        fh_label.setToolTip("Minimum vertical run length to detect the forehead segment.")
        self.spin_fh = QDoubleSpinBox()
        self.spin_fh.setRange(1.0, 1e6)
        self.spin_fh.setDecimals(2)
        self.spin_fh.setValue(50.0)
        self.spin_fh.setToolTip("Larger FH requires a longer vertical straight segment.")
        grid.addWidget(fh_label, 0, 0)
        grid.addWidget(self.spin_fh, 0, 1)

        uh_label = QLabel("UH")
        uh_label.setToolTip("Minimum horizontal run length after FH segment detection.")
        self.spin_uh = QDoubleSpinBox()
        self.spin_uh.setRange(1.0, 1e6)
        self.spin_uh.setDecimals(2)
        self.spin_uh.setValue(50.0)
        self.spin_uh.setToolTip("Larger UH requires a longer horizontal straight segment.")
        grid.addWidget(uh_label, 0, 2)
        grid.addWidget(self.spin_uh, 0, 3)

        sx_label = QLabel("SX")
        sx_label.setToolTip("X offset applied to Mv before nearest-point BSP search.")
        self.spin_sx = QDoubleSpinBox()
        self.spin_sx.setRange(-1e6, 1e6)
        self.spin_sx.setDecimals(2)
        self.spin_sx.setValue(0.0)
        grid.addWidget(sx_label, 1, 0)
        grid.addWidget(self.spin_sx, 1, 1)

        sy_label = QLabel("SY")
        sy_label.setToolTip("Y offset applied to Mv before nearest-point BSP search.")
        self.spin_sy = QDoubleSpinBox()
        self.spin_sy.setRange(-1e6, 1e6)
        self.spin_sy.setDecimals(2)
        self.spin_sy.setValue(0.0)
        grid.addWidget(sy_label, 1, 2)
        grid.addWidget(self.spin_sy, 1, 3)

        pbl_label = QLabel("PBL")
        pbl_label.setToolTip("Number of indexed bending points to export.")
        self.spin_pbl = QSpinBox()
        self.spin_pbl.setRange(10, 100000)
        self.spin_pbl.setValue(500)
        grid.addWidget(pbl_label, 2, 0)
        grid.addWidget(self.spin_pbl, 2, 1)

        step_label = QLabel("Sample Step (pixel)")
        step_label.setToolTip("Spatial interval when sampling along turtle-line path.")
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.01, 100.0)
        self.spin_step.setDecimals(2)
        self.spin_step.setValue(1.0)
        grid.addWidget(step_label, 2, 2)
        grid.addWidget(self.spin_step, 2, 3)
        return group

    def _create_guide_group(self) -> QGroupBox:
        group = QGroupBox("How Strategy 2 Works + Auto Diagnostics")
        layout = QVBoxLayout(group)
        self.text_guide = QTextEdit()
        self.text_guide.setReadOnly(True)
        self.text_guide.setPlainText(
            "1) Build connected components from edge points (8-neighborhood).\n"
            "2) Pick the two longest components; the lower one becomes turtle line.\n"
            "3) Detect TLSP endpoint, FH run, UH run, then compute Mv and shifted Mv'.\n"
            "4) Compute BSP by nearest point to Mv' and sample indexed trajectory.\n"
            "5) Export TXT format: # x,y,index / x,y,1 ...\n\n"
            "Diagnostics severity levels:\n"
            "  INFO    : informational status\n"
            "  WARNING : attention required\n"
            "  CRITICAL: immediate corrective action needed (shown in red)\n\n"
            "Critical points are also marked in red in the visualization image."
        )
        layout.addWidget(self.text_guide)
        return group

    def _create_example_group(self) -> QGroupBox:
        group = QGroupBox("Example Data Generator")
        layout = QHBoxLayout(group)
        self.btn_generate_example = QPushButton("Generate Example TXT...")
        layout.addWidget(self.btn_generate_example)
        layout.addStretch(1)
        return group

    def _create_run_group(self) -> QGroupBox:
        group = QGroupBox("Execution")
        layout = QHBoxLayout(group)
        layout.addStretch(1)
        self.btn_run = QPushButton("Run Processing")
        self.btn_run.setProperty("primary", True)
        layout.addWidget(self.btn_run)
        return group

    def _create_preview_group(self) -> QGroupBox:
        group = QGroupBox("Last Visualization Preview (up to 100 images)")
        layout = QVBoxLayout(group)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_host = QWidget()
        self.preview_grid = QGridLayout(self.preview_host)
        self.preview_grid.setHorizontalSpacing(14)
        self.preview_grid.setVerticalSpacing(16)
        self.preview_scroll.setWidget(self.preview_host)
        layout.addWidget(self.preview_scroll)
        return group

    def _create_log_group(self) -> QGroupBox:
        group = QGroupBox("Execution Log (Severity Colored)")
        layout = QVBoxLayout(group)
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        layout.addWidget(self.text_log)
        return group

    def _apply_professional_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #16181D; color: #EAEFF8; font-size: 11pt; }
            QGroupBox {
                border: 1px solid #384150;
                border-radius: 8px;
                margin-top: 12px;
                padding: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px 0 6px;
                color: #C8D7F2;
            }
            QTextEdit, QListWidget, QLineEdit, QDoubleSpinBox, QSpinBox, QScrollArea {
                background: #1F232C;
                border: 1px solid #424B5C;
                border-radius: 6px;
            }
            QPushButton {
                background: #2A64BE;
                border: 1px solid #4A85DE;
                border-radius: 6px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover { background: #3471CE; }
            QPushButton:pressed { background: #265AB0; }
            QLabel#TitleLabel {
                font-size: 19pt;
                font-weight: 700;
                color: #F5F9FF;
                padding-bottom: 2px;
            }
            QLabel#StatusBadge {
                background: #253041;
                border: 1px solid #446182;
                border-radius: 6px;
                padding: 6px 10px;
                color: #D9E8FF;
                font-weight: 600;
            }
            QWidget#PreviewTile {
                border: 1px solid #3E4C63;
                border-radius: 8px;
                background: #1C212A;
                padding: 6px;
            }
            QSplitter::handle {
                background: #33445E;
            }
            QSplitter::handle:horizontal {
                width: 7px;
            }
            QSplitter::handle:vertical {
                height: 7px;
            }
            """
        )

    def _default_output_dir(self) -> str:
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "output")

    def _severity_color(self, severity: str) -> str:
        mapping = {
            "info": "#9EC5FF",
            "warning": "#FFCC66",
            "critical": "#FF3B3B",
            "error": "#FF3B3B",
            "success": "#56D364",
        }
        return mapping.get(severity, "#EAEFF8")

    def log(self, msg: str, severity: str = "info") -> None:
        color = self._severity_color(severity)
        safe = html.escape(msg)
        self.text_log.append(f"<span style='color:{color};'>{safe}</span>")
        cursor = self.text_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_log.setTextCursor(cursor)

    def _clear_preview_grid(self) -> None:
        while self.preview_grid.count():
            item = self.preview_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_preview_grid(self) -> None:
        self._clear_preview_grid()
        cols = 4
        for idx, img_path in enumerate(self.preview_paths):
            row = idx // cols
            col = idx % cols

            tile = QWidget()
            tile.setObjectName("PreviewTile")
            tile_layout = QVBoxLayout(tile)

            image_label = ClickableImageLabel(img_path)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            pix = QPixmap(img_path)
            if not pix.isNull():
                image_label.setPixmap(
                    pix.scaled(
                        360,
                        260,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

            name_label = QLabel(os.path.basename(img_path))
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setWordWrap(True)
            tile_layout.addWidget(image_label)
            tile_layout.addWidget(name_label)
            self.preview_grid.addWidget(tile, row, col)

    def _append_preview(self, img_path: str) -> None:
        self.preview_paths.append(img_path)
        if len(self.preview_paths) > 100:
            self.preview_paths = self.preview_paths[-100:]
        self._refresh_preview_grid()

    def on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select TXT Files",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not files:
            return
        current = {self.file_list.item(i).text() for i in range(self.file_list.count())}
        for path in files:
            if path in current:
                continue
            if self.file_list.count() >= 500:
                QMessageBox.warning(self, "Limit Exceeded", "You can select up to 500 files.")
                break
            self.file_list.addItem(path)
        self.log(f"Selected file count: {self.file_list.count()}", "info")

    def on_remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self.log(f"Selected file count: {self.file_list.count()}", "info")

    def on_select_output_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self.edit_output_dir.text().strip() or self._default_output_dir(),
        )
        if not chosen:
            return
        self.edit_output_dir.setText(chosen)
        self.log(f"Output folder set to: {chosen}", "info")

    def on_run(self) -> None:
        count = self.file_list.count()
        if count == 0:
            QMessageBox.information(self, "No Input Files", "Please add at least one TXT file before running.")
            return

        config = Strategy2Config(
            FH=self.spin_fh.value(),
            UH=self.spin_uh.value(),
            SX=self.spin_sx.value(),
            SY=self.spin_sy.value(),
            PBL=self.spin_pbl.value(),
            sample_step=self.spin_step.value(),
        )

        output_dir = self.edit_output_dir.text().strip() or self._default_output_dir()
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Invalid Output Folder", f"Cannot access output folder:\n{output_dir}\n\n{e}")
            return

        self.log(
            f"Run started: files={count}, FH={config.FH}, UH={config.UH}, SX={config.SX}, "
            f"SY={config.SY}, PBL={config.PBL}, step={config.sample_step}",
            "info",
        )
        self.log(f"Output folder: {output_dir}", "info")
        self.log("=" * 65, "info")
        self.status_badge.setText("UI/UX Quality Status: Running analysis...")

        num_success = 0
        num_fail = 0
        warn_count = 0
        critical_count = 0

        for i in range(count):
            path = self.file_list.item(i).text()
            stem = os.path.splitext(os.path.basename(path))[0]
            out_txt = os.path.join(output_dir, stem + "_bending_points.txt")
            out_img = os.path.join(output_dir, stem + "_visualization.png")

            self.log(f"\n[{i+1}/{count}] Processing: {os.path.basename(path)}", "info")
            try:
                self.log("  - Reading input points...", "info")
                original_points = parse_txt_points(path)
                self.log(f"  - Loaded points: {len(original_points)}", "info")

                self.log("  - Running Strategy 2...", "info")
                result = run_strategy2_on_file(path, config)
                if result.longest_two_lines_info:
                    for idx, (lowest, length) in enumerate(result.longest_two_lines_info, start=1):
                        self.log(
                            f"  - Longest line #{idx}: lowest=({lowest[0]}, {lowest[1]}), length={length}",
                            "info",
                        )

                self.log(
                    f"  - Turtle line: lowest=({result.turtle_lowest_point[0]}, {result.turtle_lowest_point[1]}), "
                    f"length={result.turtle_line_length}",
                    "info",
                )
                self.log(f"  - Mv: ({result.mv[0]}, {result.mv[1]})", "info")
                self.log(f"  - Mv': ({result.mv_shifted[0]}, {result.mv_shifted[1]})", "info")
                self.log(f"  - BSP: ({result.bsp[0]}, {result.bsp[1]})", "info")

                self.log("  - Saving TXT output...", "info")
                save_result_points_txt(out_txt, result)
                self.log(f"  - Bending points exported: {len(result.bending_points)}", "success")

                self.log("  - Rendering visualization...", "info")
                visualize_result(original_points, result, out_img)
                self._append_preview(out_img)
                self.log(f"  - Visualization saved: {os.path.basename(out_img)}", "success")

                self.log("  - Auto diagnostics:", "info")
                for d in result.diagnostics:
                    sev = d.severity.lower()
                    prefix = sev.upper()
                    if sev == "warning":
                        warn_count += 1
                    elif sev == "critical":
                        critical_count += 1
                    if d.point is not None:
                        self.log(f"    [{prefix}] {d.message} @ {d.point}", sev)
                    else:
                        self.log(f"    [{prefix}] {d.message}", sev)

                self.log(f"[SUCCESS] {os.path.basename(path)}", "success")
                self.log(f"  -> {os.path.basename(out_txt)} ({len(result.bending_points)} points)", "success")
                self.log(f"  -> {os.path.basename(out_img)}", "success")
                num_success += 1
            except Strategy2Error as e:
                self.log(f"[FAILED] {os.path.basename(path)}", "critical")
                self.log(f"  Immediate action required: {e}", "critical")
                import traceback

                self.log(f"  Details: {traceback.format_exc()}", "error")
                num_fail += 1
            except Exception as e:  # noqa: BLE001
                self.log(f"[ERROR] {os.path.basename(path)}", "critical")
                self.log(f"  Immediate action required: {type(e).__name__}: {e}", "critical")
                import traceback

                self.log(f"  Details:\n{traceback.format_exc()}", "error")
                num_fail += 1

        QMessageBox.information(
            self,
            "Completed",
            f"Processing complete.\nSuccess: {num_success}\nFailed: {num_fail}\nOutput: {output_dir}",
        )
        self.status_badge.setText(
            "UI/UX Quality Status: Completed | "
            f"Success={num_success}, Failed={num_fail}, Warnings={warn_count}, Critical={critical_count}"
        )
        self.log("=== Processing complete ===", "info")

    def on_generate_example(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Save Path for Example TXT",
            "example_turtle.txt",
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return
        points = generate_turtle_and_partner(SyntheticParams())
        try:
            save_points_txt(path, points)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Example Generation Failed", str(e))
            return
        self.file_list.addItem(path)
        self.log(f"Example TXT generated and added: {path}", "success")


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

