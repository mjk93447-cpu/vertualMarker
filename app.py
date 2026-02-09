import os
import sys
from typing import List

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
    QListWidgetItem,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QGroupBox,
    QCheckBox,
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
        self.setWindowTitle("Vertual Marker - Strategy 2 (Turtle Head)")
        self.resize(900, 600)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # File selection group
        file_group = QGroupBox("TXT 파일 선택 (최대 500개)")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        self.file_list = QListWidget()
        file_layout.addWidget(self.file_list)

        file_btn_layout = QHBoxLayout()
        self.btn_add_files = QPushButton("파일 추가...")
        self.btn_remove_selected = QPushButton("선택 삭제")
        self.btn_clear_files = QPushButton("전체 삭제")
        file_btn_layout.addWidget(self.btn_add_files)
        file_btn_layout.addWidget(self.btn_remove_selected)
        file_btn_layout.addWidget(self.btn_clear_files)
        file_layout.addLayout(file_btn_layout)

        main_layout.addWidget(file_group)

        # Parameters group
        param_group = QGroupBox("전략 2 설정값")
        param_layout = QHBoxLayout()
        param_group.setLayout(param_layout)

        # FH
        fh_layout = QVBoxLayout()
        fh_label = QLabel("FH (Forehead, 세로 길이)")
        self.spin_fh = QDoubleSpinBox()
        self.spin_fh.setRange(1.0, 1e6)
        self.spin_fh.setDecimals(2)
        self.spin_fh.setValue(50.0)
        fh_layout.addWidget(fh_label)
        fh_layout.addWidget(self.spin_fh)
        param_layout.addLayout(fh_layout)

        # UH
        uh_layout = QVBoxLayout()
        uh_label = QLabel("UH (Upperhead, 가로 길이)")
        self.spin_uh = QDoubleSpinBox()
        self.spin_uh.setRange(1.0, 1e6)
        self.spin_uh.setDecimals(2)
        self.spin_uh.setValue(50.0)
        uh_layout.addWidget(uh_label)
        uh_layout.addWidget(self.spin_uh)
        param_layout.addLayout(uh_layout)

        # SX
        sx_layout = QVBoxLayout()
        sx_label = QLabel("SX (Mv x축 평행 이동)")
        self.spin_sx = QDoubleSpinBox()
        self.spin_sx.setRange(-1e6, 1e6)
        self.spin_sx.setDecimals(2)
        self.spin_sx.setValue(0.0)
        sx_layout.addWidget(sx_label)
        sx_layout.addWidget(self.spin_sx)
        param_layout.addLayout(sx_layout)

        # SY
        sy_layout = QVBoxLayout()
        sy_label = QLabel("SY (Mv y축 평행 이동)")
        self.spin_sy = QDoubleSpinBox()
        self.spin_sy.setRange(-1e6, 1e6)
        self.spin_sy.setDecimals(2)
        self.spin_sy.setValue(0.0)
        sy_layout.addWidget(sy_label)
        sy_layout.addWidget(self.spin_sy)
        param_layout.addLayout(sy_layout)

        # PBL
        pbl_layout = QVBoxLayout()
        pbl_label = QLabel("PBL (Bending length, 출력 포인트 수)")
        self.spin_pbl = QSpinBox()
        self.spin_pbl.setRange(10, 100000)
        self.spin_pbl.setValue(500)
        pbl_layout.addWidget(pbl_label)
        pbl_layout.addWidget(self.spin_pbl)
        param_layout.addLayout(pbl_layout)

        # Angle tolerances & sample step
        angle_layout = QVBoxLayout()
        ang_v_label = QLabel("세로 각도 허용 (deg)")
        self.spin_ang_v = QDoubleSpinBox()
        self.spin_ang_v.setRange(0.0, 45.0)
        self.spin_ang_v.setDecimals(1)
        self.spin_ang_v.setValue(5.0)

        ang_h_label = QLabel("가로 각도 허용 (deg)")
        self.spin_ang_h = QDoubleSpinBox()
        self.spin_ang_h.setRange(0.0, 45.0)
        self.spin_ang_h.setDecimals(1)
        self.spin_ang_h.setValue(5.0)

        step_label = QLabel("샘플 간 거리 (pixel)")
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.01, 100.0)
        self.spin_step.setDecimals(2)
        self.spin_step.setValue(1.0)

        angle_layout.addWidget(ang_v_label)
        angle_layout.addWidget(self.spin_ang_v)
        angle_layout.addWidget(ang_h_label)
        angle_layout.addWidget(self.spin_ang_h)
        angle_layout.addWidget(step_label)
        angle_layout.addWidget(self.spin_step)

        param_layout.addLayout(angle_layout)

        main_layout.addWidget(param_group)

        # Example data generator group
        example_group = QGroupBox("예시 TXT 데이터 생성")
        example_layout = QHBoxLayout()
        example_group.setLayout(example_layout)
        self.btn_generate_example = QPushButton("예시 TXT 생성...")
        example_layout.addWidget(self.btn_generate_example)
        example_layout.addStretch(1)
        main_layout.addWidget(example_group)

        # Run button
        run_layout = QHBoxLayout()
        run_layout.addStretch(1)
        self.btn_run = QPushButton("선택한 파일들 처리 실행")
        run_layout.addWidget(self.btn_run)
        main_layout.addLayout(run_layout)

        # Log output
        log_group = QGroupBox("로그")
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
        self.btn_run.clicked.connect(self.on_run)
        self.btn_generate_example.clicked.connect(self.on_generate_example)

    # File list helpers
    def on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "TXT 파일 선택",
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
                        "제한 초과",
                        "최대 500개의 파일만 선택할 수 있습니다.",
                    )
                    break
                self.file_list.addItem(path)

        self.log(f"현재 선택된 파일 수: {self.file_list.count()}")

    def on_remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
        self.log(f"현재 선택된 파일 수: {self.file_list.count()}")

    def log(self, msg: str) -> None:
        self.text_log.append(msg)
        cursor = self.text_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_log.setTextCursor(cursor)

    # Run processing
    def on_run(self) -> None:
        count = self.file_list.count()
        if count == 0:
            QMessageBox.information(self, "알림", "먼저 TXT 파일을 하나 이상 선택하세요.")
            return

        config = Strategy2Config(
            FH=self.spin_fh.value(),
            UH=self.spin_uh.value(),
            SX=self.spin_sx.value(),
            SY=self.spin_sy.value(),
            PBL=self.spin_pbl.value(),
            vertical_angle_tol_deg=self.spin_ang_v.value(),
            horizontal_angle_tol_deg=self.spin_ang_h.value(),
            sample_step=self.spin_step.value(),
        )

        self.log(
            f"실행 시작: 파일 {count}개, "
            f"FH={config.FH}, UH={config.UH}, SX={config.SX}, SY={config.SY}, PBL={config.PBL}"
        )

        num_success = 0
        num_fail = 0

        for i in range(count):
            path = self.file_list.item(i).text()
            base, ext = os.path.splitext(path)
            out_txt = base + "_bending_points.txt"
            out_img = base + "_visualization.png"

            try:
                original_points = parse_txt_points(path)
                result = run_strategy2_on_file(path, config)
                save_result_points_txt(out_txt, result)
                visualize_result(original_points, result, out_img)
                self.log(f"[성공] {os.path.basename(path)} -> {os.path.basename(out_txt)}, {os.path.basename(out_img)}")
                num_success += 1
            except Strategy2Error as e:
                self.log(f"[실패] {os.path.basename(path)}: {e}")
                num_fail += 1
            except Exception as e:  # noqa: BLE001
                self.log(f"[오류] {os.path.basename(path)}: {e}")
                num_fail += 1

        QMessageBox.information(
            self,
            "완료",
            f"처리 완료.\n성공: {num_success}개\n실패: {num_fail}개",
        )
        self.log("=== 처리 완료 ===")

    def on_generate_example(self) -> None:
        """Generate a synthetic example TXT close to real edge shape."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "예시 TXT 저장 위치 선택",
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
            QMessageBox.critical(self, "예시 생성 실패", str(e))
            return

        self.log(f"예시 TXT 생성: {path}")
        # 방금 생성한 파일을 리스트에 자동 추가
        self.file_list.addItem(path)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

