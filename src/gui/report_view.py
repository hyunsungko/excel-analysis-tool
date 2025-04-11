import os
from typing import Optional, List, Dict, Any, Union
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QTabWidget, QTextEdit, QFileDialog, QMessageBox,
    QTreeView, QScrollArea, QFrame, QProgressBar, QDialog,
    QFormLayout, QLineEdit, QDialogButtonBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont

class ReportSettingsDialog(QDialog):
    """리포트 설정 다이얼로그"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("리포트 설정")
        self.setMinimumWidth(400)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # 리포트 제목
        self.title_edit = QLineEdit()
        self.title_edit.setText("데이터 분석 리포트")
        form_layout.addRow("리포트 제목:", self.title_edit)
        
        # 리포트 하위 제목
        self.subtitle_edit = QLineEdit()
        self.subtitle_edit.setText("자동 생성된 데이터 분석 결과")
        form_layout.addRow("하위 제목:", self.subtitle_edit)
        
        # 저자 정보
        self.author_edit = QLineEdit()
        form_layout.addRow("저자:", self.author_edit)
        
        # 포함 항목
        include_layout = QVBoxLayout()
        
        self.include_toc = QCheckBox("목차 포함")
        self.include_toc.setChecked(True)
        include_layout.addWidget(self.include_toc)
        
        self.include_summary = QCheckBox("요약 정보 포함")
        self.include_summary.setChecked(True)
        include_layout.addWidget(self.include_summary)
        
        self.include_stats = QCheckBox("기술 통계 포함")
        self.include_stats.setChecked(True)
        include_layout.addWidget(self.include_stats)
        
        self.include_corr = QCheckBox("상관관계 분석 포함")
        self.include_corr.setChecked(True)
        include_layout.addWidget(self.include_corr)
        
        self.include_viz = QCheckBox("시각화 포함")
        self.include_viz.setChecked(True)
        include_layout.addWidget(self.include_viz)
        
        form_layout.addRow("포함 항목:", include_layout)
        
        # 리포트 형식
        self.format_combo = QComboBox()
        self.format_combo.addItems(["HTML", "PDF", "TXT"])
        form_layout.addRow("리포트 형식:", self.format_combo)
        
        layout.addLayout(form_layout)
        
        # 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def getSettings(self) -> Dict[str, Any]:
        """설정 값 반환"""
        return {
            "title": self.title_edit.text(),
            "subtitle": self.subtitle_edit.text(),
            "author": self.author_edit.text(),
            "include_toc": self.include_toc.isChecked(),
            "include_summary": self.include_summary.isChecked(),
            "include_stats": self.include_stats.isChecked(),
            "include_corr": self.include_corr.isChecked(),
            "include_viz": self.include_viz.isChecked(),
            "format": self.format_combo.currentText().lower()
        }

class ReportGeneratorThread(QThread):
    """리포트 생성을 위한 백그라운드 스레드"""
    progress_updated = pyqtSignal(int)
    finished_with_result = pyqtSignal(str, bool)
    
    def __init__(self, report_engine, settings, parent=None):
        super().__init__(parent)
        self.report_engine = report_engine
        self.settings = settings
        self.output_path = None
        
    def run(self):
        try:
            # 기본 설정
            self.report_engine.set_title(self.settings["title"])
            if self.settings.get("subtitle"):
                self.report_engine.set_subtitle(self.settings["subtitle"])
            if self.settings.get("author"):
                self.report_engine.set_author(self.settings["author"])
                
            # 진행 상황 업데이트
            self.progress_updated.emit(10)
            
            # 포함 항목에 따라 추가
            # 진행률 계산을 위한 전체 단계 수
            total_steps = 1
            current_step = 1
            
            # 필요한 포함 항목 카운트
            if self.settings.get("include_summary", True):
                total_steps += 1
            if self.settings.get("include_stats", True):
                total_steps += 1
            if self.settings.get("include_corr", True):
                total_steps += 1
            if self.settings.get("include_viz", True):
                total_steps += 1
                
            # 항목 추가
            try:
                if self.settings.get("include_summary", True):
                    self.report_engine.add_summary_section()
                    self.progress_updated.emit(int(current_step / total_steps * 100))
                    current_step += 1
                    
                if self.settings.get("include_stats", True):
                    self.report_engine.add_statistics_section()
                    self.progress_updated.emit(int(current_step / total_steps * 100))
                    current_step += 1
                    
                if self.settings.get("include_corr", True):
                    self.report_engine.add_correlation_section()
                    self.progress_updated.emit(int(current_step / total_steps * 100))
                    current_step += 1
                    
                if self.settings.get("include_viz", True):
                    self.report_engine.add_visualizations_section()
                    self.progress_updated.emit(int(current_step / total_steps * 100))
                    current_step += 1
            except Exception as e:
                self.finished_with_result.emit(f"리포트 생성 중 오류: {str(e)}", False)
                return
                
            # 리포트 생성
            format_type = self.settings.get("format", "html")
            
            try:
                output_path = self.report_engine.export_report(format_type=format_type)
                self.output_path = output_path
                self.progress_updated.emit(100)
                self.finished_with_result.emit(output_path, True)
            except Exception as e:
                self.finished_with_result.emit(f"리포트 생성 중 오류: {str(e)}", False)
                
        except Exception as e:
            self.finished_with_result.emit(f"리포트 생성 중 예기치 않은 오류: {str(e)}", False)

class ReportView(QWidget):
    """리포트 뷰 위젯"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report_engine = None
        self.report_integrator = None
        self.data_loader = None
        self.processor = None 
        self.analysis_engine = None
        self.visualization_engine = None
        self.generated_reports = []
        self.initUI()
        
    def initUI(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 상단 도구 모음
        top_layout = QHBoxLayout()
        
        self.generate_button = QPushButton("새 리포트 생성")
        self.generate_button.clicked.connect(self.generateReport)
        top_layout.addWidget(self.generate_button)
        
        self.view_button = QPushButton("파일로 열기")
        self.view_button.clicked.connect(self.openReportFile)
        self.view_button.setEnabled(False)
        top_layout.addWidget(self.view_button)
        
        top_layout.addStretch()
        
        self.refresh_button = QPushButton("새로고침")
        self.refresh_button.clicked.connect(self.refreshReportList)
        top_layout.addWidget(self.refresh_button)
        
        layout.addLayout(top_layout)
        
        # 리포트 생성 상태 표시
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("상태:"))
        
        self.status_label = QLabel("준비됨")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # 내용 영역 - 분할 레이아웃
        content_layout = QHBoxLayout()
        
        # 왼쪽 패널 - 리포트 목록
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        left_layout.addWidget(QLabel("생성된 리포트"))
        
        self.report_tree = QTreeView()
        self.report_tree.setHeaderHidden(True)
        self.tree_model = QStandardItemModel()
        self.report_tree.setModel(self.tree_model)
        self.report_tree.clicked.connect(self.onReportSelected)
        left_layout.addWidget(self.report_tree)
        
        # 오른쪽 패널 - 리포트 미리보기
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_layout.addWidget(QLabel("리포트 미리보기"))
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        right_layout.addWidget(self.preview_text)
        
        # 레이아웃에 패널 추가
        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(right_panel, 2)
        
        layout.addLayout(content_layout)
        
    def setComponents(self, report_engine=None, report_integrator=None, 
                     data_loader=None, processor=None, 
                     analysis_engine=None, visualization_engine=None):
        """구성 요소 설정"""
        self.report_engine = report_engine
        self.report_integrator = report_integrator
        self.data_loader = data_loader
        self.processor = processor
        self.analysis_engine = analysis_engine
        self.visualization_engine = visualization_engine
        
        self.refreshReportList()
        
    def refreshReportList(self):
        """생성된 리포트 목록 새로고침"""
        self.tree_model.clear()
        root = self.tree_model.invisibleRootItem()
        
        # 기본 리포트 디렉토리
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            # 디렉토리가 없는 경우 생성
            try:
                os.makedirs(reports_dir)
            except Exception as e:
                return
                
        # 리포트 파일 검색
        self.generated_reports = []
        formats = [".html", ".pdf", ".txt"]
        
        for file in os.listdir(reports_dir):
            file_path = os.path.join(reports_dir, file)
            if os.path.isfile(file_path) and any(file.endswith(ext) for ext in formats):
                self.generated_reports.append(file_path)
                
                # 확장자에 따라 그룹화
                ext = os.path.splitext(file)[1].lower()
                format_item = None
                
                # 해당 형식의 그룹 항목 찾기
                for i in range(root.rowCount()):
                    item = root.child(i)
                    if item and item.text() == ext[1:].upper():
                        format_item = item
                        break
                        
                # 그룹 항목이 없으면 생성
                if not format_item:
                    format_item = QStandardItem(ext[1:].upper())
                    root.appendRow(format_item)
                    
                # 파일 항목 추가
                file_item = QStandardItem(os.path.basename(file))
                file_item.setData(file_path, Qt.UserRole)
                format_item.appendRow(file_item)
                
        # 트리 확장
        self.report_tree.expandAll()
        
        # 버튼 활성화 여부 설정
        self.view_button.setEnabled(len(self.generated_reports) > 0)
        
    def onReportSelected(self, index):
        """리포트 선택 시 동작"""
        item = self.tree_model.itemFromIndex(index)
        if not item:
            return
            
        # 파일 경로 가져오기
        file_path = item.data(Qt.UserRole)
        if not file_path or not os.path.isfile(file_path):
            return
            
        # 파일 형식에 따라 미리보기
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.html':
                # HTML 파일 내용 표시
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    self.preview_text.setHtml(html_content)
            elif ext == '.txt':
                # 텍스트 파일 내용 표시
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                    self.preview_text.setPlainText(text_content)
            elif ext == '.pdf':
                # PDF는 미리보기 제한
                self.preview_text.setPlainText("PDF 형식은 외부 뷰어로 확인하세요.\n\n파일 경로: " + file_path)
        except Exception as e:
            self.preview_text.setPlainText(f"파일 미리보기 중 오류: {str(e)}")
            
    def generateReport(self):
        """새 리포트 생성"""
        if not self.report_engine:
            QMessageBox.warning(self, "오류", "리포트 엔진이 설정되지 않았습니다.")
            return
            
        # 리포트 설정 다이얼로그 표시
        dialog = ReportSettingsDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
            
        settings = dialog.getSettings()
        
        # 리포트 통합기 설정
        if self.report_integrator:
            self.report_integrator.set_components(
                data_loader=self.data_loader,
                processor=self.processor,
                analysis_engine=self.analysis_engine,
                visualization_engine=self.visualization_engine,
                report_engine=self.report_engine
            )
            
            # 리포트 생성 프로세스 시작
            self.status_label.setText("리포트 생성 중...")
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # 백그라운드 스레드에서 처리
            self.report_thread = ReportGeneratorThread(self.report_engine, settings, self)
            self.report_thread.progress_updated.connect(self.updateProgressBar)
            self.report_thread.finished_with_result.connect(self.onReportGenerated)
            self.report_thread.start()
        else:
            QMessageBox.warning(self, "오류", "리포트 통합기가 설정되지 않았습니다.")
            
    def updateProgressBar(self, value):
        """진행률 업데이트"""
        self.progress_bar.setValue(value)
        
    def onReportGenerated(self, result, success):
        """리포트 생성 완료 처리"""
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_label.setText(f"리포트 생성 완료: {os.path.basename(result)}")
            self.refreshReportList()
            
            # 생성된 리포트 찾기 및 선택
            for i in range(self.tree_model.rowCount()):
                format_item = self.tree_model.item(i)
                if format_item:
                    for j in range(format_item.rowCount()):
                        file_item = format_item.child(j)
                        if file_item and file_item.data(Qt.UserRole) == result:
                            index = file_item.index()
                            self.report_tree.setCurrentIndex(index)
                            self.onReportSelected(index)
                            break
        else:
            self.status_label.setText("리포트 생성 실패")
            QMessageBox.warning(self, "오류", result)
            
    def openReportFile(self):
        """선택된 리포트 파일 외부 프로그램으로 열기"""
        # 현재 선택된 항목
        indexes = self.report_tree.selectedIndexes()
        if not indexes:
            return
            
        item = self.tree_model.itemFromIndex(indexes[0])
        if not item:
            return
            
        # 파일 경로 가져오기
        file_path = item.data(Qt.UserRole)
        if not file_path or not os.path.isfile(file_path):
            return
            
        # Windows 시스템 기본 프로그램으로 파일 열기
        try:
            os.startfile(file_path)
        except Exception as e:
            QMessageBox.warning(self, "오류", f"파일을 열 수 없습니다: {str(e)}")
            logger.error(f"파일 열기 오류: {str(e)}") 