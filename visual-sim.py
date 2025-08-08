import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QSlider, QLabel, QFileDialog,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsRectItem, QMessageBox) # Added QMessageBox
from PyQt5.QtCore import Qt, QRect, QPropertyAnimation, QEasingCurve, pyqtProperty, QPoint, QRectF, pyqtSignal
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QFont, QPalette,
                         QBrush, QLinearGradient)
import numpy as np
import cv2


class AnimatedButton(QPushButton):
    def __init__(self, text, button_type="normal"):
        super().__init__(text)
        self.button_type = button_type
        self._corner_opacity = 0
        self.setup_style()
        
        # Animation for corner effect
        self.animation = QPropertyAnimation(self, b"corner_opacity")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def setup_style(self):
        if self.button_type == "close":
            base_color = "rgba(180, 50, 50, 200)"
            hover_color = "rgba(200, 70, 70, 220)"
        elif self.button_type == "find":
            base_color = "rgba(50, 100, 180, 200)"
            hover_color = "rgba(70, 120, 200, 220)"
        else:
            base_color = "rgba(60, 60, 60, 200)"
            hover_color = "rgba(80, 80, 80, 220)"
            
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {base_color};
                border: 1px solid rgba(100, 100, 100, 100);
                border-radius: 0px;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-color: rgba(150, 150, 150, 150);
            }}
            QPushButton:pressed {{
                background-color: rgba(40, 40, 40, 240);
            }}
            QPushButton:disabled {{
                background-color: rgba(40, 40, 40, 100);
                color: rgba(150, 150, 150, 150);
            }}
        """)
    
    @pyqtProperty(float)
    def corner_opacity(self):
        return self._corner_opacity
    
    @corner_opacity.setter
    def corner_opacity(self, value):
        self._corner_opacity = value
        self.update()
    
    def enterEvent(self, event):
        if self.button_type == "normal" and self.isEnabled():
            self.animation.setStartValue(0)
            self.animation.setEndValue(1)
            self.animation.start()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        if self.button_type == "normal":
            self.animation.setStartValue(self._corner_opacity)
            self.animation.setEndValue(0)
            self.animation.start()
        super().leaveEvent(event)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.button_type == "normal" and self._corner_opacity > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw corner elbows
            pen = QPen(QColor(255, 255, 255, int(self._corner_opacity * 150)))
            pen.setWidth(2)
            painter.setPen(pen)
            
            rect = self.rect()
            corner_size = 8
            
            # Top-left corner
            painter.drawLine(rect.left() + 1, rect.top() + corner_size, 
                             rect.left() + 1, rect.top() + 1)
            painter.drawLine(rect.left() + 1, rect.top() + 1, 
                             rect.left() + corner_size, rect.top() + 1)
            
            # Top-right corner
            painter.drawLine(rect.right() - corner_size, rect.top() + 1, 
                             rect.right() - 1, rect.top() + 1)
            painter.drawLine(rect.right() - 1, rect.top() + 1, 
                             rect.right() - 1, rect.top() + corner_size)
            
            # Bottom-left corner
            painter.drawLine(rect.left() + 1, rect.bottom() - corner_size, 
                             rect.left() + 1, rect.bottom() - 1)
            painter.drawLine(rect.left() + 1, rect.bottom() - 1, 
                             rect.left() + corner_size, rect.bottom() - 1)
            
            # Bottom-right corner
            painter.drawLine(rect.right() - corner_size, rect.bottom() - 1, 
                             rect.right() - 1, rect.bottom() - 1)
            painter.drawLine(rect.right() - 1, rect.bottom() - 1, 
                             rect.right() - 1, rect.bottom() - corner_size)


class ImageDisplayWidget(QGraphicsView):
    anchorBoxDrawn = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        self.pixmap_item = None
        self.current_rect_item = None # Renamed to avoid confusion with QRect
        self.drawing = False
        self.start_point = QPoint()
        self.anchor_rect_items = [] # Renamed to store QGraphicsRectItem
        
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setStyleSheet("""
            QGraphicsView {
                background-color: rgba(30, 30, 30, 200);
                border: 2px solid rgba(70, 70, 70, 150);
                border-radius: 8px;
            }
        """)
        
        # Initial message
        self.show_message("Please load an image to begin.")
        
    def show_message(self, message):
        self.scene.clear()
        self.pixmap_item = None
        text_item = self.scene.addText(message, QFont("Arial", 14))
        text_item.setDefaultTextColor(QColor(150, 150, 150))
        # Center the text
        text_item_rect = text_item.boundingRect()
        view_rect = self.viewport().rect()
        text_item.setPos((view_rect.width() - text_item_rect.width()) / 2,
                         (view_rect.height() - text_item_rect.height()) / 2)
        
    def load_image(self, image_path):
        self.scene.clear()
        self.anchor_rect_items = [] # Clear previous anchor boxes
        
        pixmap = QPixmap(image_path)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(QRectF(pixmap.rect())) # Important: scene rect matches image size
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap_item:
            scene_pos = self.mapToScene(event.pos())
            # Check if click is within the image bounds (pixmap_item)
            if self.pixmap_item.contains(scene_pos):
                self.drawing = True
                self.start_point = scene_pos.toPoint() # QPointF to QPoint
                self.current_rect_item = QGraphicsRectItem()
                # Anchor box color
                self.current_rect_item.setPen(QPen(QColor(246, 130, 59), 3)) # Orange color for anchor
                self.scene.addItem(self.current_rect_item)
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.drawing and self.current_rect_item:
            scene_pos = self.mapToScene(event.pos())
            # Ensure the drawing rectangle stays within image bounds
            image_rect = self.pixmap_item.boundingRect()
            clamped_scene_pos = QPoint(
                int(max(image_rect.left(), min(scene_pos.x(), image_rect.right()))),
                int(max(image_rect.top(), min(scene_pos.y(), image_rect.bottom())))
            )
            rect = QRect(self.start_point, clamped_scene_pos).normalized()
            self.current_rect_item.setRect(QRectF(rect)) # Convert QRect to QRectF
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.drawing and self.current_rect_item:
            self.drawing = False
            rect = self.current_rect_item.rect()
            # A minimum size for the drawn box
            if rect.width() > 10 and rect.height() > 10:
                self.anchor_rect_items.append(self.current_rect_item)
                self.anchorBoxDrawn.emit() 
            else:
                self.scene.removeItem(self.current_rect_item)
            self.current_rect_item = None
        
        super().mouseReleaseEvent(event)
    
    def clear_results(self):
        # Remove all similarity result rectangles (green ones)
        items_to_remove = []
        for item in self.scene.items():
            # Check if it's a QGraphicsRectItem and not one of the anchor_rect_items
            if isinstance(item, QGraphicsRectItem) and item not in self.anchor_rect_items:
                # Assuming green color for similarity results
                if item.pen().color() == QColor(74, 163, 22): # Specific green used in drawing results
                    items_to_remove.append(item)
        
        for item in items_to_remove:
            self.scene.removeItem(item)
    
    def add_similarity_result(self, x, y, w, h, score):
        # Create QRect from (x, y, w, h) for drawing
        result_rect_q = QRectF(float(x), float(y), float(w), float(h))
        result_rect_item = QGraphicsRectItem(result_rect_q)
        result_rect_item.setPen(QPen(QColor(74, 163, 22), 3)) # Green color for results

        self.scene.addItem(result_rect_item)
        
        # Add score text
        text_item = self.scene.addText(f'{score:.2f}', QFont("Arial", 10, QFont.Bold))
        text_item.setDefaultTextColor(QColor(74, 163, 22)) # Green text
        text_item.setPos(float(x), float(y) - 20) # Position above the box
        # Store a reference to the text item so it can be cleared with the box
        # For now, we'll rely on clearing all non-anchor rects, which should implicitly clear texts too if we filter them.
        # A more robust solution might link text to its box.
        
    def wheelEvent(self, event):
        # Zoom functionality
        zoom_factor = 1.15
        if event.angleDelta().y() < 0:
            zoom_factor = 1.0 / zoom_factor
        
        self.scale(zoom_factor, zoom_factor)


class ImageSimilarityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.current_image = None # This will store the OpenCV image (numpy array)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Image Similarity Search")
        self.setGeometry(100, 100, 1000, 700)
        
        # Set window attributes for translucency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # Main widget
        main_widget = QWidget()
        main_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(20, 20, 20, 220);
                border-radius: 12px;
            }
        """)
        self.setCentralWidget(main_widget)
        
        # Main layout
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title bar
        title_layout = QHBoxLayout()
        title_label = QLabel("Image Similarity Search")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
                background-color: transparent;
            }
        """)
        
        close_btn = AnimatedButton("×", "close")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.close)
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(close_btn)
        layout.addLayout(title_layout)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Load button
        self.btn_load = AnimatedButton("1. Load Image")
        self.btn_load.clicked.connect(self.load_image)
        control_layout.addWidget(self.btn_load)
        
        # Find similar button
        self.btn_find = AnimatedButton("2. Find Similar Objects", "find")
        self.btn_find.setEnabled(False)
        self.btn_find.clicked.connect(self.find_similar)
        control_layout.addWidget(self.btn_find)
        
        # Threshold slider
        threshold_layout = QVBoxLayout()
        self.threshold_label = QLabel("Similarity Threshold: 0.75")
        self.threshold_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 12px;
                background-color: transparent;
            }
        """)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100) # Corresponds to 0.50 to 1.00
        self.threshold_slider.setValue(75)
        self.threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid rgba(100, 100, 100, 100);
                height: 6px;
                background-color: rgba(50, 50, 50, 150);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: rgba(70, 130, 200, 200);
                border: 1px solid rgba(70, 130, 200, 255);
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background-color: rgba(90, 150, 220, 240);
            }
        """)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        
        control_layout.addStretch()
        control_layout.addLayout(threshold_layout)
        layout.addLayout(control_layout)
        
        # Image display
        self.image_display = ImageDisplayWidget()
        layout.addWidget(self.image_display)

        self.image_display.anchorBoxDrawn.connect(self.update_find_button_state)
        
        # Make window draggable
        self.old_pos = self.pos()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = QPoint(event.globalPos() - self.old_pos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPos()
    
    def update_threshold_label(self):
        value = self.threshold_slider.value() / 100.0
        self.threshold_label.setText(f"Similarity Threshold: {value:.2f}")
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.TIF *.TIFF)"
        )
        
        if file_path:
            self.current_image_path = file_path
            # Load image using OpenCV
            self.current_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.current_image.dtype != np.uint8:
                self.current_image = cv2.normalize(self.current_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            if self.current_image.ndim == 3:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

            if self.current_image is None:
                QMessageBox.critical(self, "Error", "Failed to load image with OpenCV. Check file path or format.")
                self.current_image_path = None
                self.current_image = None
                return
            
            # Clear existing anchor boxes as a new image is loaded
            self.image_display.anchor_rect_items = []
            self.image_display.load_image(file_path)
            self.update_find_button_state()
    
    def update_find_button_state(self):
        # Enable find button if image is loaded AND at least one anchor box exists
        has_image = self.current_image is not None
        has_anchor_boxes = len(self.image_display.anchor_rect_items) > 0
        self.btn_find.setEnabled(has_image and has_anchor_boxes)
    
    def find_similar(self):
        if self.current_image is None or not self.image_display.anchor_rect_items:
            QMessageBox.warning(self, "Warning", "Please load an image and draw at least one anchor box.")
            return
        
        # Clear previous results (green boxes) but keep anchor boxes
        self.image_display.clear_results()
        
        threshold = self.threshold_slider.value() / 100.0
        
        all_similar_boxes = [] # To collect results from all anchor boxes
        
        for anchor_rect_item in self.image_display.anchor_rect_items:
            # Get QRect from QGraphicsRectItem
            anchor_qrect = anchor_rect_item.rect()
            
            # Convert QRect to (x, y, w, h) for OpenCV processing
            x, y, w, h = int(anchor_qrect.x()), int(anchor_qrect.y()), int(anchor_qrect.width()), int(anchor_qrect.height())
            
            # Ensure anchor ROI is within image bounds (important for robustness)
            img_h, img_w = self.current_image.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            if w <= 0 or h <= 0:
                QMessageBox.warning(self, "Invalid Anchor Box", "Drawn anchor box is too small or invalid. Please draw a larger box.")
                continue

            anchor_roi = self.current_image[y : y + h, x : x + w]
            
            # 1. Feature Extraction (Anchor)
            anchor_feature = self.extract_features(anchor_roi)

            # 2. Proposal Generation (Sliding Window)
            proposals = self.generate_proposals(self.current_image, w, h)
            
            # List to store similar boxes for this anchor
            similar_boxes_for_anchor = []

            for (px, py, pw, ph) in proposals:
                # Ensure proposal ROI is valid
                if pw <= 0 or ph <= 0 or py + ph > img_h or px + pw > img_w:
                    continue # Skip invalid proposals

                proposal_roi = self.current_image[py : py + ph, px : px + pw]
                proposal_feature = self.extract_features(proposal_roi)
                
                # 3. Comparison
                similarity = self.calculate_similarity(anchor_feature, proposal_feature)
                
                if similarity >= threshold:
                    similar_boxes_for_anchor.append({'box': [px, py, pw, ph], 'score': similarity})
            
            # Add the anchor box itself to the list of candidates for NMS
            # This helps avoid finding the anchor box as a "new" match,
            # but it will be filtered by NMS if another box is too similar.
            # A better approach might be to exclude the exact anchor box during drawing.
            # For simplicity, we include it and let NMS handle it.
            # However, for a clearer UI, we only want to show *new* matches.
            # So, we'll apply NMS and then exclude the anchor box's *exact* coordinates.
            
            # 4. Non-Maximum Suppression for results from THIS anchor
            # Note: We collect all matches from all anchor boxes and then apply NMS globally.
            # This prevents duplicate matches if multiple anchor boxes find the same object.
            # But let's apply NMS per anchor first for simpler logic, then a final NMS.
            
            # Let's collect all candidate boxes from ALL anchors first, then apply one global NMS.
            all_similar_boxes.extend(similar_boxes_for_anchor)

        if not all_similar_boxes:
            QMessageBox.information(self, "No Matches", "No similar objects found with the current threshold.")
            self.btn_find.setEnabled(False) # Re-enable when a new box is drawn
            return
            
        # Global NMS across all collected similar boxes
        final_boxes = self.non_max_suppression(all_similar_boxes, iou_threshold=0.2)
        
        # Filter out the original anchor boxes from final_boxes if they are too close to matches
        # This is a heuristic. A robust solution might use IoU with anchor_box_coords to exclude.
        
        # 5. Draw results on the image_display
        found_matches_count = 0
        for item in final_boxes:
            bx, by, bw, bh = item['box']
            score = item['score']

            # Check if this result box is *very* similar to any of the anchor boxes
            # This is to prevent drawing the anchor box itself as a "match"
            is_anchor = False
            for anchor_item in self.image_display.anchor_rect_items:
                anchor_x, anchor_y, anchor_w, anchor_h = int(anchor_item.rect().x()), int(anchor_item.rect().y()), \
                                                         int(anchor_item.rect().width()), int(anchor_item.rect().height())
                
                # Simple check for very close match (can be improved with IoU)
                if abs(bx - anchor_x) < 5 and abs(by - anchor_y) < 5 and \
                   abs(bw - anchor_w) < 5 and abs(bh - anchor_h) < 5:
                    is_anchor = True
                    break
            
            if not is_anchor: # Only draw if it's not the anchor itself
                self.image_display.add_similarity_result(bx, by, bw, bh, score)
                found_matches_count += 1

        if found_matches_count == 0:
            QMessageBox.information(self, "No Matches", "No *new* similar objects found with the current threshold after filtering anchor boxes.")

        self.btn_find.setEnabled(False) # Disable until a new box is drawn or image is loaded

    # --- Core OpenCV Pipeline Methods ---

    def extract_features(self, roi):
        """Extracts a 3D color histogram as a feature vector."""
        # Check if ROI is empty or invalid
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            # Return an empty array or handle error appropriately
            return np.zeros((8*8*8), dtype=np.float32) # Return a zero-filled histogram

        # Convert to HSV color space, which is often better for color-based features
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Calculate histogram across all 3 channels
        # [8, 8, 8] bins for H, S, V respectively
        # [0, 180, 0, 256, 0, 256] ranges for H, S, V respectively
        hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        # Normalize and flatten the histogram
        cv2.normalize(hist, hist).flatten()
        return hist.flatten()

    def calculate_similarity(self, hist1, hist2):
        """Calculates similarity using histogram comparison (Correlation)."""
        # Ensure histograms are not empty
        if hist1.size == 0 or hist2.size == 0:
            return 0.0 # Return 0 similarity if either histogram is invalid
        # OpenCV's compareHist is efficient for this
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def generate_proposals(self, image, anchor_w, anchor_h):
        """Generates proposal boxes using a sliding window."""
        (img_h, img_w) = image.shape[:2]
        proposals = []
        
        # Define different window sizes relative to the anchor
        scales = [0.8, 1.0, 1.2] # Smaller, same, larger than anchor
        # scales = [1.0] # same size as the current anchor box
        
        for scale in scales:
            win_w = int(anchor_w * scale)
            win_h = int(anchor_h * scale)
            
            # Ensure window size is at least 1x1
            if win_w < 1: win_w = 1
            if win_h < 1: win_h = 1

            # Define step size (stride) - Adjust based on desired density
            # A smaller step size means more proposals but slower processing.
            # Let's make it relative to the window size, e.g., 25% of the smaller dimension
            step_x = max(1, int(win_w * 0.2)) # At least 1 pixel step
            step_y = max(1, int(win_h * 0.2))

            for y in range(0, img_h - win_h + 1, step_y): # +1 to include the last possible step
                for x in range(0, img_w - win_w + 1, step_x):
                    proposals.append((x, y, win_w, win_h))
        return proposals

    def non_max_suppression(self, boxes_with_scores, iou_threshold):
        """Filters overlapping boxes based on score and IoU."""
        if not boxes_with_scores:
            return []

        # Extract boxes and scores
        boxes = np.array([d['box'] for d in boxes_with_scores])
        scores = np.array([d['score'] for d in boxes_with_scores])

        # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by score (descending)
        order = scores.argsort()[::-1]
        
        keep_indices = []
        while order.size > 0:
            i = order[0]
            keep_indices.append(i)
            
            if order.size == 1: # Only one box left
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w_inter = np.maximum(0.0, xx2 - xx1 + 1)
            h_inter = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w_inter * h_inter
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU less than the threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1] # +1 because order[1:] is a slice starting from index 1

        return [boxes_with_scores[i] for i in keep_indices]


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application palette for dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = ImageSimilarityApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()