# YOLO Model Evaluation Audit & Retraining Plan Evidence

> **Date:** September 2026  
> **Status:** Baseline Audited | Root Cause Identified | Retraining Planned  
> **Target Model:** `src/ml/yolo26m_v1/best.pt`
> **Evaluation Dataset:** `benchmark_results/dataset/20260824` (200 images, 10,000 ground truth objects)

---

## 1. Executive Summary

Trang tài liệu này lưu trữ toàn bộ **kết quả đánh giá thực tế**, **chẩn đoán nguyên nhân gốc rễ (Root Cause Analysis)** và **kế hoạch huấn luyện lại (Retraining Plan)** cho mô hình phát hiện vi nhựa YOLO.

Mục tiêu chính:
1. Ghi nhận sự khác biệt giữa kết quả benchmark cũ (lỗi công thức đếm số lượng, F1 = 0.984) và benchmark thực tế chuẩn không gian (IoU ≥ 0.50, F1 = 0.609).
2. Lưu lại bằng chứng thực nghiệm giải thích tại sao lớp **Fiber/Filament** chỉ đạt **F1 = 11.69%**, chứng minh mô hình không bị nhầm lẫn phân loại mà bị lỗi sai lệch kích thước bounding box từ dữ liệu train cũ.
3. Thiết lập lộ trình kỹ thuật chuẩn xác để huấn luyện lại mô hình đạt **F1 tổng thể ≥ 85%**.

---

## 2. Kết quả Đánh giá Hiện tại (Baseline Evaluation)

### 2.1. So sánh Tổng quan (Overall Metrics)

| Tiêu chí đánh giá | YOLO (Count-only cũ) | **YOLO (Chuẩn IoU ≥ 0.50)** | Quick Analysis (CV) | Deep Analysis (CV) |
|---|---:|---:|---:|---:|
| **True Positives (TP)** | 9,914 | **6,138** | 7,165 | 6,655 |
| **False Positives (FP)** | 243 | **4,019** | 1,031 | 1,687 |
| **False Negatives (FN)** | 86 | **3,862** | 2,835 | 3,345 |
| **Precision** | 0.9761 | **0.6043** | 0.8742 | 0.7978 |
| **Recall** | 0.9914 | **0.6138** | 0.7165 | 0.6655 |
| **F1-Score** | **0.9837** | **0.6090** | **0.7875** | **0.7257** |

*Điều kiện kiểm thử:*
- Dataset cố định: 200 ảnh huỳnh quang nhân tạo (`1280x1280`), 10,000 hạt ground truth (`SHA-256: d5d55d44aa9a8cff6f39a292a7f99e186887a15b8c0c5ac300399b8e5f4d5a23`).
- Ngưỡng tin cậy (Confidence Threshold): `0.25`.
- Ngưỡng IoU: `0.50` (1-to-1 greedy matching theo thứ tự confidence giảm dần).

### 2.2. Chi tiết theo từng lớp hạt (Per-Class Breakdown)

| Canonical Class | GT Count | Predictions | TP | FP | FN | Precision | Recall | **F1-Score** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Irregular** | 3,334 | 3,279 | 3,065 | 214 | 269 | 0.9347 | 0.9193 | **0.9270** |
| **Microbead/Pellet** | 3,302 | 3,413 | 2,674 | 739 | 628 | 0.7835 | 0.8098 | **0.7964** |
| **Fiber/Filament** | 3,364 | 3,465 | 399 | 3,066 | 2,965 | 0.1152 | 0.1186 | **0.1169** |

---

## 3. Bằng chứng Phân tích Thực nghiệm (Empirical Audit)

### 3.1. Nghịch lý của lớp Fiber/Filament

Số lượng dự đoán lớp `Fiber/Filament` của YOLO (3,465) rất sát với số lượng Ground Truth (3,364). Tuy nhiên, số lượng True Positive đạt chuẩn IoU ≥ 0.50 chỉ có **399 hạt**.

Qua kiểm tra từng cặp đối tượng trên toàn bộ 200 ảnh, kết quả thực nghiệm như sau:

1. **Khả năng định vị và phân loại thực tế:**
   - **99.3%** hạt Fiber GT (3,342 / 3,364) có ít nhất một bounding box dự đoán của YOLO bao phủ không gian (IoU > 0).
   - Trong số các dự đoán trùng khớp vị trí nhất với hạt Fiber GT:
     - **96.3% (3,240 hạt)** được YOLO gán đúng nhãn `Fiber/Filament`.
     - Chỉ có **2.2% (74 hạt)** bị nhầm thành `Irregular`.
     - Chỉ có **0.8% (28 hạt)** bị nhầm thành `Microbead/Pellet`.
   $\rightarrow$ **Kết luận 1:** Mô hình học đặc trưng hình thái (feature representation) của sợi rất tốt, hoàn toàn không bị nhầm lẫn class.

2. **Phân bố IoU giữa dự đoán và Ground Truth:**
   - IoU $\ge$ 0.10: **3,305 hạt (98.2%)**
   - IoU $\ge$ 0.20: **3,025 hạt (89.9%)**
   - IoU $\ge$ 0.30: **1,942 hạt (57.7%)**
   - IoU $\ge$ 0.40: **1,003 hạt (29.8%)**
   - IoU $\ge$ 0.50: **402 hạt (12.0%)** (chính sách greedy 1-1 ghi nhận 399 TP)

3. **Sai lệch kích thước hình học Bounding Box:**
   - Đối với 3,240 hạt được dự đoán đúng nhãn `Fiber/Filament`:
     - **Tỷ lệ diện tích (Pred Area / GT Area):** Median = **0.424** (Mean = 0.532).
     - **Tỷ lệ chiều rộng (Pred Width / GT Width):** Median = **0.446** (Mean = 0.619).
     - **Tỷ lệ chiều cao (Pred Height / GT Height):** Median = **0.751** (Mean = 1.088).
   $\rightarrow$ **Kết luận 2:** Hộp bao dự đoán của YOLO chỉ bằng **~42% diện tích thực tế** của hạt vi sợi trên ảnh. Vì hộp dự đoán nằm lọt thỏm bên trong hộp GT, tỷ lệ Intersection / Union chỉ dao động từ `0.25 - 0.40`, dẫn đến việc bị đánh trượt khi ngưỡng chuẩn là `0.50`.

---

## 4. Nguyên nhân Kỹ thuật Cốt lõi (Root Cause Identification)

### 4.1. Lỗi công thức xuất nhãn trong Pipeline sinh dữ liệu cũ

Trong file [`src/data_generation/yolo_exporter.py`](file:///Users/nghiant51/Fluorescence-based-Microplastic-analyzer/src/data_generation/yolo_exporter.py#L84-L92):

```python
# Đoạn code tạo bounding box cho nhãn train trong yolo_exporter.py cũ:
if 'Fiber' in shape_name or 'Filament' in shape_name:
    # Cố định giả định sợi luôn đứng thẳng: bề rộng hẹp, chiều dài dọc
    bbox_width = max(size_hint * 0.3, 8)  # Rất hẹp
    bbox_height = size_hint * 1.2
```

Trong khi trong hàm sinh hình thái hạt [`src/data_generation/synthetic_generator.py`](file:///Users/nghiant51/Fluorescence-based-Microplastic-analyzer/src/data_generation/synthetic_generator.py#L233):

```python
# Fiber xoay ngẫu nhiên mọi góc độ:
angle = random.uniform(0, 180)
rect = ((cx, cy), (length, width_f), angle)
box = cv2.boxPoints(rect).astype(np.int32)
```

**Hậu quả:**
- Khi sợi nằm nghiêng $45^\circ$ hoặc nằm ngang $90^\circ$, bounding box chuẩn `cv2.boundingRect` của nó là một hình chữ nhật lớn bao quanh toàn bộ đoạn nghiêng.
- Bộ exporter cũ lại xuất nhãn huấn luyện cho YOLO dưới dạng một khung chữ nhật thẳng đứng hẹp.
- YOLO học theo nhãn bị bóp méo này và luôn dự đoán bounding box sợi nhỏ hơn nhiều so với footprint thực tế trên ảnh kiểm thử.

---

## 5. Kế hoạch Huấn luyện lại Mô hình (Retraining Plan)

```
+-------------------------------------------------------------------------------+
|                        QUY TRÌNH RETRAIN CHUẨN HÓA                            |
+-------------------------------------------------------------------------------+
| 1. SỬA DATA PIPELINE                                                          |
|    - Dùng cv2.boundingRect(mask) trực tiếp, bỏ ước lượng size_hint * 0.3      |
|    - Chuẩn hóa 3 canonical classes: Microbead/Pellet, Fiber/Filament, Irregular|
|    - Cân đối vùng phát quang huỳnh quang (visible glow footprint)             |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| 2. SINH BỘ DỮ LIỆU MỚI (DATASET SYNTHESIS)                                    |
|    - 1,200 - 1,500 ảnh (1280x1280), ~60,000 particles                         |
|    - Phân chia: Train (70%), Val (15%), Test (15%) với seed độc lập           |
|    - Đa dạng hóa: Sợi uốn cong, góc xoay 0-360°, chạm hạt, nền nhiễu quang    |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| 3. HUẤN LUYỆN YOLO                                                            |
|    - Kiến trúc: YOLO26m (hoặc YOLO26m nếu cần tăng mAP)                       |
|    - imgsz: 1024 / 1280 (giữ nguyên độ phân giải để không mất sợi mỏng)       |
|    - Hyperparameters: epochs=120, patience=20, box=7.5, cls=0.5               |
|    - Augmentation: Mosaic, Mixup, Random Perspective, Scale                   |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| 4. BENCHMARK & KIỂM TRA CHẤP THUẬN                                            |
|    - Đo bằng src/analysis/benchmark_batch.py (IoU >= 0.50)                    |
|    - Tiêu chí: Fiber F1 >= 80%, Overall F1 >= 85% (vượt Quick Analysis 78.7%) |
|    - Cập nhật weights vào src/ml/Yolo26m_v2/best.pt                           |
+-------------------------------------------------------------------------------+
```

### Tiêu chí Nghiệm thu (Acceptance Criteria)

1. **Khắc phục triệt để Fiber/Filament:**
   - F1 của Fiber/Filament tăng từ **11.69%** lên **≥ 80.00%**.
   - Median tỷ lệ diện tích `Pred / GT Area` đạt từ `0.80 - 1.20`.
2. **Hiệu năng Tổng thể:**
   - Micro Precision $\ge$ 85%.
   - Micro Recall $\ge$ 85%.
   - Overall F1 $\ge$ 85% (vượt mốc 78.75% của Quick Analysis).
3. **Tính toàn vẹn & Tái lập (Reproducibility):**
   - Bộ trọng số mới `best.pt` vượt qua bộ test tự động của hệ thống (`pytest tests/`).
   - Có báo cáo HTML và sidecar `.benchmark.json` lưu giữ SHA-256 đầy đủ.

---

## 6. Danh mục Bằng chứng Hiện vật (Artifacts Reference)

- **Báo cáo chuẩn sau sửa:** [`benchmark_results/evidence/after_fix_20260825_231305/after_fix_final.html`](file:///Users/nghiant51/Fluorescence-based-Microplastic-analyzer/benchmark_results/evidence/after_fix_20260825_231305/after_fix_final.html)
- **Sidecar JSON kiểm toán:** [`benchmark_results/evidence/after_fix_20260825_231305/after_fix_final.benchmark.json`](file:///Users/nghiant51/Fluorescence-based-Microplastic-analyzer/benchmark_results/evidence/after_fix_20260825_231305/after_fix_final.benchmark.json)
- **Bảng số liệu so sánh:** [`benchmark_results/evidence/after_fix_20260825_231305/before_after_comparison.json`](file:///Users/nghiant51/Fluorescence-based-Microplastic-analyzer/benchmark_results/evidence/after_fix_20260825_231305/before_after_comparison.json)
- **Biểu đồ per-class:** [`benchmark_results/evidence/after_fix_20260825_231305/after_fix_ml_per_class.png`](file:///Users/nghiant51/Fluorescence-based-Microplastic-analyzer/benchmark_results/evidence/after_fix_20260825_231305/after_fix_ml_per_class.png)

---

## 7. Kết quả Nghiệm thu Thực tế Sau khi Retrain (Post-Retraining Validation)

> **Mô hình mới:** `src/ml/yolo26m_v2/best.pt`  
> **SHA-256:** `1481da60a6e7fa6d2823defb6b982aabe4f39c0f407ec18cb1e47545252a8b05`  
> **Kiểm định:** Chạy trực tiếp trên tập chuẩn 200 ảnh (`benchmark_results/dataset/20260824`, 10,000 GT hạt), IoU $\ge$ 0.50, confidence $\ge$ 0.25, 1-to-1 greedy matching.

### 7.1. Bảng So sánh Trước và Sau khi Retrain

| Lớp (Class) | F1 Trước Retrain (v1) | F1 Sau Retrain (v2) | Chênh lệch (Delta) | Trạng thái Nghiệm thu |
|:---|:---:|:---:|:---:|:---:|
| **Fiber/Filament** | **11.69%** | **97.39%** | **+85.70%** | **ĐẠT XUẤT SẮC** (Vượt mục tiêu $\ge$ 80%) |
| **Microbead/Pellet** | **79.64%** | **97.43%** | **+17.79%** | **ĐẠT XUẤT SẮC** |
| **Irregular** | **92.70%** | **99.33%** | **+6.63%** | **ĐẠT XUẤT SẮC** |
| **TỔNG THỂ (Overall)** | **60.90%** | **98.05%** | **+37.15%** | **ĐẠT XUẤT SẮC** (Vượt Quick Analysis 78.75%) |

### 7.2. Số liệu Chi tiết của Mô hình `yolo26m_v2`

| Lớp | GT | Dự đoán | TP | FP | FN | Precision | Recall | F1-Score |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Microbead/Pellet** | 3,302 | 3,341 | 3,236 | 105 | 66 | 96.86% | 98.00% | **97.43%** |
| **Fiber/Filament** | 3,364 | 3,390 | 3,289 | 101 | 75 | 97.02% | 97.77% | **97.39%** |
| **Irregular** | 3,334 | 3,345 | 3,317 | 28 | 17 | 99.16% | 99.49% | **99.33%** |
| **Toàn bộ Dataset** | **10,000** | **10,076** | **9,842** | **234** | **158** | **97.68%** | **98.42%** | **98.05%** |

**Kết luận nghiệm thu:** Tất cả các tiêu chuẩn nghiệm thu đề ra đều được hoàn thành vượt mức cam kết. Điểm nghẽn nghiêm trọng của lớp Fiber/Filament đã được giải quyết triệt để.
