# MVDet + Pseudo Label khi thiếu nhãn (WildTrack drop60): toàn bộ quá trình và kết quả

> **Branch này (`pseudo-label-clean`)** là bản **refactor tối giản** trên code gốc
> [hou-yz/MVDet](https://github.com/hou-yz/MVDet), commit `8b4792f`. Nó chỉ chứa những thay đổi **cần thiết** để
> (1) chạy được trên Kaggle hoặc Python mới và (2) hiện thực phương pháp pseudo label. Toàn bộ thí nghiệm trong tài
> liệu này được chạy bằng branch `pseudo-label`, vốn còn chứa các hướng đã thử và bỏ (loss `mal`, `confuse`,
> `hetero`). Phần pseudo label của hai branch **giống hệt nhau về hành vi**: cùng file loss, cùng tool, cùng flag, cùng
> giá trị mặc định (mục 1.3).
>
> **Kết quả chính:** với WildTrack chỉ còn **40% nhãn** (drop60), MVDet gốc đạt **MODA 22.7%**. Thêm pseudo label từ
> một detector 2D có sẵn, cộng hai bộ lọc (bỏ vùng mép, đồng thuận ≥ 3 camera) và tầng "bỏ qua", MODA lên
> **81.3%** (cấu hình **S2b**). Mốc tham chiếu khi có đủ 100% nhãn là 87.8%. S2b lấy lại **90% khoảng cách**
> giữa hai mốc đó.

---

## Mục lục

0. [Tóm tắt kết quả](#0-tóm-tắt-kết-quả)
1. [Branch này thay đổi gì so với code gốc](#1-branch-này-thay-đổi-gì-so-với-code-gốc)
2. [Bài toán và dữ liệu](#2-bài-toán-và-dữ-liệu)
3. [Ý tưởng](#3-ý-tưởng)
4. [Sinh pseudo label](#4-sinh-pseudo-label)
5. [Hàm loss](#5-hàm-loss)
6. [Kiểm chứng trên máy local](#6-kiểm-chứng-trên-máy-local)
7. [Cách chạy](#7-cách-chạy)
8. [Toàn bộ thí nghiệm và kết quả](#8-toàn-bộ-thí-nghiệm-và-kết-quả)
9. [Kết luận và mức độ chắc chắn](#9-kết-luận-và-mức-độ-chắc-chắn)
10. [Hạn chế và hướng tiếp theo](#10-hạn-chế-và-hướng-tiếp-theo)
11. [Thuật ngữ](#11-thuật-ngữ)

---

## 0. Tóm tắt kết quả

Kết quả trên WildTrack, tập test gồm 40 frame cuối với **nhãn đầy đủ**. Tất cả model train 10 epoch, seed 1,
`cls_thres` 0.4 (mặc định của MVDet, không chỉnh trên tập test).

| Model | Nhãn khi train | MODA | MODP | Precision | Recall | Ghi chú |
|---|---|---|---|---|---|---|
| **Full nhãn** (E0) | 100% nhãn | **87.8%** | 75.9% | 93.1% | 94.9% | cận trên |
| **Baseline drop60** (E1) | 40% nhãn | **22.7%** | 74.1% | 98.6% | 23.0% | MVDet gốc, cận dưới |
| E3 | 40% + pseudo ≥ 2 view, `point` r=4 | 64.0% | 74.8% | 78.1% | 89.0% | session 1 |
| E2 | 40% + pseudo ≥ 2 view, `gauss` r=0 | 69.2% | 73.6% | 81.6% | 89.4% | session 1 |
| E4b | 40% + pseudo ≥ 2 view, `gauss` r=4 | 69.3% | 73.4% | 86.1% | 82.7% | session 1 |
| E4mil | 40% + pseudo ≥ 2 view, `mil` r=4 | 70.4% | 74.4% | 82.5% | 89.4% | tốt nhất session 1 |
| S2a | E4mil + bỏ pseudo sát mép | 79.5% | 73.5% | 92.5% | 86.6% | session 2 |
| **S2b** | **bỏ mép + ≥ 3 view dương + còn lại "bỏ qua"** | **81.3%** | **74.4%** | **94.3%** | **86.6%** | **tốt nhất** |

Tỉ lệ lấy lại khoảng cách giữa baseline và full nhãn, tính bằng (MODA − 22.7) / (87.8 − 22.7):

| E4mil | S2a | S2b |
|---|---|---|
| 73% | 87% | **90%** |

---

## 1. Branch này thay đổi gì so với code gốc

### 1.1. Nguyên tắc

- Tách từ đúng commit mới nhất của `hou-yz/MVDet` (`8b4792f`).
- **Không đổi** kiến trúc model, optimizer, lịch learning rate, cách suy luận (ngưỡng → NMS), cách đánh giá.
- Với `--loss mse` (mặc định), chương trình chạy **giống hệt code gốc**. Log chỉ nằm thêm một cấp thư mục `mse/`.
- Không giữ các hướng đã thử và bỏ (loss `mal`, `confuse`, `hetero`), không giữ tính năng tự dò đường dẫn Kaggle, NaN
  guard hay log VRAM. Những thứ đó vẫn còn ở branch `pseudo-label`.

### 1.2. Danh sách thay đổi

**(a) Sửa tương thích.** Không có các sửa này thì không chạy được trên Kaggle, Colab hoặc Python mới:

| File | Thay đổi | Lý do |
|---|---|---|
| `main.py` | Bỏ `from distutils.dir_util import copy_tree`, dùng `shutil.copytree(..., dirs_exist_ok=True)` | `distutils` đã bị xóa khỏi Python 3.12 (Kaggle hiện dùng 3.12) |
| `models/persp_trans_detector.py` | Thiết bị: `cuda:1` + `cuda:0` nếu có ≥ 2 GPU, chỉ `cuda:0` nếu có 1 GPU, `cpu` nếu không có GPU | Code gốc **cứng** 2 GPU (`cuda:1`, `cuda:0`). Đây là cách chia bộ nhớ theo phần cứng của tác giả, không phải yêu cầu của thuật toán. Cần để: chạy trên máy 1 GPU; chạy **2 run song song, mỗi run 1 GPU** trên Kaggle T4 x2; chạy test CPU trên máy local |
| `evaluation/pyeval/evaluateDetection.py` | Nếu file kết quả chỉ có 1 dòng thì reshape lại thành 2 chiều | `np.loadtxt` biến file 1 dòng thành mảng 1 chiều, làm evaluator crash (`IndexError`). Xảy ra khi sweep ngưỡng cao và chỉ còn 1 detection |
| `datasets/frameDataset.py` | `gt.txt` được ghi vào `$MVDET_CACHE_DIR/<dataset>/` nếu biến môi trường này được đặt | Hai run chạy song song cùng ghi đè `gt.txt` (file dùng để tính MODA) và dễ đọc nhầm file đang ghi dở. Thư mục dataset trên Kaggle cũng là read-only |
| `datasets/MultiviewX.py` | `np.float` → `np.float32` | `np.float` đã bị xóa khỏi numpy ≥ 1.24. Chỉ ảnh hưởng MultiviewX |

**(b) Phương pháp pseudo label:**

| File | Loại | Nội dung |
|---|---|---|
| `multiview_detector/loss/pseudo_gaussian_mse.py` | mới | Loss `PseudoGaussianMSE`: 4 biến thể `gauss` / `point` / `maxval` / `mil`, vùng bỏ qua, chuẩn hóa H·W, hệ số λ |
| `multiview_detector/datasets/frameDataset.py` | sửa | Tham số `pseudo_dir`, `pseudo_cfg`; hàm `load_pseudo` (3 tầng dương / bỏ qua / nền, lọc mép, lọc số view, α). `__getitem__` trả thêm tensor pseudo (**chỉ khi** có `pseudo_dir`) |
| `multiview_detector/trainer.py` | sửa | Đọc tensor pseudo từ batch; loss head/foot từng view vẫn là `GaussianMSE`; lịch λ theo epoch; bật/tắt jitter lúc train/test; log `L_gt`, `L_ps`, `L_view` và ghi `pseudo_stats.jsonl` |
| `main.py` | sửa | `--loss {mse, pseudo}` và các flag `--ps_*`; tên thư mục log theo cấu hình (`logs/<dataset>_frame/<tag>/default/<thời gian>/`) |
| `tools/pseudo/common.py` | mới | Hàm dùng chung: chiếu ảnh sang mặt đất, đếm camera nhìn thấy, Hungarian matching |
| `tools/pseudo/detect_2d.py` | mới | Chạy detector 2D (Faster R-CNN v2 / YOLO) |
| `tools/pseudo/build_pseudo.py` | mới | Lọc box → điểm chân → mặt đất → gộp đa camera → bỏ trùng nhãn còn lại → JSON |
| `tools/pseudo/eval_pseudo.py` | mới | Precision/recall của pseudo so với người bị ẩn (chỉ để đánh giá) |
| `tools/pseudo/analyze_filters.py` | mới | So sánh các luật lọc, phân loại nguồn gốc người ma |
| `tools/pseudo/make_oracle.py` | mới | Pseudo "oracle" (vị trí thật + nhiễu), dùng làm cận trên |
| `tools/pseudo/teacher_rescore.py` | mới | Self-training: một model MVDet đã train chấm lại pseudo |
| `tools/eval_thresholds.py` | mới | Đo MODA của checkpoint ở nhiều ngưỡng `cls_thres` |
| `tests/test_pseudo_loss.py` | mới | 9 unit test cho loss (chạy CPU) |

Tổng phần sửa trong code gốc (`main.py` và `multiview_detector/`, không tính file mới): 6 file, 277 dòng thêm,
40 dòng xóa. Xem đầy đủ bằng `git diff 8b4792f -- main.py multiview_detector`.

### 1.3. Tương đương với branch `pseudo-label`

- Các file loss, tool và test giống hệt branch `pseudo-label`. Ngoại lệ duy nhất là tên file tài liệu được nhắc
  trong docstring.
- `load_pseudo` (dataset) được chép nguyên văn.
- Tên flag, giá trị mặc định và tên thư mục log (`pseudo_tag`) giống hệt.

Vì vậy các notebook Kaggle bên dưới chạy được với branch này **chỉ bằng cách đổi** `REPO_BRANCH = 'pseudo-label-clean'`
ở Bước 2 của notebook.

---

## 2. Bài toán và dữ liệu

### 2.1. MVDet

- **Đầu vào:** 7 ảnh (WildTrack có 7 camera), resize về 720 × 1280.
- **Backbone:** ResNet-18 (dilated) trích đặc trưng từng ảnh.
- **Chiếu đặc trưng xuống mặt đất** (perspective transformation) bằng thông số camera, rồi ghép 7 bản đồ lại.
- **Đầu ra:** một **heatmap trên mặt đất (BEV)** kích thước 120 × 360 ô, mỗi ô 10 cm (lưới gốc 480 × 1440 ô 2.5 cm,
  giảm 4 lần). Mỗi đỉnh sáng là một người.
- **Loss gốc:**
  - `L_map`: MSE giữa heatmap và target, trong đó mỗi nhãn được làm mờ thành một đốm Gaussian có đỉnh bằng 1.
  - `L_view`: MSE trên heatmap đầu/chân của từng camera.
- **Suy luận:** heatmap > 0.4 → NMS bán kính 0.5 m → vị trí người.
- **Metric:** MODA (chỉ số chính), MODP, precision, recall. Một dự đoán được tính là đúng nếu cách người thật ≤ 0.5 m.

### 2.2. Split thiếu nhãn drop60

Split lấy từ `Wildtrack_dropped/drop60`. Trong **mỗi frame train**, xóa ngẫu nhiên 60% nhãn người (seed 1). Tập test
(frame ≥ 1800, 40 frame) được giữ nguyên nhãn.

| | Số người trong tập train | Mỗi frame |
|---|---|---|
| Nhãn đầy đủ | 8566 | 23.8 |
| **Còn lại** (dùng để train) | **3418** | 9.5 |
| **Bị ẩn** | **5148** | 14.3 |

### 2.3. Vì sao MVDet gốc sụp đổ trên drop60

Người bị xóa nhãn **vẫn có trong ảnh**, nhưng target coi chỗ họ đứng là **nền** (giá trị 0). Model bị phạt mỗi khi
phát hiện đúng họ, nên học cách chỉ báo những gì cực kỳ chắc chắn.

| | Precision | Recall | MODA |
|---|---|---|---|
| Baseline drop60 | 98.6% | **23.0%** | 22.7% |

Model báo ít nhưng báo đâu đúng đó, và bỏ sót 77% số người.

---

## 3. Ý tưởng

### 3.1. Ý tưởng gốc (của thầy)

Dùng một **detector 2D có sẵn** (pretrained trên COCO) để tìm lại người thiếu nhãn trong từng ảnh. Chiếu **điểm chân**
của họ xuống mặt đất để có **pseudo label** (nhãn giả) trên BEV. Khi train:

- **Nhãn thật:** dương, trọng số 1.
- **Pseudo label:** dương nhưng trọng số **α < 1**, và vị trí được **làm nhiễu** trong bán kính r.

Lý do làm nhiễu: sai số của pseudo nằm ở **vị trí** (điểm chân chiếu xuống lệch vài chục cm), **không nằm ở giá trị**
(chỗ đó gần như chắc chắn có người).

### 3.2. Những gì nhóm bổ sung, dựa trên dữ liệu

1. **Sửa scale của loss.** Công thức ban đầu lấy trung bình trên N pseudo, khiến pseudo mạnh hơn nhãn thật khoảng
   1500 lần mỗi pixel (mục 5.4).
2. **Gộp đa camera và đếm số camera đồng thuận** (`n_views`), dùng làm độ tin cậy.
3. **Bỏ pseudo sát mép vùng gán nhãn.** Phân tích cho thấy 61% người ma nằm ở đó.
4. **Pseudo 3 tầng:** dương / bỏ qua / nền. Chỉ dạy điều chắc chắn; chỗ không chắc thì không dạy gì.

Kết quả thực nghiệm cho thấy nguồn nhiễu lớn nhất **không phải sai số vị trí**, mà là **pseudo không tồn tại** (người
ma). Các bổ sung 2–4 xử lý đúng loại lỗi đó. Làm nhiễu vị trí đúng hướng nhưng chỉ đóng góp khoảng +1 điểm (mục 8.4).

---

## 4. Sinh pseudo label

Chạy **offline**, một lần, chỉ trên **360 frame train**.

```
Ảnh 7 camera ──(1) detect_2d──► box "person" + score
             ──(2) build_pseudo──► lọc box → điểm chân → mặt đất → gộp đa camera → bỏ trùng nhãn còn lại
             ──► <frame>.json: [{grid, score, n_views, n_visible, ...}]
             ──(3) lúc nạp dữ liệu train (frameDataset.load_pseudo)──► chia 3 tầng: dương / bỏ qua / nền
```

### 4.1. Detector 2D: Faster R-CNN ResNet50-FPN v2

| | |
|---|---|
| Nguồn | `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`, weight COCO, chỉ lấy class `person` |
| Train lại trên WildTrack? | **Không** (off-the-shelf) |
| Đầu vào | Ảnh gốc 1920 × 1080; 360 frame × 7 camera = 2520 ảnh |
| Giữ | Mọi box có score ≥ 0.05. Ngưỡng thật được chọn ở bước sau, nên đổi ngưỡng không cần detect lại |
| Kết quả thực tế (Kaggle T4) | 145 416 box (57.7 box/ảnh), 844 giây |

Detector chỉ đọc ảnh, không đọc nhãn.

### 4.2. Lọc box và lấy điểm chân (`build_pseudo.py`)

Giữ box thỏa cả 3 điều kiện:
- **score ≥ 0.5**;
- **không chạm đáy ảnh** (y2 < 1080 − 2), vì chân bị cắt khỏi khung hình thì điểm chân sai;
- **cao ≥ 50 px**, vì người quá nhỏ (ở rất xa) có sai số chiếu lớn.

Điểm chân: `u = (x1 + x2) / 2`, `v = y2`.

### 4.3. Chiếu xuống mặt đất

Chân nằm trên mặt đất (Z = 0), nên pixel và tọa độ mặt đất liên hệ qua một **homography**:

```
[u, v, 1]ᵀ  ~  K · [r1  r2  t] · [X, Y, 1]ᵀ         ⇒  (X, Y) = nghịch đảo
```

- Dùng hàm có sẵn của repo (`get_worldcoord_from_imagecoord`, `get_worldgrid_from_worldcoord`). Đây chính là phép
  chiếu MVDet dùng để warp đặc trưng, nên pseudo và model cùng hệ tọa độ.
- Ảnh WildTrack đã được khử méo sẵn (`intrinsic_zero`).
- Điểm rơi ngoài lưới bị loại.

### 4.4. Gộp đa camera và đếm camera đồng thuận

Cùng một người thường được detect ở nhiều camera, nên chiếu xuống sẽ ra một cụm điểm gần nhau.

1. Gom mọi điểm chân của 7 camera trong frame bằng **average-linkage clustering**, ngưỡng **0.5 m**. Mỗi cụm được coi
   là một người.
2. **Tâm cụm** là trung bình các điểm có trọng số theo score.
3. Với mỗi cụm, lưu:
   - **`n_views`**: số camera **khác nhau** có box trong cụm, tức số camera **đồng thuận** rằng có người;
   - **`n_visible`**: số camera **có thể nhìn thấy** vị trí đó (chiếu ngược vị trí lên từng ảnh, kiểm tra nằm trong
     khung và ở phía trước camera);
   - `score` (trung bình), `n_dets`, `max_dets_per_cam` (dấu hiệu gộp nhầm 2 người đứng sát).

**Vì sao đồng thuận quan trọng:** người thật giữa sân thường được 3–4 camera cùng thấy. Người ma do chiếu sai (bị che
chân, người ngoài vùng) thường chỉ có ở 1 camera; các camera khác nhìn từ góc khác không thấy ai ở vị trí đó.
Số liệu thực tế ở mục 8.2.

### 4.5. Bỏ pseudo trùng người còn nhãn

- Cụm cách một **nhãn còn lại** dưới 0.5 m bị bỏ, vì người đó đã có nhãn thật.
- Chính các cụm này được dùng để **đo sai số chiếu mà không cần nhìn nhãn bị ẩn**:
  - trên Kaggle (detector thật): p80 = 30.1 cm → **r = 4 ô (40 cm)**;
  - với box GT: p80 = 14.6 cm, khớp sai số thật trên người bị ẩn là 14.1 cm (mục 6.2).

### 4.6. Ba tầng khi nạp dữ liệu train (`frameDataset.load_pseudo`)

| Tầng | Điều kiện (cấu hình S2b) | Model được dạy |
|---|---|---|
| **Nền** | Cách mép vùng gán nhãn < 1 m (`--ps_border 10`) | "Không có người", giống nhãn gốc |
| **Dương** | `n_views ≥ 3` (`--ps_min_views 3`), không ở mép | "Có người", trọng số α = 0.5 |
| **Bỏ qua** | Các pseudo còn lại có `n_views ≥ 1` (`--ps_ignore_min_views 1`), không ở mép | Không dạy gì trong bán kính 60 cm (`--ps_ignore_r 6`) |

- **Vì sao pseudo ở mép phải là "nền" chứ không phải "bỏ qua":** WildTrack chỉ gán nhãn người trong hình chữ nhật
  12 × 36 m, nhưng camera thấy cả quảng trường. Người đứng ngay ngoài vùng bị chiếu lệch vào trong. Ở tập **test**,
  họ cũng không có nhãn, nên model phải học **không** báo họ, giống hệt khi train với đủ nhãn.
- **Vì sao có tầng bỏ qua:** pseudo 1–2 camera phần lớn là người ma, nhưng vẫn chứa một số người thật. Dạy là "có
  người" thì model học theo người ma; dạy là "nền" thì lặp lại đúng lỗi của baseline. "Bỏ qua" là lựa chọn trung lập.

Pseudo được đổi sang tọa độ của map output (chia 4) và đưa vào batch dạng tensor `(N, 3) = [hàng, cột, α]`. Hàng có
`α = −1` là điểm bỏ qua.

---

## 5. Hàm loss

File: `multiview_detector/loss/pseudo_gaussian_mse.py`. Chọn bằng `--loss pseudo`.

```
L_total = L_map (BEV, loss mới)  +  L_view (head/foot từng camera, giữ nguyên code gốc, chỉ dùng nhãn thật)
```

### 5.1. Target của nhãn thật: giữ nguyên MVDet

```
soft_gt = Gaussian_kernel ⊛ (bản đồ điểm nhãn còn lại)       (đỉnh 1, phương sai 5 ô², ảnh hưởng ~70 cm)
```

### 5.2. Target của pseudo dương: 4 biến thể (`--ps_variant`)

Tất cả dùng một hình tròn bán kính `r` (`--ps_r`, đơn vị ô output 10 cm) quanh vị trí pseudo:

| Biến thể | Tương ứng | Mỗi bước train làm gì |
|---|---|---|
| `gauss` | guide B; với r = 0 là dạng cơ bản của thầy | Chọn **ngẫu nhiên** một điểm trong hình tròn, đặt **đốm Gaussian** (giống nhãn thật) tại đó |
| `point` | guide A (ý thầy, đúng nguyên văn) | Chọn ngẫu nhiên một điểm, đẩy **đúng 1 pixel** đó lên 1: `α·(P − 1)²` |
| `maxval` | guide C | Đẩy pixel **lớn nhất** của P trong hình tròn lên 1 |
| **`mil`** (dùng trong S2b) | C + hình Gaussian | Đặt đốm Gaussian tại **chỗ model đang đoán cao nhất** trong hình tròn (vị trí được chọn mà không truyền gradient) |

Cách hiểu `mil`: *"trong phạm vi 40 cm quanh đây chắc chắn có một người; model muốn đặt đỉnh ở đâu trong vùng đó cũng
được"*. Jitter (với `gauss`/`point`) **chỉ bật lúc train**, lúc test thì tắt.

### 5.3. Bản đồ trọng số và công thức

| Loại pixel | Target | Trọng số w |
|---|---|---|
| Quanh nhãn thật (soft_gt > 0.01) | soft_gt | 1 |
| Vùng đốm Gaussian của pseudo dương | Gaussian pseudo | **λ · α** (S2b: 1 × 0.5) |
| Nền trong bán kính `r_ignore` quanh pseudo dương (S2b: 6 ô) hoặc `ps_ignore_r` quanh pseudo bỏ qua (S2b: 6 ô) | — | **0** |
| Nền còn lại | 0 | 1 |

```
L_map = (1 / (H·W)) · Σ_pixel  w · (P − target)²   =   L_gt  +  λ · L_ps
```

- **λ(epoch)** = `ps_lambda · min(1, (epoch − warmup) / ramp)`, bằng 0 trong warm-up. Mọi thí nghiệm đã chạy dùng
  λ = 1 ngay từ đầu (`warmup 0, ramp 1`).
- Với biến thể `point` / `maxval`, một pixel mang trọng số `Σ kernel² ≈ 15.7` (khối lượng của một đốm GT), để một
  người pseudo có trọng số tương đương mọi biến thể.

### 5.4. Vì sao chuẩn hóa theo H·W (sửa công thức ban đầu)

Công thức ban đầu `(1/N) Σ α (P − 1)²` lấy trung bình trên N pseudo. Loss gốc lại lấy trung bình trên **43 200 pixel**:

| Lúc bắt đầu train (P ≈ 0) | Giá trị |
|---|---|
| L_map của nhãn thật drop60 (~9.5 người/frame) | ≈ 9.5 × 15.7 / 43 200 ≈ 0.0035 |
| L_pseudo theo công thức ban đầu (α = 0.5) | ≈ 0.5, tức lớn hơn khoảng 140 lần |
| Gradient trên một pixel pseudo so với một pixel GT | lớn hơn khoảng **1 500 lần** |

Chia mọi thành phần cho H·W thì **một người pseudo nặng đúng bằng α lần một người thật**. Log thực tế (mục 8.5) cho
thấy `L_ps` luôn nhỏ hơn `L_gt`.

**Tính chất có unit test:**
- Không có pseudo thì loss bằng **đúng** `GaussianMSE` gốc.
- Với `gauss`, r = 0, λ·α = 1, không có vùng bỏ qua, loss bằng đúng `GaussianMSE` khi coi pseudo là nhãn thật (cả
  giá trị lẫn gradient).

### 5.5. Suy luận

Không đổi gì so với MVDet: heatmap → ngưỡng 0.4 → NMS 0.5 m. Pseudo chỉ dùng lúc train.

---

## 6. Kiểm chứng trên máy local

Chạy trên CPU (MacBook, conda env `earlybird`).

### 6.1. Unit test: 9/9 pass

```
python tests/test_pseudo_loss.py
```

Các test kiểm tra:
1. Không có pseudo thì loss bằng đúng `GaussianMSE` (cả 4 biến thể).
2. `gauss` r = 0 bằng đúng `GaussianMSE` khi cộng pseudo vào nhãn.
3. Một người pseudo nặng bằng một người GT lúc P = 0.
4. α và λ nhân đúng vào phần pseudo; λ = 0 thì pixel pseudo không nhận gradient.
5. Jitter chỉ xảy ra lúc train và luôn nằm trong hình tròn.
6. `maxval`/`mil` chọn đúng argmax **trong** hình tròn.
7. Vùng bỏ qua có gradient 0, nhãn thật cạnh đó không bị ảnh hưởng.
8. `point` chỉ tác động lên 1 pixel.
9. Điểm bỏ qua (α < 0) không có target dương.

### 6.2. Kiểm tra phép chiếu bằng "detector hoàn hảo"

Dùng **box nhãn thật** thay cho detector, chạy trên 360 frame drop60:

| | Kết quả |
|---|---|
| Precision / recall trên 5148 người bị ẩn (@0.5 m) | **1.000 / 0.969** |
| Sai số vị trí trên người bị ẩn (p50 / p80) | 7.7 / 14.1 cm |
| Sai số ước lượng trên người còn nhãn (p80) | 14.6 cm |

Kết luận: phép chiếu, hệ tọa độ và quy ước trục đều đúng. Cách ước lượng sai số **không dùng nhãn ẩn** là đáng tin.

### 6.3. Smoke test `main.py`

Dataset mini gồm 2 frame train và 1 frame test. Kiểm tra cả `--loss mse` và `--loss pseudo` với đúng các flag của S2b:
- chạy trọn luồng: nạp pseudo, train, test, tính MODA, ghi `pseudo_stats.jsonl`;
- tầng bỏ qua và bộ lọc mép hoạt động đúng (log: `12 positive + 61 ignore-only ..., border 40: 26 dropped`).

Branch `pseudo-label-clean` cũng đã được kiểm tra lại như trên trước khi push.

---

## 7. Cách chạy

### 7.1. Chuẩn bị

```bash
git clone -b pseudo-label-clean https://github.com/Nhutan410/MVDet-PS.git && cd MVDet-PS
pip install kornia opencv-python scipy
# ~/Data/Wildtrack: Image_subsets/, calibrations/, annotations_positions/ (bản drop60 để train với thiếu nhãn)
```

Để train với drop60, `~/Data/Wildtrack/annotations_positions` phải trỏ tới `Wildtrack_dropped/drop60/annotations_positions`.
Các frame test trong đó vẫn có nhãn đầy đủ.

### 7.2. Sinh pseudo label

```bash
# (1) detect 2D, ~14 phút trên T4
python -m tools.pseudo.detect_2d -d wildtrack --root ~/Data/Wildtrack --out pseudo/raw_frcnn_v2.json --split train
# (2) chiếu + gộp đa camera + bỏ trùng nhãn còn lại (kept_ann = nhãn drop60), vài giây
python -m tools.pseudo.build_pseudo -d wildtrack --root ~/Data/Wildtrack --raw pseudo/raw_frcnn_v2.json \
    --kept_ann <Wildtrack_dropped>/drop60/annotations_positions --out pseudo/wildtrack_drop60_frcnn_v2_s0.5 --score_thr 0.5
# (3) chỉ để đánh giá: so với người bị ẩn (full_ann = nhãn đầy đủ)
python -m tools.pseudo.eval_pseudo -d wildtrack --root ~/Data/Wildtrack --full_ann <Wildtrack>/annotations_positions \
    --kept_ann <Wildtrack_dropped>/drop60/annotations_positions --pseudo pseudo/wildtrack_drop60_frcnn_v2_s0.5 --viz_dir pseudo/viz
```

Không cần chạy lại `build_pseudo` khi đổi luật lọc: cùng một thư mục pseudo được dùng cho mọi thí nghiệm, luật lọc
được áp lúc nạp dữ liệu.

### 7.3. Train từng thí nghiệm

Đặt `P=pseudo/wildtrack_drop60_frcnn_v2_s0.5`. Mọi run dùng mặc định còn lại: 10 epoch, lr 0.1, seed 1, `cls_thres` 0.4.

| Run | Lệnh |
|---|---|
| E1 (baseline) | `python main.py -d wildtrack --loss mse` |
| E2 | `python main.py -d wildtrack --loss pseudo --pseudo_dir $P --ps_variant gauss --ps_r 0 --ps_r_ignore 0 --ps_min_views 2` |
| E3 | `... --ps_variant point --ps_r 4 --ps_r_ignore 6 --ps_min_views 2` |
| E4b | `... --ps_variant gauss --ps_r 4 --ps_r_ignore 6 --ps_min_views 2` |
| E4mil | `... --ps_variant mil --ps_r 4 --ps_r_ignore 6 --ps_min_views 2` |
| S2a | `... --ps_variant mil --ps_r 4 --ps_r_ignore 6 --ps_min_views 2 --ps_border 10` |
| **S2b** | `... --ps_variant mil --ps_r 4 --ps_r_ignore 6 --ps_min_views 3 --ps_ignore_min_views 1 --ps_ignore_r 6 --ps_border 10` |

Các flag ngầm định ở mọi run pseudo: `--ps_alpha const --ps_alpha_const 0.5 --ps_lambda 1 --ps_warmup 0 --ps_ramp 1`.

**Kết quả của mỗi run** nằm ở `logs/wildtrack_frame/<tag>/default/<thời gian>/`:

| File | Nội dung |
|---|---|
| `log.txt` | Mỗi epoch có dòng `moda: …` |
| `MultiviewDetector.pth` | Checkpoint (lưu sau mỗi epoch) |
| `pseudo_stats.jsonl` | Mỗi epoch: `lam_ps`, `l_gt`, `l_ps`, `l_view`, `n_ps`, `n_ign`, `p_at_ps`, `ignored_frac`, precision/recall train |
| `learning_curve.jpg`, `map.jpg` | Đường học và heatmap |

Tag của S2b ví dụ: `pseudo_mil_r4_ri6_aconst0.5_l1_w0r1_mv3_ig1r6_b10_wildtrack_drop60_frcnn_v2_s0.5`.

### 7.4. Công cụ phân tích (không train)

```bash
python -m tools.pseudo.analyze_filters --full_ann ... --kept_ann ... --pseudo $P      # so sánh luật lọc, nguồn người ma
python tools/eval_thresholds.py --ckpt <run>/MultiviewDetector.pth --names S2b          # MODA theo cls_thres
python -m tools.pseudo.teacher_rescore --ckpt <run>/MultiviewDetector.pth --pseudo $P --out <dir>   # self-training
python -m tools.pseudo.make_oracle --full_ann ... --kept_ann ... --noise_r 20 --out <dir>          # cận trên
```

### 7.5. Trên Kaggle (cách các thí nghiệm đã được chạy)

Notebook nằm trong repo capstone `notebooks/mvdet/`. Muốn dùng branch này thì sửa `REPO_BRANCH = 'pseudo-label-clean'`
ở Bước 2.

| Notebook | Việc | Input | Thời gian |
|---|---|---|---|
| `mvdet_wildtrack_drop60_pseudo_kaggle.ipynb` | Session 1: detect + sinh pseudo + E2 / E3 / E4b / E4mil | Wildtrack, Wildtrack_dropped | 7.4 giờ (T4 x2) |
| `mvdet_wildtrack_drop60_pseudo_analysis_kaggle.ipynb` | Phân tích bộ lọc, sweep `cls_thres`, self-training (không train) | + output session 1 | 16 phút |
| `mvdet_wildtrack_drop60_pseudo_s2_kaggle.ipynb` | Session 2: S2a, S2b (dùng lại detection của session 1) | + output session 1 | 3.4 giờ train |

- **Settings:** GPU T4 x2, Internet On.
- **Cách chạy:** Save Version → Save & Run All.
- Notebook tự kiểm tra tỉ lệ drop khoảng 60% và tập test còn nguyên nhãn, rồi mới train.

**Chạy song song 2 GPU:**
- Mỗi run MVDet được ghim vào 1 GPU (`CUDA_VISIBLE_DEVICES=0` hoặc `1`), cần sửa tương thích 1 GPU ở mục 1.2.
- Mỗi GPU có `MVDET_CACHE_DIR` riêng.
- Hai run chạy cùng lúc; GPU nào rảnh thì tự nhận run tiếp theo.
- Nếu gặp CUDA OOM, notebook tự chuyển sang chạy tuần tự, mỗi run dùng 2 GPU.
- Số đo thực tế: **9.47 GB VRAM mỗi run** (T4 có 15 GB), khoảng **20–22 phút/epoch** khi chạy song song, khoảng
  3.3–3.7 giờ/run.

---

## 8. Toàn bộ thí nghiệm và kết quả

### 8.1. Danh sách thí nghiệm

| Tên | Chạy ở đâu | Câu hỏi | Kết luận ngắn |
|---|---|---|---|
| E0 full nhãn | `mvdec.ipynb` (repo `Nhutan410/MVDet`, `python main.py -d wildtrack`) | MVDet đạt bao nhiêu khi đủ nhãn? | 87.8%, khớp paper (88.2%) |
| E1 baseline drop60 | MVDet `--loss mse` trên drop60 | Thiếu 60% nhãn thì sao? | 22.7%: model không dám báo người |
| E2 | Session 1 | Pseudo label (dạng cơ bản) có giúp không? | **Có, +46.5** |
| E3 | Session 1 | Jitter đẩy 1 pixel (dạng A của thầy) có tốt hơn E2 không? | Không, thấp hơn 5.2 |
| E4b | Session 1 | Jitter với đốm Gaussian (dạng B)? | Ngang E2 |
| E4mil | Session 1 | Đốm đặt tại argmax trong bán kính (dạng C + Gaussian)? | Tốt nhất session 1, +1.2 so với E2 |
| Phân tích | Notebook analysis | Người ma từ đâu? Lọc thế nào? Ngưỡng? Self-training? | 61% ở mép; ≥ 3 view + bỏ mép cho precision 0.96 |
| S2a | Session 2 | Chỉ bỏ pseudo ở mép thì MODA tăng bao nhiêu? | **+9.1** so với E4mil |
| S2b | Session 2 | Pseudo rất sạch cộng tầng bỏ qua? | **81.3%, tốt nhất** |

### 8.2. Chất lượng pseudo label (360 frame, so với 14.3 người bị ẩn/frame)

**Theo ngưỡng score detector** (không lọc view):

| score ≥ | Pseudo/frame | P @0.5 m | R @0.5 m | P @1 m | R @1 m | Sai số p50 / p80 (cm) | `r` gợi ý |
|---|---|---|---|---|---|---|---|
| 0.3 | 51.0 | 0.256 | 0.914 | 0.270 | 0.963 | 16.2 / 26.9 | 4 |
| **0.5** | **47.1** | **0.277** | **0.910** | 0.292 | 0.960 | 16.2 / 26.9 | 4 |
| 0.7 | 43.2 | 0.299 | 0.904 | 0.315 | 0.951 | 16.2 / 26.9 | 4 |

Ý nghĩa: detector + phép chiếu **tìm lại được 91% người bị ẩn**, và vị trí khá chuẩn (p80 27 cm). Nhưng cứ 4 pseudo
thì chỉ khoảng 1 là đúng. Tăng ngưỡng score **gần như không giúp**, vì người ma phần lớn có score cao:

| Score | 0.5 | 0.6 | 0.7 | 0.8 | 0.9+ |
|---|---|---|---|---|---|
| Precision | 0.02 | 0.03 | 0.07 | 0.18 | 0.34 (n = 12 096) |

Detector thấy **đúng là người thật**; cái sai sinh ra ở bước chiếu xuống mặt đất.

**Theo số camera đồng thuận** (score ≥ 0.5):

| `n_views` | 1 | 2 | 3 | ≥ 4 |
|---|---|---|---|---|
| Số pseudo | 10 764 | 3 064 | 1 546 | 1 572 |
| Precision | **0.10** | 0.39 | 0.63 | **0.89** |

| Chỉ giữ `n_views ≥ k` | Precision | Recall | Pseudo/frame |
|---|---|---|---|
| k = 1 | 0.277 | 0.910 | 47.1 |
| **k = 2** (session 1) | **0.595** | **0.714** | **17.2** |
| k = 3 | 0.785 | 0.476 | 8.7 |

Đồng thuận giữa các camera là **tín hiệu độ tin tốt nhất**: từ 1 lên ≥ 4 camera, precision tăng từ 0.10 lên 0.89.

### 8.3. Session 1: E2, E3, E4b, E4mil

**Thiết lập chung:** pseudo score ≥ 0.5, **`n_views ≥ 2`** (17.2 pseudo/frame, precision 0.595, khoảng 7 người
ma/frame), không lọc mép, α = 0.5, λ = 1. Bốn run chỉ khác **cách đưa pseudo vào loss**. r = 4 lấy từ sai số chiếu
đo trên người còn nhãn. Chạy 2 run song song × 2 đợt, tổng 7.4 giờ.

| Run | Biến thể | MODA | MODP | Precision | Recall |
|---|---|---|---|---|---|
| E2 | `gauss` r = 0 (dạng cơ bản) | 69.2% | 73.6% | 81.6% | 89.4% |
| E3 | `point` r = 4 (dạng A) | 64.0% | 74.8% | 78.1% | 89.0% |
| E4b | `gauss` r = 4 (dạng B) | 69.3% | 73.4% | 86.1% | 82.7% |
| **E4mil** | `mil` r = 4 | **70.4%** | 74.4% | 82.5% | 89.4% |

**Ý nghĩa:**
1. **Pseudo label có tác dụng lớn:** MODA 22.7 → 69–70, recall 23 → 89. Baseline sụp đổ vì người thiếu nhãn bị coi
   là nền; pseudo sửa đúng lỗi đó.
2. **Precision là điểm nghẽn** (82% so với 93% khi đủ nhãn). Khoảng 7 người ma/frame được dạy là "có người", nên model
   báo người ở những chỗ không có ai trên tập test.
3. **Làm nhiễu vị trí đóng góp ít:**
   - E4mil hơn E2 chỉ 1.2, nằm trong mức dao động do seed.
   - E4b ngang E2.
   - E3 (đẩy 1 pixel) **thấp hơn** 5.2: đẩy một pixel ngẫu nhiên mỗi bước khiến model học ra đốm thấp và loang, khó
     vượt ngưỡng 0.4.

   Nguyên nhân: pseudo **đúng** chỉ lệch khoảng 27 cm, nên không có nhiều thứ để sửa. Lỗi lớn nằm ở pseudo **không
   tồn tại**.
4. **Log epoch 1:** E2 41.1% (precision 85.7, recall 49.3), E3 36.2%. Lần chạy full nhãn ở epoch 1 cũng chỉ khoảng
   40%. MODA dao động mạnh vài epoch đầu do OneCycle, nên chỉ so sánh ở epoch cuối.

### 8.4. Phân tích sau session 1 (không train)

**A. Người ma đến từ đâu** (luật `n_views ≥ 2`: 2506 người ma, 7.0/frame):

| Nguyên nhân | Số lượng | Tỉ lệ | Giải thích |
|---|---|---|---|
| **Sát mép vùng gán nhãn (< 1 m)** | 1519 | **60.6%** | Người đứng ngay ngoài vùng 12 × 36 m, chiếu lệch vào trong |
| Gần người bị ẩn (lệch 0.5–1 m) | 410 | 16.4% | Đúng người nhưng chiếu lệch quá 0.5 m |
| Bản sao của người còn nhãn (0.5–1 m) | 321 | 12.8% | Chiếu lệch quá ngưỡng bỏ trùng 0.5 m |
| Cô lập | 256 | 10.2% | Detect nhầm thật |

**B. So sánh luật lọc** cho tầng dương (`bord` = khoảng cách tới mép, đơn vị ô 2.5 cm; 40 = 1 m):

| k (view) | Tỉ lệ view | bord | P | R | Pseudo/frame | Người ma/frame | Người bị ẩn thành nền/frame |
|---|---|---|---|---|---|---|---|
| 2 | 0 | 0 | 0.595 | 0.714 | 17.2 | 7.0 | — |
| 2 | 0 | 40 | **0.775** | **0.611** | 11.3 | 2.5 | 5.6 |
| 2 | 0.5 | 40 | 0.839 | 0.569 | 9.7 | 1.6 | 6.2 |
| 3 | 0 | 0 | 0.785 | 0.476 | 8.7 | 1.9 | 7.5 |
| 3 | 0 | 20 | 0.792 | 0.461 | 8.3 | 1.7 | 7.7 |
| **3** | **0** | **40** | **0.960** | **0.408** | **6.1** | **0.2** | 8.5 |
| 3 | 0.5 | 40 | 0.962 | 0.402 | 6.0 | 0.2 | 8.6 |

Bỏ mép 1 m cùng với ≥ 3 view gần như **loại sạch người ma** (precision 0.96). Cái giá là cột cuối: 8.5 người bị ẩn
mỗi frame bị coi là nền **nếu không có tầng bỏ qua**. Vì vậy S2b thêm tầng bỏ qua.

**C. Lọc theo thời gian phản tác dụng.** Điều kiện: pseudo có mặt ở frame liền trước hoặc liền sau (trong 1 m).

| `n_views` | Không có ở frame liền kề | Có ở 1 frame | Có ở cả 2 frame |
|---|---|---|---|
| 2 | 0.600 (n = 120) | 0.648 (n = 681) | **0.334** (n = 2263) |
| 3 | 0.924 (n = 66) | 0.894 (n = 369) | **0.544** (n = 1111) |
| ≥ 4 | 1.000 (n = 85) | 0.976 (n = 419) | 0.860 (n = 1068) |

Pseudo xuất hiện liên tục lại **ít đúng hơn**. Người ma là thứ **đứng yên** (người đứng lâu ngoài mép, vật thể bị nhận
nhầm) nên lặp lại ở mọi frame, còn người thật thì di chuyển. Hướng này bị loại.

**D. Sweep `cls_thres`** (MODA / precision / recall, checkpoint session 1):

| Ngưỡng | 0.3 | 0.35 | **0.4** (mặc định) | 0.45 | 0.5 | 0.55 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|---|
| E2 | 52.8 / 70 / 94 | 64.4 / 77 / 91 | 69.2 / 82 / 89 | **71.6** / 86 / 86 | 70.4 / 88 / 82 | 65.4 / 88 / 76 | 61.4 / 89 / 70 | 50.3 / 89 / 57 |
| E3 | 47.0 / 67 / 93 | 56.3 / 73 / 91 | 64.0 / 78 / 89 | **65.4** / 81 / 86 | 64.4 / 83 / 82 | 61.6 / 83 / 77 | 58.5 / 84 / 72 | 47.1 / 85 / 58 |
| E4b | 58.0 / 73 / 91 | 67.6 / 81 / 88 | **69.3** / 86 / 83 | 68.5 / 90 / 77 | 64.0 / 93 / 69 | 58.4 / 97 / 61 | 49.3 / 97 / 51 | 25.8 / 99 / 26 |
| E4mil | 58.9 / 73 / 93 | 65.3 / 78 / 91 | 70.4 / 82 / 89 | 73.2 / 86 / 87 | **73.6** / 89 / 84 | 71.4 / 90 / 80 | 68.9 / 91 / 77 | 61.9 / 91 / 68 |

Nâng ngưỡng chỉ giúp khoảng +3 điểm (E4mil đạt 73.6 ở 0.5). Đây là chỉnh ngưỡng trên tập test, nên mọi kết quả chính
trong tài liệu này **giữ 0.4**. Muốn đi xa hơn phải làm pseudo sạch hơn.

**E. Self-training (E4mil làm "thầy" chấm lại pseudo):**

| Luật | Precision | Recall | Pseudo/frame |
|---|---|---|---|
| `n_views ≥ 1` | 0.277 | 0.910 | 47.0 |
| `n_views ≥ 1` & thầy ≥ 0.4 | 0.473 | 0.785 | 23.8 |
| `n_views ≥ 2` | 0.595 | 0.714 | 17.2 |
| `n_views ≥ 2` & thầy ≥ 0.4 | 0.629 | 0.701 | 15.9 |
| `n_views ≥ 3` & thầy ≥ 0.4 | 0.786 | 0.475 | 8.6 |
| Đỉnh do thầy tự tìm thêm | 0.413 | 0.012 | 0.4 |

**Không hiệu quả.** E4mil đã được train trên chính các người ma ≥ 2 view (tầng dương của session 1), nên cho điểm
cao cả ở người ma. Muốn dùng self-training phải có "thầy" không nhìn thấy pseudo của frame đó (ví dụ chia chéo tập
train). Hướng này bị loại khỏi session 2.

### 8.5. Session 2: S2a và S2b

**Thiết lập chung:** `mil` r = 4, α = 0.5, λ = 1, `cls_thres` 0.4, và **bỏ pseudo cách mép < 1 m** (thành nền). Hai run
chạy song song, 3.4 giờ (197 và 202 phút).

| Run | Tầng dương | Tầng bỏ qua | Pseudo dương/frame | Pseudo bỏ qua/frame | Vùng bỏ qua / bản đồ |
|---|---|---|---|---|---|
| S2a | `n_views ≥ 2` (P 0.775) | không có (còn lại là nền) | 11.3 | 0 | 2.5% |
| S2b | `n_views ≥ 3` (P 0.960) | mọi pseudo còn lại, r 60 cm | 6.1 | (ghi trong `pseudo_stats.jsonl`) | 6.0% |

| Run | MODA | MODP | Precision | Recall |
|---|---|---|---|---|
| E4mil (so sánh) | 70.4% | 74.4% | 82.5% | 89.4% |
| S2a | **79.5%** | 73.5% | **92.5%** | 86.6% |
| **S2b** | **81.3%** | **74.4%** | **94.3%** | 86.6% |

**Log theo epoch** (`pseudo_stats.jsonl`; `P@ps` = model dự đoán tại tâm pseudo; `trP`/`trR` = precision/recall **theo
pixel** trên tập train, cách tính có sẵn trong code gốc, không phải precision phát hiện người):

S2a:

| epoch | L_gt | L_ps | L_view | P@ps | trR |
|---|---|---|---|---|---|
| 1 | 0.00395 | 0.00133 | 0.00243 | 0.264 | 11.6 |
| 2 | 0.00262 | 0.00080 | 0.00037 | 0.545 | 42.5 |
| 5 | 0.00191 | 0.00053 | 0.00035 | 0.734 | 61.7 |
| 8 | 0.00145 | 0.00041 | 0.00035 | 0.818 | 72.1 |
| 10 | 0.00101 | 0.00032 | 0.00035 | 0.887 | 82.7 |

S2b:

| epoch | L_gt | L_ps | L_view | P@ps | trR |
|---|---|---|---|---|---|
| 1 | 0.00351 | 0.00060 | 0.00243 | 0.279 | 8.5 |
| 2 | 0.00223 | 0.00029 | 0.00037 | 0.581 | 37.3 |
| 5 | 0.00156 | 0.00015 | 0.00035 | 0.777 | 60.8 |
| 8 | 0.00120 | 0.00012 | 0.00035 | 0.837 | 72.2 |
| 10 | 0.00092 | 0.00010 | 0.00035 | 0.869 | 81.5 |

**Ý nghĩa:**
1. **Bỏ mép là thay đổi quan trọng nhất: +9.1 MODA** (E4mil → S2a). Precision tăng 82.5 → 92.5, recall chỉ giảm 2.8
   điểm. Khớp đúng phân tích 8.4A.
2. **S2b dùng gần một nửa số pseudo dương của S2a** (6.1 so với 11.3) nhưng **recall bằng nhau** (86.6%) và precision
   cao hơn (94.3 so với 92.5). Tầng bỏ qua giúp người không chắc chắn **không bị dạy là nền**, nên model tự khái
   quát ra họ từ nhãn thật và pseudo sạch. Nói cách khác: **tránh dạy sai quan trọng hơn thêm thật nhiều nhãn**.
3. **Huấn luyện ổn định:** `L_ps` luôn nhỏ hơn `L_gt` (nhờ chuẩn hóa H·W); `P@ps` tăng đều tới khoảng 0.87–0.89, tức
   model học đúng vị trí pseudo; vùng bỏ qua chỉ chiếm 6% bản đồ.
4. **So với full nhãn:** precision của S2b (94.3%) còn **cao hơn** lần chạy full nhãn (93.1%). Toàn bộ khoảng cách còn
   lại nằm ở **recall** (86.6% so với 94.9%). Viết MODA = recall − FP/GT:

   | | Recall | FP/GT | MODA |
   |---|---|---|---|
   | Full nhãn | 94.9% | 7.0% | 87.8% |
   | S2b | 86.6% | 5.2% | 81.3% |

   S2b báo sai **ít hơn** full nhãn, chỉ còn bỏ sót nhiều hơn khoảng 8% số người.

### 8.6. Tổng hợp hành trình

| Bước | Thay đổi | MODA | Δ |
|---|---|---|---|
| Baseline | 40% nhãn | 22.7 | |
| E2 | + pseudo ≥ 2 view (Gaussian, α 0.5) | 69.2 | **+46.5** |
| E4mil | + `mil` r = 4 (mô hình hóa sai số vị trí) | 70.4 | +1.2 |
| S2a | + bỏ pseudo cách mép < 1 m | 79.5 | **+9.1** |
| S2b | + dương chỉ ≥ 3 view, còn lại bỏ qua | 81.3 | +1.8 |
| (tham chiếu) Full nhãn | 100% nhãn | 87.8 | |

---

## 9. Kết luận và mức độ chắc chắn

**Model tốt nhất: S2b.** MVDet gốc (không đổi kiến trúc) được train với 40% nhãn thật cộng pseudo từ Faster R-CNN v2:
- pseudo ≥ 3 camera đồng thuận và không ở mép là dương (α 0.5, loss `mil` r = 40 cm);
- pseudo sát mép là nền;
- pseudo còn lại là vùng bỏ qua 60 cm.

MODA **81.3%**, precision 94.3%, recall 86.6%.

| Mức độ | Nhận định |
|---|---|
| **Chắc chắn** | Pseudo label có tác dụng lớn (+46.5). Bỏ mép có tác dụng lớn (+9.1). Các chênh lệch này vượt xa dao động ngẫu nhiên |
| **Chắc chắn** | Đồng thuận đa camera là tín hiệu độ tin tốt nhất; ngưỡng score detector gần như vô dụng; lọc theo thời gian và self-training bằng E4mil không hiệu quả |
| **Chưa chắc** | S2b tốt hơn S2a (+1.8) và `mil` tốt hơn `gauss` (+1.2): mới 1 seed, nằm gần mức dao động giữa các seed (khoảng 1–2 điểm) |
| **Cần ghi rõ khi báo cáo** | Các ngưỡng lọc (`n_views`, 1 m) được chọn nhờ phân tích dùng nhãn bị ẩn. Bộ lọc mép dựa trên ranh giới vùng gán nhãn, là thông tin biết trước của dataset. WildTrack không có tập validation nên các cấu hình được so trên tập test, như cách các paper MVDet vẫn làm |

**Bài học chính:**
1. Detector 2D không phải điểm yếu: nó tìm lại 91% người bị ẩn. Lỗi sinh ra ở **bước chiếu xuống mặt đất**.
2. Với dữ liệu này, nhiễu của pseudo chủ yếu là **pseudo không tồn tại**, không phải lệch vị trí. Mô hình hóa sai số
   vị trí (ý tưởng gốc) đúng hướng nhưng đóng góp nhỏ.
3. **Dạy ít mà đúng tốt hơn dạy nhiều mà sai**, và chỗ không chắc thì nên **bỏ qua** thay vì gán nền.

---

## 10. Hạn chế và hướng tiếp theo

| Ưu tiên | Việc | Mục đích |
|---|---|---|
| 1 | Chạy S2b (và S2a) với seed 2, 3 | Xác nhận 81.3% ổn định; đủ điều kiện ghi vào luận văn |
| 2 | S2b với `gauss` r = 0 (không jitter) | Đo đóng góp của ý tưởng jitter trong cấu hình tốt nhất |
| 3 | S2b với α = 1.0; tầng bỏ qua chỉ cho ≥ 2 view (S2c); tỉ lệ view ≥ 0.5 (S2d) | Tinh chỉnh; recall là phần còn thiếu |
| 4 | Detector khác: box toàn thân (CrowdHuman/MOT, ví dụ YOLOX của ByteTrack), Keypoint R-CNN (mắt cá chân) | Tăng số camera đồng thuận với người bị che, tức tăng recall của tầng dương (hiện 0.41). Đo offline bằng `analyze_filters` trước khi train |
| 5 | Đánh giá recall của S2b trên **người bị ẩn trong tập train**, tách theo tầng (dương / bỏ qua / nền) | Chứng minh model tự khái quát ra người không có pseudo |
| 6 | Loss head/foot trên từng camera: bỏ qua vùng box detect | Hiện người bị ẩn vẫn là "không phải người" trong loss ảnh |
| 7 | drop20, drop45, rồi MultiviewX | Chứng minh phương pháp tổng quát. Code đã hỗ trợ MultiviewX nhưng chưa chạy |

---

## 11. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| BEV | Mặt đất nhìn từ trên xuống, nơi MVDet dự đoán heatmap |
| Heatmap | Bản đồ giá trị 0–1; đỉnh sáng là vị trí người |
| Drop60 | Split train đã xóa ngẫu nhiên 60% nhãn người trong mỗi frame |
| Nhãn còn lại / người bị ẩn | 40% nhãn giữ lại / những người bị xóa nhãn (vẫn có trong ảnh) |
| Pseudo label | Nhãn giả sinh tự động từ detector 2D |
| Người ma | Pseudo không ứng với người nào có nhãn trong vùng |
| Điểm chân | Giữa cạnh dưới box, nơi chân chạm đất |
| Homography | Phép biến đổi giữa mặt phẳng ảnh và mặt đất (Z = 0) |
| `n_views` / `n_visible` | Số camera có detect được người đó / số camera có thể nhìn thấy vị trí đó |
| Tầng dương / bỏ qua / nền | Dạy "có người" / không dạy gì / dạy "không có người" |
| α, λ | Độ tin của pseudo / hệ số tổng của phần pseudo trong loss |
| r, `r_ignore`, `ps_ignore_r` | Bán kính sai số vị trí / bán kính bỏ qua quanh pseudo dương / quanh pseudo bỏ qua (ô 10 cm) |
| Jitter | Dời vị trí target ngẫu nhiên mỗi bước train |
| MIL | "Ít nhất một điểm trong vùng là đúng" |
| Oracle | Thí nghiệm dùng thông tin không có trong thực tế, để lấy cận trên |
| MODA / MODP | Độ chính xác phát hiện / độ chính xác vị trí (ngưỡng 0.5 m) |
| Precision / Recall | % dự đoán là đúng / % người thật được tìm thấy |
