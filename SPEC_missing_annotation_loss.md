# Spec: Loss chống nhiễu thiếu nhãn (Missing-Annotation Robust Loss) cho MVDet

## 1. Bối cảnh & mục tiêu

MVDet-PS (Partly-Supervised) huấn luyện trên tập dữ liệu multi-view pedestrian detection
trong đó **một phần vị trí người thật KHÔNG có nhãn** — các vị trí này bị heatmap ground-truth
gán nhầm thành background (giá trị 0), gây nhiễu khi train theo loss chuẩn (Focal loss trên
BEV heatmap).

**Mục tiêu:** thiết kế một loss thay thế, trong đó mô hình **tự học** mức độ tin cậy của nhãn
"background" tại mỗi vị trí BEV — không dùng ngưỡng cố định (threshold) — dựa trên **bằng
chứng đồng thuận độc lập giữa nhiều camera view**, để tránh vòng lặp tự tham chiếu
(model tự đánh giá rồi tự tin vào chính nó).

**Ràng buộc thiết kế bắt buộc (đọc kỹ trước khi code):**
1. Không dùng bất kỳ ngưỡng (threshold) cố định nào để quyết định "tin nhãn hay tin model".
2. Tín hiệu quyết định độ tin cậy PHẢI đến từ một nguồn **độc lập về gradient** với nhánh dự
   đoán chính — không được để mô hình tự phán xét chính nó (nguy cơ collapse: model học cách
   "nói dối" để giảm loss).
3. Positive labels (nhãn người đã có) luôn được coi là sạch, không áp dụng cơ chế nghi ngờ lên
   chúng — chỉ áp dụng cho vùng `y_obs = 0` (bị gán background).

## 2. Kiến trúc nền (giả định MVDet gốc)

Pipeline MVDet chuẩn:

```
Mỗi camera view v (N views)
  → CNN backbone (chia sẻ trọng số giữa các view)
  → feature map per-view F_v  [C, H, W]
  → homography projection lên ground plane
  → concat/fuse các F_v đã chiếu → BEV feature [C, X, Y]
  → CNN head trên BEV feature
  → heatmap dự đoán p(x,y)  [1, X, Y]  (qua sigmoid)
```

Loss gốc: Focal loss / BCE giữa `p(x,y)` và heatmap ground-truth (Gaussian quanh mỗi
vị trí người có nhãn).

## 3. Kiến trúc mới cần thêm

Thêm **2 nhánh mới**, song song với pipeline gốc, không thay đổi nhánh `p` hiện có:

```
Mỗi camera view v
  → CNN backbone RIÊNG, KHÔNG chia sẻ trọng số với backbone chính
    (hoặc: dùng backbone pretrained self-supervised, ĐÓNG BĂNG — khuyến nghị, xem mục 6)
  → feature per-view F_v'
  → head nhỏ (1-2 conv layer) → evidence map s_v(u, v) ∈ (0,1), qua sigmoid
    [đây là "bằng chứng độc lập": xác suất pixel (u,v) trên ảnh view v là người]

  → chiếu s_v lên ground plane BẰNG CHÍNH phép homography đã có sẵn trong MVDet
    (tái sử dụng ma trận camera calibration, KHÔNG train thêm phép chiếu)
    → s_v_bev(x,y)

Với mỗi vị trí BEV (x,y), tổng hợp qua các view nhìn thấy được vị trí đó:
  c(x,y) = MEDIAN hoặc PERCENTILE THẤP (vd: 25th) của {s_v_bev(x,y)}_{v: visible}
  [KHÔNG dùng mean — mean dễ bị 1-2 view "ảo giác" kéo điểm lên cao]
  c(x,y) → STOP GRADIENT (detach khỏi computational graph)

Trên nhánh BEV feature chính (đã có sẵn trong MVDet):
  → thêm 1 head nhỏ mới → q(x,y) ∈ (0,1), qua sigmoid
    [gate: "mức độ nghi ngờ nhãn 0 tại vị trí này là sai do thiếu nhãn"]
```

## 4. Công thức loss

Ký hiệu: `p(x,y)` = dự đoán chính (đã có), `q(x,y)` = gate mới, `c(x,y)` = evidence đa-view
(đã stop-gradient), `y_obs(x,y) ∈ {0,1}` = nhãn quan sát được (có thể sai ở vùng =0).

### 4.1. Loss cho vùng có nhãn `y_obs = 1` (giữ nguyên, không đổi)
```
L_pos(x,y) = -log( p(x,y) )
```

### 4.2. Loss cho vùng có nhãn `y_obs = 0` (thay đổi — đây là phần chính)
```
L_conf(x,y) = (1 - q(x,y)) * [ -log(1 - p(x,y)) ]     # nhánh "tin nhãn 0"
            +       q(x,y)  * [ -log(    p(x,y)) ]     # nhánh "tin model, coi như positive"
```

### 4.3. Loss huấn luyện gate `q` (bắt buộc — nếu thiếu, `q` không học được gì)
```
L_q(x,y) = ( q(x,y) - stopgrad(c(x,y)) )^2
```

### 4.4. Loss huấn luyện evidence per-view `s_v` (độc lập, KHÔNG dùng vùng `y_obs=0`)
```
L_s = sum_v [
    -log( s_v(u,v) )       tại các pixel positive ĐÃ CÓ NHÃN, chiếu ngược từ BEV xuống ảnh view v
  + -log( 1 - s_v(u,v) )   tại các pixel "easy negative": CÁCH XA mọi vị trí có nhãn
                            hoặc vùng confusion (định nghĩa bằng khoảng cách tối thiểu
                            trên BEV, xem tham số MIN_DIST_EASY_NEG bên dưới)
]
```
Pixel nằm trong vùng "confusion" (gần vị trí có nhãn nhưng bản thân không có nhãn) **KHÔNG**
được dùng để train `s_v` — loại hẳn khỏi `L_s`.

### 4.5. (Tùy chọn) Regularizer neo trung bình
```
L_prior = ( mean(q(x,y) trên vùng y_obs=0) - PI )^2
```
`PI`: tỷ lệ thiếu nhãn ước lượng trước (siêu tham số config, có thể để 0 nếu không ước lượng
được — L_q ở 4.3 đã là cơ chế neo chính).

### 4.6. Loss tổng
```
L_total = sum_{y_obs=1} L_pos
        + sum_{y_obs=0} L_conf
        + LAMBDA_Q   * sum_{x,y} L_q
        + LAMBDA_S   * L_s
        + LAMBDA_PRIOR * L_prior   (nếu dùng)
```

## 5. Định nghĩa "view visible" và phép chiếu

- Với mỗi vị trí BEV `(x,y)`, một view `v` được coi là "visible" nếu điểm đó nằm trong
  field-of-view của camera `v` (dùng lại logic mask khả kiến đã có sẵn trong MVDet, nếu có;
  nếu chưa có, tính bằng cách chiếu `(x,y,0)` qua ma trận camera và kiểm tra tọa độ pixel nằm
  trong kích thước ảnh).
- Nếu số view visible tại một vị trí < 2, đặt `c(x,y) = 0` (không đủ bằng chứng độc lập, mặc
  định không tin) — KHÔNG suy diễn từ 1 view duy nhất.

## 6. Khuyến nghị triển khai (ưu tiên theo thứ tự)

1. **Ưu tiên cao nhất — tránh collapse:** backbone của `s_v` PHẢI khác backbone chính. Cách an
   toàn nhất: dùng một backbone self-supervised pretrained (vd: DINOv2 nhỏ) cho `s_v`, ĐÓNG
   BĂNG (freeze) toàn bộ, chỉ train 1 linear/conv head nhỏ phía trên bằng `L_s`. Nếu vì lý do
   tài nguyên không dùng được DINOv2, tối thiểu phải tách riêng optimizer/không share layer
   nào giữa backbone của `s_v` và backbone chính của `p`.
2. **Warm-up bắt buộc:** train `p` (và pipeline gốc) độc lập trước N epoch đầu, `q` cố định = 0
   (tương đương loss gốc). Chỉ bật `L_conf` với `q` thật và `L_q` sau khi `p` đã ổn định.
3. `stopgrad(c)` — dùng `.detach()` (PyTorch) khi đưa `c` vào `L_q`. Đây là bước bắt buộc,
   không được bỏ qua.
4. Log riêng giá trị trung bình của `q(x,y)` trên vùng `y_obs=0` qua từng epoch để giám sát:
   nếu `q` tiến nhanh về 1 ở khắp nơi ngay từ đầu → dấu hiệu collapse, cần kiểm tra lại việc
   tách gradient giữa `s_v` và `p`.

## 7. Tham số cần thêm vào config

```yaml
missing_annotation_loss:
  enabled: true
  warmup_epochs: <int>            # số epoch train p thuần trước khi bật q
  min_dist_easy_neg: <float>      # khoảng cách tối thiểu (đơn vị BEV) để coi là easy negative
  consensus_agg: "median"          # "median" | "percentile25" | "min"
  min_visible_views: 2             # số view tối thiểu để tính c(x,y), nếu ít hơn -> c=0
  lambda_q: <float>
  lambda_s: <float>
  lambda_prior: 0.0                 # để 0 nếu không dùng L_prior
  pi_prior: 0.0
  s_v_backbone: "frozen_dinov2" | "separate_trainable"
```

## 8. Việc CẦN LÀM khi implement (checklist cho Claude Code)

- [ ] Thêm module `EvidenceHead` (nhánh `s_v`) — nhận feature per-view, xuất `s_v` qua sigmoid.
- [ ] Tái sử dụng hàm homography projection có sẵn trong MVDet để chiếu `s_v` sang BEV
      (không viết lại phép chiếu mới).
- [ ] Thêm hàm `aggregate_consensus(s_v_bev_list, visible_mask, method="median")` trả về `c(x,y)`,
      đã `.detach()`.
- [ ] Thêm module `GateHead` (nhánh `q`) trên BEV feature chính.
- [ ] Viết hàm loss `missing_annotation_loss(p, q, c, y_obs, s_v_list, positive_coords)` trả về
      dict các thành phần loss riêng biệt (để log từng phần, không chỉ tổng).
- [ ] Thêm cờ `warmup_epochs` vào training loop: chỉ gọi `L_conf`/`L_q` sau epoch này; trước đó
      dùng loss gốc.
- [ ] Thêm logging: `mean(q)` trên vùng `y_obs=0`, `mean(c)`, histogram phân bố `q` — mỗi epoch.
- [ ] Viết unit test nhỏ: kiểm tra gradient của `c` không lan vào backbone chính (dùng
      `retain_grad()`/kiểm tra `requires_grad` sau `.detach()`).

## 9. Tài liệu tham khảo (để đối chiếu khi cần)

- Blum & Mitchell, "Combining Labeled and Unlabeled Data with Co-Training", COLT 1998 —
  cơ sở lý thuyết cho việc dùng đồng thuận giữa các nguồn độc lập.
- Han et al., "Co-teaching: Robust training of deep neural networks with extremely noisy
  labels", NeurIPS 2018 — nguyên lý chống collapse bằng nguồn đánh giá độc lập.
- Zhang et al., "Solving Missing-Annotation Object Detection with Background Recalibration
  Loss", ICASSP 2020 (arXiv:2002.05274) — nguồn gốc ý tưởng "mirror branch" cho `L_conf`.
- Simon et al., "Hand Keypoint Detection in Single Images using Multiview Bootstrapping",
  CVPR 2017 — cơ sở cho việc dùng đồng thuận đa-camera.
- VSRD (Instance-Aware Volumetric Silhouette Rendering for Weakly Supervised 3D Object
  Detection) — mẫu thiết kế gần nhất cho việc dùng multi-view consistency làm confidence
  weight trực tiếp trong loss.

## 10. Lưu ý quan trọng

Đây là một thiết kế **tổng hợp có cơ sở lý thuyết** (xem mục 9), nhưng **chưa có paper nào
áp dụng đúng tổ hợp này cho bài toán MVDet-PS cụ thể** — cần chạy thực nghiệm để xác nhận
hiệu quả trước khi coi đây là kết quả cuối cùng. Khi implement, ưu tiên giữ mọi thành phần
loss có thể bật/tắt độc lập qua config (mục 7) để dễ chạy ablation study so sánh với baseline
gốc.
