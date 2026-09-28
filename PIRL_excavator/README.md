# PIRL cho máy xúc hybrid có thu hồi năng lượng hạ cần

Code: [`excavator_model.py`](excavator_model.py) (vật lý), [`duty_cycles.py`](duty_cycles.py) (tải), [`agents.py`](agents.py) (PIRL và các baseline), [`bench.py`](bench.py) (train, đánh giá), [`report.py`](report.py) (bảng, hình). Kết quả benchmark và phân tích: [`BENCHMARK.md`](BENCHMARK.md).

## 1. Đối tượng: máy xúc hybrid song song 20 tấn

```
 Động cơ diesel ═══ EMG1 (motor/máy phát 45 kW) ═══ Bơm thủy lực chính ──► cần / tay gầu / gầu / quay toa / di chuyển
                     │                                                          │ hạ cần: dầu hồi áp suất cao
                     │ DC bus 350 V                                              ▼
 Pin Li-ion 9 kWh ◄══╪══◄ EMG2 (máy phát 30 kW) ◄══ motor thủy lực HM ◄══ dầu hồi từ xi lanh cần
 (nhiệt + lão hoá)   └── phụ tải 1 kW
```

| Thành phần | Mô hình |
|---|---|
| Động cơ diesel 5.2 L, 110 kW @ 2000 rpm | **Bản đồ nhiên liệu 2D** (15 tốc độ × 29 mô-men, sinh từ mô hình Willans: ma sát theo tốc độ, hiệu suất chỉ thị theo tốc độ và tải), đường mô-men lớn nhất. BSFC từ 205 g/kWh (1400 rpm, tải cao) đến trên 330 g/kWh (tải thấp, tốc độ cao) |
| EMG1 PMSM 45 kW / 350 Nm | Bản đồ tổn hao `kc·T² + ki·ω + kw·ω³ + C0` (đồng, sắt, quạt gió), mô-men lớn nhất / công suất không đổi |
| Bơm biến lưu lượng | Hiệu suất theo tốc độ và tải. **Lưu lượng tối đa tỉ lệ với tốc độ trục** (70 W thủy lực/rpm): hạ tốc độ quá mức thì bơm thiếu dầu, máy chạy chậm, mất năng suất |
| Trục (động cơ + EMG1 + bơm) | Quán tính 2.5 kg·m², tốc độ đổi tối đa ±300 rpm mỗi 0.5 s |
| Thu hồi năng lượng cần | Dầu hạ cần → motor thủy lực (hiệu suất theo tốc độ) → EMG2 (bản đồ tổn hao, giới hạn 30 kW) → DC bus. **Nếu pin không nhận thêm được (giới hạn dòng sạc, SOC, nhiệt), phần dư bị tiết lưu thành nhiệt** |
| Pin Li-ion NMC 96s 25 Ah | OCV(SOC) phi tuyến, R(SOC, T) theo Arrhenius, dòng xả/sạc 5C/3C, cửa sổ SOC 0.3–0.8 |
| Nhiệt pin | `C·dT/dt = I²R − hA(T − T_amb)`, **giảm định mức dòng tuyến tính từ 40 °C xuống 0 ở 50 °C** |
| Lão hoá pin | Ah-throughput có trọng số Arrhenius theo nhiệt độ và C-rate, quy ra $ và g diesel tương đương |

**Bài toán EMS:** mỗi 0.5 s, chọn **(tốc độ động cơ n, công suất động cơ P_eng)**.

```
P_EMG1 = P_bơm(n, P_hyd) + P_quán_tính − P_eng          (cân bằng trục)
P_pin  = P_EMG1_điện + P_phụ_tải − P_thu_hồi_dùng       (cân bằng DC bus)
r      = −( nhiên liệu + 7.85·Ah_hao_mòn + 1000·(SOC − 0.55)² + 0.19·kJ_thủy_lực_thiếu )   [g diesel tương đương]
```

**Tải công việc** (`duty_cycles.py`). Train trên `train_mixed`: đào–đổ tải, san gạt, đào rãnh, chờ, sinh ngẫu nhiên mỗi episode. Test trên 10 công việc cố định:

| Công việc | Đặc điểm | Có trong dữ liệu train? |
|---|---|---|
| heavy_dig | đất cứng, chế độ H 2000 rpm, đỉnh ~130 kW | gần giống |
| truck_load_180 | đổ lên xe tải, quay toa 180° | gần giống |
| trenching | đào rãnh sâu, hạ cần dài (28% năng lượng là năng lượng hạ cần) | có |
| grading | san gạt, tải liên tục thấp | có |
| pipe_lifting | cẩu ống, hạ cần có tải (40% năng lượng hạ cần) | **không** |
| breaker | búa phá đá, tải cao ổn định, không có thu hồi | **không** |
| dig_with_waiting | xen kẽ chờ xe 30–60 s | có |
| eco_dig | chế độ E 1600 rpm | gần giống |
| dig_and_travel | di chuyển xen đào | **không** |
| rock_excavation | đỉnh tải 140 kW, biến động mạnh | **không** |

Kịch bản **mismatch**: máy thật khác mô hình trong PIRL (động cơ mòn: hiệu suất −3%; bơm mòn: −3 điểm %; pin lão hoá: R +40%, Q −15%; trời 45 °C).

## 2. Vật lý được nhúng vào mạng NN của PIRL

```
# ---- Actor ----
x      = [s, n_lo, n_hi, n_bơm_cần, P_bơm, P_thu_hồi, P_eng_lo, P_eng_hi, P_pin_max, P_pin_min]   # (1) Physics feature layer
u1, u2 = sigmoid(MLP_θ(x))
n      = n_lo + u1·(n_hi − n_lo)                  # (2) Physics constraint layer:
lo, hi = bounds_phys(s, n)                        #     cửa sổ tốc độ đạt được (quán tính, lưu lượng bơm),
P_eng  = lo + u2·(hi − lo)                        #     giới hạn động cơ, EMG1, dòng pin, SOC, nhiệt
                                                  #     (giải dạng đóng bất phương trình bậc 2 tổn hao EMG1)
# ---- Critic ----
s', r_phys = f_phys(s, n, P_eng)                  # (3) bản đồ động cơ, EMG, bơm, thu hồi + tiết lưu,
Q(s, a)    = r_phys + γ·V_φ(s')                   #     pin R_int, nhiệt, lão hoá. Chỉ V_φ là NN

# ---- Train ----
L_V = (V_φ(s) − [r_phys + γ·V_φ⁻(f_phys(s, π⁻(s)))])²
L_π = −Q(s, π_θ(s))              # gradient đi xuyên qua bản đồ nhiên liệu, tổn hao EMG, nhiệt và lão hoá pin
```

Physics constraint layer ở đây quan trọng hơn so với ô tô, vì nó bảo đảm pin:
- dòng không vượt 5C/3C;
- SOC không ra ngoài cửa sổ [0.3, 0.8];
- khi pin nóng quá 40 °C thì dòng bị cắt dần.

Tất cả được bảo đảm **ngay cả khi đang khám phá lúc train**.

## 3. Nên so sánh PIRL với những thuật toán nào?

Mục tiêu là chứng minh ba điều: (a) PIRL gần tối ưu, (b) PIRL tốt hơn các DRL mạnh nhất hiện nay, (c) **từng** thành phần vật lý đều có đóng góp. Bảng dưới là bộ so sánh mình đề xuất. Các dòng có ✅ đã được cài trong benchmark này.

| Nhóm | Thuật toán | Vì sao cần | Có trong benchmark |
|---|---|---|---|
| Cận tối ưu | **DP** (quy hoạch động, biết trước chu trình) | Chuẩn tham chiếu, cho biết PIRL cách tối ưu bao xa | ✅ |
| | PMP (nguyên lý cực tiểu Pontryagin) | Tối ưu offline nhanh hơn DP, hay xuất hiện trong bài báo EMS | – |
| Dựa trên mô hình, chạy online | **A-ECMS** (tương đương tiêu thụ, thích nghi theo SOC) | Baseline công nghiệp mạnh nhất. Dùng **cùng** mô hình vật lý với PIRL, nên nếu PIRL thắng thì phần thắng đến từ khả năng nhìn xa (giá trị dài hạn V), không phải từ việc có mô hình | ✅ |
| | MPC (dự báo tải bằng Markov/NN) | Reviewer thường hỏi. Chi phí tính toán online cao | – |
| Luật | **Rule-based** load-leveling / thermostat | Chuẩn của máy thương mại | ✅ |
| | Fuzzy logic | Hay gặp trong bài báo máy xúc hybrid | – |
| DRL model-free | **DDPG** | Baseline của repo gốc | ✅ |
| | **TD3** | Sửa lỗi ước lượng quá mức Q của DDPG | ✅ |
| | **SAC** | DRL liên tục mạnh nhất hiện nay, nhờ entropy tự điều chỉnh | ✅ |
| | PPO | On-policy, ổn định nhưng tốn mẫu | – |
| | DQN / Double DQN | Action rời rạc, hay gặp trong EMS | – |
| DRL có kiến thức | RI-DDPG (bài báo gốc của repo: chèn đường BSFC tối ưu) | Chứng minh cách nhúng vật lý của PIRL tốt hơn cách "chèn luật" | – |
| | Safe RL: SAC-Lagrangian, action shielding | So sánh ràng buộc cứng của PIRL với ràng buộc mềm hoặc lớp an toàn gắn ngoài | một phần: ablation "PIRL w/o constraint" |
| | Model-based RL (MBPO, Dyna) | Học mô hình từ dữ liệu so với nhúng mô hình vật lý | – |
| **Ablation** | PIRL bỏ physics feature / bỏ constraint layer / bỏ physics critic | **Bắt buộc**: chứng minh từng thành phần có đóng góp | ✅ |

Tiêu chí nên báo cáo:
1. Nhiên liệu quy đổi SOC và khoảng cách tới DP
2. Chi phí vận hành (nhiên liệu + hao mòn pin)
3. Tận dụng năng lượng hạ cần
4. Tuổi thọ pin, T_max, thời gian bị giảm định mức
5. Số lần vi phạm ràng buộc khi train và khi chạy
6. Số episode để hội tụ
7. Tổng quát hoá sang công việc chưa thấy và sang máy có tham số khác (mismatch)
8. Thời gian tính toán mỗi bước (PIRL rất nhẹ, A-ECMS/MPC nặng hơn)
