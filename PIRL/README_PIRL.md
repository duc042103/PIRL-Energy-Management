# PIRL – Physics-Informed Reinforcement Learning (ví dụ đơn giản cho HEV)

Code: [`pirl_hev.py`](pirl_hev.py) · Chạy: `python PIRL/pirl_hev.py` (cần `torch numpy scipy matplotlib`)

Bài toán giống repo gốc: mỗi giây, agent chọn **công suất động cơ `P_eng`**, phần còn lại do pin bù.

- State `s = (SOC, v, a)`
- Action `P_eng` (W)
- Reward `r = -(α·fuel + β·(SOC_ref − SOC')²)`

Khác với DDPG thông thường (mạng NN "hộp đen" phải tự học hết), trong PIRL **các phương trình vật lý là một phần của mạng**.

---

## 1. Pseudo code

```
# ---------- PHYSICS (khả vi, không có tham số học) ----------
function P_req(v, a):                                  # cân bằng lực dọc trục
    F = m·a + m·g·Cr + ½·ρ·A·Cd·v²
    return F·v / η_drive  (nếu ≥ 0)   hoặc   F·v·η_regen  (khi phanh)

function battery_limits(SOC):                          # giới hạn dòng điện + cửa sổ SOC
    I_max = min(I_dis, (SOC − SOC_min)·Q/dt)
    I_min = −min(I_chg, (SOC_max − SOC)·Q/dt)
    return P_b_min = Voc·I_min − R·I_min²,  P_b_max = Voc·I_max − R·I_max²

function engine_bounds(s):                             # miền khả thi của action
    lo = clip(P_req − P_b_max, 0, P_eng_max)
    hi = clip(P_req − P_b_min, 0, P_eng_max)
    return lo, hi

function f_phys(s, P_eng):                             # mô hình pin R_int
    P_batt = clip(P_req − P_eng, P_b_min, P_b_max)
    I      = (Voc − sqrt(Voc² − 4·R·P_batt)) / (2R)
    SOC'   = SOC − I·dt/Q
    fuel   = on(P_eng)·(f0 + f1·P_eng + f2·P_eng²)
    return SOC', fuel

# ---------- NN CÓ VẬT LÝ BÊN TRONG ----------
Actor π_θ(s):
    x  = [s, P_req, lo, hi]                   # (1) Physics feature layer
    u  = sigmoid(MLP_θ(x))                    #     u ∈ (0,1)
    return lo + u·(hi − lo)                   # (2) Physics constraint layer → luôn khả thi

Critic Q_φ(s, P_eng | v', a'):
    SOC', fuel = f_phys(s, P_eng)             # (3) vật lý
    r          = −(α·fuel + β·(SOC_ref − SOC')²)
    return r + γ·V_φ(SOC', v', a')            #     chỉ V_φ là NN

# ---------- TRAINING ----------
khởi tạo θ, φ, target θ⁻ ← θ, φ⁻ ← φ, replay buffer D
for episode = 1..N:
    SOC ← random(0.5, 0.7)
    for t = 1..T (theo chu trình lái):
        s    = (SOC, v_t, a_t)
        P    = π_θ(s) với noise trong không gian logit          # vẫn khả thi
        SOC  = f_phys(s, P)                                     # môi trường
        D ← D ∪ {(s, v_{t+1}, a_{t+1})}                         # không cần lưu r, s'
        lấy batch B từ D
        y       = Q_φ⁻(s, π_θ⁻(s) | v', a')                     # target (vật lý tính r, s')
        L_V     = mean( (V_φ(s) − y)² )                         # cập nhật critic
        L_π     = −mean( Q_φ(s, π_θ(s) | v', a') )              # gradient đi XUYÊN vật lý
        θ⁻ ← τθ + (1−τ)θ⁻ ;  φ⁻ ← τφ + (1−τ)φ⁻
```

---

## 2. Giải thích: vật lý nằm ở đâu trong mạng NN?

```
           ┌──────────────── ACTOR ─────────────────┐
 s ──► [Physics feature] ──► MLP_θ ──► sigmoid ──► [Physics constraint] ──► P_eng
        (P_req, lo, hi)                              lo + u·(hi−lo)
                                                                              │
           ┌──────────────── CRITIC ────────────────┐                         ▼
           │  [f_phys: pin R_int, nhiên liệu] ──► r_phys + γ·V_φ(SOC', v', a')  ──► Q
           └────────────────────────────────────────┘
```

**(1) Physics feature layer.** Mạng không phải tự học `P_req = (m·a + lăn + gió)·v`. Lớp này tính sẵn công suất yêu cầu và giới hạn `lo`, `hi` rồi đưa vào MLP. Nhờ vậy mạng chỉ cần học *chiến lược chia công suất*, không phải học lại định luật Newton.

**(2) Physics constraint layer (lớp cuối của actor).** Output của NN là `u ∈ (0,1)`, sau đó được ánh xạ vào `[lo, hi]`. Hai giá trị này được suy ra từ ràng buộc `P_req = P_eng + P_batt`, giới hạn dòng điện của pin và cửa sổ SOC `[0.4, 0.8]`. Vì vậy **mọi action, kể cả khi đang khám phá với noise, đều hợp lệ về mặt vật lý**: SOC không bao giờ vượt ngưỡng, không quá dòng, không nổ máy khi đang phanh. Đây là ràng buộc cứng (hard constraint) chứ không phải phạt trong reward.

**(3) Critic có mô hình vật lý.** Một critic DDPG thường là `Q_φ(s,a)` hộp đen. Ở đây:

`Q(s,a) = r_phys(s,a) + γ·V_φ(f_phys(s,a))`

- Phần tức thời (nhiên liệu, SOC kế tiếp) được tính **chính xác** bằng phương trình, không phải xấp xỉ.
- NN chỉ phải học giá trị dài hạn `V_φ(SOC, v, a)`, một hàm 3 chiều trơn và dễ học hơn nhiều so với `Q(s,a)`.
- Khi cập nhật actor, `∂Q/∂P_eng = ∂r/∂P_eng + γ·∂V/∂SOC' · ∂SOC'/∂P_eng`. Các đạo hàm `∂fuel/∂P_eng` và `∂SOC'/∂P_eng` là **gradient vật lý thật** do autograd tính qua phương trình pin R_int, nên actor nhận tín hiệu học tốt ngay từ đầu.
- Replay buffer chỉ cần lưu `(s, v', a')` vì reward và SOC' luôn được tính lại bằng vật lý.

**Mẹo để hàm vật lý khả vi.** Trạng thái bật/tắt máy `on(P_eng)` được viết dạng `sigmoid((P−1500)/400)` thay cho hàm bậc thang, để gradient đi qua được. Căn trong công thức dòng pin được `clamp` để tránh NaN.

---

## 3. Kết quả (train 30 episode trên UDDS, CPU ~4 phút)

| Chu trình | Policy | Fuel (g) | SOC cuối | Fuel quy đổi SOC (g) |
|---|---|---|---|---|
| UDDS (train) | **PIRL** | 363.0 | 0.591 | **366.1** |
| UDDS (train) | Rule | 394.1 | 0.606 | 392.0 |
| NEDC (test) | **PIRL** | 395.5 | 0.654 | **376.3** |
| NEDC (test) | Rule | 416.0 | 0.657 | 395.6 |
| WVUSUB (test) | **PIRL** | 348.8 | 0.593 | **351.5** |
| WVUSUB (test) | Rule | 375.7 | 0.607 | 373.4 |

PIRL tiết kiệm khoảng 5–7% nhiên liệu so với rule-based trên cả chu trình chưa từng thấy. Suốt quá trình train, SOC luôn nằm trong `[0.4, 0.8]` nhờ lớp ràng buộc. Hình SOC trên NEDC: `pirl_soc_nedc.png`.

---

## 4. Giới hạn và hướng mở rộng

- Để giữ ví dụ đơn giản, môi trường dùng **cùng** mô hình `HEVPhysics` với mạng. Trong thực tế, môi trường nên là `Prius_model_new.py` (bản đồ BSFC, hiệu suất motor/generator). Khi đó `HEVPhysics` là mô hình gần đúng, và có thể thêm một NN residual `ΔSOC_φ(s,a)` học phần sai lệch giữa hai mô hình.
- Có thể biến tham số vật lý (ví dụ `R`, `Q` của pin) thành `nn.Parameter` và học từ dữ liệu. Khi đó physics layer trở thành *grey-box*.
- Có thể thêm loss dạng PINN, ví dụ phạt `‖SOC'_pred − (SOC − I·dt/Q)‖²` nếu dùng một mạng dự đoán động học.

---

## 5. Benchmark với DDPG và DP

Xem [`BENCHMARK.md`](BENCHMARK.md). Bộ benchmark gồm 10 chu trình lái chưa dùng để train, cộng thêm kịch bản xe có tham số khác mô hình, và đo các chỉ số về năng lượng và pin.
