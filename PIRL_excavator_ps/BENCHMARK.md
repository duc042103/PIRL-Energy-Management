# Benchmark máy xúc hybrid power-split: PIRL, PIRL-P, PIRL-P+ và các baseline

Mô hình: [`ps_model.py`](ps_model.py). Bảng đầy đủ: [`results/tables.md`](results/tables.md).

## Cấu hình
- **Bộ hành tinh:** Carrier = ICE (qua hộp số i_e = 1.6), Sun = trục ra dẫn bơm (i_p = 2.4, qua ly hợp C1), Ring = EMG (i_m = 3.0, qua ly hợp CM). k = 78/30.
- **Nâng / làm việc (M1):** CM đóng, C2 mở. ICE và EMG chia công suất qua bộ hành tinh. EMS chọn (tốc độ bơm, tốc độ ICE). Mô-men ICE và mô-men EMG do tải bơm quyết định.
- **Hạ cần (M2):** van chính đóng, CM mở, C2 đóng. Motor thủy lực 63 cm³/vòng quay EMG phát điện nạp pin. ICE chạy không tải.
- **Chu trình tải:** pha hạ cần được tách riêng, không có thao tác bơm chạy song song.
- **Protocol:** như benchmark trước (10 công việc test, kịch bản mismatch, 3 seed đánh giá, γ = 0.99).
- **DP:** chạy trên mô hình nới lỏng (bỏ giới hạn tốc độ đổi và bỏ quán tính), nên là **cận dưới**.

## Ba biến thể PIRL
| | Cách train critic | Cách chọn action lúc chạy |
|---|---|---|
| PIRL | target theo actor | actor |
| PIRL-P | target theo actor | argmax Q_phys trên lưới 9×9 + ứng viên actor |
| PIRL-P+ | target = max Q_phys trên lưới 5×7 + ứng viên actor | argmax Q_phys trên lưới 9×9 + ứng viên actor |

Actor đã được sửa lỗi bão hoà: loss chuẩn hoá theo |Q|, phạt biên độ logit, cắt chuẩn gradient.

## Kết quả (TB 10 công việc test, 3 seed)

| | DP | PIRL-P | PIRL-P+ | PIRL | SAC | TD3 | DDPG | A-ECMS | Rule |
|---|---|---|---|---|---|---|---|---|---|
| Lệch so với DP, g/kWh công (nominal) | 0 | +6.4% | **+5.8%** | +8.5% | +8.1% | +6.6% | +7.5% | **+0.6%** | +10.9% |
| Lệch so với DP (mismatch) | 0 | **+5.6%** | +5.8% | +8.3% | +7.6% | +6.2% | +7.6% | **+0.4%** | +10.6% |
| Công việc không đáp ứng (nominal) | 0.12% | 1.88% | 1.30% | 0.86% | 0.83% | 2.05% | 1.58% | 0.84% | **0.22%** |
| Chi phí ($/h) | 17.77 | 18.65 | 18.75 | 19.04 | 19.08 | 18.70 | 18.99 | **17.66** | 19.70 |
| Hao mòn pin (Ah/h) | 40 | 67 | 70 | 67 | 48 | 72 | 72 | 40 | **38** |
| Vi phạm ràng buộc khi train | – | **0** | **0** | **0** | 27 137 | 28 900 | 28 052 | – | – |

Số công việc PIRL-P thắng, tính theo g/kWh công:

| Kịch bản | PIRL | SAC | TD3 | DDPG | Rule | A-ECMS |
|---|---|---|---|---|---|---|
| Nominal | 6/10 | 6/10 | 6/10 | 7/10 | 7/10 | **0/10** |
| Mismatch | 7/10 | 7/10 | 8/10 | 10/10 | 9/10 | **0/10** |

## Đánh giá
- **PIRL-P và PIRL-P+ tốt hơn PIRL gốc** (khoảng 6% so với 8.5% khi so với DP). Bộ chọn action bằng physics critic có tác dụng thật.
- **So với DRL model-free, PIRL-P chỉ nhỉnh hơn nhẹ và không nhất quán.** Nó thắng TD3 ở 6/10 công việc. Ưu điểm rõ ràng duy nhất vẫn là 0 vi phạm ràng buộc.
- **Về pin, PIRL-P kém SAC:** hao mòn cao hơn khoảng 40%, và thiếu công nhiều hơn (1.9% so với 0.8%).
- **A-ECMS gần như tối ưu** (chỉ cách DP 0.6%) và thắng mọi phương pháp học ở cả 10 công việc. Với cấu hình này, tối ưu tức thời bằng mô hình vật lý đã đủ tốt: vai trò đệm năng lượng của pin nhỏ, và pha hạ cần tách riêng khỏi pha làm việc.

## Nguyên nhân và hướng tiếp theo
1. **Bộ chọn dùng tải hiện tại làm ước lượng cho tải bước sau.** Khi tải tăng đột ngột, nó hạ tốc độ bơm hoặc ICE quá mức nên thiếu công, rõ nhất ở heavy_dig, rock_excavation và dig_and_travel. Cần thêm biên an toàn tốc độ vào miền khả thi, hoặc một bộ dự báo tải ngắn hạn.
2. **Critic V học chưa đủ chính xác.** argmax khuếch đại sai số của V (chọn đúng những action mà V đánh giá quá cao). Hướng xử lý: dùng ensemble V, hoặc thêm phạt bảo thủ.
3. **Hướng hứa hẹn nhất: PIRL học phần hiệu chỉnh cho A-ECMS.** Giữ cấu trúc tối ưu tức thời của A-ECMS, và để mạng học hệ số tương đương s_eq(SOC, T_pin, tải, pha công việc) thay cho s₀ + k_p·ΔSOC. Như vậy PIRL không phải học lại phần A-ECMS đã làm tốt, chỉ cần học phần nhìn xa: dành chỗ trong pin trước pha hạ cần, và giảm hao mòn pin khi nóng.
