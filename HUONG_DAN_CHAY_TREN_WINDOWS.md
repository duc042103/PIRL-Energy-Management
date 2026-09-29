# Hướng dẫn tải và chạy lại code trên Windows

Tài liệu này hướng dẫn từng bước để tải toàn bộ code từ GitHub về máy Windows và chạy lại ba thí
nghiệm PIRL (`PIRL/`, `PIRL_excavator/`, `PIRL_excavator_ps/`). Viết cho người chưa quen dùng
dòng lệnh, nên mỗi bước đều ghi rõ lệnh gõ vào và cách kiểm tra đã đúng chưa.

Áp dụng cho Windows 10/11. Thời gian ước tính cho toàn bộ phần cài đặt: 15–20 phút.

---

## Phần 1. Cài các phần mềm cần thiết

### 1.1. Cài Python

1. Vào https://www.python.org/downloads/ , bấm nút tải bản mới nhất (Python 3.11 hoặc 3.12 đều
   được — tránh Python 3.13 nếu có thể vì một số thư viện có thể chưa hỗ trợ kịp).
2. Chạy file cài đặt vừa tải. **Ở màn hình đầu tiên, nhất định phải tích vào ô "Add python.exe to
   PATH"** ở dưới cùng trước khi bấm "Install Now". Nếu bỏ qua bước này, các lệnh `python` ở
   dưới sẽ báo lỗi "not recognized".
3. Cài xong, mở **PowerShell** (bấm nút Start, gõ `powershell`, Enter) và gõ:
   ```powershell
   python --version
   ```
   Nếu hiện ra ví dụ `Python 3.11.9` là đã cài đúng. Nếu báo lỗi, xem mục **Xử lý sự cố** ở cuối
   tài liệu (mục A1).

### 1.2. Cài Git

1. Vào https://git-scm.com/download/win , tải bản Windows (thường tự động bắt đầu tải).
2. Chạy file cài đặt, cứ bấm "Next" theo mặc định cho tới khi cài xong (không cần đổi gì).
3. Đóng PowerShell đang mở và mở lại một cửa sổ **PowerShell mới** (để nó nhận Git vừa cài), gõ:
   ```powershell
   git --version
   ```
   Thấy hiện ra số phiên bản (ví dụ `git version 2.46.0`) là đã cài đúng.

### 1.3. (Không bắt buộc) Cài một trình soạn thảo code

Nếu muốn mở đọc code, khuyên cài **VS Code**: https://code.visualstudio.com/ — tải bản Windows,
cài mặc định. Không bắt buộc để chạy được code, chỉ giúp đọc/sửa code dễ hơn.

---

## Phần 2. Tải code từ GitHub về máy

### 2.1. Chọn nơi lưu code

Mở PowerShell, chọn một thư mục để chứa code, ví dụ thư mục `Code` trong ổ D:
```powershell
cd D:\
mkdir Code
cd D:\Code
```
(Nếu đã có thư mục rồi thì bỏ qua `mkdir`, chỉ cần `cd D:\Code`.)

### 2.2. Tải code bằng Git

Gõ lệnh sau (thay đúng địa chỉ repo của bạn):
```powershell
git clone https://github.com/duc042103/PIRL-Energy-Management.git
```
Đợi vài giây tới vài chục giây tuỳ tốc độ mạng. Xong sẽ thấy dòng `done.` hiện ra, và có một
thư mục mới tên `PIRL-Energy-Management` trong `D:\Code`.

> **Cách khác nếu không muốn dùng Git:** vào trang GitHub của repo trên trình duyệt, bấm nút xanh
> **Code** → **Download ZIP**, tải về rồi giải nén (chuột phải → Extract All...) vào `D:\Code`.
> Cách này đơn giản hơn nhưng sau này muốn cập nhật code mới thì phải tải lại từ đầu, còn cách
> dùng `git clone` ở trên cho phép cập nhật bằng lệnh `git pull` (xem mục 5.3).

### 2.3. Vào đúng thư mục code

```powershell
cd D:\Code\PIRL-Energy-Management
dir
```
Lệnh `dir` sẽ liệt kê các thư mục/tệp. Kiểm tra phải thấy đủ:
```
Data_Standard Driving Cycles
PIRL
PIRL_excavator
PIRL_excavator_ps
PIRL_README.md
README.md
```
Nếu thấy đủ như trên là tải code thành công.

> **Lưu ý về khoảng trắng trong tên thư mục:** thư mục `Data_Standard Driving Cycles` có dấu cách
> trong tên. Khi gõ lệnh `cd` vào thư mục này thì phải để trong ngoặc kép, ví dụ
> `cd "Data_Standard Driving Cycles"`. Các script Python trong repo đã tự xử lý việc này, bạn
> không cần `cd` vào đó — chỉ cần chạy mọi lệnh Python từ thư mục gốc
> `D:\Code\PIRL-Energy-Management` là được.

---

## Phần 3. Tạo môi trường Python riêng và cài thư viện

Bước này tạo một "môi trường ảo" (virtual environment) riêng cho project, để không ảnh hưởng tới
các phần mềm Python khác trên máy.

### 3.1. Tạo môi trường ảo

Đứng ở thư mục `D:\Code\PIRL-Energy-Management`, gõ:
```powershell
python -m venv venv
```
Lệnh này tạo ra một thư mục con tên `venv` chứa Python riêng cho project. Mất khoảng 10–20 giây.

### 3.2. Kích hoạt môi trường ảo

```powershell
.\venv\Scripts\Activate.ps1
```
Nếu thành công, đầu dòng lệnh trong PowerShell sẽ xuất hiện chữ `(venv)` ở phía trước, ví dụ:
```
(venv) PS D:\Code\PIRL-Energy-Management>
```

Nếu PowerShell báo lỗi đỏ dạng **"...cannot be loaded because running scripts is disabled on this
system..."**, đó là do Windows chặn chạy script theo mặc định. Cách xử lý — xem mục **A2** ở phần
Xử lý sự cố cuối tài liệu.

> Mỗi lần mở PowerShell mới để chạy lại code, phải kích hoạt lại môi trường ảo bằng đúng lệnh ở
> bước 3.2 này (chỉ cần làm bước 3.1 một lần duy nhất).

### 3.3. Cài các thư viện cần thiết

Khi đã thấy `(venv)` ở đầu dòng lệnh, gõ:
```powershell
python -m pip install --upgrade pip
pip install torch numpy scipy matplotlib
```
Lệnh `pip install torch ...` sẽ tải khoảng 200–800 MB tuỳ phiên bản, có thể mất vài phút tuỳ tốc
độ mạng. Đợi tới khi thấy dòng `Successfully installed ...` là xong.

### 3.4. Kiểm tra cài đặt đúng

```powershell
python -c "import torch, numpy, scipy, matplotlib; print('OK, torch', torch.__version__)"
```
Nếu in ra dòng `OK, torch 2.x.x` (không có dòng lỗi màu đỏ nào) là mọi thứ đã cài đúng, sẵn sàng
chạy code.

---

## Phần 4. Chạy thử code

Từ đây, mọi lệnh đều chạy từ thư mục gốc `D:\Code\PIRL-Energy-Management`, và **phải đang ở trong
môi trường ảo** (thấy `(venv)` ở đầu dòng lệnh — nếu tắt PowerShell rồi mở lại, làm lại bước 3.2).

### 4.1. Chạy thử nhanh nhất (khoảng 2 phút)

```powershell
python PIRL\pirl_hev.py
```
Lệnh này train một agent PIRL trên mô hình xe hybrid và so sánh với một luật đơn giản. Chạy xong
sẽ thấy log in ra từng episode, và tạo ra 2 file:
- `PIRL\pirl_actor.pt` — trọng số mạng đã train
- `PIRL\pirl_soc_nedc.png` — hình vẽ SOC pin theo thời gian

Mở file `.png` bằng Photos (bấm đúp chuột vào file trong File Explorer) để xem kết quả.

### 4.2. Xem lại kết quả benchmark đã có sẵn (không cần chạy lại)

Toàn bộ kết quả train và đánh giá của cả 3 thí nghiệm đã được lưu sẵn trong repo (thư mục `runs/`
và `results/` của mỗi thí nghiệm), **không cần train lại vẫn xem được**:

```powershell
notepad PIRL\BENCHMARK.md
notepad PIRL_excavator\BENCHMARK.md
notepad PIRL_excavator_ps\BENCHMARK.md
```
(hoặc mở các file này trực tiếp bằng VS Code / trình duyệt để hiển thị markdown đẹp hơn). Các
hình `.png` trong từng thư mục `results\` xem trực tiếp bằng Photos.

### 4.3. Chạy lại toàn bộ benchmark (train từ đầu)

Việc train từ đầu tốn nhiều thời gian (xem bảng thời gian ước tính bên dưới) và không bắt buộc.
Nếu muốn thử, làm theo đúng các bước trong file **`PIRL_README.md`** ở thư mục gốc — file đó liệt
kê chính xác từng lệnh cho cả 3 thí nghiệm. Trên Windows, chỉ cần đổi 2 điều so với hướng dẫn
trong file đó:
- Dấu `/` trong đường dẫn thư mục đổi thành `\` khi gõ trong PowerShell, ví dụ
  `PIRL_excavator\bench.py` thay vì `PIRL_excavator/bench.py`. (Nhưng bên trong các lệnh Python
  `--plant`, `--job`, `--algo`, ... thì giữ nguyên như trong tài liệu, không đổi gì.)
- Nếu tài liệu chỉ dẫn `cd PIRL_excavator` rồi chạy `python bench.py ...`, có thể copy y nguyên
  vì `cd` trong PowerShell cũng hoạt động như vậy.

Ví dụ chạy một lượt train PIRL cho thí nghiệm máy xúc power-split:
```powershell
cd PIRL_excavator_ps
python bench.py train --algo pirl --gamma 0.99 --seed 1
cd ..
```

### 4.4. Bảng thời gian ước tính (máy 4 nhân CPU, không cần GPU)

| Việc | Thời gian |
|---|---|
| `python PIRL\pirl_hev.py` | ~2 phút |
| 1 lượt train `PIRL\benchmark.py train` (30 episode) | ~4 phút |
| 1 lượt train `PIRL_excavator\bench.py train` | ~4–15 phút tuỳ thuật toán |
| 1 lượt train `PIRL_excavator_ps\bench.py train --algo pirl` | ~25 phút |
| 1 lượt train `PIRL_excavator_ps\bench.py train --algo pirlp` | ~35 phút |
| 1 lượt `bench.py evaluate` (1 công việc × 1 kịch bản) | ~2–3.5 phút |
| `report.py` (vẽ lại bảng + hình) | vài giây tới ~1 phút |

Máy càng nhiều nhân CPU thì có thể chạy song song nhiều lượt train cùng lúc để nhanh hơn — xem ví
dụ trong `PIRL_README.md`.

---

## Phần 5. Các thao tác thường dùng về sau

### 5.1. Mở lại project vào lần sau

Mỗi lần mở máy lại và muốn chạy code:
```powershell
cd D:\Code\PIRL-Energy-Management
.\venv\Scripts\Activate.ps1
```
rồi chạy lệnh Python như bình thường (không cần cài lại thư viện, không cần `git clone` lại).

### 5.2. Thoát môi trường ảo

```powershell
deactivate
```

### 5.3. Cập nhật code mới nhất từ GitHub

Nếu sau này repo trên GitHub có cập nhật, tải bản mới về bằng:
```powershell
cd D:\Code\PIRL-Energy-Management
git pull
```
(chỉ dùng được nếu ban đầu tải bằng `git clone` ở mục 2.2, không dùng được nếu tải bằng ZIP).

---

## Phần 6. Xử lý sự cố thường gặp

**A1. Gõ `python --version` báo "python is not recognized as an internal or external command"**
→ Khi cài Python đã quên tích ô "Add python.exe to PATH". Gỡ cài đặt Python (vào Settings → Apps),
cài lại từ đầu, nhớ tích đúng ô đó ở màn hình cài đặt đầu tiên.

**A2. Kích hoạt venv báo lỗi "running scripts is disabled on this system"**
→ Mở PowerShell **với quyền Administrator** (chuột phải vào biểu tượng PowerShell → "Run as
administrator"), gõ:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
gõ `Y` rồi Enter khi được hỏi xác nhận. Sau đó mở lại PowerShell bình thường (không cần quyền
Administrator nữa), `cd` vào lại thư mục project và làm lại bước 3.2.

**A3. `pip install torch ...` chạy rất chậm hoặc báo lỗi mạng**
→ Kiểm tra lại kết nối mạng. Nếu mạng công ty/trường học chặn, thử dùng mạng khác (ví dụ phát wifi
từ điện thoại) rồi chạy lại đúng lệnh ở mục 3.3.

**A4. Chạy code báo lỗi `ModuleNotFoundError: No module named 'torch'` (hoặc numpy/scipy/matplotlib)**
→ Quên kích hoạt môi trường ảo. Kiểm tra đầu dòng lệnh có chữ `(venv)` chưa; nếu chưa, làm lại
bước 3.2 rồi chạy lại.

**A5. Chạy code báo lỗi liên quan tới đường dẫn có chứa "Data_Standard Driving Cycles"**
→ Đảm bảo bạn đang chạy lệnh Python **từ đúng thư mục gốc**
`D:\Code\PIRL-Energy-Management` (kiểm tra bằng lệnh `pwd` xem có đúng thư mục này không), không
phải từ một thư mục con. Các script tự tìm tới thư mục dữ liệu theo đường dẫn tương đối kể từ nơi
bạn đứng chạy lệnh.

**A6. Không thấy hình `.png` mới được tạo ra, hoặc chạy `report.py` báo lỗi liên quan tới
matplotlib/hiển thị**
→ Các script đã tự đặt `matplotlib` chạy ở chế độ không cần màn hình (chỉ lưu file ảnh, không mở
cửa sổ), nên không cần cài thêm gì. Nếu vẫn báo lỗi, gửi lại đúng dòng lỗi để kiểm tra tiếp — có
thể phiên bản matplotlib vừa cài không tương thích, thử `pip install --upgrade matplotlib`.

**A7. Muốn gỡ bỏ toàn bộ để cài lại từ đầu**
→ Xoá thư mục `D:\Code\PIRL-Energy-Management` (chuột phải → Delete), rồi làm lại từ Phần 2.
