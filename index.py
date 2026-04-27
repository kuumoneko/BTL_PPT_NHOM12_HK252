import numpy as np
import matplotlib 
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ==========================================
# THÔNG SỐ BÀI TOÁN
# ==========================================
L = 1.0       # Chiều dài miền [0, L]
N = 50        # Số khoảng chia lưới (số nút bên trong là N-1)

# Định nghĩa hàm tải trọng (vế phải phương trình)
def f_load(x):
    return np.full_like(x,10.0)
    # return (np.pi**2) * np.sin(np.pi * x)

# Định nghĩa nghiệm chính xác (để so sánh)
def u_exact(x):
    return np.full_like(x,1.0)
    # return np.sin(np.pi * x)

# ==========================================
# BƯỚC 1: RỜI RẠC HÓA KHÔNG GIAN
# ==========================================
h = L / N                        # Kích thước bước lưới
x = np.linspace(0, L, N + 1)     # Tọa độ các điểm nút toàn cục
x_interior = x[1:-1]             # Tọa độ các nút bên trong (bỏ 2 nút biên)
n_interior = N - 1               # Số lượng ẩn số cần tìm

# ==========================================
# BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU TẢI
# ==========================================
# Tính giá trị tải trọng tại các nút bên trong
F = f_load(x_interior)

# ==========================================
# BƯỚC 3: LẮP RÁP MA TRẬN ĐỘ CỨNG THƯA
# ==========================================
# Dùng sai phân trung tâm: -u''(x) ~ (-u_{i-1} + 2u_i - u_{i+1}) / h^2
# Tạo các đường chéo của ma trận A
main_diag = np.full(n_interior, 2.0 / h**2)    # Đường chéo chính
off_diag = np.full(n_interior - 1, -1.0 / h**2) # Đường chéo phụ

# Khởi tạo ma trận thưa A định dạng CSR để tính toán cực nhanh
diagonals = [main_diag, off_diag, off_diag]
offsets = [0, -1, 1]
A = diags(diagonals, offsets, format='csr')

# ==========================================
# BƯỚC 4: GIẢI HỆ PHƯƠNG TRÌNH ĐẠI SỐ
# ==========================================
# Giải hệ A * U = F bằng bộ giải ma trận thưa
U_interior = spsolve(A, F)

# ==========================================
# BƯỚC 5: LẮP RÁP NGHIỆM TOÀN CỤC
# ==========================================
# Khởi tạo mảng nghiệm với giá trị 0 (bao hàm luôn điều kiện biên u(0)=0, u(L)=0)
U_num = np.zeros(N + 1)
# Gắn nghiệm của các nút bên trong vào mảng toàn cục
U_num[1:-1] = U_interior

# ==========================================
# BƯỚC 6: TÍNH SAI SỐ VÀ TRỰC QUAN HÓA
# ==========================================
# Tính nghiệm chính xác tại các nút
U_ex = u_exact(x)

# Tính sai số chuẩn L2 rời rạc (có nhân với bước lưới h)
L2_error = np.sqrt(h * np.sum((U_num - U_ex)**2))
print(f"Số lượng phần tử lưới (N) : {N}")
print(f"Chuẩn sai số L2           : {L2_error:.4e}")

# Vẽ đồ thị
plt.figure(figsize=(9, 5))
plt.plot(x, U_ex, 'r-', linewidth=2, label='Nghiệm chính xác (Giải tích)')
plt.plot(x, U_num, 'bo', markersize=5, markerfacecolor='none', label='Nghiệm số trị (FDM)')

plt.title('Giải Bài toán Giá trị Biên 1D bằng FDM')
plt.xlabel('Tọa độ x')
plt.ylabel('Giá trị u(x)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()