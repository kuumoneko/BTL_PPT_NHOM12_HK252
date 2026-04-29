import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from numba import njit
import gc

# ==========================================
# 1. HÀM THUẬT TOÁN THOMAS (TDMA)
# ==========================================
@njit
def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n-1, dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    if n == 1:
        x[0] = d[0] / b[0]
        return x
    # Bước 1: Khử tiến (Forward sweep)
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n-1):
        m = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / m
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / m
    m = b[n-1] - a[n-2] * c_prime[n-2]
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / m
    # Bước 2: Thế ngược (Back substitution)
    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    return x

# ==========================================
# 2. HÀM GIẢI FDM TÍCH HỢP THOMAS
# ==========================================
def fdm_thomas(L_val, N, g_func_num, u_exact_func_num):
    x_num = np.linspace(0, L_val, N)
    h = L_val / (N - 1)
    g_discrete = g_func_num(x_num)
    if np.isscalar(g_discrete):
        g_discrete = np.full_like(x_num, g_discrete)
    num_interior = N - 2
    d = g_discrete[1:-1] * (h**2)
    b = np.full(num_interior, -2.0)
    a = np.full(num_interior - 1, 1.0)
    c = np.full(num_interior - 1, 1.0)
    u_interior = thomas_algorithm(a, b, c, d)
    # Ghép điều kiện biên u(0) = 0, u(L) = 0
    u_fdm = np.zeros(N)
    u_fdm[1:-1] = u_interior
    # Tính sai số L2
    u_exact_vals = u_exact_func_num(x_num)
    l2_error = np.sqrt(h * np.sum((u_fdm - u_exact_vals)**2))
    return x_num, u_fdm, u_exact_vals, l2_error, g_discrete

    
# ==========================================
# 3. THIẾT LẬP BÀI TOÁN & GIẢI TÍCH (CHỈ CHẠY 1 LẦN)
# ==========================================    
x = sp.Symbol('x')
g_input = input("Nhập hàm g(x) (ví dụ: exp(x), x**2, sin(x)): ")
L = float(input("Nhập chiều dài thanh sắt L: "))
N_single = int(input("Nhập số nút rời rạc hóa N: "))

try:
    g = sp.sympify(g_input)
    # Kiểm tra nếu g vẫn là Equality sau khi sympify (trường hợp hiếm)
    if isinstance(g, sp.Equality):
        g = g.rhs # Lấy vế phải của phương trình
    print(f"Hàm g(x) đã nhận: {g}")
except Exception as e:
    print(f"Lỗi: Hàm nhập vào không hợp lệ! Chi tiết: {e}")
    exit()

print("Đang giải phương trình giải tích bằng SymPy...")
# Tích phân lần 1: u'(x) = integral(g) + C1
C1, C2 = sp.symbols('C1 C2')
du = sp.integrate(g, x) + C1

# Tích phân lần 2: u(x) = integral(u'(x)) + C2
sol_exact_expr_gen = sp.integrate(du, x) + C2

# Giải hệ phương trình tìm hằng số từ điều kiện biên u(0)=0 và u(L)=0
constants = sp.solve([
    sol_exact_expr_gen.subs(x, 0), 
    sol_exact_expr_gen.subs(x, L)
], (C1, C2))

# Thế hằng số vào để có nghiệm cuối cùng
sol_exact_expr = sol_exact_expr_gen.subs(constants)
g_func_num = sp.lambdify(x, g, "numpy")
u_exact_func_num = sp.lambdify(x, sol_exact_expr, "numpy")    
    
print("Hoàn tất thiết lập!\n")

# ==========================================
# 4. IN BẢNG GIÁ TRỊ VÀ SAI SỐ CHO N=51
# ==========================================
x_vals, u_fdm_vals, u_exact_vals, l2_err, g_vals = fdm_thomas(L, N_single, g_func_num, u_exact_func_num)

print("--- THÔNG TIN BÀI TOÁN ---")
print(f"Chiều dài L = {L}, Số nút N = {N_single}")
print(f"Hàm u(x) = {sol_exact_expr}")
print(f"Sai số chuẩn L2: {l2_err:.4e}\n")

print("--- BẢNG GIÁ TRỊ SO SÁNH ---")
header = f"| {'Tọa độ x':^10} | {'g(x)':^18} | {'u_FDM (Giải số)':^22} | {'u_Exact (Giải tích)':^22} | {'Sai số tuyệt đối':^18} |"
print("-" * len(header))
print(header)
print("-" * len(header))

for i in range(0, N_single, 2):
    xi = x_vals[i]
    gi = g_vals[i]
    u_num = u_fdm_vals[i]
    u_ex = u_exact_vals[i]
    err_abs = abs(u_num - u_ex)
    
    row = f"| {xi:10.2f} | {gi:18.4f} | {u_num:22.15f} | {u_ex:22.15f} | {err_abs:18.4e} |"
    print(row)
print("-" * len(header))

# ==========================================
# 5. ĐÁNH GIÁ BẬC HỘI TỤ 
# ==========================================
print("\n--- ĐANG TÍNH TOÁN ĐÁNH GIÁ HỘI TỤ BẰNG THUẬT TOÁN THOMAS ---")
num_levels = 25
N_list = [10 * (2**i) + 1 for i in range(num_levels)]

h_list = []
err_list = []

for n_nodes in N_list:
    print(n_nodes)
    _, _, _, err, _ = fdm_thomas(L, n_nodes, g_func_num, u_exact_func_num)
    h = L / (n_nodes - 1)
    h_list.append(h)
    err_list.append(err)
    gc.collect()

print(f"Đã tính xong sai số cho các lưới: {N_list}")

# ==========================================
# 6. VẼ ĐỒ THỊ 
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Đồ thị so sánh nghiệm
ax1.plot(x_vals, u_fdm_vals, 'ro', label='Nghiệm FDM', markersize=5)
ax1.plot(x_vals, u_exact_vals, 'b-', label='Nghiệm giải tích', linewidth=1.5)
ax1.set_title(f'So sánh nghiệm u(x) | N={N_single}')
ax1.set_xlabel('Tọa độ x')
ax1.set_ylabel('Độ võng u(x)')
ax1.axhline(0, color='black', linewidth=1)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Subplot 2: Đồ thị sai số theo h (Tuyến tính)
ax2.plot(h_list, err_list, 'b-o', linewidth=2, markersize=8)
ax2.set_title('Sai số L2 theo bước lưới h')
ax2.set_xlabel('Bước lưới h')
ax2.set_ylabel('Sai số L2')
ax2.grid(True, linestyle='--', alpha=0.7)

# Subplot 3: Đồ thị hội tụ Log-Log
ax3.loglog(h_list, err_list, 'k-o', label='Sai số L2 (Thực tế)', linewidth=2, markersize=8)

# Tham chiếu O(h^2)
h_ref = np.array(h_list)
err_ref = err_list[0] * (h_ref / h_ref[0])**2  
ax3.loglog(h_ref, err_ref, 'r--', label='Tham chiếu lý thuyết $\mathcal{O}(h^2)$')

ax3.set_title('Đánh giá bậc hội tụ (Thang đo Log-Log)')
ax3.set_xlabel('Bước lưới h')
ax3.set_ylabel('Sai số L2')
ax3.grid(True, which="both", linestyle='--', alpha=0.7)
ax3.legend()

plt.tight_layout()
plt.show()