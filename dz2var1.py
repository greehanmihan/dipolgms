import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import math

# ========== ПАРАМЕТРЫ ВАРИАНТА 25 ==========
frequency = 0.5e9  # 0.5 ГГц (переименовано из f)
c = 3e8
lam = c / frequency
ratio = 1.4  # 2l/λ
l = ratio * lam / 2
k = 2 * np.pi / lam
kl = k * l

print("=" * 70)
print("РАСЧЁТ ДИАГРАММЫ НАПРАВЛЕННОСТИ СИММЕТРИЧНОГО ВИБРАТОРА")
print(f"Вариант 25: f = {frequency/1e9} ГГц, 2l/λ = {ratio}")
print("=" * 70)

# ========== РАСЧЁТ ДИАГРАММЫ НАПРАВЛЕННОСТИ ==========
# Углы от 0 до 180 градусов с шагом 0.5 градуса
theta = np.linspace(0, np.pi, 361)  # в радианах
theta_deg = np.degrees(theta)

# Нормированная характеристика направленности по полю
def F(theta_val):
    """Нормированная ДН по полю"""
    if np.abs(np.sin(theta_val)) < 1e-10:
        return 0.0
    numerator = np.cos(kl * np.cos(theta_val)) - np.cos(kl)
    denominator = np.sin(theta_val) * (1 - np.cos(kl))
    return numerator / denominator

# Вычисляем F(theta)
F_vals = np.array([F(th) for th in theta])
F_vals = np.nan_to_num(F_vals, nan=0.0)

# ========== РАСЧЁТ МАКСИМАЛЬНОГО КНД ==========
# Интегрируем F^2 * sin(theta) dtheta
integrand = F_vals**2 * np.sin(theta)
I = simpson(integrand, theta)
Dmax = 2 / I if I > 0 else 1.0

# КНД как функция угла
D_theta = F_vals**2 * Dmax
D_theta_dB = 10 * np.log10(np.maximum(D_theta, 1e-10))

print(f"\nПАРАМЕТРЫ АНТЕННЫ:")
print(f"  Частота: {frequency/1e9} ГГц")
print(f"  Длина волны: {lam:.6f} м")
print(f"  Длина вибратора 2l: {2*l:.6f} м")
print(f"  Длина плеча l: {l:.6f} м")
print(f"  Волновое число k: {k:.6f} рад/м")
print(f"  kl: {kl:.6f} рад")

print(f"\nРЕЗУЛЬТАТЫ РАСЧЁТА:")
print(f"  Максимальный КНД Dmax = {Dmax:.6f} раз")
print(f"  Максимальный КНД Dmax = {10*np.log10(Dmax):.6f} дБ")
print(f"  Угол с максимальным КНД: {theta_deg[np.argmax(D_theta)]:.1f} градусов")
print(f"  Значение КНД на 90°: {D_theta[180]:.6f} раз ({D_theta_dB[180]:.6f} дБ)")

# ========== СОХРАНЕНИЕ ДАННЫХ ==========
# Сохраняем результаты в CSV файл
data = np.column_stack([theta_deg, F_vals, D_theta, D_theta_dB])
np.savetxt('dipole_v25_analytic.csv', data,
           delimiter=',',
           header='theta_deg,F_norm,D_ratio,D_dB',
           fmt='%.6f',
           comments='')

# Сохраняем Dmax отдельно
with open('dipole_v25_Dmax.txt', 'w') as file_out:
    file_out.write(f"Dmax (разах) = {Dmax:.6f}\n")
    file_out.write(f"Dmax (дБ) = {10*np.log10(Dmax):.6f}\n")

print("\nСохранены файлы:")
print("  - dipole_v25_analytic.csv (полные данные)")
print("  - dipole_v25_Dmax.txt (максимальный КНД)")

# ========== ПОСТРОЕНИЕ ГРАФИКОВ ==========
# Создаем фигуру с 4 графиками (2x2)
fig = plt.figure(figsize=(14, 10))

# 1. Декартова система: КНД в разах
ax1 = plt.subplot(2, 2, 1)
ax1.plot(theta_deg, D_theta, 'b-', linewidth=2, label='Аналитический расчет')
ax1.set_xlabel('θ, градусы', fontsize=12)
ax1.set_ylabel('D(θ), разы', fontsize=12)
ax1.set_title('Диаграмма направленности (разы)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 180)

# 2. Декартова система: КНД в децибелах
ax2 = plt.subplot(2, 2, 2)
ax2.plot(theta_deg, D_theta_dB, 'r-', linewidth=2, label='Аналитический расчет')
ax2.set_xlabel('θ, градусы', fontsize=12)
ax2.set_ylabel('D(θ), дБ', fontsize=12)
ax2.set_title('Диаграмма направленности (дБ)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(0, 180)

# 3. Полярная система: КНД в разах
ax3 = plt.subplot(2, 2, 3, projection='polar')
ax3.plot(theta, D_theta, 'g-', linewidth=2, label='Разы')
ax3.set_theta_zero_location('N')  # 0 градусов сверху
ax3.set_theta_direction(-1)  # По часовой стрелке
ax3.set_title('Полярная диаграмма (разы)', fontsize=14, fontweight='bold', pad=20)
ax3.grid(True)
ax3.legend(loc='upper right')

# 4. Полярная система: КНД в децибелах
ax4 = plt.subplot(2, 2, 4, projection='polar')
ax4.plot(theta, D_theta_dB, 'm-', linewidth=2, label='дБ')
ax4.set_theta_zero_location('N')
ax4.set_theta_direction(-1)
ax4.set_title('Полярная диаграмма (дБ)', fontsize=14, fontweight='bold', pad=20)
ax4.grid(True)
ax4.legend(loc='upper right')

# Добавляем информацию об антенне (исправленная строка)
fig.suptitle(f'Симметричный вибратор: f = {frequency/1e9} ГГц, 2l/λ = {ratio}\n' +
             f'Dmax = {Dmax:.3f} раз ({10*np.log10(Dmax):.3f} дБ)',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('dipole_v25_patterns.png', dpi=300, bbox_inches='tight')
print("\nСоздан график: dipole_v25_patterns.png")

# ========== ТЕКСТОВАЯ ВИЗУАЛИЗАЦИЯ В КОНСОЛИ ==========
print("\n" + "=" * 70)
print("ТЕКСТОВАЯ ДИАГРАММА НАПРАВЛЕННОСТИ")
print("=" * 70)

max_D = np.max(D_theta)
angles_to_show = [0, 30, 60, 90, 120, 150, 180]
indices = [int(angle/0.5) for angle in angles_to_show]

for idx in indices:
    deg = theta_deg[idx]
    val = D_theta[idx]
    dB = D_theta_dB[idx]
    percentage = int(100 * val / max_D)
    bar = '█' * (percentage // 2)
    print(f"θ = {deg:3.0f}°: {bar:25} {val:7.3f} раз ({dB:7.3f} дБ)")

# ========== ГЕНЕРАЦИЯ ОТЧЁТА ==========
print("\n" + "=" * 70)
print("ИНСТРУКЦИЯ ДЛЯ ОТЧЁТА:")
print("=" * 70)
print("1. Для отчёта используйте файлы:")
print("   - dipole_v25_patterns.png (все графики)")
print("   - dipole_v25_analytic.csv (табличные данные)")
print("   - dipole_v25_Dmax.txt (значение Dmax)")
print("\n2. Для моделирования в программе ЭМ моделирования:")
print(f"   Частота: {frequency/1e9} ГГц")
print(f"   Длина вибратора: {2*l:.6f} м")
print(f"   Длина плеча: {l:.6f} м")
print("\n3. После моделирования добавьте данные в графики:")
print("   - Загрузите данные моделирования в CSV формате")
print("   - Добавьте вторую линию на графики с пометкой 'Моделирование'")

plt.show()