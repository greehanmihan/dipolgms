import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import os

# ПАРАМЕТРЫ ВАРИАНТА 25
def print_header():
    print("=" * 70)
    print("РАСЧЕТ ДИАГРАММЫ НАПРАВЛЕННОСТИ СИММЕТРИЧНОГО ВИБРАТОРА")
    print("=" * 70)
    print("ВАРИАНТ 25")
    print("-" * 70)
    print(f"Рабочая частота: f = {freq / 1e9} ГГц")
    print(f"Отношение длины вибратора к длине волны: 2l/λ = {ratio}")
    print(f"Длина волны: λ = {lam:.6f} м")
    print(f"Длина одного плеча: l = {l:.6f} м")
    print("=" * 70)


# Параметры (переименовал f в freq чтобы избежать конфликта)
freq = 0.5e9  # ГГц -> Гц (было f, переименовал в freq)
c = 3e8  # скорость света
lam = c / freq  # длина волны
ratio = 1.4  # 2l / λ
l = ratio * lam / 2  # длина одного плеча
k = 2 * np.pi / lam  # волновое число

# Вывод параметров
print_header()

# РАСЧЕТНЫЕ ФУНКЦИИ
def F_theta(theta):
    if np.isclose(theta, 0) or np.isclose(theta, np.pi):
        return 0.0
    num = np.cos(k * l * np.cos(theta)) - np.cos(k * l)
    den = np.sin(theta)
    return num / den


def F_sq(theta):
    """Квадрат нормированной ДН."""
    val = F_theta(theta)
    return val ** 2


def calculate_Dmax():
    def integrand(theta):
        return F_sq(theta) * np.sin(theta)

    # Интегрирование по θ от 0 до π
    result, error = integrate.quad(integrand, 0, np.pi)

    if result > 0:
        Dmax = 4 * np.pi / result
    else:
        Dmax = 1.0  # минимальное значение

    return Dmax, error


def calculate_D_pattern(theta_values):
    D_linear = np.array([F_sq(t) * Dmax for t in theta_values])
    D_dB = 10 * np.log10(D_linear)

    D_dB_fixed = np.where(np.isinf(D_dB),
                          np.nanmin(D_dB[np.isfinite(D_dB)]),
                          D_dB)

    return D_linear, D_dB_fixed

# ОСНОВНЫЕ РАСЧЕТЫ
print("\nВЫПОЛНЕНИЕ РАСЧЕТОВ...")
print("-" * 70)

# Расчет Dmax
Dmax, error = calculate_Dmax()
print(f"Максимальное значение КНД:")
print(f"  Dmax = {Dmax:.6f} (в разах)")
print(f"  Dmax = {10 * np.log10(Dmax):.6f} дБ")
print(f"  Погрешность интегрирования: {error:.2e}")

# Подготовка углов для расчетов
theta_rad = np.linspace(0, np.pi, 361)  # 0-180° с шагом 0.5°
theta_deg = np.rad2deg(theta_rad)  # в градусах

# Расчет ДН
D_analytic_linear, D_analytic_dB = calculate_D_pattern(theta_rad)

print(f"\nДиапазон значений КНД:")
print(f"  В разах: от {np.min(D_analytic_linear):.6f} до {np.max(D_analytic_linear):.6f}")
print(f"  В дБ: от {np.min(D_analytic_dB):.6f} до {np.max(D_analytic_dB):.6f}")


# ЗАГРУЗКА ДАННЫХ ИЗ CST STUDIO
def load_cst_data():
    cst_files = {
        'cart_linear': 'cst_cartesian_linear.txt',
        'cart_dB': 'cst_cartesian_dB.txt',
        'polar_linear': 'cst_polar_linear.txt',
        'polar_dB': 'cst_polar_dB.txt'
    }

    print("\n" + "-" * 70)
    print("ЗАГРУЗКА ДАННЫХ ИЗ CST STUDIO")
    print("-" * 70)

    try:
        # Загрузка декартовых данных
        cst_cart_linear = np.loadtxt(cst_files['cart_linear'])
        theta_cst_cart_deg = cst_cart_linear[:, 0]
        D_cst_cart_linear = cst_cart_linear[:, 1]

        cst_cart_dB = np.loadtxt(cst_files['cart_dB'])
        D_cst_cart_dB = cst_cart_dB[:, 1]

        # Загрузка полярных данных
        cst_polar_linear = np.loadtxt(cst_files['polar_linear'])
        theta_cst_polar_deg = cst_polar_linear[:, 0]
        D_cst_polar_linear = cst_polar_linear[:, 1]

        cst_polar_dB = np.loadtxt(cst_files['polar_dB'])
        D_cst_polar_dB = cst_polar_dB[:, 1]

        print("Данные успешно загружены:")
        print(f"  Декартовы данные: {len(theta_cst_cart_deg)} точек")
        print(f"  Полярные данные: {len(theta_cst_polar_deg)} точек")

        return (theta_cst_cart_deg, D_cst_cart_linear, D_cst_cart_dB,
                theta_cst_polar_deg, D_cst_polar_linear, D_cst_polar_dB)

    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None


# Загрузка данных CST
cst_data = load_cst_data()


# ============================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================
def create_cartesian_plots():
    """Создание декартовых графиков."""
    print("\n" + "-" * 70)
    print("СОЗДАНИЕ ДЕКАРТОВЫХ ГРАФИКОВ")
    print("-" * 70)

    # 1. График в разах
    plt.figure(figsize=(12, 6))

    # Аналитический расчет - синяя сплошная линия
    plt.plot(theta_deg, D_analytic_linear, 'b-', linewidth=3,
             label='Аналитический расчет', alpha=0.7)

    if cst_data is not None:
        theta_cst_cart_deg, D_cst_cart_linear, _, _, _, _ = cst_data
        # Данные из CST - красная пунктирная линия (будет совпадать с синей)
        plt.plot(theta_cst_cart_deg, D_cst_cart_linear, 'r--',
                 linewidth=2, label='CST Studio', alpha=0.9)

    plt.xlabel('Угол θ, градусы', fontsize=12)
    plt.ylabel('Коэффициент направленного действия D(θ), разы', fontsize=12)
    plt.title(f'Диаграмма направленности симметричного вибратора (разы)\n'
              f'Вариант 25: f = {freq / 1e9} ГГц, 2l/λ = {ratio}, Dmax = {Dmax:.3f}',
              fontsize=14, pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.xlim(0, 180)
    plt.xticks(np.arange(0, 181, 30))
    plt.tight_layout()
    plt.savefig('cartesian_linear_v25.png', dpi=300, bbox_inches='tight')
    print("  Сохранен: cartesian_linear_v25.png")

    if cst_data is not None:
        theta_cst_cart_deg, D_cst_cart_linear, _, _, _, _ = cst_data
        D_analytic_interp = np.interp(theta_cst_cart_deg, theta_deg, D_analytic_linear)
        diff = np.max(np.abs(D_cst_cart_linear - D_analytic_interp))
        print(f"  Максимальное расхождение (разы): {diff:.6e}")

    # 2. График в дБ
    plt.figure(figsize=(12, 6))

    # Аналитический расчет - синяя сплошная линия
    plt.plot(theta_deg, D_analytic_dB, 'b-', linewidth=3,
             label='Аналитический расчет', alpha=0.7)

    if cst_data is not None:
        theta_cst_cart_deg, _, D_cst_cart_dB, _, _, _ = cst_data
        # Данные из CST - красная пунктирная линия
        plt.plot(theta_cst_cart_deg, D_cst_cart_dB, 'r--',
                 linewidth=2, label='CST Studio', alpha=0.9)

    plt.xlabel('Угол θ, градусы', fontsize=12)
    plt.ylabel('Коэффициент направленного действия D(θ), дБ', fontsize=12)
    plt.title(f'Диаграмма направленности симметричного вибратора (дБ)\n'
              f'Вариант 25: f = {freq / 1e9} ГГц, 2l/λ = {ratio}, Dmax = {10 * np.log10(Dmax):.2f} дБ',
              fontsize=14, pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.xlim(0, 180)
    plt.xticks(np.arange(0, 181, 30))
    plt.tight_layout()
    plt.savefig('cartesian_dB_v25.png', dpi=300, bbox_inches='tight')
    print("  Сохранен: cartesian_dB_v25.png")

    if cst_data is not None:
        theta_cst_cart_deg, _, D_cst_cart_dB, _, _, _ = cst_data
        # Интерполируем аналитические данные на сетку CST для сравнения
        D_analytic_dB_interp = np.interp(theta_cst_cart_deg, theta_deg, D_analytic_dB)
        diff_dB = np.max(np.abs(D_cst_cart_dB - D_analytic_dB_interp))
        print(f"  Максимальное расхождение (дБ): {diff_dB:.6e}")


def create_polar_plots():
    """Создание полярных графиков."""
    print("\n" + "-" * 70)
    print("СОЗДАНИЕ ПОЛЯРНЫХ ГРАФИКОВ")
    print("-" * 70)

    # Для полярных графиков используем данные CST как основу
    if cst_data is not None:
        _, _, _, theta_cst_polar_deg, _, _ = cst_data
        theta_polar_rad = np.deg2rad(theta_cst_polar_deg)
    else:
        # Если данных CST нет, создаем свою сетку
        theta_polar_rad = np.linspace(0, 2 * np.pi, 721)

    # Подготовка аналитических данных для полного круга
    theta_half_rad = np.linspace(0, np.pi, 181)
    D_half_linear, D_half_dB = calculate_D_pattern(theta_half_rad)

    D_polar_analytic_linear = np.concatenate([D_half_linear[:-1], D_half_linear[::-1]])
    D_polar_analytic_dB = np.concatenate([D_half_dB[:-1], D_half_dB[::-1]])

    theta_analytic_polar_rad = np.linspace(0, 2 * np.pi, len(D_polar_analytic_linear))

    # 1. Полярный график в разах
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')

    # Аналитический расчет - синяя сплошная линия
    ax.plot(theta_analytic_polar_rad, D_polar_analytic_linear, 'b-',
            linewidth=3, label='Аналитический расчет', alpha=0.7)

    if cst_data is not None:
        _, _, _, theta_cst_polar_deg, D_cst_polar_linear, _ = cst_data
        theta_cst_polar_rad = np.deg2rad(theta_cst_polar_deg)
        # Данные из CST - красная пунктирная линия (будет совпадать с синей)
        ax.plot(theta_cst_polar_rad, D_cst_polar_linear, 'r--',
                linewidth=2, label='CST Studio', alpha=0.9)

    ax.set_theta_zero_location('N')  # 0° сверху
    ax.set_theta_direction(-1)  # по часовой стрелке
    ax.set_title(f'Полярная диаграмма направленности (разы)\n'
                 f'Вариант 25: f = {freq / 1e9} ГГц, 2l/λ = {ratio}',
                 fontsize=14, pad=25)
    ax.legend(loc='upper right', fontsize=11, bbox_to_anchor=(1.3, 1.0))
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('polar_linear_v25.png', dpi=300, bbox_inches='tight')
    print("  Сохранен: polar_linear_v25.png")

    # 2. Полярный график в дБ
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')

    # Аналитический расчет - синяя сплошная линия
    ax.plot(theta_analytic_polar_rad, D_polar_analytic_dB, 'b-',
            linewidth=3, label='Аналитический расчет', alpha=0.7)

    if cst_data is not None:
        _, _, _, theta_cst_polar_deg, _, D_cst_polar_dB = cst_data
        theta_cst_polar_rad = np.deg2rad(theta_cst_polar_deg)
        # Данные из CST - красная пунктирная линия (будет совпадать с синей)
        ax.plot(theta_cst_polar_rad, D_cst_polar_dB, 'r--',
                linewidth=2, label='CST Studio', alpha=0.9)

    ax.set_theta_zero_location('N')  # 0° сверху
    ax.set_theta_direction(-1)  # по часовой стрелке
    ax.set_title(f'Полярная диаграмма направленности (дБ)\n'
                 f'Вариант 25: f = {freq / 1e9} ГГц, 2l/λ = {ratio}',
                 fontsize=14, pad=25)
    ax.legend(loc='upper right', fontsize=11, bbox_to_anchor=(1.3, 1.0))
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('polar_dB_v25.png', dpi=300, bbox_inches='tight')
    print("  Сохранен: polar_dB_v25.png")


# ============================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================
def save_results():
    """Сохранение результатов в файл."""
    print("\n" + "-" * 70)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-" * 70)

    # Используем другую переменную для файла (не f)
    with open('results_variant_25.txt', 'w', encoding='utf-8') as result_file:
        result_file.write("=" * 70 + "\n")
        result_file.write("РЕЗУЛЬТАТЫ РАСЧЕТА ДИАГРАММЫ НАПРАВЛЕННОСТИ\n")
        result_file.write("=" * 70 + "\n")
        result_file.write("ВАРИАНТ 25\n")
        result_file.write("-" * 70 + "\n")
        result_file.write(f"Рабочая частота: f = {freq / 1e9} ГГц\n")
        result_file.write(f"Отношение 2l/λ = {ratio}\n")
        result_file.write(f"Длина волны: λ = {lam:.6f} м\n")
        result_file.write(f"Длина одного плеча: l = {l:.6f} м\n")
        result_file.write(f"Волновое число: k = {k:.6f} рад/м\n")
        result_file.write("-" * 70 + "\n")
        result_file.write("РАСЧЕТНЫЕ ПАРАМЕТРЫ:\n")
        result_file.write(f"Максимальный КНД: Dmax = {Dmax:.6f} (разы)\n")
        result_file.write(f"Максимальный КНД: Dmax = {10 * np.log10(Dmax):.6f} дБ\n")
        result_file.write(f"Минимальный КНД: {np.min(D_analytic_linear):.6f} (разы)\n")
        result_file.write(f"Минимальный КНД: {np.min(D_analytic_dB):.6f} дБ\n")
        result_file.write("-" * 70 + "\n")
        result_file.write("СОЗДАННЫЕ ГРАФИКИ:\n")
        result_file.write("1. cartesian_linear_v25.png - декартов график в разах\n")
        result_file.write("2. cartesian_dB_v25.png - декартов график в дБ\n")
        result_file.write("3. polar_linear_v25.png - полярный график в разах\n")
        result_file.write("4. polar_dB_v25.png - полярный график в дБ\n")
        result_file.write("5. results_variant_25.txt - этот файл\n")
        result_file.write("=" * 70 + "\n")

    print("  Сохранен: results_variant_25.txt")


# ============================================
# ОСНОВНОЙ БЛОК ВЫПОЛНЕНИЯ
# ============================================
if __name__ == "__main__":
    # Создание графиков
    create_cartesian_plots()
    create_polar_plots()

    # Сохранение результатов
    save_results()

    # Финальное сообщение
    print("\n" + "=" * 70)
    print("РАСЧЕТ УСПЕШНО ЗАВЕРШЕН!")
    print("=" * 70)
    print("\nСозданы следующие файлы:")
    print("  1. cartesian_linear_v25.png - декартов график в разах")
    print("  2. cartesian_dB_v25.png     - декартов график в дБ")
    print("  3. polar_linear_v25.png     - полярный график в разах")
    print("  4. polar_dB_v25.png         - полярный график в дБ")
    print("  5. results_variant_25.txt   - файл с результатами")

    print("\n" + "=" * 70)

    # Показать все графики
    plt.show()