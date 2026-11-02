"""Векторизованная модель сферического жидкосцинтилляционного детектора.

Все длины задаются в метрах, энергия — в МэВ. Фотоумножители моделируются
неперекрывающимися круглыми фотокатодами на внутренней поверхности сферы.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from scipy.spatial import cKDTree


Array = np.ndarray


def isotropic_directions(n: int, rng: np.random.Generator) -> Array:
    """Вернуть ``n`` единичных изотропных направлений, форма массива ``(n, 3)``."""
    if n < 0:
        raise ValueError("Количество направлений не может быть отрицательным")
    cos_theta = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    return np.column_stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)
    )


def fibonacci_sphere(n: int) -> Array:
    """Квазирaвномерные точки на единичной сфере методом сферы Фибоначчи."""
    if n <= 0:
        raise ValueError("Количество точек должно быть положительным")
    i = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    rho = np.sqrt(1.0 - z**2)
    return np.column_stack((rho * np.cos(phi), rho * np.sin(phi), z))


@dataclass(frozen=True)
class PhotomultiplierResponse:
    """Стохастический отклик фотокатода.

    По условию число фотоэлектронов от каждого попавшего фотона распределено
    по Пуассону со средним ``mean_pe``. Сумма откликов от ``k`` фотонов имеет
    распределение ``Poisson(k * mean_pe)``.
    """

    mean_pe: float = 0.5

    def __post_init__(self) -> None:
        if self.mean_pe < 0:
            raise ValueError("Средний отклик не может быть отрицательным")

    def sample_total(
        self, incident_photons: int | Array, rng: np.random.Generator
    ) -> Array:
        incident = np.asarray(incident_photons, dtype=int)
        if np.any(incident < 0):
            raise ValueError("Количество фотонов не может быть отрицательным")
        return rng.poisson(self.mean_pe * incident)


@dataclass
class Detector:
    """Сферический детектор с одинаковыми круглыми фотоумножителями."""

    R: float = 2.0
    N: int = 40
    r: float = 0.10
    a: float = 0.5
    unit_centers: Array = field(init=False, repr=False)
    centers: Array = field(init=False, repr=False)
    response: PhotomultiplierResponse = field(init=False, repr=False)
    _spatial_tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.R <= 0 or self.r <= 0:
            raise ValueError("Радиусы должны быть положительными")
        if self.r >= self.R:
            raise ValueError("Радиус фотокатода должен быть меньше радиуса детектора")
        if self.N <= 0:
            raise ValueError("Количество фотоумножителей должно быть положительным")
        self.unit_centers = fibonacci_sphere(self.N)
        self.centers = self.R * self.unit_centers
        self.response = PhotomultiplierResponse(self.a)
        self._spatial_tree = cKDTree(self.unit_centers)

    @property
    def angular_radius(self) -> float:
        """Угловой радиус фотокатода, видимого как круг на поверхности сферы."""
        return float(np.arcsin(self.r / self.R))

    @property
    def geometric_coverage_fraction(self) -> float:
        """Доля поверхности, покрытая неперекрывающимися сферическими кругами."""
        return float(self.N * (1.0 - np.cos(self.angular_radius)) / 2.0)

    @property
    def chord_radius(self) -> float:
        """Радиус фотокатода как хорда на единичной сфере."""
        return float(2.0 * np.sin(self.angular_radius / 2.0))

    def _unit_surface_points(self, points: Array) -> tuple[Array, tuple[int, ...]]:
        pts = np.asarray(points, dtype=float)
        if pts.shape[-1] != 3:
            raise ValueError("Последняя размерность массива точек должна быть равна 3")
        original_shape = pts.shape[:-1]
        flat = pts.reshape(-1, 3)
        norms = np.linalg.norm(flat, axis=1)
        if np.any(norms == 0):
            raise ValueError("Нулевая точка не лежит на поверхности")
        unit = flat / norms[:, None]
        return unit, original_shape

    def point_to_pmt_index_dense(self, points: Array) -> Array:
        """Плотный векторизованный поиск по всем ФЭУ."""
        unit, original_shape = self._unit_surface_points(points)
        cosines = unit @ self.unit_centers.T
        nearest = np.argmax(cosines, axis=1)
        best = cosines[np.arange(len(unit)), nearest]
        nearest[best < np.cos(self.angular_radius)] = -1
        return nearest.reshape(original_shape)

    def point_to_pmt_index_fast(self, points: Array) -> Array:
        """Найти ФЭУ через k-d-дерево, не перебирая все центры."""
        unit, original_shape = self._unit_surface_points(points)
        distances, nearest = self._spatial_tree.query(
            unit, k=1, distance_upper_bound=self.chord_radius
        )
        result = np.asarray(nearest, dtype=int)
        result[(result == self.N) | ~np.isfinite(distances)] = -1
        return result.reshape(original_shape)

    def point_to_pmt_index(self, points: Array) -> Array:
        """Найти номер ФЭУ для точек на поверхности; ``-1`` означает промах."""
        pts = np.asarray(points)
        n_points = int(np.prod(pts.shape[:-1]))
        if self.N >= 200 or n_points * self.N > 5_000_000:
            return self.point_to_pmt_index_fast(points)
        return self.point_to_pmt_index_dense(points)

    def ray_distances(self, source_position: Array, directions: Array) -> Array:
        """Расстояния от внутренней точки до сферы вдоль заданных лучей."""
        x0 = np.asarray(source_position, dtype=float)
        if x0.shape != (3,):
            raise ValueError("Координаты источника должны иметь форму (3,)")
        if np.linalg.norm(x0) >= self.R:
            raise ValueError("Источник должен находиться внутри детектора")
        dirs = np.asarray(directions, dtype=float)
        if dirs.ndim != 2 or dirs.shape[1] != 3:
            raise ValueError("Направления должны иметь форму (n, 3)")
        norms = np.linalg.norm(dirs, axis=1)
        if np.any(norms == 0):
            raise ValueError("Направление не может быть нулевым")
        dirs = dirs / norms[:, None]
        projection = dirs @ x0
        discriminant = projection**2 + self.R**2 - float(x0 @ x0)
        return -projection + np.sqrt(discriminant)

    def boundary_intersection(self, source_position: Array, directions: Array) -> Array:
        """Пересечь лучи из внутренней точки с поверхностью сферы."""
        dirs = np.asarray(directions, dtype=float)
        norms = np.linalg.norm(dirs, axis=1)
        dirs = dirs / norms[:, None]
        x0 = np.asarray(source_position, dtype=float)
        distance = self.ray_distances(x0, dirs)
        return x0 + distance[:, None] * dirs

    def trace_photons(self, source_position: Array, directions: Array) -> Array:
        """Вернуть номер ФЭУ для каждого фотона или ``-1`` при промахе."""
        boundary_points = self.boundary_intersection(source_position, directions)
        return self.point_to_pmt_index(boundary_points)

    def trace_photons_scalar(self, source_position: Array, directions: Array) -> Array:
        """Контрольная скалярная реализация полного перебора ФЭУ."""
        x0 = np.asarray(source_position, dtype=float)
        result = []
        threshold = np.cos(self.angular_radius)
        for direction in np.asarray(directions, dtype=float):
            direction = direction / np.linalg.norm(direction)
            projection = float(direction @ x0)
            distance = -projection + np.sqrt(
                projection**2 + self.R**2 - float(x0 @ x0)
            )
            surface_point = (x0 + distance * direction) / self.R
            cosines = self.unit_centers @ surface_point
            best = int(np.argmax(cosines))
            result.append(best if cosines[best] >= threshold else -1)
        return np.asarray(result, dtype=int)

    def detect_photons(
        self,
        source_position: Array,
        directions: Array,
        rng: np.random.Generator,
        attenuation_length: float | None = None,
    ) -> Array:
        """Трассировать фотоны и, при необходимости, применить затухание."""
        indices = self.trace_photons(source_position, directions)
        if attenuation_length is None:
            return indices
        if attenuation_length <= 0:
            raise ValueError("Длина затухания должна быть положительной")
        distances = self.ray_distances(source_position, directions)
        survives = rng.random(len(indices)) < np.exp(-distances / attenuation_length)
        return np.where(survives, indices, -1)

    def estimate_hit_probability(
        self,
        source_position: Array,
        n_rays: int,
        rng: np.random.Generator,
        batch_size: int = 100_000,
    ) -> tuple[float, float]:
        """Оценить вероятность попадания и её биномиальную стандартную ошибку."""
        if n_rays <= 0:
            raise ValueError("Число лучей должно быть положительным")
        hits = 0
        done = 0
        while done < n_rays:
            n_batch = min(batch_size, n_rays - done)
            directions = isotropic_directions(n_batch, rng)
            hits += int(np.count_nonzero(self.trace_photons(source_position, directions) >= 0))
            done += n_batch
        probability = hits / n_rays
        error = np.sqrt(probability * (1.0 - probability) / n_rays)
        return float(probability), float(error)

    def estimate_detection_probability(
        self,
        source_position: Array,
        n_rays: int,
        rng: np.random.Generator,
        attenuation_length: float | None = None,
        batch_size: int = 100_000,
    ) -> tuple[float, float]:
        """Оценить вероятность регистрации с учётом затухания."""
        if n_rays <= 0:
            raise ValueError("Число лучей должно быть положительным")
        hits = 0
        done = 0
        while done < n_rays:
            n_batch = min(batch_size, n_rays - done)
            directions = isotropic_directions(n_batch, rng)
            detected = self.detect_photons(
                source_position, directions, rng, attenuation_length
            )
            hits += int(np.count_nonzero(detected >= 0))
            done += n_batch
        probability = hits / n_rays
        error = np.sqrt(probability * (1.0 - probability) / n_rays)
        return float(probability), float(error)


@dataclass(frozen=True)
class Source:
    """Изотропный источник сцинтилляционных фотонов."""

    Y: float = 10_000.0
    position: Array = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        if self.Y <= 0:
            raise ValueError("Световыход должен быть положительным")
        position = np.asarray(self.position, dtype=float)
        if position.shape != (3,):
            raise ValueError("Координаты источника должны иметь форму (3,)")
        object.__setattr__(self, "position", position)

    def photon_count(self, energy_mev: float) -> int:
        """Ожидаемое число фотонов, округлённое до целого (задачи 2--4)."""
        if energy_mev < 0:
            raise ValueError("Энергия не может быть отрицательной")
        return int(np.rint(self.Y * energy_mev))

    def sample_photon_count(
        self, energy_mev: float, rng: np.random.Generator
    ) -> int:
        """Случайное число фотонов ``Poisson(E*Y)`` (задача 5)."""
        if energy_mev < 0:
            raise ValueError("Энергия не может быть отрицательной")
        return int(rng.poisson(self.Y * energy_mev))

    def sample_directions(
        self,
        energy_mev: float,
        rng: np.random.Generator,
        stochastic_count: bool = True,
    ) -> Array:
        count = (
            self.sample_photon_count(energy_mev, rng)
            if stochastic_count
            else self.photon_count(energy_mev)
        )
        return isotropic_directions(count, rng)

    def photon_batches(
        self,
        energy_mev: float,
        rng: np.random.Generator,
        batch_size: int = 100,
    ) -> Iterator[Array]:
        """Генератор изотропных фотонов пакетами не более ``batch_size``."""
        if batch_size <= 0:
            raise ValueError("Размер пакета должен быть положительным")
        remaining = self.sample_photon_count(energy_mev, rng)
        while remaining:
            current = min(batch_size, remaining)
            yield isotropic_directions(current, rng)
            remaining -= current


@dataclass(frozen=True)
class EventResult:
    emitted_photons: int
    detected_photons: int
    hits_per_pmt: Array
    charge_per_pmt: Array

    @property
    def total_charge(self) -> int:
        return int(np.sum(self.charge_per_pmt))


def simulate_event(
    detector: Detector,
    source: Source,
    energy_mev: float,
    rng: np.random.Generator,
    attenuation_length: float | None = None,
) -> EventResult:
    """Полностью промоделировать одно событие с трассировкой каждого фотона."""
    directions = source.sample_directions(energy_mev, rng)
    pmt_indices = detector.detect_photons(
        source.position, directions, rng, attenuation_length
    )
    detected = pmt_indices[pmt_indices >= 0]
    hits = np.bincount(detected, minlength=detector.N)
    charge = detector.response.sample_total(hits, rng)
    return EventResult(len(directions), len(detected), hits, charge)


def simulate_charge_ensemble(
    detector: Detector,
    source: Source,
    energy_mev: float,
    n_events: int,
    hit_probability: float,
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    """Быстро смоделировать ансамбль зарядов по точной составной модели.

    Для каждого события число попавших фотонов биномиально, а условный заряд
    распределён как ``Poisson(a * n_hits)``. Это эквивалентно независимой
    трассировке при известной геометрической вероятности попадания.
    """
    if n_events <= 0:
        raise ValueError("Число событий должно быть положительным")
    if not 0.0 <= hit_probability <= 1.0:
        raise ValueError("Вероятность должна лежать в [0, 1]")
    n_photons = source.photon_count(energy_mev)
    hits = rng.binomial(n_photons, hit_probability, size=n_events)
    charges = detector.response.sample_total(hits, rng)
    return hits, charges


def simulate_charge_ensemble_poisson(
    detector: Detector,
    source: Source,
    energy_mev: float,
    n_events: int,
    detection_probability: float,
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    """Ансамбль с пуассоновским числом испущенных фотонов (задачи 5--6)."""
    if n_events <= 0:
        raise ValueError("Число событий должно быть положительным")
    if not 0.0 <= detection_probability <= 1.0:
        raise ValueError("Вероятность должна лежать в [0, 1]")
    mean_hits = source.Y * energy_mev * detection_probability
    hits = rng.poisson(mean_hits, size=n_events)
    charges = detector.response.sample_total(hits, rng)
    return hits, charges


def pmt_detection_probabilities(
    detector: Detector,
    source_position: Array,
    attenuation_length: float | None = None,
) -> Array:
    """Вероятности регистрации одного изотропного фотона каждым ФЭУ.

    Для малого круглого фотокатода используется его телесный угол:
    ``p_i = A cos(gamma_i) / (4 pi d_i**2)``. Здесь ``d_i`` — расстояние
    от источника до центра фотокатода, а ``gamma_i`` — угол падения на
    обращённую внутрь нормаль. Такая гладкая формула удобна для PyMC и в
    центре детектора воспроизводит приближение ``r**2/(4 R**2)``.
    """
    position = np.asarray(source_position, dtype=float)
    if position.shape != (3,):
        raise ValueError("Координаты источника должны иметь форму (3,)")
    if np.linalg.norm(position) >= detector.R:
        raise ValueError("Источник должен находиться внутри детектора")
    if attenuation_length is not None and attenuation_length <= 0:
        raise ValueError("Длина затухания должна быть положительной")

    displacement = detector.centers - position
    distance = np.linalg.norm(displacement, axis=1)
    incidence_cosine = (detector.R - detector.unit_centers @ position) / distance
    photocathode_area = np.pi * detector.r**2
    probabilities = (
        photocathode_area * incidence_cosine / (4.0 * np.pi * distance**2)
    )
    if attenuation_length is not None:
        probabilities *= np.exp(-distance / attenuation_length)
    return np.clip(probabilities, 0.0, 1.0)


def expected_pmt_photoelectrons(
    detector: Detector,
    source: Source,
    energy_mev: float,
    attenuation_length: float | None = None,
) -> Array:
    """Средние числа фотоэлектронов на ФЭУ для гладкой модели PyMC."""
    if energy_mev < 0:
        raise ValueError("Энергия не может быть отрицательной")
    probabilities = pmt_detection_probabilities(
        detector, source.position, attenuation_length
    )
    return detector.a * source.Y * energy_mev * probabilities


def sample_pmt_photoelectrons(
    detector: Detector,
    source: Source,
    energy_mev: float,
    rng: np.random.Generator,
    attenuation_length: float | None = None,
) -> Array:
    """Сгенерировать вектор наблюдаемых фотоэлектронов на всех ФЭУ."""
    means = expected_pmt_photoelectrons(
        detector, source, energy_mev, attenuation_length
    )
    return rng.poisson(means)


def theoretical_charge_moments(
    detector: Detector, source: Source, energy_mev: float, hit_probability: float
) -> tuple[float, float]:
    """Среднее и дисперсия полного заряда для составной модели."""
    n = source.photon_count(energy_mev)
    p = hit_probability
    a = detector.a
    mean = a * n * p
    variance = a * n * p + a**2 * n * p * (1.0 - p)
    return float(mean), float(variance)
