"""Normalized HITSZ waypoint routes and their ordered synthetic gate geometry."""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Gate:
    id: int
    station_m: float
    center_ned: tuple
    basis_ned: tuple
    aperture_m: tuple

    @property
    def normal_ned(self):
        return tuple(row[0] for row in self.basis_ned)


def _rows(array):
    return tuple(map(tuple, array.tolist()))


@dataclass(frozen=True)
class GateGraph:
    route: str
    waypoints_ned: tuple
    gates: tuple

    @classmethod
    def from_hitsz(cls, path, aperture_m=(6.0, 3.0)):
        path = Path(path)
        points = np.asarray(yaml.safe_load(path.read_text(encoding="utf-8"))["waypoints"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError(f"{path} needs at least two finite NED waypoints")
        if len(aperture_m) != 2 or not np.isfinite(aperture_m).all() or min(aperture_m) <= 0:
            raise ValueError("Gate width and height must be finite and positive")
        # HITSZ makes a pair of gates every 21.5 m, with the second 2.6 m farther.
        # Ignore duplicate consecutive points before computing the tangent basis.
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]
        if len(points) < 2:
            raise ValueError("The route has no nonzero segment")
        edges = np.diff(points, axis=0)
        length = np.linalg.norm(edges, axis=1)
        station = np.r_[0, np.cumsum(length)]
        starts = np.arange(0, station[-1], 21.5)
        gate_s = (starts[:, None] + [0, 2.6]).ravel()
        gate_s = gate_s[gate_s <= station[-1] - 0.1]
        segments = np.searchsorted(station, gate_s).clip(1, len(points) - 1) - 1
        gates = []
        for gate_id, (s, i) in enumerate(zip(gate_s, segments)):
            forward = edges[i] / length[i]
            right = np.cross([0, 0, 1], forward)
            if np.linalg.norm(right) < 1e-4:
                right = np.array([0, 1, 0]) - forward * forward[1]
            right = right / np.linalg.norm(right)
            down = np.cross(forward, right)
            basis = np.stack((forward, right, down), axis=-1)
            center = points[i] + (s - station[i]) * forward
            gates.append(Gate(gate_id, float(s), tuple(center.tolist()), _rows(basis), tuple(aperture_m)))
        return cls(path.stem, _rows(points), tuple(gates))

    def select(self, ids):
        wanted = set(ids)
        chosen = tuple(gate for gate in self.gates if gate.id in wanted)
        if not wanted or len(chosen) != len(wanted):
            raise ValueError("Select a nonempty subset of gate IDs on this route")
        return replace(self, gates=chosen)

    def transformed(self, rotation, translation):
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("Expected a 3×3 rotation and finite NED translation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8) or not np.isclose(
            np.linalg.det(rotation), 1
        ):
            raise ValueError("The rigid rotation must be in SO(3)")
        gates = tuple(replace(gate, center_ned=tuple((rotation @ gate.center_ned + translation).tolist()),
                              basis_ned=_rows(rotation @ gate.basis_ned)) for gate in self.gates)
        points = np.asarray(self.waypoints_ned) @ rotation.T + translation
        return replace(self, waypoints_ned=_rows(points), gates=gates)

    def randomized(self, seed, *, keep, max_translation_m=20.0):
        """Sample a gravity-preserving transform and a route-ordered gate subset."""
        if not 1 <= keep <= len(self.gates) or not 0 <= max_translation_m < float("inf"):
            raise ValueError("Invalid gate subset size or translation range")
        rng = np.random.default_rng(seed)
        yaw = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        shift = rng.uniform(-max_translation_m, max_translation_m, size=3)
        ids = rng.choice([gate.id for gate in self.gates], size=keep, replace=False)
        return self.select(ids).transformed(rotation, shift)

    def payload(self):
        return {"schema": 1, "frame": "NED", "route": self.route,
                "waypoints_ned": self.waypoints_ned,
                "gates": [{**asdict(gate), "normal_ned": gate.normal_ned} for gate in self.gates]}

    @property
    def sha256(self):
        canonical = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waypoints", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", type=int, help="apply a seeded yaw/translation and retain this many gates")
    args = parser.parse_args()
    graph = GateGraph.from_hitsz(args.waypoints)
    if args.keep is not None:
        graph = graph.randomized(args.seed, keep=args.keep)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({**graph.payload(), "sha256": graph.sha256}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
