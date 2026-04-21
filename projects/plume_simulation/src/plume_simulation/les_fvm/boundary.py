"""Boundary-condition application for 3-D tracer fields.

finitevolX ships 2-D BC atoms (:class:`finitevolx.Dirichlet1D`, etc.)
that expect a ``[Ny, Nx]`` field.  For the Eulerian dispersion solver
we need to update ghost rings on a ``[Nz, Ny, Nx]`` field.  The pattern
used here is:

1. A :class:`HorizontalBC` bundles a :class:`finitevolx.BoundaryConditionSet`
   and applies it per z-slice via ``eqx.filter_vmap`` — keeping the
   per-face semantics of finitevolX but vectorising over the vertical
   axis.
2. A :class:`VerticalBC` holds two ``(bc_type, value)`` pairs for the
   top and bottom faces.  We implement these directly (Dirichlet /
   Neumann / outflow / periodic) because the 2-D BC atoms assume the
   modified axis is horizontal.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
from finitevolx import (
    BoundaryConditionSet,
    Dirichlet1D,
    Neumann1D,
    Outflow1D,
    Periodic1D,
)
from jaxtyping import Array, Float

from plume_simulation.les_fvm.grid import PlumeGrid3D


VerticalBCKind = Literal["dirichlet", "neumann", "outflow", "periodic"]


class HorizontalBC(eqx.Module):
    """Apply a :class:`BoundaryConditionSet` to every z-slice of a 3-D field."""

    bc_set: BoundaryConditionSet

    def __call__(
        self,
        field: Float[Array, "Nz Ny Nx"],
        dx: float,
        dy: float,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Return ``field`` with horizontal ghost rings updated on each slice."""

        def apply_slice(slab: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
            return self.bc_set(slab, dx=dx, dy=dy)

        return eqx.filter_vmap(apply_slice)(field)


class VerticalBC(eqx.Module):
    """Top/bottom ghost-slice update for a 3-D field.

    Parameters
    ----------
    bottom_kind : {"dirichlet", "neumann", "outflow", "periodic"}
        Ground-boundary behaviour.
    bottom_value : float, default 0.0
        Dirichlet target value (ignored for ``neumann``/``outflow`` where
        the imposed gradient is zero, and for ``periodic``).
    top_kind, top_value : same
    """

    bottom_kind: VerticalBCKind = eqx.field(static=True)
    top_kind: VerticalBCKind = eqx.field(static=True)
    bottom_value: float = 0.0
    top_value: float = 0.0

    def __call__(
        self,
        field: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Return ``field`` with top and bottom ghost slices updated."""
        out = _apply_vertical_face(
            field, face="bottom", kind=self.bottom_kind, value=self.bottom_value
        )
        out = _apply_vertical_face(
            out, face="top", kind=self.top_kind, value=self.top_value
        )
        return out


def _apply_vertical_face(
    field: Float[Array, "Nz Ny Nx"],
    face: Literal["bottom", "top"],
    kind: VerticalBCKind,
    value: float,
) -> Float[Array, "Nz Ny Nx"]:
    """Update one vertical ghost slice using the requested BC flavour."""
    if face == "bottom":
        interior_slice = field[1, :, :]
        opposite_slice = field[-2, :, :]
        ghost_index = 0
    else:
        interior_slice = field[-2, :, :]
        opposite_slice = field[1, :, :]
        ghost_index = -1

    if kind == "dirichlet":
        ghost = 2.0 * value - interior_slice
    elif kind == "neumann":
        # Zero-gradient Neumann: ghost mirrors interior (no cross-boundary flux).
        ghost = interior_slice
    elif kind == "outflow":
        ghost = interior_slice
    elif kind == "periodic":
        ghost = opposite_slice
    else:
        raise ValueError(f"Unknown vertical BC kind: {kind!r}")

    return field.at[ghost_index, :, :].set(ghost)


def apply_boundary_conditions(
    field: Float[Array, "Nz Ny Nx"],
    horizontal_bc: HorizontalBC,
    vertical_bc: VerticalBC,
    plume_grid: PlumeGrid3D,
) -> Float[Array, "Nz Ny Nx"]:
    """Apply horizontal then vertical BCs to a 3-D tracer field."""
    out = horizontal_bc(field, dx=plume_grid.dx, dy=plume_grid.dy)
    out = vertical_bc(out)
    return out


def build_default_concentration_bc(
    bc_x: (
        str
        | tuple[str, str]
        | tuple[tuple[str, float], tuple[str, float]]
    ) = ("dirichlet", "outflow"),
    bc_y: (
        str
        | tuple[str, str]
        | tuple[tuple[str, float], tuple[str, float]]
    ) = "periodic",
    bc_z: (
        str
        | tuple[str, str]
        | tuple[tuple[str, float], tuple[str, float]]
    ) = ("neumann", "neumann"),
) -> tuple[HorizontalBC, VerticalBC]:
    """Build ``(HorizontalBC, VerticalBC)`` from user-facing BC specs.

    Each of ``bc_x``, ``bc_y``, ``bc_z`` can be:

    - ``"periodic"``       — periodic on both faces of that axis.
    - ``(west, east)`` (for ``bc_x``) / ``(south, north)`` (``bc_y``) /
      ``(bottom, top)`` (``bc_z``) where each entry is either a BC-kind
      string (``"dirichlet"``, ``"neumann"``, ``"outflow"``) or a
      ``(kind, value)`` tuple giving the Dirichlet / Neumann target.

    Returns
    -------
    tuple[HorizontalBC, VerticalBC]
    """
    w_bc, e_bc = _split_horizontal_spec(bc_x, faces=("west", "east"))
    s_bc, n_bc = _split_horizontal_spec(bc_y, faces=("south", "north"))

    bc_set = BoundaryConditionSet(south=s_bc, north=n_bc, west=w_bc, east=e_bc)
    bot_kind, bot_val, top_kind, top_val = _split_vertical_spec(bc_z)
    return (
        HorizontalBC(bc_set=bc_set),
        VerticalBC(
            bottom_kind=bot_kind,
            bottom_value=bot_val,
            top_kind=top_kind,
            top_value=top_val,
        ),
    )


def _as_kind_value(
    entry: str | tuple[str, float],
) -> tuple[str, float]:
    if isinstance(entry, str):
        return entry, 0.0
    kind, value = entry
    return str(kind), float(value)


def _split_horizontal_spec(
    spec: (
        str
        | tuple[str, str]
        | tuple[tuple[str, float], tuple[str, float]]
    ),
    faces: tuple[str, str],
):
    """Unpack a horizontal BC spec into two face-specific atoms.

    ``faces`` is a pair of face names ordered ``(lower, upper)``, e.g.
    ``("west", "east")`` or ``("south", "north")``.
    """
    if isinstance(spec, str) and spec.lower() == "periodic":
        return Periodic1D(faces[0]), Periodic1D(faces[1])
    if isinstance(spec, tuple) and len(spec) == 2:
        lower_kind, lower_val = _as_kind_value(spec[0])
        upper_kind, upper_val = _as_kind_value(spec[1])
        return (
            _build_1d_bc(lower_kind, lower_val, faces[0]),
            _build_1d_bc(upper_kind, upper_val, faces[1]),
        )
    raise ValueError(
        "horizontal BC spec must be 'periodic' or a 2-tuple "
        f"(lower, upper); got {spec!r}"
    )


def _build_1d_bc(kind: str, value: float, face: str):
    kind_l = kind.lower()
    if kind_l == "dirichlet":
        return Dirichlet1D(face=face, value=value)
    if kind_l == "neumann":
        return Neumann1D(face=face, value=value)
    if kind_l == "outflow":
        return Outflow1D(face=face)
    if kind_l == "periodic":
        return Periodic1D(face=face)
    raise ValueError(
        f"horizontal BC kind must be one of 'dirichlet', 'neumann', "
        f"'outflow', 'periodic'; got {kind!r}"
    )


def _split_vertical_spec(
    spec: (
        str
        | tuple[str, str]
        | tuple[tuple[str, float], tuple[str, float]]
    ),
) -> tuple[VerticalBCKind, float, VerticalBCKind, float]:
    if isinstance(spec, str) and spec.lower() == "periodic":
        return "periodic", 0.0, "periodic", 0.0
    if isinstance(spec, tuple) and len(spec) == 2:
        bot_kind, bot_val = _as_kind_value(spec[0])
        top_kind, top_val = _as_kind_value(spec[1])
        for kind in (bot_kind, top_kind):
            if kind.lower() not in {"dirichlet", "neumann", "outflow", "periodic"}:
                raise ValueError(
                    f"vertical BC kind must be one of 'dirichlet', 'neumann', "
                    f"'outflow', 'periodic'; got {kind!r}"
                )
        return bot_kind.lower(), bot_val, top_kind.lower(), top_val  # type: ignore[return-value]
    raise ValueError(
        "vertical BC spec must be 'periodic' or a 2-tuple "
        f"(bottom, top); got {spec!r}"
    )


