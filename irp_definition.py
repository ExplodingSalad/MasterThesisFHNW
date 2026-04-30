"""
Inventory Routing Problem (IRP) – Problem Definition & Encoding
================================================================
Optimised version: vectorized numpy operations, pre-computed arrays,
realistic initial inventory.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass


# ======================================================================
# Scenario Parameters
# ======================================================================
@dataclass
class IRPScenario:
    """Complete IRP scenario — all parameters from Table 4."""

    n_supplier_warehouses: int = 1
    supplier_warehouse_capacity: float = 100.0
    n_products: int = 1
    product_sizes: np.ndarray | None = None
    supplier_holding_cost: np.ndarray | None = None
    customer_holding_cost: np.ndarray | None = None
    n_vehicle_types: int = 1
    vehicles_per_type: np.ndarray | None = None
    vehicle_capacity: np.ndarray | None = None
    vehicle_cost_per_dist: np.ndarray | None = None
    n_customers: int = 5
    n_customer_warehouses: np.ndarray | None = None
    customer_warehouse_capacity: np.ndarray | None = None
    daily_consumption: np.ndarray | None = None
    replenishment_strategy: int = 1
    n_periods: int = 5
    distance_matrix: np.ndarray | None = None
    penalty_capacity: float = 1000.0
    penalty_stockout: float = 2000.0
    penalty_vehicle_cap: float = 1500.0
    penalty_supplier_stockout: float = 1500.0

    def __post_init__(self):
        nc, np_ = self.n_customers, self.n_products
        nvt = self.n_vehicle_types
        if self.product_sizes is None:
            self.product_sizes = np.ones(np_)
        if self.supplier_holding_cost is None:
            self.supplier_holding_cost = np.full(np_, 0.1)
        if self.customer_holding_cost is None:
            self.customer_holding_cost = np.full(np_, 0.2)
        if self.vehicles_per_type is None:
            self.vehicles_per_type = np.full(nvt, 3, dtype=int)
        if self.vehicle_capacity is None:
            self.vehicle_capacity = np.full(nvt, 20.0)
        if self.vehicle_cost_per_dist is None:
            self.vehicle_cost_per_dist = np.full(nvt, 1.0)
        if self.n_customer_warehouses is None:
            self.n_customer_warehouses = np.ones(nc, dtype=int)
        if self.customer_warehouse_capacity is None:
            self.customer_warehouse_capacity = np.full(nc, 50.0)
        if self.daily_consumption is None:
            self.daily_consumption = _rng_default().uniform(5, 20, size=(nc, np_))
        if self.distance_matrix is None:
            size = nc + 1
            dm = _rng_default().uniform(10, 200, size=(size, size))
            dm = (dm + dm.T) / 2
            np.fill_diagonal(dm, 0)
            self.distance_matrix = dm

        # --- Pre-compute derived arrays (used thousands of times) ---
        self._cust_total_cap = (
            self.n_customer_warehouses.astype(float)
            * self.customer_warehouse_capacity
        )  # shape (C,)
        self._sup_total_cap = float(
            self.n_supplier_warehouses * self.supplier_warehouse_capacity
        )

    @property
    def total_supplier_capacity(self) -> float:
        return self._sup_total_cap

    def customer_total_capacity_arr(self) -> np.ndarray:
        """Total capacity per customer, shape (C,)."""
        return self._cust_total_cap


_rng_cache = None
def _rng_default():
    global _rng_cache
    if _rng_cache is None:
        _rng_cache = np.random.default_rng(42)
    return _rng_cache


# ======================================================================
# Initial inventory helper
# ======================================================================
def _make_initial_inventory(scenario: IRPScenario):
    """
    Create realistic starting inventories.

    Customers start with enough stock for ~2 periods of consumption,
    capped by their warehouse capacity. This prevents the degenerate
    case where unlimited-capacity scenarios start with millions of
    free units.

    Supplier starts with enough for ~3 periods of total demand.
    """
    C, P = scenario.n_customers, scenario.n_products
    ps = scenario.product_sizes  # (P,)
    cap_arr = scenario.customer_total_capacity_arr()  # (C,)

    # Customer: 2 periods of consumption, capped by capacity
    cust_inv = scenario.daily_consumption * 2.0  # (C, P) in units

    # Cap by volume capacity
    for c in range(C):
        vol = float(np.sum(cust_inv[c] * ps))
        if vol > cap_arr[c] and vol > 0:
            cust_inv[c] *= cap_arr[c] / vol

    # Supplier: 3 periods of total demand, capped by supplier capacity
    total_demand_per_period = scenario.daily_consumption.sum(axis=0)  # (P,)
    sup_inv = total_demand_per_period * 3.0  # (P,) in units

    sup_cap = scenario.total_supplier_capacity
    sup_vol = float(np.sum(sup_inv * ps))
    if sup_vol > sup_cap and sup_vol > 0:
        sup_inv *= sup_cap / sup_vol

    return cust_inv.copy(), sup_inv.copy()


# ======================================================================
# Bounds
# ======================================================================
def get_bounds(scenario: IRPScenario):
    T = scenario.n_periods
    C = scenario.n_customers
    P = scenario.n_products
    V = scenario.n_vehicle_types

    n_qty = T * C * P
    n_vtype = T * C
    n_route = T * C

    lb = np.zeros(n_qty + n_vtype + n_route)
    ub = np.ones(n_qty + n_vtype + n_route)
    ub[n_qty: n_qty + n_vtype] = V - 1 + 0.999
    return lb, ub


# ======================================================================
# Decode
# ======================================================================
@dataclass
class IRPPlan:
    delivery_qty: np.ndarray   # (T, C, P)
    vehicle_type: np.ndarray   # (T, C)
    route_order: np.ndarray    # (T, C)


def decode(x: np.ndarray, scenario: IRPScenario) -> IRPPlan:
    T, C, P = scenario.n_periods, scenario.n_customers, scenario.n_products
    ps = scenario.product_sizes
    cap_arr = scenario.customer_total_capacity_arr()
    idx = 0

    raw_frac = np.clip(x[idx:idx + T*C*P].reshape(T, C, P), 0.0, 1.0)
    idx += T * C * P

    vehicle_type = np.clip(
        np.floor(x[idx:idx + T*C].reshape(T, C)).astype(int),
        0, scenario.n_vehicle_types - 1
    )
    idx += T * C

    route_order = np.argsort(x[idx:idx + T*C].reshape(T, C), axis=1)

    delivery_qty = np.zeros((T, C, P))

    cust_inv, _ = _make_initial_inventory(scenario)

    for t in range(T):
        # Vectorized volume calculation
        cust_vol = np.sum(cust_inv * ps[None, :], axis=1)  # (C,)
        remaining_vol = np.maximum(cap_arr - cust_vol, 0.0)  # (C,)

        for c in range(C):
            rv = remaining_vol[c]
            for p in range(P):
                psize = float(ps[p])
                max_units = rv / psize if psize > 0 else 0.0

                if scenario.replenishment_strategy == 1:
                    if raw_frac[t, c, p] > 0.5:
                        delivery_qty[t, c, p] = max(0.0, round(max_units))
                    # else: 0
                else:
                    delivery_qty[t, c, p] = max(0.0, round(
                        raw_frac[t, c, p] * max_units
                    ))

                rv = max(0.0, rv - delivery_qty[t, c, p] * psize)

        cust_inv += delivery_qty[t]
        cust_inv -= scenario.daily_consumption
        np.clip(cust_inv, 0, None, out=cust_inv)

    return IRPPlan(delivery_qty, vehicle_type, route_order)


# ======================================================================
# Objective (vectorized)
# ======================================================================
def evaluate_irp(x: np.ndarray, scenario: IRPScenario) -> float:
    plan = decode(x, scenario)
    T, C, P = scenario.n_periods, scenario.n_customers, scenario.n_products
    ps = scenario.product_sizes
    cap_arr = scenario.customer_total_capacity_arr()
    dm = scenario.distance_matrix

    total_holding = 0.0
    total_transport = 0.0
    total_penalty = 0.0

    cust_inv, sup_inv = _make_initial_inventory(scenario)
    sup_cap = scenario.total_supplier_capacity

    for t in range(T):
        pd = plan.delivery_qty[t].copy()  # (C, P)

        # --- Supplier check (vectorized over products) ---
        shipped = pd.sum(axis=0)  # (P,)
        over = shipped > sup_inv
        if over.any():
            for p in np.where(over)[0]:
                shortfall = shipped[p] - sup_inv[p]
                total_penalty += scenario.penalty_supplier_stockout * shortfall
                if shipped[p] > 0:
                    scale = sup_inv[p] / shipped[p]
                    pd[:, p] = np.round(pd[:, p] * scale)
                    shipped[p] = pd[:, p].sum()

        sup_inv -= shipped
        np.clip(sup_inv, 0, None, out=sup_inv)

        # --- Customer delivery + capacity check (vectorized) ---
        cust_inv += pd
        cust_vol = np.sum(cust_inv * ps[None, :], axis=1)  # (C,)
        overflow = cust_vol - cap_arr
        overflow_mask = overflow > 0
        if overflow_mask.any():
            total_penalty += scenario.penalty_capacity * float(overflow[overflow_mask].sum())
            for c in np.where(overflow_mask)[0]:
                if cust_vol[c] > 0:
                    cust_inv[c] *= cap_arr[c] / cust_vol[c]

        # --- Consumption ---
        cust_inv -= scenario.daily_consumption
        neg_mask = cust_inv < 0
        if neg_mask.any():
            total_penalty += scenario.penalty_stockout * float(np.abs(cust_inv[neg_mask]).sum())
            np.clip(cust_inv, 0, None, out=cust_inv)

        # --- Holding costs (fully vectorized) ---
        total_holding += float(np.sum(cust_inv * scenario.customer_holding_cost[None, :]))
        total_holding += float(np.sum(sup_inv * scenario.supplier_holding_cost))

        # --- Transportation ---
        served_mask = pd.sum(axis=1) > 0  # (C,) bool
        served = np.where(served_mask)[0]

        if len(served) > 0:
            visit_order = plan.route_order[t]
            vtypes = plan.vehicle_type[t]

            # Sort served by route priority
            positions = np.array([np.searchsorted(visit_order, c) for c in served])
            served_sorted = served[np.argsort(positions)]

            # Group by vehicle type
            vtype_groups: dict[int, list[int]] = defaultdict(list)
            for c in served_sorted:
                vtype_groups[int(vtypes[c])].append(c)

            for vt, customers in vtype_groups.items():
                cap_vol = float(scenario.vehicle_capacity[vt])
                cost_km = float(scenario.vehicle_cost_per_dist[vt])
                n_avail = int(scenario.vehicles_per_type[vt])

                routes: list[list[int]] = []
                cur_route: list[int] = []
                cur_load = 0.0

                for c in customers:
                    load_c = float(np.dot(pd[c], ps))
                    if cur_load + load_c > cap_vol and cur_route:
                        routes.append(cur_route)
                        cur_route = []
                        cur_load = 0.0
                    cur_route.append(c)
                    cur_load += load_c
                    if load_c > cap_vol:
                        total_penalty += scenario.penalty_vehicle_cap * (load_c - cap_vol)

                if cur_route:
                    routes.append(cur_route)

                if len(routes) > n_avail:
                    total_penalty += scenario.penalty_vehicle_cap * (len(routes) - n_avail) * 500

                for route in routes:
                    dist = dm[0, route[0] + 1]
                    for k in range(len(route) - 1):
                        dist += dm[route[k] + 1, route[k + 1] + 1]
                    dist += dm[route[-1] + 1, 0]
                    total_transport += dist * cost_km

        # --- Supplier resupply ---
        sup_inv += shipped * 0.8
        sup_max_per_p = sup_cap / max(P, 1) / ps
        np.minimum(sup_inv, sup_max_per_p, out=sup_inv)

    return total_holding + total_transport + total_penalty


# ======================================================================
# Convenience
# ======================================================================
def make_irp_objective(scenario: IRPScenario):
    lb, ub = get_bounds(scenario)
    def objective(x: np.ndarray) -> float:
        return evaluate_irp(x, scenario)
    return objective, lb, ub