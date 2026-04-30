"""
IRP Scenario Generator
======================
Generates IRP scenarios matching the five complexity levels
from Table 5 of the thesis proposal (pp. 30-31).

Each level has specific parameters enabled/disabled and specific
value ranges. Comments reference the exact Table 5 rows.
"""

from __future__ import annotations

import numpy as np

from irp_definition import IRPScenario


def generate_scenario(
    level: str = "basic",
    seed: int = 42,
) -> IRPScenario:
    """
    Generate an IRP scenario at the given complexity level.

    Parameters
    ----------
    level : str
        One of "basic", "simple", "intermediate", "advanced", "complex".
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    IRPScenario
    """
    rng = np.random.default_rng(seed)
    level = level.lower()

    builders = {
        "basic": _basic,
        "simple": _simple,
        "intermediate": _intermediate,
        "advanced": _advanced,
        "complex": _complex,
    }

    if level not in builders:
        raise ValueError(
            f"Unknown level: {level!r}. "
            f"Choose from: {list(builders.keys())}"
        )

    return builders[level](rng)


def _make_distance_matrix(n_customers: int, rng, lo=10, hi=200):
    """Symmetric random distance matrix with depot at index 0."""
    size = n_customers + 1
    dm = rng.uniform(lo, hi, size=(size, size))
    dm = (dm + dm.T) / 2
    np.fill_diagonal(dm, 0)
    return dm


# ==================================================================
# BASIC
# ==================================================================
# Table 5 description:
#   "No Warehouse Capacity, No Product Size, No Vehicle Capacity,
#    No Vehicle Cost for Delivery, No Replenishment Strategy"
#
# TODO NOTE:
#   The original Table 5 omits warehouse capacity and holding costs
#   for the Basic scenario. However, without these the IRP degenerates
#   into a trivial problem where "never deliver" is optimal (no cost
#   drivers exist). To ensure a meaningful optimisation problem, we
#   add realistic warehouse capacity and holding costs. Vehicle
#   capacity and vehicle cost remain disabled as per Table 5. -> change in paper
#
# Parameters used:         Value
#   W_Nr_Sup               1
#   W_Ca_Sup               200
#   P_Nr                   1
#   P_Ho_Co_Sup            0.05
#   P_Ho_Co_Cust           0.1
#   V_Ty                   1
#   V_Nr_Ty                1
#   Cust_Nr                1
#   W_Nr_Cust              1
#   W_Ca_Cust              50-100
#   P_Cons                 1-20
# ==================================================================
def _basic(rng) -> IRPScenario:
    n_cust = 1                                          # Cust_Nr = 1
    n_prod = 1                                          # P_Nr = 1
    cons = rng.integers(1, 21, size=(n_cust, n_prod)).astype(float)  # P_Cons 1-20
    w_cap_cust = rng.integers(50, 101, size=n_cust).astype(float)    # added

    return IRPScenario(
        n_supplier_warehouses=1,                        # W_Nr_Sup = 1
        supplier_warehouse_capacity=200.0,              # added (realistic)
        n_products=n_prod,
        product_sizes=np.ones(n_prod),                  # not used (=1)
        supplier_holding_cost=np.full(n_prod, 0.05),    # added
        customer_holding_cost=np.full(n_prod, 0.10),    # added
        n_vehicle_types=1,                              # V_Ty = 1
        vehicles_per_type=np.array([1]),                # V_Nr_Ty = 1
        vehicle_capacity=np.array([1e6]),               # unlimited (not in scenario)
        vehicle_cost_per_dist=np.array([0.0]),           # no cost (not in scenario)
        n_customers=n_cust,
        n_customer_warehouses=np.ones(n_cust, dtype=int),  # W_Nr_Cust = 1
        customer_warehouse_capacity=w_cap_cust,         # added
        daily_consumption=cons,
        replenishment_strategy=1,                       # not used (default)
        n_periods=5,
        distance_matrix=_make_distance_matrix(n_cust, rng),
    )


# ==================================================================
# SIMPLE
# ==================================================================
# Table 5 description:
#   "No Warehouse Capacity, No Holding Cost, No Replenishment Strategy"
#
#TODO NOTE:
#   Like Basic, the original Table 5 omits warehouse capacity and
#   holding costs for Simple. Without holding costs, there is no
#   incentive to minimize inventory, and without warehouse capacity
#   the RS=1 max-level strategy tries to deliver ~1M units (causing
#   astronomical supplier stockout penalties). We add these to make
#   the scenario a meaningful IRP. Vehicle constraints remain as
#   specified in Table 5. --> add in paper
#
# Parameters used:         Value
#   W_Nr_Sup               1
#   W_Ca_Sup               500
#   P_Nr                   1
#   P_Si                   1
#   P_Ho_Co_Sup            0.05
#   P_Ho_Co_Cust           0.1
#   V_Ty                   1
#   V_Nr_Ty                1
#   V_Ca_Ty                5-15
#   V_Ty_Co                2
#   Cust_Nr                1-4
#   W_Nr_Cust              1
#   W_Ca_Cust              30-60
#   P_Cons                 5-25
# ==================================================================
def _simple(rng) -> IRPScenario:
    n_cust = int(rng.integers(1, 5))                    # Cust_Nr 1-4
    n_prod = 1                                          # P_Nr = 1
    cons = rng.integers(5, 26, size=(n_cust, n_prod)).astype(float)   # P_Cons 5-25
    v_cap = float(rng.integers(5, 16))                  # V_Ca_Ty 5-15
    w_cap_cust = rng.integers(30, 61, size=n_cust).astype(float)      # added

    return IRPScenario(
        n_supplier_warehouses=1,                        # W_Nr_Sup = 1
        supplier_warehouse_capacity=500.0,              # added (realistic)
        n_products=n_prod,
        product_sizes=np.ones(n_prod),                  # P_Si = 1
        supplier_holding_cost=np.full(n_prod, 0.05),    # added
        customer_holding_cost=np.full(n_prod, 0.10),    # added
        n_vehicle_types=1,                              # V_Ty = 1
        vehicles_per_type=np.array([1]),                # V_Nr_Ty = 1
        vehicle_capacity=np.array([v_cap]),             # V_Ca_Ty 5-15
        vehicle_cost_per_dist=np.array([2.0]),           # V_Ty_Co = 2
        n_customers=n_cust,
        n_customer_warehouses=np.ones(n_cust, dtype=int),  # W_Nr_Cust = 1
        customer_warehouse_capacity=w_cap_cust,         # added
        daily_consumption=cons,
        replenishment_strategy=1,                       # not used (default)
        n_periods=5,
        distance_matrix=_make_distance_matrix(n_cust, rng),
    )


# ==================================================================
# INTERMEDIATE
# ==================================================================
# Table 5 description:
#   "All parameters"
#
# Parameters used:         Value
#   W_Nr_Sup               1
#   W_Ca_Sup               30-50
#   P_Nr                   1-3
#   P_Ho_Co_Sup            0.05-0.10
#   P_Si                   1
#   V_Ty                   2
#   V_Nr_Ty                2-5
#   V_Ca_Ty                5-30
#   V_Ty_Co                0.5-1
#   Cust_Nr                3-8
#   W_Nr_Cust              1
#   W_Ca_Cust              40-60
#   P_Cons                 15-30
#   P_Ho_Co_Cust           0.05-0.30
#   RS                     1
# ==================================================================
def _intermediate(rng) -> IRPScenario:
    n_cust = int(rng.integers(3, 9))                    # Cust_Nr 3-8
    n_prod = int(rng.integers(1, 4))                    # P_Nr 1-3
    n_vt = 2                                            # V_Ty = 2

    cons = rng.integers(15, 31, size=(n_cust, n_prod)).astype(float)  # P_Cons 15-30
    w_cap_sup = float(rng.integers(30, 51))             # W_Ca_Sup 30-50
    w_cap_cust = rng.integers(40, 61, size=n_cust).astype(float)  # W_Ca_Cust 40-60
    v_nr = rng.integers(2, 6, size=n_vt)                # V_Nr_Ty 2-5
    v_cap = rng.integers(5, 31, size=n_vt).astype(float)  # V_Ca_Ty 5-30
    v_cost = rng.uniform(0.5, 1.0, size=n_vt)           # V_Ty_Co 0.5-1
    h_sup = rng.uniform(0.05, 0.10, size=n_prod)        # P_Ho_Co_Sup 0.05-0.10
    h_cust = rng.uniform(0.05, 0.30, size=n_prod)       # P_Ho_Co_Cust 0.05-0.30

    return IRPScenario(
        n_supplier_warehouses=1,                        # W_Nr_Sup = 1
        supplier_warehouse_capacity=w_cap_sup,
        n_products=n_prod,
        product_sizes=np.ones(n_prod),                  # P_Si = 1
        supplier_holding_cost=h_sup,
        customer_holding_cost=h_cust,
        n_vehicle_types=n_vt,
        vehicles_per_type=v_nr,
        vehicle_capacity=v_cap,
        vehicle_cost_per_dist=v_cost,
        n_customers=n_cust,
        n_customer_warehouses=np.ones(n_cust, dtype=int),  # W_Nr_Cust = 1
        customer_warehouse_capacity=w_cap_cust,
        daily_consumption=cons,
        replenishment_strategy=1,                       # RS = 1
        n_periods=5,
        distance_matrix=_make_distance_matrix(n_cust, rng),
    )


# ==================================================================
# ADVANCED
# ==================================================================
# Table 5 description:
#   "All parameters"
#
# Parameters used:         Value
#   W_Nr_Sup               1-3
#   W_Ca_Sup               50-100
#   P_Nr                   2-6
#   P_Ho_Co_Sup            0.10-0.30
#   P_Si                   1-3
#   V_Ty                   1-3
#   V_Nr_Ty                5-10
#   V_Ca_Ty                10-40
#   V_Ty_Co                0.5-2
#   Cust_Nr                10-15
#   W_Nr_Cust              1-3
#   W_Ca_Cust              40-80
#   P_Cons                 5-20
#   P_Ho_Co_Cust           0.15-0.5
#   RS                     1
# ==================================================================
def _advanced(rng) -> IRPScenario:
    n_cust = int(rng.integers(10, 16))                  # Cust_Nr 10-15
    n_prod = int(rng.integers(2, 7))                    # P_Nr 2-6
    n_vt = int(rng.integers(1, 4))                      # V_Ty 1-3
    n_sup_wh = int(rng.integers(1, 4))                  # W_Nr_Sup 1-3

    cons = rng.integers(5, 21, size=(n_cust, n_prod)).astype(float)   # P_Cons 5-20
    w_cap_sup = float(rng.integers(50, 101))            # W_Ca_Sup 50-100
    n_cust_wh = rng.integers(1, 4, size=n_cust)         # W_Nr_Cust 1-3
    w_cap_cust = rng.integers(40, 81, size=n_cust).astype(float)  # W_Ca_Cust 40-80
    p_sizes = rng.integers(1, 4, size=n_prod).astype(float)  # P_Si 1-3
    v_nr = rng.integers(5, 11, size=n_vt)               # V_Nr_Ty 5-10
    v_cap = rng.integers(10, 41, size=n_vt).astype(float)  # V_Ca_Ty 10-40
    v_cost = rng.uniform(0.5, 2.0, size=n_vt)           # V_Ty_Co 0.5-2
    h_sup = rng.uniform(0.10, 0.30, size=n_prod)        # P_Ho_Co_Sup 0.10-0.30
    h_cust = rng.uniform(0.15, 0.50, size=n_prod)       # P_Ho_Co_Cust 0.15-0.5

    return IRPScenario(
        n_supplier_warehouses=n_sup_wh,
        supplier_warehouse_capacity=w_cap_sup,
        n_products=n_prod,
        product_sizes=p_sizes,
        supplier_holding_cost=h_sup,
        customer_holding_cost=h_cust,
        n_vehicle_types=n_vt,
        vehicles_per_type=v_nr,
        vehicle_capacity=v_cap,
        vehicle_cost_per_dist=v_cost,
        n_customers=n_cust,
        n_customer_warehouses=n_cust_wh,
        customer_warehouse_capacity=w_cap_cust,
        daily_consumption=cons,
        replenishment_strategy=1,                       # RS = 1
        n_periods=5,
        distance_matrix=_make_distance_matrix(n_cust, rng),
    )


# ==================================================================
# COMPLEX
# ==================================================================
# Table 5 description:
#   "All parameters"
#
# Parameters used:         Value
#   W_Nr_Sup               2-5
#   W_Ca_Sup               100-300
#   P_Nr                   4-10
#   P_Ho_Co_Sup            0.10-0.40
#   P_Si                   1-7
#   V_Ty                   2-6
#   V_Nr_Ty                10-20
#   V_Ca_Ty                10-100
#   V_Ty_Co                0.3-3
#   Cust_Nr                30-50
#   W_Nr_Cust              5-20
#   W_Ca_Cust              50-90
#   P_Cons                 20-30
#   P_Ho_Co_Cust           0.10-0.9
#   RS                     1 or 2
# ==================================================================
def _complex(rng) -> IRPScenario:
    n_cust = int(rng.integers(30, 51))                  # Cust_Nr 30-50
    n_prod = int(rng.integers(4, 11))                   # P_Nr 4-10
    n_vt = int(rng.integers(2, 7))                      # V_Ty 2-6
    n_sup_wh = int(rng.integers(2, 6))                  # W_Nr_Sup 2-5

    cons = rng.integers(20, 31, size=(n_cust, n_prod)).astype(float)  # P_Cons 20-30
    w_cap_sup = float(rng.integers(100, 301))           # W_Ca_Sup 100-300
    n_cust_wh = rng.integers(5, 21, size=n_cust)        # W_Nr_Cust 5-20
    w_cap_cust = rng.integers(50, 91, size=n_cust).astype(float)  # W_Ca_Cust 50-90
    p_sizes = rng.integers(1, 8, size=n_prod).astype(float)  # P_Si 1-7
    v_nr = rng.integers(10, 21, size=n_vt)              # V_Nr_Ty 10-20
    v_cap = rng.integers(10, 101, size=n_vt).astype(float)  # V_Ca_Ty 10-100
    v_cost = rng.uniform(0.3, 3.0, size=n_vt)           # V_Ty_Co 0.3-3
    h_sup = rng.uniform(0.10, 0.40, size=n_prod)        # P_Ho_Co_Sup 0.10-0.40
    h_cust = rng.uniform(0.10, 0.90, size=n_prod)       # P_Ho_Co_Cust 0.10-0.9
    rs = int(rng.choice([1, 2]))                        # RS = 1 or 2

    return IRPScenario(
        n_supplier_warehouses=n_sup_wh,
        supplier_warehouse_capacity=w_cap_sup,
        n_products=n_prod,
        product_sizes=p_sizes,
        supplier_holding_cost=h_sup,
        customer_holding_cost=h_cust,
        n_vehicle_types=n_vt,
        vehicles_per_type=v_nr,
        vehicle_capacity=v_cap,
        vehicle_cost_per_dist=v_cost,
        n_customers=n_cust,
        n_customer_warehouses=n_cust_wh,
        customer_warehouse_capacity=w_cap_cust,
        daily_consumption=cons,
        replenishment_strategy=rs,
        n_periods=5,
        distance_matrix=_make_distance_matrix(n_cust, rng),
    )