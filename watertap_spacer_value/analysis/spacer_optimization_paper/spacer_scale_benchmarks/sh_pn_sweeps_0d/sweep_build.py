# Author : Charan
from pyomo.environ import (
    NonNegativeReals,
)
from watertap.costing import WaterTAPCosting
from idaes.core import UnitModelCostingBlock
from idaes.models.unit_models import Feed, Product
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.unit_models.reverse_osmosis_0D import ReverseOsmosis0D
from idaes.models.unit_models import Feed, Mixer, Product
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import *
from watertap_spacer_value.analysis.spacer_optimization_paper.spacer_scale_benchmarks.literature_correlations.RO_case_studies_across_correlations import solve, solve_for_recovery


def build_swro_flowsheet():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Define property package
    m.fs.properties = SeawaterParameterBlock()

    # Feed, Product, and brine blocks
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.brine = Product(property_package=m.fs.properties)

    # Pump and Energy Recovery Device (ERD)
    m.fs.pump = Pump(property_package=m.fs.properties)
    m.fs.erd = EnergyRecoveryDevice(property_package=m.fs.properties)

    # RO Kwargs
    ro_kwargs = {
        "concentration_polarization_type": ConcentrationPolarizationType.calculated,
        "mass_transfer_coefficient": MassTransferCoefficient.calculated,
        "pressure_change_type": PressureChangeType.calculated,
        "module_type": ModuleType.flat_sheet,
        "has_pressure_change": True,
        "has_full_reporting": True,
    }

    # Define RO unit model
    m.fs.ro_stages = Set(initialize=[1], doc="RO stages")
    m.fs.ro = ReverseOsmosis0D(
        m.fs.ro_stages, property_package=m.fs.properties, **ro_kwargs
    )
    # Add spiral wound width constraint and change Sherwood number and friction factor correlations as required
    for stage in m.fs.ro.values():
        add_geometry_constraints(stage)
        change_sh_and_f_correlations(stage, correlation_type="parameterized")

    # Arcs for connections
    m.fs.feed_to_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)

    # Connect RO stage to pump and product
    m.fs.pump_to_ro = Arc(source=m.fs.pump.outlet, destination=m.fs.ro[1].inlet)
    m.fs.ro_to_product = Arc(source=m.fs.ro[1].permeate, destination=m.fs.product.inlet)

    m.fs.retentate_to_erd = Arc(source=m.fs.ro[1].retentate, destination=m.fs.erd.inlet)
    m.fs.erd_to_brine = Arc(source=m.fs.erd.outlet, destination=m.fs.brine.inlet)

    # Expand the arcs in the flowsheet
    TransformationFactory("network.expand_arcs").apply_to(m)

    # Add water recovery variable
    m.fs.water_recovery = Var(
        initialize=0.75,
        domain=NonNegativeReals,
        doc="Water recovery across the RO stages",
        units=pyunits.dimensionless,
    )

    @m.fs.Constraint(doc="Constraint to enforce water recovery across RO stages",)
    def water_recovery_constraint(b):
        return (
            b.water_recovery * b.feed.properties[0].flow_vol_phase["Liq"]
            == b.product.properties[0].flow_vol_phase["Liq"]
        )

    return m


def build_bwro_flowsheet():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Define property package
    m.fs.properties = SeawaterParameterBlock()

    # Feed, Product, and brine blocks
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.brine = Product(property_package=m.fs.properties)

    # Pump and Energy Recovery Device (ERD)
    m.fs.pump = Pump(property_package=m.fs.properties)
    m.fs.erd = EnergyRecoveryDevice(property_package=m.fs.properties)

    # RO Kwargs
    ro_kwargs = {
        "concentration_polarization_type": ConcentrationPolarizationType.calculated,
        "mass_transfer_coefficient": MassTransferCoefficient.calculated,
        "pressure_change_type": PressureChangeType.calculated,
        "module_type": ModuleType.flat_sheet,
        "has_pressure_change": True,
        "has_full_reporting": True,
    }

    # Define RO unit model
    m.fs.ro_stages = Set(initialize=[1, 2, 3], doc="RO stages")
    m.fs.ro = ReverseOsmosis0D(
        m.fs.ro_stages, property_package=m.fs.properties, **ro_kwargs
    )
    # Add spiral wound width constraint and change Sherwood number and friction factor correlations as required
    for stage in m.fs.ro.values():
        add_geometry_constraints(stage)
        change_sh_and_f_correlations(stage, correlation_type="parameterized")

    # Mixer for permeate streams from all RO stages
    m.fs.P_mixer = Mixer(
        property_package=m.fs.properties,
        inlet_list=[f"stage_{i}" for i in m.fs.ro_stages],
        momentum_mixing_type=MomentumMixingType.minimize,
    )

    # Arcs for connections
    m.fs.feed_to_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)

    # Connect RO stages
    for i in m.fs.ro_stages:
        if i == 1:
            setattr(
                m.fs,
                f"ro_inlet_arc_{i}",
                Arc(source=m.fs.pump.outlet, destination=m.fs.ro[i].inlet),
            )
        else:
            setattr(
                m.fs,
                f"ro_inlet_arc_{i}",
                Arc(source=m.fs.ro[i - 1].retentate, destination=m.fs.ro[i].inlet),
            )

        # Permeate mixer connections
        setattr(
            m.fs,
            f"stage_{i}_to_P_mixer",
            Arc(
                source=m.fs.ro[i].permeate,
                destination=getattr(m.fs.P_mixer, f"stage_{i}"),
            ),
        )

    m.fs.P_mixer_to_product = Arc(
        source=m.fs.P_mixer.outlet, destination=m.fs.product.inlet
    )

    m.fs.retentate_to_erd = Arc(source=m.fs.ro[3].retentate, destination=m.fs.erd.inlet)
    m.fs.erd_to_brine = Arc(source=m.fs.erd.outlet, destination=m.fs.brine.inlet)

    # Expand the arcs in the flowsheet
    TransformationFactory("network.expand_arcs").apply_to(m)

    # Add water recovery variable
    m.fs.water_recovery = Var(
        initialize=0.75,
        domain=NonNegativeReals,
        doc="Water recovery across the RO stages",
        units=pyunits.dimensionless,
    )

    @m.fs.Constraint(doc="Constraint to enforce water recovery across RO stages",)
    def water_recovery_constraint(b):
        return (
            b.water_recovery * b.feed.properties[0].flow_vol_phase["Liq"]
            == b.product.properties[0].flow_vol_phase["Liq"]
        )

    return m



if __name__ == "__main__":
    m = build_swro_flowsheet()
    fix_model(m, velocity=0.2, salinity=35, ro_system="SWRO")
    scale_model(m, ro_system="SWRO")
    initialize_model(m, overpressure=2, ro_system="SWRO", verbose=False)
    add_costing(m)
    solve(m, tee=False, solver=None, display=True)
    solve_for_recovery(m, recovery=0.5, tee=False, solver=None, display=True)


    m = build_bwro_flowsheet()
    fix_model(m, velocity=0.2, salinity=5, ro_system="BWRO")
    scale_model(m, ro_system="BWRO")
    set_low_salinity_bounds(m)
    initialize_model(m, overpressure=5.5, ro_system="BWRO", verbose=False)
    add_costing(m)
    solve(m, tee=False, solver=None, display=True)
    solve_for_recovery(m, recovery=0.8, tee=False, solver=None, display=True)


