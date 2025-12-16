# Author : Charan
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import (
    NonNegativeReals,
    Var,
    units as pyunits,
    value,
)
import pandas as pd
from pyomo.opt import assert_optimal_termination
from watertap.core.solvers import get_solver
from watertap.costing import WaterTAPCosting
from idaes.core import UnitModelCostingBlock
from idaes.models.unit_models import Feed, Product
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.unit_models.reverse_osmosis_1D import ReverseOsmosis1D
from idaes.core import FlowsheetBlock
from idaes.models.unit_models import Feed, Mixer, MomentumMixingType, Product
from pyomo.core import Set, TransformationFactory, Objective
from pyomo.network import Arc
from pyomo.environ import ConcreteModel, NonNegativeReals
from watertap.core.membrane_channel_base import (
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    ModuleType,
    PressureChangeType,
)
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.unit_models.reverse_osmosis_1D import ReverseOsmosis1D
from idaes.core.util.scaling import set_scaling_factor, calculate_scaling_factors
from idaes.core.util.initialization import propagate_state




def build_two_stage_with_booster(correlation_type="guillen", nfe=60):
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
    m.fs.booster_pump = Pump(property_package=m.fs.properties)
    m.fs.erd = EnergyRecoveryDevice(property_package=m.fs.properties)

    # RO Kwargs
    ro_kwargs = {
        "concentration_polarization_type": ConcentrationPolarizationType.calculated,
        "mass_transfer_coefficient": MassTransferCoefficient.calculated,
        "pressure_change_type": PressureChangeType.calculated,
        "module_type": ModuleType.flat_sheet,
        "has_pressure_change": True,
        "transformation_method": "dae.finite_difference",
        "transformation_scheme": "BACKWARD",
        "finite_elements": nfe,
        "has_full_reporting": True,
    }

    # Define RO unit model
    m.fs.ro_stages = Set(initialize=[1, 2], doc="RO stages")
    m.fs.ro = ReverseOsmosis1D(
        m.fs.ro_stages, property_package=m.fs.properties, **ro_kwargs
    )
    # Add spiral wound width constraint and change Sherwood number and friction factor correlations as required
    for stage in m.fs.ro.values():
        add_geometry_constraints(stage)
        change_sh_and_f_correlations(stage, correlation_type=correlation_type)

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
        elif i ==2:
            setattr(
                m.fs,
                f"ro_inlet_arc_{i}",
                Arc(
                    source=m.fs.booster_pump.outlet,
                    destination=m.fs.ro[i].inlet,
                ),
            )

        # Exit connections for RO stages
        if i ==1 :
            # Connect the Brine to booster
            m.fs.ro_to_booster = Arc(
                source=m.fs.ro[i].retentate, destination=m.fs.booster_pump.inlet
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

    m.fs.retentate_to_erd = Arc(source=m.fs.ro[2].retentate, destination=m.fs.erd.inlet)
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


def fix_model(m, velocity=0.25, salinity=35, ro_system="SWRO"):
    m.fs.feed.properties[0].temperature.fix(298.15)  # K
    m.fs.feed.properties[0].pressure.fix(101325)  # Pa, 1 atm

    first_stage_pvs = 180  # Number of pressure vessels in the first stage
    if ro_system == "SWRO":
        hc = 0.7112e-3  # Channel height for SWRO
    elif ro_system == "BWRO":
        hc = 0.8636e-3  # Channel height for BWRO
    else:
        raise ValueError("Type must be either 'SWRO' or 'BWRO'")
    # Calculate the feed flow rate based on the desired velocity and membrane area
    # Based on an RO element with 8-inch diameter and 40-inch length
    channel_area = (
        (hc * 0.85 / (hc + 0.254e-3 + 0.2032e-3))
        * first_stage_pvs
        * 0.04776103732
    )
    feed_flow_rate = velocity * channel_area  # m^3/s
    calculate_feed_state(m, feed_salinity=salinity, vol_flow=feed_flow_rate)
    # Pump efficiency
    m.fs.pump.efficiency_pump.fix(0.8)  # 80% efficiency
    m.fs.booster_pump.efficiency_pump.fix(0.8)  # 80% efficiency
    # ERD efficiency
    m.fs.erd.efficiency_pump.fix(0.8)  # 80% efficiency
    m.fs.erd.control_volume.properties_out[0].pressure.fix(101325)  # Pa, 1 atm
    # Membrane performance and geometry parameters for each RO stage
    if ro_system == "SWRO":
        water_permeability = 2.97e-12  # m^2/s/bar, water permeability for SWRO
        salt_permeability = 1.58e-8  # m/s/bar, Salt permeability for SWRO
        channel_height = 0.7112e-3  # 1 mm channel height for SWRO
    elif ro_system == "BWRO":
        water_permeability = 9.36e-12
        salt_permeability = 2.38e-8
        channel_height = 0.8636e-3  # 1 mm channel height for BWRO
    for s_id, stage in m.fs.ro.items():
        if s_id == 1:
            stage.n_pressure_vessels.fix(first_stage_pvs)
        elif s_id == 2:
            stage.n_pressure_vessels.fix(first_stage_pvs * 2/3)

        stage.A_comp[0, "H2O"].fix(water_permeability)  # m^2/s/bar, water permeability
        stage.B_comp[0, "TDS"].fix(salt_permeability)  # m/s/bar, Salt permeability
        stage.feed_side.channel_height.fix(channel_height)  # 1 mm channel height
        stage.length.fix(6)  # 6 m length of the RO module
        stage.feed_side.spacer_porosity.fix(0.85)  # 85% porosity of the spacer
        stage.width.setub(None)
        stage.area.setub(None)
        stage.mixed_permeate[0].pressure.fix(101325)  # 1 atm
        # Fix the Sherwood and friction factor improvement factors to 1
        if hasattr(stage.feed_side, "Sh_improvement_factor"):
            stage.feed_side.Sh_improvement_factor.fix(1)
            stage.feed_side.friction_factor_improvement.fix(1)
        elif hasattr(stage.feed_side, "sherwood_number"):
            stage.feed_side.sherwood_number.fix(30)
            stage.feed_side.power_number.fix(1e6)


def scale_model(m, ro_system="SWRO"):
    # Feed block
    set_scaling_factor(m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"], 1e-2)
    set_scaling_factor(m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"], 1)

    # Product block
    set_scaling_factor(
        m.fs.product.properties[0].flow_mass_phase_comp["Liq", "TDS"], 1e-1
    )
    set_scaling_factor(m.fs.product.properties[0].flow_mass_phase_comp["Liq", "H2O"], 1)

    # Brine block
    set_scaling_factor(m.fs.brine.properties[0].flow_mass_phase_comp["Liq", "TDS"], 1e3)
    set_scaling_factor(m.fs.brine.properties[0].flow_mass_phase_comp["Liq", "H2O"], 1)

    set_scaling_factor(m.fs.pump.control_volume.work, 1e-3)
    set_scaling_factor(
        m.fs.pump.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "H2O"], 1
    )
    set_scaling_factor(
        m.fs.pump.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "TDS"],
        1e2,
    )
    set_scaling_factor(
        m.fs.pump.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"], 1
    )
    set_scaling_factor(
        m.fs.pump.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
        1e2,
    )

    # Energy Recovery Device (ERD)
    set_scaling_factor(m.fs.erd.control_volume.work, 1e-3)
    set_scaling_factor(
        m.fs.erd.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "H2O"], 1
    )
    set_scaling_factor(
        m.fs.erd.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "TDS"], 1e3
    )
    set_scaling_factor(
        m.fs.erd.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"], 1
    )
    set_scaling_factor(
        m.fs.erd.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
        1e3,
    )

    # Reverse Osmosis (RO) unit models
    for ro_stage in m.fs.ro.values():
        for pos in ro_stage.feed_side.properties.keys():
            # Feed side flow variables
            for (p, j), v in ro_stage.feed_side.properties[
                pos
            ].flow_mass_phase_comp.items():
                if p == "Liq" and j == "H2O":
                    set_scaling_factor(v, 1)
                elif p == "Liq" and j == "TDS":
                    set_scaling_factor(v, 1e2)
                else:
                    raise ValueError(
                        "Unexpected phase or component in feed side flow variables."
                    )

            # Interface flow variables
            for (p, j), v in ro_stage.feed_side.properties_interface[
                pos
            ].flow_mass_phase_comp.items():
                if p == "Liq" and j == "H2O":
                    set_scaling_factor(v, 1)
                elif p == "Liq" and j == "TDS":
                    set_scaling_factor(v, 1e2)
                else:
                    raise ValueError(
                        "Unexpected phase or component in feed side flow variables."
                    )

            # Permeate side flow variables
            for (p, j), v in ro_stage.permeate_side[pos].flow_mass_phase_comp.items():
                if p == "Liq" and j == "H2O":
                    set_scaling_factor(v, 1)
                elif p == "Liq" and j == "TDS":
                    set_scaling_factor(v, 1e3)
                else:
                    raise ValueError(
                        "Unexpected phase or component in feed side flow variables."
                    )

        # Mixed permeate flow variables
        set_scaling_factor(
            ro_stage.mixed_permeate[0].flow_mass_phase_comp["Liq", "H2O"], 1
        )
        set_scaling_factor(
            ro_stage.mixed_permeate[0].flow_mass_phase_comp["Liq", "TDS"], 1e3
        )

        set_scaling_factor(ro_stage.area, 1e-3)  # Area of the RO module
        set_scaling_factor(ro_stage.length, 1e-1)  # Length of the RO module
        set_scaling_factor(ro_stage.feed_side.channel_height, 1e-3)  # Channel height
        set_scaling_factor(ro_stage.feed_side.spacer_porosity, 1)  # Spacer porosity
        set_scaling_factor(ro_stage.A_comp[0, "H2O"], 1e12)  # Water permeability
        set_scaling_factor(ro_stage.B_comp[0, "TDS"], 1e8)  # Salt permeability
        set_scaling_factor(ro_stage.width, 1e-2)  # Width of the RO module
        set_scaling_factor(ro_stage.feed_side.area, 1e-2)  # Area of the feed side

        if hasattr(ro_stage.feed_side, "pressure_dx"):
            for v in ro_stage.feed_side.pressure_dx.values():
                v.set_value(0.0, skip_validation=True)

        if hasattr(ro_stage.feed_side, "material_flow_dx"):
            for v in ro_stage.feed_side.material_flow_dx.values():
                v.set_value(0.0, skip_validation=True)

        calculate_scaling_factors(ro_stage)

    if ro_system == "BWRO":
        # Mixer
        for i in m.fs.ro_stages:
            state = getattr(m.fs.P_mixer, f"stage_{i}_state")
            set_scaling_factor(state[0].flow_mass_phase_comp["Liq", "H2O"], 1)
            set_scaling_factor(state[0].flow_mass_phase_comp["Liq", "TDS"], 1e2)

        set_scaling_factor(
            m.fs.P_mixer.mixed_state[0].flow_mass_phase_comp["Liq", "H2O"], 1
        )
        set_scaling_factor(
            m.fs.P_mixer.mixed_state[0].flow_mass_phase_comp["Liq", "TDS"], 1e2
        )


def initialize_model(m, overpressure=2, ro_system="SWRO", verbose=False):
    # Touch the osmotic pressure variables to ensure they are initialized
    m.fs.feed.properties[0].pressure_osm_phase[...]
    m.fs.feed.properties[0].flow_vol_phase[...]
    m.fs.feed.properties[0].conc_mass_phase_comp[...]
    m.fs.feed.initialize()

    propagate_state(m.fs.feed_to_pump)
    osm_pressure = m.fs.feed.properties[0].pressure_osm_phase["Liq"].value
    print(f"Osmotic pressure of feed: {osm_pressure * 1e-5:.2f} bar")
    m.fs.pump.outlet.pressure[0].fix(overpressure * osm_pressure)
    m.fs.pump.initialize()
    if verbose:
        m.fs.pump.report()


    # Initialize each RO stage sequentially
    for s_id, stage in m.fs.ro.items():

        if s_id ==2:
            propagate_state(m.fs.ro_to_booster)
            booster_osm_pressure = osm_pressure * overpressure * 1.5
            m.fs.booster_pump.outlet.pressure[0].fix(booster_osm_pressure)
            m.fs.booster_pump.initialize()

        propagate_state(getattr(m.fs, "ro_inlet_arc_" + str(s_id)))
        stage.initialize()
        if verbose:
            stage.report()
        # propagate the permeate stream to the mixer
        propagate_state(getattr(m.fs, f"stage_{s_id}_to_P_mixer"))




        # Initialize the permeate mixer after all RO stages are initialized
        m.fs.P_mixer.initialize()

    # Initialize product and brine streams
    m.fs.product.properties[0].flow_vol_phase[...]
    m.fs.product.properties[0].conc_mass_phase_comp[...]
    if ro_system == "SWRO":
        propagate_state(m.fs.ro_to_product)
    elif ro_system == "BWRO":
        propagate_state(m.fs.P_mixer_to_product)
    m.fs.product.initialize()

    propagate_state(m.fs.retentate_to_erd)
    m.fs.erd.initialize()

    m.fs.brine.properties[0].flow_vol_phase[...]
    m.fs.brine.properties[0].conc_mass_phase_comp[...]
    propagate_state(m.fs.erd_to_brine)
    m.fs.brine.initialize()

    if verbose:
        m.fs.product.report()
        m.fs.erd.report()
        m.fs.brine.report()


def set_low_salinity_bounds(m):

    for ro in m.fs.ro.values():
        # Set low bounds for velocity variables
        for x, v in ro.feed_side.velocity.items():
            v.setlb(1e-6)
            v.setub(1)
        #
        # # Set lower bounds for osmotic pressure variables
        # for x in ro.feed_side.properties:
        #     for p, v in ro.feed_side.properties[x].pressure_osm_phase.items():
        #         v.setlb(1e3)
        # for x in ro.feed_side.properties_interface:
        #     for p, v in ro.feed_side.properties_interface[x].pressure_osm_phase.items():
        #         v.setlb(1e3)
        # for x in ro.permeate_side:
        #     for p, v in ro.permeate_side[x].pressure_osm_phase.items():
        #         v.setlb(1e3)


def add_costing(m):
    # Add Costing
    m.fs.product.properties[0].flow_vol[...]
    m.fs.costing = WaterTAPCosting()
    m.fs.pump.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    for stage in m.fs.ro.values():
        stage.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.erd.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.product.properties[0].flow_vol)
    m.fs.costing.initialize()
    return


def add_lcow_objective(m):
    m.fs.lcow_objective = Objective(expr=m.fs.costing.LCOW)


def change_sh_and_f_correlations(ro, correlation_type="guillen"):

    if hasattr(ro.feed_side, "eq_N_Sh_comp"):
        del ro.feed_side.eq_N_Sh_comp

    if hasattr(ro.feed_side, "eq_friction_factor"):
        del ro.feed_side.eq_friction_factor

    if correlation_type == "parameterized":
        if not hasattr(ro.feed_side, "sherwood_number"):
            ro.feed_side.sherwood_number = Var(
                ro.flowsheet().config.time,
                domain=NonNegativeReals,
                initialize=10,
                doc="User-defined Sherwood number",
                units=pyunits.dimensionless,
            )
        if not hasattr(ro.feed_side, "power_number"):
            ro.feed_side.power_number = Var(
                ro.flowsheet().config.time,
                domain=NonNegativeReals,
                initialize=1e6,
                doc="User-defined Power number",
                units=pyunits.dimensionless,
            )

    if correlation_type != "parameterized":

        if not hasattr(ro.feed_side, "Sh_improvement_factor"):

            # Add improvement factors for Sherwood number and Darcy friction factor
            ro.feed_side.Sh_improvement_factor = Var(
                initialize=1.0,
                domain=NonNegativeReals,
                doc="Improvement factor for Sherwood number",
                units=pyunits.dimensionless,
            )

        if not hasattr(ro.feed_side, "friction_factor_improvement"):

            ro.feed_side.friction_factor_improvement = Var(
                initialize=1.0,
                domain=NonNegativeReals,
                doc="Improvement factor for Darcy friction factor",
                units=pyunits.dimensionless,
            )

    solute_set = ro.config.property_package.solute_set

    @ro.feed_side.Constraint(
        ro.flowsheet().config.time,
        ro.length_domain,
        solute_set,
        doc="Sherwood number equation based on " + str(correlation_type),
    )
    def eq_N_Sh_comp(b, t, x, j):
        ct = correlation_type.lower()
        if ct == "guillen":
            return (
                b.N_Sh_comp[t, x, j]
                == 0.46
                * (b.N_Re[t, x] * b.N_Sc_comp[t, x, j]) ** 0.36
                * b.Sh_improvement_factor
            )
        elif ct == "schock":
            return (
                b.N_Sh_comp[t, x, j]
                == 0.065
                * b.N_Re[t, x] ** 0.875
                * b.N_Sc_comp[t, x, j] ** 0.25
                * b.Sh_improvement_factor
            )
        elif ct == "dacosta":
            angle_rad = 3.141592653589793 / 2  # pi/2
            return b.N_Sh_comp[t, x, j] == 0.644 * b.N_Re[t, x] ** 0.5 * b.N_Sc_comp[
                t, x, j
            ] ** 0.33 * (
                1.654
                * 0.5
                ** -0.039  # 0.5 is the ratio of filament diameter to channel height
                * b.spacer_porosity ** 0.75
                * __import__("math").sin(angle_rad / 2) ** 0.086
                * b.Sh_improvement_factor
            )
        elif ct == "koustou":
            return (
                b.N_Sh_comp[t, x, j]
                == 0.20
                * b.N_Re[t, x] ** 0.57
                * b.N_Sc_comp[t, x, j] ** 0.40
                * b.Sh_improvement_factor
            )
        elif ct == "kuroda":
            return (
                b.N_Sh_comp[t, x, j]
                == 0.50
                * b.N_Re[t, x] ** 0.50
                * b.N_Sc_comp[t, x, j] ** 0.33
                * b.Sh_improvement_factor
            )
        elif ct == "parameterized":
            return b.N_Sh_comp[t, x, j] == b.sherwood_number[t]
        else:
            raise ValueError(
                f"Unrecognized correlation {correlation_type} for Sherwood number"
            )

    @ro.feed_side.Constraint(
        ro.flowsheet().config.time,
        ro.length_domain,
        doc="Darcy friction factor in feed channel based on " + str(correlation_type),
    )
    def eq_friction_factor(b, t, x):
        ct = correlation_type.lower()
        if ct == "guillen":
            return (
                b.friction_factor_darcy[t, x]
                == (0.42 + 189.3 * b.N_Re[t, x] ** -1) * b.friction_factor_improvement
            )
        elif ct == "schock":
            return (
                b.friction_factor_darcy[t, x]
                == 6.23 * b.N_Re[t, x] ** -0.3 * b.friction_factor_improvement
            )
        elif ct == "dacosta":
            return (
                b.friction_factor_darcy[t, x]
                == 3.40 * b.N_Re[t, x] ** -0.24 * b.friction_factor_improvement
            )
        elif ct == "koustou":
            return (
                b.friction_factor_darcy[t, x]
                == 5.30 * b.N_Re[t, x] ** -0.14 * b.friction_factor_improvement
            )
        elif ct == "kuroda":
            return (
                b.friction_factor_darcy[t, x]
                == 4.15 * b.N_Re[t, x] ** -0.21 * b.friction_factor_improvement
            )
        elif ct == "parameterized":
            return (
                b.friction_factor_darcy[t, x] * b.N_Re[t, x] ** 3
                == 2 * b.power_number[t]
            )
        elif ct == "parameterized":
            return (
                b.friction_factor_darcy[t, x] * b.N_Re[t, x] ** 3
                == b.power_number[t] * 2
            )
        else:
            raise ValueError(
                f"Unrecognized correlation {correlation_type} for friction factor"
            )

    if not hasattr(ro.feed_side, "Sh_avg"):
        # Add expression for average CP, Sherwood, and friction factor
        @ro.feed_side.Expression(
            ro.flowsheet().config.time,
            solute_set,
            doc="Average Sherwood number across the channel",
        )
        def Sh_avg(b, t, j):
            return sum(b.N_Sh_comp[t, x, j] for x in b.length_domain if x > 0) / b.nfe

        @ro.feed_side.Expression(
            ro.flowsheet().config.time,
            doc="Average Darcy friction factor across the channel",
        )
        def f_avg(b, t):
            return (
                sum(b.friction_factor_darcy[t, x] for x in b.length_domain if x > 0)
                / b.nfe
            )

        @ro.feed_side.Expression(
            ro.flowsheet().config.time,
            solute_set,
            doc="Average concentration polarization coefficient across the channel",
        )
        def CP_avg(b, t, j):
            return sum(b.cp_modulus[t, x, j] for x in b.length_domain if x > 0) / b.nfe

        @ro.feed_side.Expression(
            ro.flowsheet().config.time, doc="Average power number across the channel",
        )
        def Pn_avg(b, t):
            return (
                sum(
                    b.friction_factor_darcy[t, x] * b.N_Re[t, x] ** 3 * 0.5
                    for x in b.length_domain
                    if x > 0
                )
                / b.nfe
            )


def add_geometry_constraints(ro):
    ro.n_pressure_vessels = Var(
        initialize=100,
        domain=NonNegativeReals,
        doc="Number of pressure vessels in the RO stage",
        units=pyunits.dimensionless,
    )

    @ro.Constraint(
        doc="Constraint to define the width of the RO stage based on the number of vessels"
    )
    def eq_width(b):
        return b.width * (
            b.feed_side.channel_height
            + 0.254e-3 * pyunits.m  # Permeate Spacer Height
            + 0.2032e-3 * pyunits.m  # Membrane Thickness
        ) == b.n_pressure_vessels * (0.04776103732 * pyunits.m ** 2)


def print_solved_state(m):
    # System-level metrics print as a formatted table
    system_data = [
        {
            "Feed flow rate": f"{value(m.fs.feed.properties[0].flow_vol_phase['Liq']):.2e}"
            f" {pyunits.get_units(m.fs.feed.properties[0].flow_vol_phase['Liq'])}/s",
            "Feed salinity": f"{value(m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'TDS']):.2f}"
            f" {pyunits.get_units(m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'TDS'])}",
            "Pump outlet pressure": f"{value(m.fs.pump.outlet.pressure[0]) * 1e-5:.2f} bar",
            "LCOW": f"{value(m.fs.costing.LCOW):.2f} $/m^3",
            "Specific Energy Consumption": f"{value(m.fs.costing.specific_energy_consumption):.2f} kWh/m^3",
            "Overall water recovery": f"{value(m.fs.water_recovery) * 100:.2f}%",
        }
    ]
    print("\n--- System Level Metrics ---")
    print(pd.DataFrame(system_data).to_markdown(index=False))
    data = []
    for stage in m.fs.ro.values():
        data.append(
            {
                "Stage": stage.name,
                "Area (m^2)": value(stage.area),
                "Length (m)": value(stage.length),
                "Pressure vessels": value(stage.n_pressure_vessels),
                "Inlet velocity (cm/s)": value(stage.feed_side.velocity[0, 0]) * 1e2,
                "Channel Height (mm)": value(stage.feed_side.channel_height) * 1e3,
                "Recovery (%)": value(stage.recovery_vol_phase[0, "Liq"]) * 100,
                "Average f": stage.feed_side.f_avg[0](),
                "Average Pn": stage.feed_side.Pn_avg[0](),
                "DeltaP (bar)": value(stage.deltaP[0]) * 1e-5,
                "Average K": stage.feed_side.K_avg[0, "TDS"](),
                "Average Sh": stage.feed_side.Sh_avg[0, "TDS"](),
                "Average CP": stage.feed_side.CP_avg[0, "TDS"](),
            }
        )
    df = pd.DataFrame(data)
    print("\n--- RO Stage Metrics ---")
    print(df.to_markdown(index=False))


def get_variables_with_key(blk, key, descend_into=True):
    var_list = []
    for var in blk.component_data_objects(Var, descend_into=descend_into):
        if key in var.name:
            var_list.append(var)
    return var_list


def collect_macro_variables(m):
    n_stages = len(m.fs.ro)
    outputs = {
        "LCOW": m.fs.costing.LCOW(),
        "Energy Consumption": m.fs.costing.specific_energy_consumption(),
        "Volumetric Recovery": m.fs.water_recovery.value,
        "Total membrane area": sum(stage.area.value for stage in m.fs.ro.values()),
        "Average cp (TDS)": sum(stage.feed_side.CP_avg[0, "TDS"]() for stage in m.fs.ro.values()) / n_stages,
        "Total pressure drop": sum(stage.deltaP[0].value for stage in m.fs.ro.values()),
        "Average K (TDS)": sum(stage.feed_side.K_avg[0, "TDS"]() for stage in m.fs.ro.values()) / n_stages,
        "Average f": sum(stage.feed_side.f_avg[0]() for stage in m.fs.ro.values()) / n_stages,
        "Average Re": sum(stage.feed_side.N_Re_avg[0]() for stage in m.fs.ro.values()) / n_stages,
        "Average Sh": sum(stage.feed_side.Sh_avg[0, "TDS"]() for stage in m.fs.ro.values()) / n_stages,
        "Average Pn": sum(stage.feed_side.Pn_avg[0]() for stage in m.fs.ro.values()) / n_stages,
        "Operating pressure": m.fs.pump.outlet.pressure[0].value,
        "Outlet bulk concentration": m.fs.ro[n_stages].feed_side.properties[0, 1].conc_mass_phase_comp["Liq", "TDS"].value,
        "Outlet interface concentration": m.fs.ro[n_stages].feed_side.properties_interface[0, 1].conc_mass_phase_comp["Liq", "TDS"].value,
    }

    outputs_stagewise = {
        "Membrane area list": [stage.area.value for stage in m.fs.ro.values()],
        "cp (TDS) list": [stage.feed_side.CP_avg[0, "TDS"]() for stage in m.fs.ro.values()],
        "Pressure drop list": [stage.deltaP[0].value for stage in m.fs.ro.values()],
        "K (TDS) list": [stage.feed_side.K_avg[0, "TDS"]() for stage in m.fs.ro.values()],
        "f list": [stage.feed_side.f_avg[0]() for stage in m.fs.ro.values()],
        "Re list": [stage.feed_side.N_Re_avg[0]() for stage in m.fs.ro.values()],
        "Sh (TDS) list": [stage.feed_side.Sh_avg[0, "TDS"]() for stage in m.fs.ro.values()],
        "Pn list": [stage.feed_side.Pn_avg[0]() for stage in m.fs.ro.values()],
        "Inlet velocity list": [stage.feed_side.velocity[0, 0].value for stage in m.fs.ro.values()],
        "Outlet velocity list": [stage.feed_side.velocity[0, 1].value for stage in m.fs.ro.values()],
        "Number of pressure vessels list": [stage.n_pressure_vessels.value for stage in m.fs.ro.values()],
        "Recovery list": [stage.recovery_vol_phase[0, "Liq"].value for stage in m.fs.ro.values()],
        "Channel height list": [stage.feed_side.channel_height.value for stage in m.fs.ro.values()],
        "Length list": [stage.length.value for stage in m.fs.ro.values()],
        "Porosity list": [stage.feed_side.spacer_porosity.value for stage in m.fs.ro.values()],
        "Sh_improvement_list": [getattr(stage.feed_side, "Sh_improvement_factor", None).value if hasattr(stage.feed_side, "Sh_improvement_factor") else None for stage in m.fs.ro.values()],
        "f_improvement_list": [getattr(stage.feed_side, "friction_factor_improvement", None).value if hasattr(stage.feed_side, "friction_factor_improvement")
        else None for stage in m.fs.ro.values()],
        "k_error_list": [getattr(stage.feed_side, "k_error_factor", None).value if hasattr(stage.feed_side, "k_error_factor") else None for stage in m.fs.ro.values()],
        "f_error_list": [getattr(stage.feed_side, "f_error_factor", None).value if hasattr(stage.feed_side, "f_error_factor") else None for stage in m.fs.ro.values()],
    }

    if all(hasattr(stage, "telescoping_potential") for stage in m.fs.ro.values()):
        outputs["Average telescoping potential"] = (
            sum(stage.telescoping_potential.value for stage in m.fs.ro.values()) / n_stages
        )
        outputs_stagewise["Telescoping potential list"] = [
            stage.telescoping_potential.value for stage in m.fs.ro.values()
        ]
    all_outputs = {**outputs, **outputs_stagewise}
    return all_outputs


def collect_micro_trend(m, keys=None):
    """
    Collects micro-scale trends for multiple keys across all RO stages,
    returning a structured table suitable for export to Excel.

    Args:
        m: Pyomo model with RO stages.
        keys: List of attribute names (strings) to collect from feed_side.

    Returns:
        pd.DataFrame with columns: 'global_length_domain', <key1>, <key2>, ...
    """
    if keys is None:
        keys = ["velocity", "Re", "Sh", "cp_modulus", "K", "deltaP", "f", "Pn"]
    global_length_domain = []
    cumulative_length = 0.0
    # Build the global length domain
    for i in m.fs.ro_stages:
        stage = m.fs.ro[i]
        ld = list(stage.feed_side.length_domain)
        if len(ld) > 1:
            scaled_domain = [
                cumulative_length + x * value(stage.length) for x in ld[1:]
            ]
            global_length_domain.extend(scaled_domain)
            cumulative_length += value(stage.length)

    # Initialize dictionary for each key
    data = {"global_length_domain": global_length_domain}
    for key in keys:
        values = []
        for i in m.fs.ro_stages:
            stage = m.fs.ro[i]
            ld = list(stage.feed_side.length_domain)
            if key == "velocity":
                var = stage.feed_side.velocity
                for x in ld[1:]:
                    if (0, x) in var:
                        values.append(var[0, x].value)
            elif key == "Re":
                var = stage.feed_side.N_Re
                for x in ld[1:]:
                    if (0, x) in var:
                        values.append(var[0, x].value)
            elif key == "Sh":
                var = stage.feed_side.N_Sh_comp
                for x in ld[1:]:
                    if (0, x, "TDS") in var:
                        values.append(var[0, x, "TDS"].value)
            elif key == "cp_modulus":
                var = stage.feed_side.cp_modulus
                for x in ld[1:]:
                    if (0, x, "TDS") in var:
                        values.append(var[0, x, "TDS"].value)
            elif key == "K":
                var = stage.feed_side.K
                for x in ld[1:]:
                    if (0, x, "TDS") in var:
                        values.append(var[0, x, "TDS"].value)
            elif key == "deltaP":
                var = stage.feed_side.deltaP
                for x in ld[1:]:
                    if (0, x) in var:
                        values.append(var[0, x].value)
            elif key == "f":
                var = stage.feed_side.friction_factor_darcy
                for x in ld[1:]:
                    if (0, x) in var:
                        values.append(var[0, x].value)
            elif key == "Pn":
                for x in ld[1:]:
                    f_val = stage.feed_side.friction_factor_darcy[0, x].value
                    re_val = stage.feed_side.N_Re[0, x].value
                    values.append(f_val * re_val ** 3 * 0.5)
        data[key] = values
    df = pd.DataFrame(data)
    return df


def calculate_feed_state(m, feed_salinity, vol_flow):
    m.fs.feed.properties[0].flow_mass_phase_comp[
        ...
    ].unfix()  # Unfix the mass flow rates to recalculate them based on the new conditions
    m.fs.feed.properties.calculate_state(
        var_args={
            (
                "conc_mass_phase_comp",
                ("Liq", "TDS"),
            ): feed_salinity,  # feed mass concentration
            ("flow_vol_phase", "Liq"): vol_flow,
        },  # volumetric feed flowrate [-]
        hold_state=True,  # fixes the calculated component mass flow rates
    )
    print(
        f"Fixed the feed conditions to salinity: "
        f"{value(m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'TDS']):.2f}"
        f"{pyunits.get_units(m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'TDS'])}"
        f" and volumetric flow rate: {value(m.fs.feed.properties[0].flow_vol_phase['Liq']):.2e}"
        f"{pyunits.get_units(m.fs.feed.properties[0].flow_vol_phase['Liq'])}"
    )
    return



def solve_for_recovery(m, solver=None, tee=False, display=True, strategy='simulation', recovery=0.5):
    # Always ensure that bounds and constraints are applied before solving
    if strategy  == 'simulation':
        # Ensure LCOW objective is not active during simulation solves
        if hasattr(m.fs, "lcow_objective") and m.fs.lcow_objective.active:
            m.fs.lcow_objective.deactivate()
        # Unfix the pressure and fix the recovery as needed
        m.fs.pump.outlet.pressure[0].unfix()
        m.fs.water_recovery.fix(recovery)
        assert degrees_of_freedom(m) == 0, "DOF is not zero before simulation solve."


    elif strategy == 'optimization':
        if hasattr(m.fs, "lcow_objective") and not m.fs.lcow_objective.active:
            m.fs.lcow_objective.activate()
        elif not hasattr(m.fs, "lcow_objective"):
            m.fs.lcow_objective = Objective(
                expr=m.fs.costing.LCOW
            )

    # Solve the model
    results = solve(m, solver=solver, tee=tee, display=display)
    return results


def solve(m, solver=None, tee=False, display=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(m, tee=tee)
    assert_optimal_termination(results)
    if display:
        print_solved_state(m)
    return results


if __name__ == "__main__":
    model = build_two_stage_with_booster(correlation_type="guillen", nfe = 10)

    # Print all arcs to verify connections
    for arc in model.fs.component_objects(Arc, descend_into=True):
        print(f"Arc: {arc.name}, Source: {arc.source}, Destination: {arc.destination}")

    fix_model(model, velocity=0.25, salinity=35, ro_system="SWRO")

    scale_model(model, ro_system="SWRO")

    initialize_model(model, overpressure=4, ro_system="BWRO", verbose=True)

    solve(model, tee=True, display=False)

    add_costing(model)

    solve(model, tee=True, display=True)

    solve_for_recovery(model, strategy='simulation', tee=True, display=True, recovery=0.75)

    model.fs.costing.pprint()






