# Author: Charan Samineni
from idaes.models.unit_models import Feed, Product
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.unit_models.reverse_osmosis_1D import ReverseOsmosis1D
from idaes.core import FlowsheetBlock
from idaes.models.unit_models import Feed, Mixer, MomentumMixingType, Product
from pyomo.environ import (
    ConcreteModel,
    NonNegativeReals,
    Set,
    Var,
    TransformationFactory,
    value,
    units as pyunits,
)
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.unit_models.reverse_osmosis_1D import ReverseOsmosis1D
from watertap.core.membrane_channel_base import (
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
    ModuleType,
)
from pyomo.network import Arc
from pyomo.core import TransformationFactory
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import (
    add_geometry_constraints,
    calculate_feed_state,
    scale_model,
    initialize_model,
    add_costing, print_solved_state,
)
from pyomo.opt import assert_optimal_termination

def build_am_swro_flowsheet(nfe=60):
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
        "mass_transfer_coefficient": MassTransferCoefficient.fixed,
        "pressure_change_type": PressureChangeType.calculated,
        "module_type": ModuleType.flat_sheet,
        "has_pressure_change": True,
        "transformation_method": "dae.finite_difference",
        "transformation_scheme": "BACKWARD",
        "finite_elements": nfe,
        "has_full_reporting": True,
    }

    # Define RO unit model
    m.fs.ro_stages = Set(initialize=[1], doc="RO stages")
    m.fs.ro = ReverseOsmosis1D(
        m.fs.ro_stages, property_package=m.fs.properties, **ro_kwargs
    )
    # Add spiral wound width constraint and change Sherwood number and friction factor correlations as required
    for stage in m.fs.ro.values():
        add_geometry_constraints(stage)
        remove_mass_transfer_and_friction_factor_parameters(stage)
        add_cfd_correlations(stage)

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


def build_am_bwro_flowsheet(nfe=60):
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
        "mass_transfer_coefficient": MassTransferCoefficient.fixed,
        "pressure_change_type": PressureChangeType.calculated,
        "module_type": ModuleType.flat_sheet,
        "has_pressure_change": True,
        "transformation_method": "dae.finite_difference",
        "transformation_scheme": "BACKWARD",
        "finite_elements": nfe,
        "has_full_reporting": True,
    }

    # Define RO unit model
    m.fs.ro_stages = Set(initialize=[1, 2, 3], doc="RO stages")
    m.fs.ro = ReverseOsmosis1D(
        m.fs.ro_stages, property_package=m.fs.properties, **ro_kwargs
    )
    # Add spiral wound width constraint and change Sherwood number and friction factor correlations as required
    for stage in m.fs.ro.values():
        add_geometry_constraints(stage)
        remove_mass_transfer_and_friction_factor_parameters(stage)
        add_cfd_correlations(stage)

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


def remove_mass_transfer_and_friction_factor_parameters(ro_stage):
    feed_side = ro_stage.feed_side

    if hasattr(feed_side, "eq_K"):
        # feed_side.eq_K.pprint()
        del feed_side.eq_K
    else:
        print(f"eq_K not found in {ro_stage.name}, skipping removal.")

    if hasattr(feed_side, "friction_factor_darcy"):
        # feed_side.eq_friction_factor.pprint()
        del feed_side.eq_friction_factor
    else:
        print(f"eq_friction_factor not found in {ro_stage.name}, skipping removal.")


def add_cfd_correlations(ro_stage):
    feed_side = ro_stage.feed_side
    solute_set = feed_side.config.property_package.solute_set

    feed_side.k_error_factor = Var(
        initialize=1.0, doc="Error factor for mass transfer coefficient correlation",
    )

    @feed_side.Constraint(
        ro_stage.flowsheet().config.time,
        feed_side.length_domain,
        solute_set,
        doc="Mass transfer coefficient correlation based on CFD",
    )
    def eq_K(b, t, x, j):
        return b.K[t, x, j] == b.k_error_factor * b.velocity[t, x] ** 0.20 * (
            7.83e-3 * b.channel_height + 3.41e-5
        )

    feed_side.f_error_factor = Var(
        initialize=1.0, doc="Error factor for friction factor correlation",
    )

    @feed_side.Constraint(
        ro_stage.flowsheet().config.time,
        feed_side.length_domain,
        doc="Friction factor correlation based on CFD",
    )
    def eq_friction_factor(b, t, x):
        return b.friction_factor_darcy[t, x] == b.f_error_factor * b.velocity[
            t, x
        ] ** -0.64 * (-1137.16 * b.channel_height + 1.26)

    # Add necessary variables and average calculations for reporting

    # add sherwood number constraint
    ro_stage.feed_side.N_Sh_comp = Var(
        ro_stage.flowsheet().config.time,
        ro_stage.feed_side.length_domain,
        solute_set,
        initialize=5,
        domain=NonNegativeReals,
        doc="Sherwood number",
    )

    @ro_stage.feed_side.Constraint(
        ro_stage.flowsheet().config.time,
        ro_stage.feed_side.length_domain,
        solute_set,
        doc="Sherwood number calculation",
    )
    def eq_sherwood_number(b, t, x, j):
        return (
            b.N_Sh_comp[t, x, j] * b.properties[t, x].diffus_phase_comp["Liq", j]
            == b.K[t, x, j] * b.dh
        )

    # Add average expressions if not already present
    if not hasattr(ro_stage.feed_side, "K_avg"):
        # Add expression for average mass transfer coefficient
        @ro_stage.feed_side.Expression(
            ro_stage.flowsheet().config.time,
            solute_set,
            doc="Average mass transfer coefficient across the channel",
        )
        def K_avg(b, t, j):
            return sum(b.K[t, x, j] for x in b.length_domain if x > 0) / b.nfe

    if not hasattr(ro_stage.feed_side, "Sh_avg"):
        # Add expression for average CP, Sherwood, and friction factor
        @ro_stage.feed_side.Expression(
            ro_stage.flowsheet().config.time,
            solute_set,
            doc="Average Sherwood number across the channel",
        )
        def Sh_avg(b, t, j):
            return sum(b.N_Sh_comp[t, x, j] for x in b.length_domain if x > 0) / b.nfe

        @ro_stage.feed_side.Expression(
            ro_stage.flowsheet().config.time,
            doc="Average Darcy friction factor across the channel",
        )
        def f_avg(b, t):
            return (
                sum(b.friction_factor_darcy[t, x] for x in b.length_domain if x > 0)
                / b.nfe
            )

        @ro_stage.feed_side.Expression(
            ro_stage.flowsheet().config.time,
            solute_set,
            doc="Average concentration polarization coefficient across the channel",
        )
        def CP_avg(b, t, j):
            return sum(b.cp_modulus[t, x, j] for x in b.length_domain if x > 0) / b.nfe

        @ro_stage.feed_side.Expression(
            ro_stage.flowsheet().config.time,
            doc="Average power number across the channel",
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

        @ro_stage.feed_side.Expression(
            ro_stage.flowsheet().config.time,
            doc="Average Reynolds number across the channel",
        )
        def Re_avg(b, t):
            return sum(b.N_Re[t, x] for x in b.length_domain if x > 0) / b.nfe


def fix_aqua_membrane_case(
    m,
    velocity=0.25,
    salinity=35,
    channel_height=0.5e-3,
    porosity=0.95,
    n_vessels=200,
    ro_system="SWRO",
):
    m.fs.feed.properties[0].temperature.fix(298.15)  # K
    m.fs.feed.properties[0].pressure.fix(101325)  # Pa, 1 atm

    hc = channel_height  # m, channel height
    eps = porosity  # spacer porosity
    channel_area = (hc * eps / (hc + 0.254e-3 + 0.2032e-3)) * n_vessels * 0.04776103732
    feed_flow_rate = velocity * channel_area  # m^3/s
    calculate_feed_state(m, feed_salinity=salinity, vol_flow=feed_flow_rate)

    # Pump efficiency
    m.fs.pump.efficiency_pump.fix(0.8)  # 80% efficiency

    # ERD efficiency
    m.fs.erd.efficiency_pump.fix(0.8)  # 80% efficiency
    m.fs.erd.control_volume.properties_out[0].pressure.fix(101325)  # Pa, 1 atm

    # Membrane performance and geometry parameters for each RO stage
    if ro_system == "SWRO":
        water_permeability = 2.97e-12  # m^2/s/bar, water permeability for SWRO
        salt_permeability = 1.58e-8  # m/s/bar, Salt permeability for SWRO
    elif ro_system == "BWRO":
        water_permeability = 9.36e-12
        salt_permeability = 2.38e-8
    for s_id, stage in m.fs.ro.items():
        if s_id == 1:
            stage.n_pressure_vessels.fix(n_vessels)
        elif s_id == 2:
            stage.n_pressure_vessels.fix(n_vessels * 2 / 3)
        elif s_id == 3:
            stage.n_pressure_vessels.fix(n_vessels * 1 / 3)

        stage.A_comp[0, "H2O"].fix(water_permeability)  # m^2/s/bar, water permeability
        stage.B_comp[0, "TDS"].fix(salt_permeability)  # m/s/bar, Salt permeability
        stage.feed_side.channel_height.fix(channel_height)  # 1 mm channel height
        stage.length.fix(6)  # 6 m length of the RO module
        stage.feed_side.spacer_porosity.fix(eps)  # 85% porosity of the spacer
        stage.width.setub(None)
        stage.area.setub(None)
        stage.mixed_permeate[0].pressure.fix(101325)  # 1 atm
        # Fix the Sherwood and friction factor improvement factors to 1

        if hasattr(stage.feed_side, "k_error_factor"):
            stage.feed_side.k_error_factor.fix(1.0)
            stage.feed_side.f_error_factor.fix(1.0)
        elif hasattr(stage.feed_side, "sherwood_number"):
            stage.feed_side.sherwood_number[0].fix(20)
            stage.feed_side.power_number[0].fix(1e6)


