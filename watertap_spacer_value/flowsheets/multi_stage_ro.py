from pyomo.environ import (
    NonNegativeReals,
    Var,
    units as pyunits,
    value,
)
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
from watertap.unit_models.reverse_osmosis_0D import ReverseOsmosis0D
from idaes.core.util.scaling import set_scaling_factor, calculate_scaling_factors
from idaes.core.util.initialization import propagate_state


def build_multi_stage_ro(n_stages=3):
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

    # Create RO stages set from n_stages
    stages = list(range(1, n_stages+1))
    print("stages", stages)
    m.fs.ro_stages = Set(initialize=stages, doc="RO stages")
    m.fs.ro = ReverseOsmosis0D(m.fs.ro_stages, property_package=m.fs.properties, **ro_kwargs)


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

    last_stage = stages[-1]
    m.fs.retentate_to_erd = Arc(source=m.fs.ro[last_stage].retentate, destination=m.fs.erd.inlet)
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


def fix_model(m, flow_rate=1e-3, salinity=35, ro_system="SWRO"):
    m.fs.feed.properties[0].temperature.fix(298.15)  # K
    m.fs.feed.properties[0].pressure.fix(101325)  # Pa, 1 atm

    if ro_system == "SWRO":
        hc = 1e-3  # Channel height for SWRO
    elif ro_system == "BWRO":
        hc = 1e-3  # Channel height for BWRO
    else:
        raise ValueError("Type must be either 'SWRO' or 'BWRO'")
    # Calculate the feed flow rate based on the desired velocity and membrane area
    # Based on an RO element with 8-inch diameter and 40-inch length

    calculate_feed_state(m, feed_salinity=salinity, vol_flow=flow_rate)
    # Pump efficiency
    m.fs.pump.efficiency_pump.fix(0.8)  # 80% efficiency
    # ERD efficiency
    m.fs.erd.efficiency_pump.fix(0.8)  # 80% efficiency
    m.fs.erd.control_volume.properties_out[0].pressure.fix(101325)  # Pa, 1 atm
    # Membrane performance and geometry parameters for each RO stage
    if ro_system == "SWRO":
        water_permeability = 2.97e-12  # m^2/s/bar, water permeability for SWRO
        salt_permeability = 1.58e-8  # m/s/bar, Salt permeability for SWRO
        channel_height = 1e-3  # 1 mm channel height for SWRO
    elif ro_system == "BWRO":
        water_permeability = 9.36e-12
        salt_permeability = 2.38e-8
        channel_height = 1e-3  # 1 mm channel height for BWRO

    for s_id, stage in m.fs.ro.items():
        stage.A_comp[0, "H2O"].fix(water_permeability)  # m^2/s/bar, water permeability
        stage.B_comp[0, "TDS"].fix(salt_permeability)  # m/s/bar, Salt permeability
        stage.feed_side.channel_height.fix(channel_height)  # 1 mm channel height
        stage.length.fix(6)  # 6 m length of the RO module
        stage.feed_side.spacer_porosity.fix(0.85)  # 85% porosity of the spacer
        stage.width.setub(None)
        stage.area.setub(None)
        stage.mixed_permeate[0].pressure.fix(101325)  # 1 atm


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

    # Pump
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
            ].flow_mass_phase_comp.items(): # Change to 0D Reverse Osmosis
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

    # Initialize RO stages
    if ro_system == "SWRO":
        propagate_state(m.fs.pump_to_ro)
        m.fs.ro[1].initialize()
        if verbose:
            m.fs.ro[1].report()
    elif ro_system == "BWRO":
        # Initialize each RO stage sequentially
        for s_id, stage in m.fs.ro.items():
            if hasattr(m.fs, "ro_inlet_arc_" + str(s_id)):
                if verbose:
                    # print the arc
                    print(f"Initializing RO stage {s_id} with inlet arc")
                    getattr(m.fs, "ro_inlet_arc_" + str(s_id)).pprint()

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


if __name__ == "__main__":
    model = build_multi_stage_ro(n_stages=3)

    # Print all arcs to verify connections
    for arc in model.fs.component_objects(Arc, descend_into=True):
        print(f"Arc: {arc.name}, Source: {arc.source}, Destination: {arc.destination}")

