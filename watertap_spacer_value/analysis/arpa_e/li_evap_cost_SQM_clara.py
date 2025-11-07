# SQM's 2020 numbers
inflation_factor = 1.23 # USD_2020 to USD_2025

sqm_operating_total = 500000000 * inflation_factor # USD/year
sqm_capital_watertap_factor = (27+17+13+7+(27+17+13+7)/(28+27+17+13+7)*8)/100  # fraction of SQM's capital cost covered by WaterTAP
sqm_operating_watertap_factor = sqm_capital_watertap_factor*(.18+.14+.12+.04)+(27+17+13+7)/(28+27+17+13+7)*.25+.14+.05+.06  # fraction of SQM's operating cost covered by WaterTAP
sqm_operating_watertap = sqm_operating_total * sqm_operating_watertap_factor  # SQM's operating cost covered by WaterTAP. This includes depreciation (capital costs).

sqm_revenue_solids = 209300000 * inflation_factor # USD/year

utilization_factor = 0.98  # factor

# Q calculation
inlet_li_conc = 2  # g/kg
li_recovery = 0.6  # fraction
inlet_flow_vol = 1461  # m^3/s (2020)
li_outflow = inlet_li_conc * li_recovery * inlet_flow_vol / 1000  # kg/s
Q = li_outflow * 3600 * 24 * 365 / 1000  # mt/year

# Results
LCOLi_sqm_total = (sqm_operating_total - sqm_revenue_solids) / (utilization_factor * Q)
LCOLi_sqm_watertap = (sqm_operating_watertap - sqm_revenue_solids) / (utilization_factor * Q)  # USD/mt Li

print(f"LCOLi_sqm_total: {LCOLi_sqm_total:.2f} $/mt Li")
print(f"LCOLi_sqm_watertap: {LCOLi_sqm_watertap:.2f} $/mt Li")
print(f"LCOLi_sqm_processing: {(LCOLi_sqm_total-LCOLi_sqm_watertap):.2f} $/mt Li")
