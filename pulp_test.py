from functions import *
## The idea for heat pumps is this:
# Tharun suggests a temperature lift of +60C is possible with HPs. I have his reboiler with its pressure and temp. https://www.sciencedirect.com/science/article/pii/S030626192300291X?via%3Dihub
# Electrification of amine is possible using HPs. If waste heat is around 60-70C and we have same pressure^. https://www.sciencedirect.com/science/article/pii/S2772656823000581?via%3Dihub#bib0028
# There is waste heat in Pulp plants at above 60C, which Elin found through regression. https://research.chalmers.se/publication/508644 
# Combining these, we can make the argument that pulp plants can recover heat around 60C for either a reboiler or a DH network! Using HPs. NOTE: If I do this, I NEED to account for signifciantly higher CAPEX.

# Useful heat avialbe for steam production from black liquor combustion = 6.046 MWh/air-dried tonne of pulp (Svensson, 2018)
# CEPCI source from JUdits MEX thesis: https://toweringskills.com/financial-analysis/cost-indices/

### THE CALCULATIONS THAT WILL BE PERFORMED FOR ONE PULP (AND PAPER) PLANT ###
# I need to have this fixed data for each plant:
# - Qliquor (=f(tPulp,yr?))
# - Qdemands
# - Qbark,max and Qbark,baseline (difficult? but try to find which matches operations of the Qliquor)
# - LP and steam states of Kraft and Bark boilers
# - tPulp/yr, tPaper/yr production volumes
# - DH connection possible?

# (1) Calculate: Qrest = Qliquor - sum(Qdemands)
# (2) Decide how bark boiler is operated in BAU: Qrest + Qbark = f(Qbark,baseline, maybe compensate -Qrest?) NOTE: if Qrest+Qbark is low, print warning to terminal! CCS is not appropriate!
# (3) Estimate the Rankine BAU cycle: 
#       - Calculate mrest and mbark = f(Qrest,Qbark,primary_states,LP) NOTE: mbark is just "excess steam", that isn't tied up to compensate for -Qrest
#       - Calculate nominal Pel and Qdh (if DH exists)
#       - Calculate an "extra emissions" factor: CO2/MWh_steam_bark (Some of this MWh (in the form of steam) may be used for the Qreb. If it is, allocate these emissions to CCS)
# (4) Calculate Vfluegas and mCO2 totals
# (5) Apply regression: Qreb, Wcc, QTcurves = f(Vfluegases,%CO2,CaptureRate)
# (OLD 6) Determine available heat from:
#       - Process: Q60 = f(tPulp/yr, tPaper/yr, regression) and Qmax,process = Q60 at the cost of COP=2.5 ish
#       - OLD, IGNORE THIS: Capture: Q70 = f(QTcurves) and Qmax,capture = Q70 at the cost of COP. NOTE: Search for >60C here as well. Temperature lift of +60C => 120C. We need slightly more, but it's acceptable (and comparable to Svenssons regression!)
#       - OLD, IGNORE THIS: Capture=>DH: Q5X = f(existing function), now we know maximum heats that could be used!

# (6) Heat now needs to be supplied to Qreb and the nominal Rankine will be affected. This is done using a SupplyStrategy:
# If HP steam:
#       just subtract Qreb from AvailableSteam, then condense all the rest!
# If LP steam:
#       calculate available energy from Kraft and Bark at LP, we will then use this merit order:
#       if Qreb =< KraftLP:
#           just use Kraft LP steam, and subtract this massflow from Kraft condensing steam. Bark is unaffected. 
#       elif Qreb > Kraft+Bark:
#           use all Kraft and Bark LP steam, and supply remaining with Welc,grid NOTE: I think this never happens, maybe unneccesary case???
#       else: we have a medium case where we need some extra BarkLP
#           use all Kraft LP steam, so the Kraft condensing steam is zero. Use some BarkLP by subtracting required amount from mass of Bark, then condense the rest.
# if HeatPumps:
#       estimate available_excess_heat, and subtract it from Qreb. Anything remains? Assume KraftLP steam is sufficient (if not, print a warning message). Condense remaining LP steam.

# In each case, calculate penalties on net power! And heat input etc. Then we are done :)





# (OLD 7) Now we manage Qreb by assuming an EnergyPriority = MaximizeTurbineUse (always run steam through turbines) or MinimizeFuelUse (avoid using additional biomass) NOTE: We chose this deliberately to reveal tensions between KPI2 and KPI3.
# If MinimizingFuelUse:
#   If MeritOrder == Steam:
#       reduce Qreb by mrest*dh(at HP), by mbark*dh(at HP), by Qmax_capture, by Qmax_process, by GridElectricity
#       save all power and heat penalties, any emissions from extra combustion, maybe some indicator for calculating HP costs! Was extra biomass burned due to the CCS? No!
#   If MeritOrder == HeatPumps:
#       reduce Qreb by Qmax_capture, by Qmax_process, by mrest*dh(at HP), by mbark*dh(at HP), by GridElectricity
#       save all power and heat penalties, any emissions from extra combustion, maybe some indicator for calculating HP costs! Was extra biomass burned due to the CCS? No!
##   If MeritOrder == GridElectricity:
##       this is weird, grid elec is "infinite", so steam is never used in this MeritOrder choice?

# If MaximizingTurbineUse:
#   If MeritOrder == Steam:
#       reduce Qreb by mrest*dh(at LP), by mbark*dh(at LP), by mbark_max*dh(at LP), by Qmax_capture, by Qmax_process, by GridElectricity. 
#       save all power and heat penalties, any emissions from extra combustion, maybe some indicator for calculating HP costs! Was extra biomass burned due to the CCS? Yes!
#   If MeritOrder == HeatPumps:
#       reduce Qreb by Qmax_capture, by Qmax_process, by mrest*dh(at LP), by mbark*dh(at LP), by mbark_max*dh(at LP), by GridElectricity
#       save all power and heat penalties, any emissions from extra combustion, maybe some indicator for calculating HP costs! Was extra biomass burned due to the CCS? Yes!
##   If MeritOrder == GridElectricity:

# (8) If MaximizingDHrecovery == True:, we will use HEXs and HPs
#       Check if CaptureHeat>60C was used for Qreb. Use what is possible for DH by heat exchange.
#       Also add some extra heat, i.e. the last low temp. heat, by heat pumping to +60C. Maybe add a cost to this?
# (9) Calculate final energy balances (e.g. subtracting Wcc, Pheatpumps, biomass used etc.), and emissions for KPI1,2,3.