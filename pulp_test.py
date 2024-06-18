from functions import *

# # This is my intended usage:
# PP.estimate_rankine() # But maybe here, we need to tell how much CHP is already produced?
# Vfg, fCO2 = PP.burn_fuel(technology_assumptions) #Here we should divide based on "fuel strategy"??? Do we maximize CHP or minimize fuel use? ... or no, this is just to determine the flue gases!
#  ... MEA is here sized based on Vfg etc. and returns a Qreboiler. This incurrs an energy penalty!
# Plost, Qlost = CHP.energy_penalty(MEA)  #The energy penalty is divided into a scenario tree:
#                                         #(A) Fuel use is minimized already. Then:
#                                             A1: if there is sufficient CHP cycle energy, then this is used, and any remains produce more CHP 
#                                             A2: if energy is insufficient, we cover what we can with existing CHP. The deficit is covered by:
#                                                 A21: burning more in bark boiler
#                                                 A22: buying elc and raise low grade heat using HPs

#                                         #(B) CHP power output is maximized. Then: .... but maybe this should be determined earlier, in "burn_fuel" function?

# if DH exists? Then direct MEA waste heat to the DH network!
#     Qrecovered = MEA.available_heat(composite_curve)

# Useful heat avialbe for steam production from black liquor combustion = 6.046 MWh/air-dried tonne of pulp (Svensson, 2018)


### THE CALCULATIONS THAT WILL BE PERFORMED FOR ONE PULP (AND PAPER) PLANT ###
# I need to have this fixed data for each plant:
# - Qliquor (=f(tPulp,yr?))
# - Qdemands
# - Qbark,max and Qbark,baseline (difficult? but try to find which matches operations of the Qliquor)
# - LP and steam states of Kraft and Bark boilers
# - tPulp/yr, tPaper/yr production volumes

# (1) Calculate: Qrest = Qliquor - sum(Qdemands)
# (2) Decide how bark boiler is operated in BAU: Qrest + Qbark = f(Min/MaxFuel) NOTE: if Qrest+Qbark is low, print warning to terminal! CCS is not appropriate!
# (3) Estimate the Rankine BAU cycle: 
#       - Calculate mrest and mbark = f(Qrest,Qbark,primary_states)
#       - Calculate nominal Pel and Qdh (if DH exists)
#       - Calculate an "extra emissions" factor: CO2/MWh_steam_bark (Some of this MWh (in the form of steam) may be used for the Qreb. If it is, allocate these emissions to CCS)
# (4) Calculate Vfluegas and mCO2 totals
# (5) Apply regression: Qreb, Wcc, QTcurves = f(Vfluegases,%CO2,CaptureRate)

## Now we manage Qreb:
# (6) If mrest*dh(at LP) > Qreb, then just use available excess steam to produce some power and then meet Qreb. Done!
# (6) If mrest*dh(at HP) > Qreb AND MinimizingFuelUse, then we use this available excess steam. Done! But not done if we are MaximizingCHP, then the steam is expanded to LP.

## If we are not done thus far, this means we have to use a MeritOrder of energy supply strategies, to cover Qreb:
                # OLD STUFF(7) Determine available heat from:
                #       - Process: Q60 = f(tPulp/yr, tPaper/yr, regression) and Qmax,process = Q60 at the cost of COP=2.5 ish
                #       - Capture: Q70 = f(QTcurves) and Qmax,capture = Q70 at the cost of COP. NOTE: Search for >60C here as well. Temperature lift of +60C => 120C. We need slightly more, but it's acceptable (and comparable to Svenssons regression!)
                #       - Capture=>DH: Q5X = f(existing function), now we know maximum heats that could be used!
                # (8) If MeritOrder == Steam: TODO: MAYBE CHANGE OUTER LOGIC TO If MinimizingFuelUse/MaximigingTurbineUse: then choose a MeritOrder?
                #       - We will prioritize steam. But from rest or bark, and from HP or LP? Depends on FuelStrategy:
                #       - If MinimizingFuelUse: NOTE: then we don't have extra bark capacity... but we may have some extra mrest from excess Recovery boiler heat.
                #               reduce Qreb by mrest*dh(at HP), calculate new mass flows
                #               reduce Qreb by Qmax,capture and then by Qmax,process, save the electricity penalties
                #               reduce Qreb by grid electricity (any remaining reboiler demand)
                #       - If MaximizingCHP: NOTE: all available steam is expanded as much as possible first, then supply CC... (I think this scenario should be re-labeled to: "maximizing turbine usage"! Then we don't optimize, but we prioritize some CHP stuff)
                #               reduce Qreb by mrest*dh(at LP)
                #               reduce Qreb by mbark*dh(at LP), calculate new mass flows
                #               reduce Qreb by Qmax,capture and then by Qmax,process, save the electricity penalties
                #               reduce Qreb by grid electricity (any remaining reboiler demand)

# TODO: My Levers have to connect in a logical sense to my KPIs. They are not, right now. Some should lead to increased biomassusage, e.g.. A solution: in BAU, the plant should not operate at "min/max" usage. It should have a baseline. Then, the response to CCS (i.e. the FuelStrategy) is relative to this baseline.
# TODO: I would like a function which "reduces Qreb by max, or by just the amound required to reduce it to zero". Such a function would be used many times! Or, just make it a real MACC, by adding capacities until they sum to Qreb?
# If MinimizingFuelUse:
#   If MeritOrder == Steam:
#       reduce Qreb by mrest*dh(at HP) (not by mbark, this is zero!), by Qmax_capture, by Qmax_process, by GridElectricity
#       save all power and heat penalties, any emissions from extra combustion, maybe some indicator for calculating HP costs! Was extra biomass burned due to the CCS? NO!
#   If MeritOrder == HeatPumps:
#       reduce Qreb by Qmax_capture, by Qmax_process, by Qreb by mrest*dh(at HP), by GridElectricity
#       save all power and heat penalties, any emissions from extra combustion, maybe some indicator for calculating HP costs!
#   If MeritOrder == GridElectricity:
#       this is weird, grid elec is "infinite", so steam is never used in this MeritOrder choice?

# If MaximizingTurbineUse:
#   If MeritOrder == Steam:
#       reduce Qreb by mrest*dh(at LP), by mbark*dh(at LP), by Qmax_capture, by Qmax_process, by GridElectricity. Was extra biomass burned due to the CCS? Not really... but we could "allocate" extra biomass use to the CCS?
#       save all power and heat penalties, any emissions from extra combustion, maybe some indicator for calculating HP costs!
#   If MeritOrder == HeatPumps:
#
#   If MeritOrder == GridElectricity:
#


# If DH-HP == True:
#       use HPs to recover heat for DH