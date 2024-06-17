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

