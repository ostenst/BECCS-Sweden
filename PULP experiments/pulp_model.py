"""This is what we do here."""

class State:
    def __init__(self, Name, p=None, T=None, s=None, satL=False, satV=False, mix=False):
        self.Name = Name

class PulpPlant():
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity
        print("Initiating plant")
    
    def estimate_nominal_cycle(self):
        print("Estimating the Rankine Cycle")

class MEA():
    def __init__(self, Name):
        self.Name = Name

# ------------ ABOVE THIS LINE WE DEFINE ALL CLASSES AND FUNCTIONS NEEDED FOR THE CCS_Pulp() MODEL --------

def CCS_Pulp(
    # Set Uncertainties, Levers and Constants (e.g. regressions or the initial/nominal plant values)
    bark_usage = 130,
    t = 25,
    EnergySupply = "Steam",
    pulp_interpolation=None,
    PulpPlant=None
):
    # technology_assumptions = {
    #     "eta_boiler": eta_boiler,
    #     "rate": rate
    # }
    # economic_assumptions = {
    #     'alpha': alpha,
    #     'cMEA': cMEA
    # }¨
    print(f"Testing {PulpPlant.name} for bark usage={bark_usage}, t={t} and supply={EnergySupply}")

    if EnergySupply == "Steam":
        capture_cost = bark_usage
        penalty_services = 140-bark_usage
        penalty_biomass = bark_usage-100

    elif EnergySupply == "HeatPumps":
        capture_cost = 40
        penalty_services = 100
        penalty_biomass = t

    return capture_cost, penalty_services, penalty_biomass

if __name__ == "__main__":
    # Load PulpPlant data
    # Load PulpAspen data
    # Construct a PulpAspenInterpolator here, which will be re-used many times.
    interpolations = ["Interp1", "Interp2"]

    # Initate a PulpPlant and calculate its nominal energy balance and volume flow.
    Plant = PulpPlant("Varo", 900000)
    Plant.estimate_nominal_cycle() 

    # The RDM evaluation starts below:
    capture_cost, penalty_services, penalty_biomass = CCS_Pulp(PulpPlant = Plant, pulp_interpolation = interpolations)
    print("Outcomes: ", capture_cost, penalty_services, penalty_biomass)