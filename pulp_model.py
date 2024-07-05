"""This is what we do here."""
from functions import *

def CCS_Pulp(
    # Set Uncertainties, Levers and Constants (e.g. regressions or the initial/nominal plant values)
    eta_boiler=0.87,
    last=0,
    pulp_interpolation=None,
    PulpPlant=None
):
    technology_assumptions = {
        "eta_boiler": eta_boiler,
        "rate": rate
    }
    economic_assumptions = {
        'alpha': alpha,
        'cMEA': cMEA
    }




    return capture_cost, penalty_services, penalty_biomass







if __name__ == "__main__":
    # Load PulpPlant data
    # Load PulpAspen data
    # Construct a PulpAspenInterpolator here, which will be re-used many times.
    interpolations = 999

    # Initate a PulpPlant and send it to the model
    Plant = PulpPlant(data)
    Plant.estimate_nominal_cycle() #TODO: Where should the RDM evaluation actually begin? Before or after nominal_cycle? I think after... we consider Pulp to be "stationary", i.e. the Vfluegases are approx. constant always! Unlike bio-CHPs.
    # This is important, because the production volume (adt pulp/yr) must be considered constant, for everything to make sense. This means that Vfluegas is constant across model runs. But Rcapturerate is not! So the regression is still useful, and should be sent as a constant to the model :)

    # We will do this later, i.e. the RDM evaluation:
    # capture_cost, penalty_services, penalty_biomass = CCS_Pulp(PulpPlant = Plant, pulp_interpolation = interpolations)
    # print(capture_cost, penalty_services, penalty_biomass)