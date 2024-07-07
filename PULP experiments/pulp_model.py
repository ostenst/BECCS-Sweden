"""This is what we do here."""
import pandas as pd
from pyXSteam.XSteam import XSteam

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
class State:
    def __init__(self, Name, p=None, T=None, s=None, satL=False, satV=False, mix=False):
        self.Name = Name
        if satL==False and satV==False and mix==False:
            self.p = p
            self.T = T
            self.s = steamTable.s_pt(p,T)
            self.h = steamTable.h_pt(p,T)
        if satL==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = steamTable.sL_p(p)
            self.h = steamTable.hL_p(p) 
        if satV==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = steamTable.sV_p(p)
            self.h = steamTable.hV_p(p)
        if mix==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = s
            self.h = steamTable.h_ps(p,s)
        if self.p is None or self.T is None or self.s is None or self.h is None:
            raise ValueError("Steam properties cannot be determined")

class PulpPlant:
    def __init__(self, name, pulp_capacity, bark_share, recovery_boiler, bark_boiler, heat_demand, electricity_demand, rp, rt, bp, bt, lp, energybalance_assumptions=None):
        self.name = name
        self.pulp_capacity = pulp_capacity
        self.bark_share = bark_share
        self.recovery_boiler = recovery_boiler
        self.bark_boiler = bark_boiler
        self.heat_demand = heat_demand
        self.electricity_demand = electricity_demand
        self.rp = rp
        self.rt = rt
        self.bp = bp
        self.bt = bt
        self.lp = lp

        self.energybalance_assumptions = energybalance_assumptions
        self.technology_assumptions = None
        self.economic_assumptions = None

        self.states = None
        self.energybalance = None
        self.gases = None
    
    def estimate_nominal_cycle(self):

        # Calculating available steam
        recovery_capacity = self.energybalance_assumptions["recovery_intensity"] /3.6 * self.pulp_capacity  #[MWh/yr]
        bark_capacity = recovery_capacity * self.bark_share
        heat_demand = self.energybalance_assumptions["heat_intensity"] /3.6 * self.pulp_capacity          
        available_steam = (recovery_capacity + bark_capacity) - heat_demand
        electricity_demand = self.energybalance_assumptions["electricity_intensity"] * self.pulp_capacity

        # Calculating nominal mass flows and energy balance
        cp = self.energybalance_assumptions["condensing_pressure"]
        boiler          = State("boiler", p=cp, satL=True)
        live_recovery   = State("recovery", p=self.rp, T=self.rt)
        mix_recovery    = State("-", p=cp, s=live_recovery.s, mix=True)
        if self.bp == 0:
            print("No bark boiler found, verify its capacity is zero")
            self.bp = 100
            self.bt = 500
        live_bark       = State("bark", p=self.bp, T=self.bt)
        mix_bark        = State("-", p=cp, s=live_bark.s, mix=True)
        self.states = {
            "boiler":           boiler,
            "live_recovery":    live_recovery,
            "mix_recovery":     mix_recovery,
            "live_bark":        live_bark,
            "mix_bark":         mix_bark,
        }
        
        time = self.energybalance_assumptions["time"]
        m_recovery = available_steam*1000 * (recovery_capacity/(recovery_capacity+bark_capacity)) /time /(live_recovery.h-boiler.h) #[kg/s]
        m_bark     = available_steam*1000 * (bark_capacity    /(recovery_capacity+bark_capacity)) /time /(live_bark.h-boiler.h)     
        P_recovery = m_recovery * (live_recovery.h - mix_recovery.h) /1000 #[MW]
        P_bark     = m_bark * (live_bark.h - mix_bark.h) /1000 
        self.energybalance = {
            "recovery_capacity": recovery_capacity, #[MWh/yr]
            "bark_capacity": bark_capacity,         #[MWh/yr]
            "available_steam": available_steam,     #[MWh/yr] NOTE: if needed later, you can save other capacities in this dict! And don't calculate biomass GWh here, do it later within RDM analysis.
            "m_recovery":   m_recovery,             #[kg/s]
            "m_bark":       m_bark,                 #[kg/s]
            "P_recovery":   P_recovery*time,        #[MWh/yr]   
            "P_bark":       P_bark*time,            #[MWh/yr]
            "P_demand":     electricity_demand      #[MWh/yr]
        }
    
    def burn_fuel(self, technology_assumptions):

        b = technology_assumptions["bark_increase"]
       
        recovery_emissions = self.energybalance["recovery_capacity"] * technology_assumptions["factor_recovery"] /1000         
        bark_emissions = self.energybalance["bark_capacity"] * technology_assumptions["factor_bark"] /1000  
        extra_emissions = bark_emissions * b
        bark_emissions += extra_emissions  

        extra_biomass = self.energybalance["bark_capacity"] * b #[MWh/yr]
        self.energybalance["bark_capacity"] += extra_biomass
        self.energybalance["available_steam"] += extra_biomass
        self.energybalance["extra_biomass"] = extra_biomass
        self.energybalance["m_bark"] *= (1+b)
        self.energybalance["P_bark"] *= (1+b)

        V_fluegas = self.pulp_capacity * technology_assumptions["fluegas_intensity"] /self.energybalance_assumptions["time"] /3600 

        self.gases = {
            "recovery_emissions": recovery_emissions, #[ktCO2/yr]
            "bark_emissions": bark_emissions,         #[ktCO2/yr]
            "extra_emissions": extra_emissions,       #[ktCO2/yr]
            "V_fluegas": V_fluegas                    #[kg/s]
        }

    def size_MEA(self, rate, pulp_interpolation):
        
        print("Making a simplified MEA calculation for now, but an Aspen interpolator is required!")
        Q_reboiler = 3.6 * (self.gases["recovery_emissions"]*1000 * rate) /3.6 #[MWh/yr]
        W_captureplant = 0.15*Q_reboiler

        self.energybalance["Q_reboiler"] = Q_reboiler
        self.energybalance["W_captureplant"] = W_captureplant

    def feed_then_condense(self):
        
        # Re-calculate all capacities, after available steam has been reduced by Q_reboiler
        recovery_capacity = self.energybalance["recovery_capacity"]
        bark_capacity = self.energybalance["bark_capacity"]
        available_steam = self.energybalance["available_steam"] - self.energybalance["Q_reboiler"]

        time = self.energybalance_assumptions["time"]
        m_recovery = available_steam*1000 * (recovery_capacity/(recovery_capacity+bark_capacity)) /time /(self.states["live_recovery"].h-self.states["boiler"].h) #[kg/s]
        m_bark     = available_steam*1000 * (bark_capacity    /(recovery_capacity+bark_capacity)) /time /(self.states["live_bark"].h-self.states["boiler"].h)     
        P_recovery = m_recovery * (self.states["live_recovery"].h - self.states["mix_recovery"].h)
        P_bark     = m_bark * (self.states["live_bark"].h - self.states["mix_bark"].h) 

        dP_recovery = self.energybalance["P_recovery"] - P_recovery
        dP_bark = self.energybalance["P_bark"] - P_bark

        P_lost = dP_recovery + dP_bark + self.energybalance["W_captureplant"]
        print("Plost is too large? Somethings is wrong...=", P_lost)
        self.energybalance["P_lost"] = P_lost

    def __repr__(self):
        return (f"PulpPlant(Name={self.name}, Pulp Capacity={self.pulp_capacity}, Bark Capacity={self.bark_share}, "
                f"Recovery Boiler={self.recovery_boiler}, Bark Boiler={self.bark_boiler}, Heat Demand={self.heat_demand}, "
                f"Electricity Demand={self.electricity_demand}, RP={self.rp}, RT={self.rt}, BP={self.bp}, BT={self.bt}, LP={self.lp})")


class MEA():
    def __init__(self, Name):
        self.Name = Name

# ------------ ABOVE THIS LINE WE DEFINE ALL CLASSES AND FUNCTIONS NEEDED FOR THE CCS_Pulp() MODEL --------

def CCS_Pulp(
        
    # Set Uncertainties, Levers and Constants (e.g. regressions or the initial/nominal plant values)
    BarkIncrease = "40",            #[%increase]
    factor_recovery = 0.4106,       #[tCO2/MWh]
    factor_bark = 0.322285714,
    fluegas_intensity = 10188.75,   #[kg/t]
    rate = 0.90,
    SupplyStrategy = "SteamHP",
    pulp_interpolation=None,
    PulpPlant=None
):
    technology_assumptions = {
        "bark_increase": int(BarkIncrease)/100,
        "factor_recovery": factor_recovery,
        "factor_bark": factor_bark,
        "fluegas_intensity": fluegas_intensity,
        "capture_rate": rate
    }
    # economic_assumptions = {
    #     'alpha': alpha,
    #     'cMEA': cMEA
    # }¨


    PulpPlant.burn_fuel(technology_assumptions)

    PulpPlant.size_MEA(rate, pulp_interpolation)
    print(PulpPlant.energybalance)

    if SupplyStrategy == "SteamHP":
        PulpPlant.feed_then_condense()

    elif SupplyStrategy == "SteamLP":
        PulpPlant.expand_then_feed()

    elif SupplyStrategy == "HeatPumps":
        PulpPlant.recover_and_supplement()


    capture_cost, penalty_services, penalty_biomass = [1,2,3]

    print(PulpPlant.energybalance)
    print(PulpPlant.gases)
    return capture_cost, penalty_services, penalty_biomass

if __name__ == "__main__":

    plants_df = pd.read_csv("Pulp data.csv",delimiter=";")
    plant_data = plants_df.iloc[0]

    # Load PulpAspen data
    # Construct a PulpAspenInterpolator here, which will be re-used many times.
    interpolations = ["Interp1", "Interp2"]

    # Initate a PulpPlant and calculate its nominal energy balance and volume flow.
    energybalance_assumptions = {
        "recovery_intensity": 18,       #[GJ/t]
        "bark_intensity": 4.2,          #[GJ/t] Although I don't think this will be needed! Since we estimate bark as a fraction of recovery.
        "heat_intensity": 11,           #[GJ/t]
        "electricity_intensity": 0.7,   #[MWh/t]
        "condensing_pressure": 0.1,     #[bar]
        "time": 8000                    #[h/yr]
    }

    pulp_plant = PulpPlant(
        name=plant_data['Name'],
        pulp_capacity=plant_data['Pulp capacity'],
        bark_share=plant_data['Bark capacity'],
        recovery_boiler=plant_data['Recovery boiler'],
        bark_boiler=plant_data['Bark boiler'],
        heat_demand=plant_data['Heat demand'],
        electricity_demand=plant_data['Electricity demand'],
        rp=plant_data['RP'],
        rt=plant_data['RT'],
        bp=plant_data['BP'],
        bt=plant_data['BT'],
        lp=plant_data['LP'],
        energybalance_assumptions=energybalance_assumptions
    )

    pulp_plant.estimate_nominal_cycle() 

    # The RDM evaluation starts below:
    capture_cost, penalty_services, penalty_biomass = CCS_Pulp(PulpPlant = pulp_plant, pulp_interpolation = interpolations)
    print("Outcomes: ", capture_cost, penalty_services, penalty_biomass)