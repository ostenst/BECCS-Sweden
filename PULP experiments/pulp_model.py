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
        self.recovery_capacity = recovery_boiler
        self.bark_capacity = bark_boiler
        self.heat_demand = heat_demand
        self.electricity_demand = electricity_demand

        self.energybalance_assumptions = energybalance_assumptions
        self.technology_assumptions = {}
        self.economic_assumptions = {}

        self.states = {
            "rp":       rp,
            "rt":       rt,
            "bp":       bp,
            "bt":       bt,
            "lp":       lp,
        }
        
        self.available_steam = None
        self.m_recovery = None
        self.m_bark = None
        self.P_recovery = None
        self.P_bark = None
        self.P_demand = None

        self.gases = {}
        self.results = {}
    
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
        live_recovery   = State("recovery", p=self.states["rp"], T=self.states["rt"])
        mix_recovery    = State("-", p=cp, s=live_recovery.s, mix=True)
        if self.states["bp"] == 0:
            print("No bark boiler found, verify its capacity is zero")
            self.states["bp"] = 100
            self.states["bt"] = 500
        live_bark       = State("bark", p=self.states["bp"], T=self.states["bt"])
        mix_bark        = State("-", p=cp, s=live_bark.s, mix=True)
        self.states = {
            "boiler":           boiler,
            "live_recovery":    live_recovery,
            "mix_recovery":     mix_recovery,
            "live_bark":        live_bark,
            "mix_bark":         mix_bark,
            "lp":               self.states["lp"],
        }
        
        time = self.energybalance_assumptions["time"]
        m_recovery = available_steam*1000 * (recovery_capacity/(recovery_capacity+bark_capacity)) /time /(live_recovery.h-boiler.h) #[kg/s]
        m_bark     = available_steam*1000 * (bark_capacity    /(recovery_capacity+bark_capacity)) /time /(live_bark.h-boiler.h)     
        P_recovery = m_recovery * (live_recovery.h - mix_recovery.h) /1000 #[MW]
        P_bark     = m_bark * (live_bark.h - mix_bark.h) /1000 

        self.available_steam = available_steam 
        self.m_recovery = m_recovery  
        self.m_bark = m_bark  
        self.P_recovery = P_recovery * time  
        self.P_bark = P_bark * time 
        self.P_demand = electricity_demand 
    
    def burn_fuel(self, technology_assumptions):
        self.technology_assumptions = technology_assumptions
        b = technology_assumptions["bark_increase"]
       
        recovery_emissions = self.recovery_capacity * technology_assumptions["factor_recovery"] /1000         
        bark_emissions = self.bark_capacity * technology_assumptions["factor_bark"] /1000  
        extra_emissions = bark_emissions * b
        bark_emissions += extra_emissions  

        extra_biomass = self.bark_capacity * b #[MWh/yr]
        self.bark_capacity += extra_biomass
        self.available_steam += extra_biomass
        self.results["extra_biomass"] = extra_biomass
        self.m_bark *= (1+b)
        self.P_bark *= (1+b)

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

        self.results["Q_reboiler"] = Q_reboiler
        self.results["W_captureplant"] = W_captureplant

    def feed_then_condense(self):    
        # Re-calculate all capacities, after available steam has been reduced by Q_reboiler
        r = self.recovery_capacity
        b = self.bark_capacity
        a = ( self.available_steam - self.results["Q_reboiler"] )*1000
        if a < 0:
            print(self.name, " available steam is insufficient to cover Qreb, consider purchasing grid power")
            raise ValueError

        time = self.energybalance_assumptions["time"]
        m_recovery = a * (r/(r+b)) /time /(self.states["live_recovery"].h-self.states["boiler"].h)          #[kg/s]
        m_bark     = a * (b/(r+b)) /time /(self.states["live_bark"].h-self.states["boiler"].h)     
        P_recovery = m_recovery * (self.states["live_recovery"].h - self.states["mix_recovery"].h) /1000    #[MW]
        P_bark     = m_bark * (self.states["live_bark"].h - self.states["mix_bark"].h) /1000

        dP_recovery = self.P_recovery - P_recovery*time                                                     #[MWh/yr]
        dP_bark = self.P_bark/(1+self.technology_assumptions["bark_increase"]) - P_bark*time #NOTE: The loss in bark output is compared to the nominal case, therefore we need to adjust for the imagined bark increase
        P_lost = dP_recovery + dP_bark + self.results["W_captureplant"]

        self.results["P_lost"] = P_lost #TODO: Check this P_lost, is it correct with MWh units?
        self.available_steam = a/1000 
        self.m_recovery = m_recovery  
        self.m_bark = m_bark  
        self.P_recovery = P_recovery * time
        self.P_bark = P_bark * time 

    def expand_then_feed(self):
        # Form a merit order from available energy, and supply to meet Q_reboiler
        time = self.energybalance_assumptions["time"]
        live_recovery = self.states["live_recovery"]
        live_bark = self.states["live_bark"]
        mix_recovery = self.states["mix_recovery"]
        mix_bark = self.states["mix_bark"]

        LP_recovery = State("-", p=self.states["lp"], s=live_recovery.s, mix=True)
        Qmax_recovery = self.m_recovery * (LP_recovery.h - State("-", p=self.states["lp"], satL=True).h) #[kW]

        LP_bark = State("-", p=self.states["lp"], s=live_bark.s, mix=True)
        Qmax_bark = self.m_bark * (LP_bark.h - State("-", p=self.states["lp"], satL=True).h) #[kW]

        capacities = [Qmax_recovery/1000*time, Qmax_bark/1000*time]
        allocations, remaining_demand = self.merit_order_supply(self.results["Q_reboiler"], capacities)
        if remaining_demand != 0:
            print("... Qreb is too high, need to purchase Pgrid =", remaining_demand)

        # Now I must subtract the allocations from the capacities, and re-calculate the mass => power production
        m_recovery_utilized = allocations[0]/time*1000 / (LP_recovery.h - State("-", p=self.states["lp"], satL=True).h) #[kg/s]
        m_bark_utilized =     allocations[1]/time*1000 / (LP_bark.h     - State("-", p=self.states["lp"], satL=True).h) 

        P_recovery =  self.m_recovery * (live_recovery.h - LP_recovery.h) /1000                         #[MW] All mass expands to LP level
        P_recovery += (self.m_recovery - m_recovery_utilized) * (LP_recovery.h - mix_recovery.h) /1000  #[MW] Some mass expands to condensing level

        P_bark =  self.m_bark * (live_bark.h - LP_bark.h) /1000
        P_bark +=  (self.m_bark - m_bark_utilized) * (LP_bark.h - mix_bark.h) /1000

        dP_recovery = self.P_recovery - P_recovery*time                                                 #[MWh/yr]
        dP_bark = self.P_bark/(1+self.technology_assumptions["bark_increase"]) - P_bark*time
        P_lost = dP_recovery + dP_bark + self.results["W_captureplant"] + remaining_demand              # Any remaining demand needs purchased grid electricity
        
        self.results["P_lost"] = P_lost
        self.m_recovery -= m_recovery_utilized
        self.m_bark -= m_bark_utilized 
        self.P_recovery = P_recovery * time  
        self.P_bark = P_bark * time 

    def recover_and_supplement(self):
        # Recover excess heat using pumps, supply residual demand with LP steam
        time = self.energybalance_assumptions["time"]
        live_recovery = self.states["live_recovery"]
        live_bark = self.states["live_bark"]
        mix_recovery = self.states["mix_recovery"]
        mix_bark = self.states["mix_bark"]
        LP_recovery = State("-", p=self.states["lp"], s=live_recovery.s, mix=True)

        Q_60C = (-29.998 + 1.248*self.pulp_capacity/1000 + 0.879*0 )*1000 #[MWh/yr]
        P_HP = Q_60C/self.technology_assumptions["COP"]
        remaining_demand = self.results["Q_reboiler"] - Q_60C

        # Assume the remaining demand can be covered by recovery boiler LP steam
        m_recovery_utilized = remaining_demand/time*1000 / (LP_recovery.h - State("-", p=self.states["lp"], satL=True).h)   #[kg/s]
        if m_recovery_utilized > self.m_recovery:
            raise ValueError
        P_recovery =  self.m_recovery * (live_recovery.h - LP_recovery.h) /1000                                             #[MW] All mass expands to LP level
        P_recovery += (self.m_recovery - m_recovery_utilized) * (LP_recovery.h - mix_recovery.h) /1000                      #[MW] Some mass expands to condensing level

        dP_recovery = self.P_recovery - P_recovery*time                                                                     #[MWh/yr]
        dP_bark = self.P_bark/(1+self.technology_assumptions["bark_increase"]) - self.P_bark                                #Probably negative, since more power is available from bark
        P_lost = dP_recovery + dP_bark + self.results["W_captureplant"] + P_HP
        
        self.results["P_lost"] = P_lost
        self.m_recovery -= m_recovery_utilized 
        self.P_recovery = P_recovery * time

    def merit_order_supply(self, Q_reboiler, capacities):
        remaining_demand = Q_reboiler
        allocations = []

        for capacity in capacities:
            if remaining_demand > 0:
                if remaining_demand >= capacity:
                    allocations.append(capacity)
                    remaining_demand -= capacity
                else:
                    allocations.append(remaining_demand)
                    remaining_demand = 0
            else:
                allocations.append(0)

        return allocations, remaining_demand
    
    def print_energybalance(self):
        print(f"\n{'Recovery Capacity:':<20} {self.recovery_capacity}")
        print(f"{'Bark Capacity:':<20} {self.bark_capacity}")
        print(f"{'Heat Demand:':<20} {self.heat_demand}")
        print(f"{'Electricity Demand:':<20} {self.electricity_demand}")

        print(f"{'Available Steam:':<20} {self.available_steam} MWh/yr")
        print(f"{'Mass Flow Recovery:':<20} {self.m_recovery} kg/s")
        print(f"{'Mass Flow Bark:':<20} {self.m_bark} kg/s")
        print(f"{'Power Recovery:':<20} {self.P_recovery} MWh/yr")
        print(f"{'Power Bark:':<20} {self.P_bark} MWh/yr")
        print(f"{'Power Demand:':<20} {self.P_demand} MWh/yr")


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
    COP = 3,
    SupplyStrategy = "SteamHP",
    pulp_interpolation=None,
    PulpPlant=None
):
    technology_assumptions = {
        "bark_increase": int(BarkIncrease)/100,
        "factor_recovery": factor_recovery,
        "factor_bark": factor_bark,
        "fluegas_intensity": fluegas_intensity,
        "capture_rate": rate,
        "COP": COP
    }
    # economic_assumptions = {
    #     'alpha': alpha,
    #     'cMEA': cMEA
    # }¨

    PulpPlant.print_energybalance()
    PulpPlant.burn_fuel(technology_assumptions)
    PulpPlant.print_energybalance()
    PulpPlant.size_MEA(rate, pulp_interpolation)
    
    if SupplyStrategy == "SteamHP":
        PulpPlant.feed_then_condense()
        PulpPlant.print_energybalance()
        print(PulpPlant.results)
        
    elif SupplyStrategy == "SteamLP":
        PulpPlant.expand_then_feed()
        PulpPlant.print_energybalance()
        print(PulpPlant.results)

    elif SupplyStrategy == "HeatPumps":
        PulpPlant.recover_and_supplement()
        PulpPlant.print_energybalance()
        print(PulpPlant.results)


    capture_cost, penalty_services, penalty_biomass = [1,2,3]

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