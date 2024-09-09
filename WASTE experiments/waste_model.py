"""This is what we do here."""
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
from scipy.interpolate import LinearNDInterpolator

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
        
def create_interpolators(aspen_df):
    # Extract 'Flow' and 'Rcapture' columns as x values, the rest are y values
    x1 = aspen_df['Flow']
    x2 = aspen_df['Rcapture']
    x_values = np.column_stack((x1, x2))

    y_values = aspen_df.drop(columns=['Flow', 'Rcapture']).values  
    aspen_interpolators = {}

    # Create interpolation function for each y column
    for idx, column_name in enumerate(aspen_df.drop(columns=['Flow', 'Rcapture']).columns):
        y = y_values[:, idx]
        interp_func = LinearNDInterpolator(x_values, y)
        aspen_interpolators[column_name] = interp_func

    return aspen_interpolators

class W2E_plant:
    def __init__(self, name, fuel, Qdh, P, Qfgc, ybirth, Tsteam, psteam, energybalance_assumptions):
        self.name = name
        self.fuel = fuel
        self.Qfuel = None
        self.Qdh = float(Qdh)
        self.P = float(P)
        self.Qfgc = float(Qfgc)
        self.ybirth = int(ybirth)
        self.Tsteam = float(Tsteam)
        self.psteam = float(psteam)
        self.msteam = None

        self.energybalance_assumptions = energybalance_assumptions
        self.technology_assumptions = {}
        self.economic_assumptions = {}
        self.states = {
            "rp":       psteam,
            "rt":       Tsteam,
            "lp":       None,
        }
        self.aspen_data = None

        self.gases = {}
        self.results = {}
        self.nominal_state = {}
    
    def get(self, parameter):
        return np.round( self.aspen_data[parameter].values[0] )
    
    def estimate_nominal_cycle(self):
        # Estimating condensation pressure and steam states
        live = State("live", self.psteam, self.Tsteam)

        Ptarget = self.P
        max_iterations = 10000
        pcond_guess = self.psteam
        Pestimated = 0
        i = 0
        tol = 0.05
        while abs(Pestimated - Ptarget) > Ptarget*tol and i < max_iterations:
            pcond_guess = pcond_guess - 0.1
            mix = State("mix", p=pcond_guess, s=live.s, mix=True)
            boiler = State("boiler", pcond_guess, satL=True)
            msteam = self.Qdh/(mix.h-boiler.h)
            Pestimated = msteam*(live.h-mix.h)
            i += 1
        if i == max_iterations:
            raise ValueError("Couldn't estimate Rankine cycle!")

        # Calculating energy balance
        self.Qfuel = msteam*(live.h-boiler.h)
        self.P = Pestimated
        self.msteam = msteam
        self.states = {
            "boiler":   boiler,
            "mix":      mix,
            "live":     live
        }

        # This is a dict with the attribute values of the nominal state. But we have to exclude the initial 'nominal_state' from being copied!
        self.nominal_state = copy.deepcopy({k: v for k, v in self.__dict__.items() if k != 'nominal_state'})
        
        if msteam is not None and Pestimated is not None and self.Qfuel > 0 and pcond_guess > 0:
            return
        else:
            raise ValueError("One or more of the variables (msteam, Pestimated, Qfuel, pcond_guess) is not positive.")
    
    def burn_fuel(self, technology_assumptions):
        self.technology_assumptions = technology_assumptions

        # 362 MWfuel produces 172 kgfluegas/s in Tharun's study
        m_fluegas = 174/362 * self.Qfuel                               #[kg/s]
        n_fluegas = m_fluegas / technology_assumptions["molar_mass"]   #[kmol/s] Using mean molar mass [kg/kmol] 
        V_fluegas = n_fluegas * 22.414                                 #[Nm3/s]  Using molar volume of ideal gas

        # Now calculating CO2 from %CO2
        V_CO2 = V_fluegas*0.16  #[m3CO2/s]
        n_CO2 = V_CO2/22.414    #[kmolCO2/s]
        m_CO2 = n_CO2*44        #[kg/s]

        duration = technology_assumptions["time"]
        duration_increase = technology_assumptions["duration_increase"]

        self.results["Qextra"] = self.Qfuel*duration_increase #[MWh/yr]
        self.gases = {
            "nominal_emissions": m_CO2*3600/1000 *(duration) /1000,
            "boiler_emissions": m_CO2*3600/1000 *(duration+duration_increase) /1000,                                    #[ktCO2/yr]
            "captured_emissions": m_CO2*3600/1000 *(duration+duration_increase) /1000*technology_assumptions["rate"],   
            "m_fluegas": m_fluegas,                                                                                     #[kg/s]
            "V_fluegas": V_fluegas,                                                                                     #[Nm3/s]
        }
        
    def size_MEA(self, rate, chp_interpolators):        
        new_Flow = self.gases["m_fluegas"]  #[kg/s]
        if new_Flow < 3:                    #The CHP interpolators only work between 3 kg/s / above 170kgs
            new_Flow = 3
        if new_Flow > 170:
            new_Flow = 170

        new_Rcapture = rate*100
        new_y_values = {}

        for column_name, interp_func in chp_interpolators.items():
            new_y = interp_func(([new_Flow], [new_Rcapture]))
            new_y_values[column_name] = new_y

        new_data = pd.DataFrame({
            'Flow': [new_Flow],
            'Rcapture': [new_Rcapture],
            **new_y_values  # Unpack new y values dictionary
        })

        self.aspen_data = new_data
        self.results["Q_reboiler"] = self.get("Qreb")/1000 #[MW]
        return

    def power_MEA(self):
        boiler = self.states["boiler"]
        mix = self.states["mix"]
        live = self.states["live"]
        dTreb = self.technology_assumptions["dTreb"]

        # Find the reboiler states [a,d] and calculate required mass m,CCS 
        mtot = self.Qfuel*1000 / (live.h-boiler.h) 
        TCCS = self.get("Treb") + dTreb
        pCCS = steamTable.psat_t(TCCS)

        a = State("a",pCCS,s=live.s,mix=True) 
        d = State("d",pCCS,satL=True)
        mCCS = self.get("Qreb") / (a.h-d.h)
        mB = mtot-mCCS

        W = 0
        for Wi in ["Wpumps","Wcfg","Wc1","Wc2","Wc3","Wrefr1","Wrefr2","Wrecomp"]:
            W += self.get(Wi)

        # The new power output depends on the pressures of p,mix and p,CCS
        if a.p > mix.p: 
            Pnew = mtot*(live.h-a.h) + mB*(a.h-mix.h) - W
        else: 
            Pnew = mtot*(live.h-mix.h) + mCCS*(mix.h-a.h) - W

        Plost = (mtot*(live.h-mix.h) - Pnew)
        Qnew = mB*(mix.h-boiler.h)
        Qlost = (mtot*(mix.h-boiler.h) - Qnew)

        self.P = Pnew/1000
        self.Qdh = Qnew/1000
        self.reboiler_steam = [a,d]
        self.results["W_captureplant"] = W/1000 
        self.results["Plost"] = Plost/1000
        self.results["Qlost"] = Qlost/1000
        return 
    
    def select_streams(self, consider_dcc=False):
        considered_streams = ['wash', 'strip', 'lean', 'int2', 'int1', 'dhx', 'dry', 'rcond', 'rint', 'preliq'] # For CHPs
        if consider_dcc: # For industrial (pulp) cases
            considered_streams.append('dcc') 

        stream_data = {}
        for component in considered_streams:
            stream_data[component] = {
                'Q': -self.get(f"Q{component}"),
                'Tin': self.get(f"Tin{component}")-273.15,
                'Tout': self.get(f"Tout{component}")-273.15
            }
        return stream_data
    
    def find_ranges(self, stream_data):
        temperatures = []
        for component, data in stream_data.items():
            temperatures.extend([data['Tin'], data['Tout']])

        unique_temperatures = list(dict.fromkeys(temperatures)) 
        unique_temperatures.sort(reverse=True)

        temperature_ranges = []
        for i in range(len(unique_temperatures) - 1):
            temperature_range = (unique_temperatures[i + 1], unique_temperatures[i])
            temperature_ranges.append(temperature_range)

        return temperature_ranges

    def merge_heat(self, stream_data):
        temperature_ranges = self.find_ranges(stream_data)

        composite_curve = [[0, temperature_ranges[0][1]]] # First data point has 0 heat and the highest temperature
        Qranges = []
        for temperature_range in temperature_ranges:
            Ctot = 0
            for component, data in stream_data.items():
                TIN = data['Tin']
                TOUT = data['Tout']
                if TIN == TOUT:
                    TIN += 0.001 # To avoid division by zero
                Q = data['Q']
                C = Q/(TIN-TOUT)
                
                if TIN >= temperature_range[1] and TOUT <= temperature_range[0]:
                    Ctot += C

            Qrange = Ctot*(temperature_range[1]-temperature_range[0])
            Qranges.append(Qrange)
            composite_curve.append([sum(Qranges), temperature_range[0]])
        self.composite_curve = composite_curve
        return composite_curve
    
    def recover_heat(self, composite_curve):
        Tsupp = self.technology_assumptions["Tsupp"]
        Tlow = self.technology_assumptions["Tlow"]
        dTmin = self.technology_assumptions["dTmin"]
        
        shifted_curve = [[point[0], point[1] - dTmin] for point in composite_curve]
        curve = shifted_curve
        # Find the elbow point (point of maximum curvature) from the distance of each point
        def distance(p1, p2, p):
            x1, y1 = p1
            x2, y2 = p2
            x0, y0 = p
            return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances = [distance(curve[0], curve[-1], point) for point in curve]
        differences = np.diff(distances)
        max_curvature_index = np.argmax(differences) + 1

        # Finding low and high points. NOTE I think this is buggy for unfeasible, kinked composite curves
        def linear_interpolation(curve, ynew):
            # Find the nearest points, then perform inverse linear interpolation
            y_values = [point[1] for point in curve]    
            nearest_index = min(range(len(y_values)), key=lambda i: abs(y_values[i] - ynew))
            x1, y1 = curve[nearest_index]
            if nearest_index == 0:
                x2, y2 = curve[1]
            elif nearest_index == len(curve) - 1:
                x2, y2 = curve[-2]
            else:
                x2, y2 = curve[nearest_index + 1]

            if y2 == y1:
                return x1 + (x2 - x1) * (ynew - y1) / (y2 - y1)
            else:
                return x1 + (x2 - x1) * (ynew - y1) / (y2 - y1)
        
        Qsupp = linear_interpolation(curve, 86) # BUG: Hard-coded Tsupp=86C since higher temp. heat won't be available
        Qlow = linear_interpolation(curve, Tlow)
        Qpinch, Tpinch = curve[max_curvature_index][0], curve[max_curvature_index][1]
        if Qlow < Qpinch:
            Qlow = Qpinch # Sometimes Qlow is poorly estimated, then just set the low grade heat to zero
        Qhex = (Qpinch-Qsupp) + (Qlow-Qpinch)

        # Qhp = composite_curve[-1][0] - Qlow # BUG: This overestimates Qhp for smaller plants where the composite curve is less accurate
        Qmax_beiron = self.get("Qreb")*1.18 # Assumptions from (Beiron, 2022)
        Qhp = Qmax_beiron - Qlow
        if not self.technology_assumptions["heat_pump"]:
            Qhp = 0
        Php = Qhp/self.technology_assumptions["COP"]

        self.P -= Php/1000
        self.results["Plost"] += Php/1000
        self.Qdh += (Qhex + Qhp)/1000
        self.results["Qlost"] -= (Qhex + Qhp)/1000
        self.results["Qrecovered"] = (Qhex + Qhp)/1000
        self.results["Qhp"] = Qhp/1000
        self.QTdict = {
            "supp": [Qsupp, 86], # BUG: Also hard-coded
            "low": [Qlow, Tlow],
            "pinch": [Qpinch, Tpinch]
        }

        if Qhex<0 or Qhp<0:
            self.plot_hexchange(show=True)
            raise ValueError("Infeasible heat exchange")
        return

    def CAPEX_MEA(self, economic_assumptions, escalate=True):
        self.economic_assumptions = economic_assumptions
        X = economic_assumptions

        # Estimating cost of capture plant
        CAPEX = X['alpha'] * (self.gases["V_fluegas"]) ** X['beta']  #[MEUR](Eliasson, 2022)
        CAPEX *= X['CEPCI'] *1000                                    #[kEUR]
        fixed_OPEX = X['fixed'] * CAPEX

        # Adding cost of HEXs (+~2% cost)
        Qhex = self.results["Qrecovered"] - self.technology_assumptions["heat_pump"] #[MW]
        U = self.technology_assumptions["U"] /1000                                   #[kW/m2K]
        A = Qhex*1000/(U * self.technology_assumptions["dTmin"])                     # This overestimates the area as dTmin<dTln so it is pessimistic costwise
        CAPEX_hex = X["cHEX"]*A**0.9145                                                  #[kEUR] Original val: 0.571 (Eliasson, 2022)
        CAPEX += CAPEX_hex

        # Adding cost of HP (+~25% cost)
        if self.technology_assumptions["heat_pump"]:
            Qhp = self.results["Qhp"]
            CAPEX_hp = X["cHP"]*1000 * Qhp                                           #[kEUR], probably represents 2-4 pumps
            CAPEX += CAPEX_hp  

        if escalate:
            CAPEX *= 1 + X['ownercost']
            escalation = sum((1 + X['rescalation']) ** (n - 1) * (1 / X['yexpenses']) for n in range(1, X['yexpenses'] + 1))
            cfunding = sum(X['WACC'] * (X['yexpenses'] - n + 1) * (1 + X['rescalation']) ** (n - 1) * (1 / X['yexpenses']) for n in range(1, X['yexpenses'] + 1))
            CAPEX *= escalation + cfunding      

        annualization = (X['i'] * (1 + X['i']) ** X['t']) / ((1 + X['i']) ** X['t'] - 1)
        aCAPEX = annualization * CAPEX                    

        return CAPEX, aCAPEX, fixed_OPEX                             #[kEUR] and [kEUR/yr]
    
    def OPEX_MEA(self, economic_assumptions):
        X = economic_assumptions
        duration = X["time"]
        duration_increase = self.technology_assumptions["duration_increase"] 
        cheat = X["cheat"] * X["celc"]

        # The annual energy revenues used to be this:
        cash_power = (self.P + self.results["Plost"])*duration * X["celc"]                  #[EUR/yr]
        cash_heat  = (self.Qdh + self.results["Qlost"] + self.Qfgc)*duration * cheat
        cost_fuel = self.Qfuel*duration * X["cbio"]
        revenues_nominal = cash_power+cash_heat-cost_fuel

        # The annual energy revenues are now this:
        cash_power = self.P*(duration + duration_increase) * X["celc"]                      #[EUR/yr]  
        cash_heat = (self.Qdh + self.Qfgc)*(duration + duration_increase) * cheat
        cost_fuel = self.Qfuel*(duration + duration_increase) * X["cbio"]
        revenues = cash_power+cash_heat-cost_fuel

        energy_OPEX = (revenues_nominal - revenues)/1000                                    #[kEUR/yr]
        other_OPEX = self.get("Makeup") * X['cMEA'] * 3600*(duration + duration_increase) /1000

        return energy_OPEX, other_OPEX
    
    def print_energybalance(self):
        print(f"\n{'Heat output (Qdh)':<20}: {self.Qdh} MWheat")
        print(f"{'Electric output (P)':<20}: {self.P} MWe")
        print(f"{'Existing FGC (Qfgc)':<20}: {self.Qfgc} MWheat")
        print(f"{'Fuel input (Qfuel)':<20}: {self.Qfuel} MWheat")

        for key,value in self.results.items():
            print(f"{' ':<5} {key:<20} {value}")
        for key,value in self.gases.items():
            print(f"{key:<20} {value}")

    def plot_hexchange(self, show=False): 
        Qsupp, Tsupp = self.QTdict["supp"]
        Qlow, Tlow = self.QTdict["low"]
        Qpinch, Tpinch = self.QTdict["pinch"]
        dTmin = self.technology_assumptions["dTmin"]

        plt.figure(figsize=(10, 8))
        composite_curve = self.composite_curve
        shifted_curve = [[point[0], point[1] - dTmin] for point in composite_curve]
        (Qpinch-Qsupp) + (Qlow-Qpinch)

        plt.plot([0, self.get("Qreb")], [self.get("Treb"), self.get("Treb")], marker='*', color='#a100ba', label='Qreboiler')
        plt.plot([point[0] for point in composite_curve], [point[1] for point in composite_curve], marker='o', color='red', label='T of CCS streams')
        plt.plot([point[0] for point in shifted_curve], [point[1] for point in shifted_curve], marker='o', color='pink', label='T shifted')
        plt.plot([Qpinch, Qlow], [Tpinch, Tlow], marker='x', color='#069AF3', label='Qlowgrade')
        plt.plot([Qpinch, Qsupp], [Tpinch, Tsupp], marker='x', color='blue', label='Qhighgrade')
        plt.plot([Qlow, composite_curve[-1][0]], [20, 15], marker='o', color='#0000FF', label='Cooling water') # NOTE: hard-coded CW temps.

        plt.text(26000, 55, f'dTmin={round(dTmin,2)} C', color='black', fontsize=12, ha='center', va='center')
        plt.text(26000, 115, f'Qreb={round(self.get("Qreb")/1000)} MW', color='#a100ba', fontsize=12, ha='center', va='center')       
        plt.text(5000, 60, f'Qhighgrade={round((Qpinch-Qsupp)/1000)} MW', color='#0000FF', fontsize=12, ha='center', va='center')
        plt.text(5000, 40, f'Qlowgrade={round((Qlow-Qpinch)/1000)} MW', color='#069AF3', fontsize=12, ha='center', va='center')
        plt.text(10000, 15, f'Qcoolingwater={round((composite_curve[-1][0]-Qlow)/1000)} MW', color='#0000FF', fontsize=12, ha='center', va='center')

        plt.xlabel('Q [kW]')
        plt.ylabel('T [C]')
        plt.title(f'[{self.name}] Heat exchange between composite curve and district heating')
        plt.legend()
        if show:
            plt.show()
        return
    
    def reset(self):
        self.__dict__.update(copy.deepcopy(self.nominal_state)) # Resets to the nominal state values

# ------------ ABOVE THIS LINE WE DEFINE ALL CLASSES AND FUNCTIONS NEEDED FOR THE CCS_Pulp() MODEL --------

def CCS_CHP( 
    dTreb=10,
    Tsupp=86,
    Tlow=38,
    dTmin=7,
    COP = 3,
    U = 1500,

    alpha=6.12,
    beta=0.6336,
    CEPCI=600/550,
    fixed=0.06,
    ownercost=0.2,
    WACC=0.05,
    yexpenses=3,
    rescalation=0.03,
    i=0.075,
    t=25,
    celc=40,
    cheat=0.80,
    cbio=99999999,
    cMEA=2,
    cHP=0.86,                       #(Bergander & Hellander, 2024)
    cHEX=0.571,                     # check units in (Eliasson, 2022)
    
    time=8000,

    # duration_increase="1000",       #[h/yr]
    # HEX_optimization="100",         #[% of optimal] everybody optimizes this, no reason to include!
    rate=0.90,
    heat_pump=True,

    chp_interpolators=None,
    CHP=None
):
    technology_assumptions = {
        'U': U,
        "time": time,
        "duration_increase": 0,         # This is the main difference compared to wood chip fired CHP
        "rate": rate,
        "heat_pump": heat_pump,
        "dTreb": dTreb,
        "Tsupp": Tsupp,
        "Tlow": Tlow,
        "dTmin": dTmin,
        "COP": COP,
        "molar_mass": 29.55
    }

    economic_assumptions = {
        'time': time,
        'alpha': alpha,
        'beta': beta,
        'CEPCI': CEPCI,
        'fixed': fixed,
        'ownercost': ownercost,
        'WACC': WACC,
        'yexpenses': yexpenses,
        'rescalation': rescalation,
        'i': i,
        't': t,
        'celc': celc,
        'cheat': cheat,
        'cbio': cbio,
        'cMEA': cMEA,
        'cHP': cHP,
        'cHEX' : cHEX
    }

    # Size a capture plant and power it
    CHP.burn_fuel(technology_assumptions)
    CHP.size_MEA(rate, chp_interpolators)
    CHP.power_MEA()

    # Calculate recoverable heat and integrate
    stream_data = CHP.select_streams()
    composite_curve = CHP.merge_heat(stream_data)
    CHP.recover_heat(composite_curve)
    
    # Calculate CAPEX/OPEX, and KPIs
    CAPEX, aCAPEX, fixed_OPEX = CHP.CAPEX_MEA(economic_assumptions, escalate=True)
    energy_OPEX, other_OPEX = CHP.OPEX_MEA(economic_assumptions)

    costs = [
        ["CAPEX",       CAPEX], 
        ["aCAPEX",      aCAPEX], 
        ["fixed_OPEX",  fixed_OPEX], 
        ["energy_OPEX", energy_OPEX], 
        ["other_OPEX",  other_OPEX]
    ] #[kEUR]

    emissions = CHP.gases
    emissions = [
        ["nominal", emissions["nominal_emissions"]], 
        ["gross",   emissions["boiler_emissions"]], 
        ["captured",emissions["captured_emissions"]],
        ["net",     emissions["boiler_emissions"]-emissions["captured_emissions"]] 
    ] #[ktCO2/yr]

    capture_cost = (aCAPEX + fixed_OPEX + energy_OPEX + other_OPEX) / (CHP.gases["captured_emissions"])                         #[kEUR/kt], ~half is energy opex
    duration, duration_increase = economic_assumptions["time"], technology_assumptions["duration_increase"]
    penalty_power = (CHP.P + CHP.results["Plost"])*duration - CHP.P*(duration + duration_increase)                              #[MWh/yr]
    penalty_heat = (CHP.Qdh + CHP.results["Qlost"] + CHP.Qfgc)*duration - (CHP.Qdh + CHP.Qfgc)*(duration + duration_increase)   #[MWh/yr]
    penalty_services = (penalty_power + penalty_heat) / (CHP.gases["captured_emissions"])                                       #[MWh/kt]
    penalty_biomass  = CHP.results["Qextra"]          / (CHP.gases["captured_emissions"])                                       #[MWh/kt]        

    # CHP.print_energybalance()
    CHP.reset()
    return capture_cost, penalty_services, penalty_biomass, costs, emissions


if __name__ == "__main__":

    # Load plant data
    plants_df = pd.read_csv("WASTE data.csv",delimiter=";")
    plant_data = plants_df.iloc[0]

    # Load CHP Aspen data
    aspen_df = pd.read_csv("MEA-chip.csv", sep=";", decimal=',')
    aspen_interpolators = create_interpolators(aspen_df)

    # Initate a CHP and calculate its nominal energy balance
    energybalance_assumptions = {
        # "U": 1500                        #[W/m2K]
        # "m_fluegas": simplified from Tharun's study
    }

    CHP = W2E_plant(
        name=plant_data["Plant Name"],
        fuel=plant_data["Fuel (W=waste, B=biomass)"],
        Qdh=plant_data["Heat output (MWheat)"],
        P=plant_data["Electric output (MWe)"],
        Qfgc=plant_data["Existing FGC heat output (MWheat)"],
        ybirth=plant_data["Year of commissioning"],
        Tsteam=plant_data["Live steam temperature (degC)"],
        psteam=plant_data["Live steam pressure (bar)"],
        energybalance_assumptions=energybalance_assumptions
    )
    print(f"||| MODELLING {CHP.name} WASTE CHP |||")

    CHP.estimate_nominal_cycle() 
    CHP.print_energybalance()

    # The RDM evaluation starts below:
    capture_cost, penalty_services, penalty_biomass, costs, emissions = CCS_CHP(CHP=CHP, chp_interpolators=aspen_interpolators)
    print("Outcomes: ", capture_cost, penalty_services, penalty_biomass, costs, emissions)

    # plt.show()