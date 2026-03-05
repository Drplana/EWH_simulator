import numpy as np
import random
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm
import math

class StochasticDHWProfile:
    """
    Class to generate a stochastic domestic hot water consumption profile.
    The profile is based on the work of Hendron and Burch [1] and others.
    """

    def __init__(self, nday=1, day_init=0, n_users=1):
        self.nday = nday
        self.day_init = day_init
        self.n_users = n_users  # Default number of users, can be adjusted if needed

    def DHW_load_gen(self, bool_plot=False):
        """
        Create a water consumption profile starting at day day_init and stopping after self.nday.
        Return the water mass flow for 1 person

        References

        [1] Hendron, R., Burch, J., Barker, G. 2010. Tool for generating realistic
        residential hot water event schedules. Proceedings of the Fourth National Conference of
        IBPSA-USA, Simbuild 2010, New York City, USA.

        [2] Hendron, R., Engebrecht, C. 2010. Building America Research Benchmark
        Definition: Updated December 2009. NREL Report No TP-550-47246. Golden,
        CO:NREL

        [3] Spur, R., Fiala, D., Nevrala, D., Probert, D. 2006. Influence of the
        domestic hot-water daily draw-off profile on the performance of a
        hot-water store. Applied Energy, 83 pp.749-773.

        [4] Jordan, U., Vajen, K. 2001. Realistic Domestic Hot-Water Profiles in
        Different Time Scales. IEA-SHC Task 26 report, May 2001.

        Parameters
        ----------
        self.nday : int
            Number of day of the simulation.
        day_init : int, optional
            Day of the year where the simulation start. The default is 0.
        bool_plot : bool, optional
            Bool variable to plot the consumption profile generated. The default is False.

        Returns
        -------
        V_tot_litsecond : numpy array
            Array of the water consumption profile in liters per second.

        """

        # Fraction of Daily Usage [2]
        # Adjustment of the chance factors to get 50L/day/pers at 50 °C + (2x more sink A than sinkB, 8 times more shower than bath + 80/20% bathroom/kitchen load)
        f_daily_sinkA = self.n_users * 1.6 * 1 / 3.5 * 1 / 6
        f_daily_sinkB = self.n_users * 1.6 * f_daily_sinkA * 1 / 6
        f_daily_shwr = self.n_users * 0.85 * 1 / 2 * 1 / 30
        f_daily_bath = self.n_users * 0.85 * 1 / 4 * 1 / 400

        probday_shwr = [0.011, 0.005, 0.003, 0.005, 0.014, 0.052, 0.118, 0.117, 0.095, 0.074, 0.060, 0.047, 0.034,
                        0.029,
                        0.025, 0.026, 0.030, 0.039, 0.042, 0.042, 0.042, 0.041, 0.029, 0.021]
        probday_bath = [0.008, 0.004, 0.004, 0.004, 0.008, 0.019, 0.046, 0.058, 0.066, 0.058, 0.046, 0.035, 0.031,
                        0.023,
                        0.023, 0.023, 0.039, 0.046, 0.077, 0.100, 0.100, 0.077, 0.066, 0.039]
        probday_sink = [0.014, 0.007, 0.005, 0.005, 0.007, 0.018, 0.042, 0.062, 0.066, 0.062, 0.054, 0.050, 0.049,
                        0.045,
                        0.041, 0.043, 0.048, 0.065, 0.075, 0.069, 0.057, 0.048, 0.040, 0.027]

        if bool_plot == True:
            # Plot the distribution
            time_vect = np.arange(0, 24, 1)

            # Creating a smoother curve using spline interpolation
            time_new = np.linspace(time_vect.min(), time_vect.max(), 300)  # 300 points for smooth curve
            spl_shwr = make_interp_spline(time_vect, probday_shwr, k=2)
            probday_shwr_smooth = spl_shwr(time_new)
            spl_bath = make_interp_spline(time_vect, probday_bath, k=2)
            probday_bath_smooth = spl_bath(time_new)
            spl_sink = make_interp_spline(time_vect, probday_sink, k=2)
            probday_sink_smooth = spl_sink(time_new)

            plt.figure(figsize=(4, 3), constrained_layout=True)
            plt.rcParams.update({'font.size': 16})
            params = {
                "text.usetex": True,
                "font.family": "cm"}
            plt.rcParams.update(params)
            plt.grid()
            plt.plot(time_new, probday_shwr_smooth, 'b', linewidth=2, label='Shower')
            plt.plot(time_new, probday_bath_smooth, 'r', linewidth=2, label='Bath')
            plt.plot(time_new, probday_sink_smooth, 'g', linewidth=2, label='Sink')
            plt.xlabel('Time [hour]', fontsize=16, fontname="Times New Roman")
            plt.ylabel('Probability [-]', fontsize=16, fontname="Times New Roman")
            plt.legend(fontsize=12, loc='best')

            # Characteristics [3] & [4]
        # A: short draw-off; B:medium draw-off; C:shower; D:bath
        V_dot_A_m = 1 / 60 / 1000  # m^3/s
        V_dot_B_m = 2 / 60 / 1000  # m^3/s
        V_dot_C_m = 8 / 60 / 1000  # m^3/s
        V_dot_D_m = 14 / 60 / 1000  # m^3/s

        V_dot_A_std = 1 / 60 / 1000  # m^3/s
        V_dot_B_std = 1 / 60 / 1000  # m^3/s
        V_dot_C_std = 2 / 60 / 1000  # m^3/s
        V_dot_D_std = 2 / 60 / 1000  # m^3/s

        DELTAt_A_m = 1 * 60  # sec
        DELTAt_B_m = 1 * 60  # sec
        DELTAt_C_m = 5 * 60  # sec
        DELTAt_D_m = 10 * 60  # sec

        DELTAt_A_std = 0.5 * 60  # sec
        DELTAt_B_std = 0.5 * 60  # sec
        DELTAt_C_std = 2.5 * 60  # sec
        DELTAt_D_std = 5 * 60  # sec

        # Time
        ti = self.day_init * 24 * 60
        DELTAtmin = 1
        tf = ti + self.nday * 24 * 60 / DELTAtmin
        nstep = int(self.nday * 24 * 60 / DELTAtmin)
        time = np.arange(nstep) / 60  # time in hours
        # Variables
        minweek = np.zeros(nstep, dtype=int)  # minutes of the week (reset after 10080)
        fsinkA = np.zeros(nstep)  # boolean to check activation of event sinkA
        fsinkB = np.zeros(nstep)  # boolean to check activation of event sinkB
        fshwr = np.zeros(nstep)  # boolean to check activation of event fshwr
        fbath = np.zeros(nstep)  # boolean to check activation of event fbath
        V_dot_sinkA = np.zeros(nstep)  # volume flow rate of type sinkA for the whole period
        V_dot_sinkB = np.zeros(nstep)  # volume flow rate of type sinkB for the whole period
        V_dot_shwr = np.zeros(nstep)  # volume flow rate of type shower for the whole period
        V_dot_bath = np.zeros(nstep)  # volume flow rate of type bath for the whole period
        V_dot_tot = np.zeros(nstep)  # total volume flow rate of type sinkA for the whole period
        V_dot_tot_lmin = np.zeros(nstep)  # total volume flow rate of type sinkA for the whole period in l/min
        n_sinkA = np.zeros(self.nday)  # number of type sinkA occurence per day
        n_sinkB = np.zeros(self.nday)  # number of type sinkB occurence per day
        n_shwr = np.zeros(self.nday)  # number of type shower occurence per day
        n_bath = np.zeros(self.nday)  # number of type bath occurence per day
        t_sinkA = np.zeros(self.nday)  # total time of type sinkA per day
        t_sinkB = np.zeros(self.nday)  # total time of type sinkb per day
        t_shwr = np.zeros(self.nday)  # total time of type shower per day
        t_bath = np.zeros(self.nday)  # total time of type bath per day
        V_sinkA_lit = np.zeros(self.nday)  # total volume of type sinkA in liters per day
        V_sinkB_lit = np.zeros(self.nday)  # total volume of type sinkA in liters per day
        V_shwr_lit = np.zeros(self.nday)  # total volume of type sinkA in liters per day
        V_bath_lit = np.zeros(self.nday)  # total volume of type sinkA in liters per day
        self.V_tot_lit = np.zeros(self.nday)  # total volume in liters per day
        V_tot_lit_yr = np.zeros(int(np.ceil(self.nday / 365)))  # total volume per year

        flowAmat = np.zeros((int(7 * 24 * 60 / DELTAtmin), int(np.ceil(self.nday / 7))))
        flowBmat = np.zeros((int(7 * 24 * 60 / DELTAtmin), int(np.ceil(self.nday / 7))))
        flowCmat = np.zeros((int(7 * 24 * 60 / DELTAtmin), int(np.ceil(self.nday / 7))))
        flowDmat = np.zeros((int(7 * 24 * 60 / DELTAtmin), int(np.ceil(self.nday / 7))))

        fAmat = np.zeros((int(7 * 24 * 60 / DELTAtmin), int(np.ceil(self.nday / 7))))
        fBmat = np.zeros((int(7 * 24 * 60 / DELTAtmin), int(np.ceil(self.nday / 7))))
        fCmat = np.zeros((int(7 * 24 * 60 / DELTAtmin), int(np.ceil(self.nday / 7))))
        fDmat = np.zeros((int(7 * 24 * 60 / DELTAtmin), int(np.ceil(self.nday / 7))))

        # Initialize indexes
        fA = 0
        fAprev = 0
        iA = -1
        jA = -1
        flowA = 0
        fB = 0
        fBprev = 0
        iB = -1
        jB = -1
        flowB = 0
        fC = 0
        fCprev = 0
        iC = -1
        jC = -1
        flowC = 0
        fD = 0
        fDprev = 0
        iD = -1
        jD = -1
        flowD = 0

        # Initialize random variable lists + starting time vector
        bA = []
        bB = []
        bC = []
        bD = []
        cA = []
        cB = []
        cC = []
        cD = []
        tstartA = []
        tstartB = []
        tstartC = []
        tstartD = []
        tendA = []
        tendB = []
        tendC = []
        tendD = []

        cumprobdurA = norm.cdf(np.arange(0, 2400), loc=DELTAt_A_m, scale=DELTAt_A_std)  # in second
        cumprobdurB = norm.cdf(np.arange(0, 2400), loc=DELTAt_B_m, scale=DELTAt_B_std)
        cumprobdurC = norm.cdf(np.arange(0, 2400), loc=DELTAt_C_m, scale=DELTAt_C_std)
        cumprobdurD = norm.cdf(np.arange(0, 2400), loc=DELTAt_D_m, scale=DELTAt_D_std)

        # Model [4]

        for t in np.arange(int(ti), int(tf)):

            hour = t / 60
            # Hour of the day (1-24)
            hourday = (hour % 24)

            day = t / (24 * 60)
            dayi = int(np.floor(day))

            # Day of the week (1-7)
            dayweek = (day % 7)

            # day of the year (1-365)
            dayyr = day % 365

            # Year
            yr = t / (24 * 60 * 365)
            yri = int(np.floor(yr))

            # Minute of the week
            # Create a list or array to hold minweek values, assuming t is an integer
            # minweek = np.zeros(t+1)  # Pre-allocate space
            minweek[t] = t % (7 * 24 * 60)
            # Week of the year
            week = t / (24 * 60 * 7)
            weeki = int(np.floor(week))

            # probseason - Course of probabilities during the year
            P = 365 * 24 * 60
            A = 0.1
            B = - 61.75 * 24 * 60
            probseason = 1 + A * np.sin(2 * np.pi * (t - B) / DELTAtmin * 1 / P)
            # probweek - Course of probabilities during the week (variable probability for taking a bath)
            probweekA = 1
            probweekB = 1
            probweekC = 1
            if dayweek <= 4:
                probweekD = 0.5
            elif dayweek > 4 and dayweek <= 5:
                probweekD = 0.8
            elif dayweek > 5 and dayweek <= 6:
                probweekD = 1.9
            elif dayweek > 6 and dayweek <= 7:
                probweekD = 2.3

            # probday - Daily distribution of probabilities
            probdayA = probday_sink[math.trunc((hourday))]
            probdayB = probday_sink[math.trunc((hourday))]
            probdayC = probday_shwr[math.trunc((hourday))]
            probdayD = probday_bath[math.trunc((hourday))]

            # probholiday - Holiday period
            if (dayyr >= 195 and dayyr <= 208) or (dayyr >= 220 and dayyr <= 233):
                probholiday = 0.5
            else:
                probholiday = 1

            # probyear - Global probability function
            probyearA = f_daily_sinkA * probseason * probweekA * probdayA * probholiday
            probyearB = f_daily_sinkB * probseason * probweekB * probdayB * probholiday
            probyearC = f_daily_shwr * probseason * probweekC * probdayC * probholiday
            probyearD = f_daily_bath * probseason * probweekD * probdayD * probholiday

            # Stochastic process
            if t < 1:
                fsinkA[t] = 0
                fsinkB[t] = 0
                fshwr[t] = 0
                fbath[t] = 0
                V_dot_sinkA[t] = 0
                V_dot_sinkB[t] = 0
                V_dot_shwr[t] = 0
                V_dot_bath[t] = 0

            else:
                aA = random.random()
                aB = random.random()
                aC = random.random()
                aD = random.random()
                # Is appliance (water use case A) already functioning ?
                if fAprev == 0:  # NO
                    # Stochastic process to determine if an operation cycle (use of water from case A) is starting
                    if aA < probyearA:  # process is starting
                        iA = iA + 1
                        fA = 1
                        tstartA.append(t)
                        bA.append(random.random())  # duration probability index
                        cA.append(max(0, np.random.normal(V_dot_A_m, V_dot_A_std)))  # pick-up flow
                        flowA = cA[iA]
                    else:  # Nothing is starting
                        fA = 0
                        flowA = 0

                    # Is appliance already functioning ?
                else:  # YES
                    # What is the duration of the on-going cycle ?
                    DELTAtonA = max(0, (t - tstartA[iA]))
                    # print(bA[iA], cumprobdurA[DELTAtonA*60], tstartA[iA], t)
                    if bA[iA] < cumprobdurA[DELTAtonA * 60]:
                        # Cycle stops
                        jA = jA + 1
                        tendA.append(t)
                        fA = 0
                        flowA = 0
                    else:  # cycle continue
                        fA = 1
                        flowA = cA[iA]

                fAprev = fA
                # Is appliance (water use case B) already functioning ?
                if fBprev == 0:  # NO
                    # Stochastic process to determine if an operation cycle (use of water from case A) is starting
                    if aB < probyearB:  # process is starting
                        iB = iB + 1
                        fB = 1
                        tstartB.append(t)
                        bB.append(random.random())  # duration probability index
                        cB.append(max(0, np.random.normal(V_dot_B_m, V_dot_B_std)))  # pick-up flow
                        flowB = cB[iB]
                    else:  # Nothing is starting
                        fB = 0
                        flowB = 0

                    # Is appliance already functioning ?
                else:  # YES
                    # What is the duration of the on-going cycle ?
                    DELTAtonB = max(0, (t - tstartB[iB]))
                    if bB[iB] < cumprobdurB[DELTAtonB * 60]:
                        # Cycle stops
                        jB = jB + 1
                        tendB.append(t)
                        fB = 0
                        flowB = 0
                    else:  # cycle continue
                        fB = 1
                        flowB = cB[iB]

                fBprev = fB
                # Is appliance (water use case C) already functioning ?
                if fCprev == 0:  # NO
                    # Stochastic process to determine if an operation cycle (use of water from case A) is starting
                    if aC < probyearC:  # process is starting
                        iC = iC + 1
                        fC = 1
                        tstartC.append(t)
                        bC.append(random.random())  # duration probability index
                        cC.append(max(0, np.random.normal(V_dot_C_m, V_dot_C_std)))  # pick-up flow
                        flowC = cC[iC]
                    else:  # Nothing is starting
                        fC = 0
                        flowC = 0

                    # Is appliance already functioning ?
                else:  # YES
                    # What is the duration of the on-going cycle ?
                    DELTAtonC = max(0, (t - tstartC[iC]))
                    if bC[iC] < cumprobdurC[DELTAtonC * 60]:
                        # Cycle stops
                        jC = jC + 1
                        tendC.append(t)
                        fC = 0
                        flowC = 0
                    else:  # cycle continue
                        fC = 1
                        flowC = cC[iC]

                fCprev = fC
                # Is appliance (water use case D) already functioning ?
                if fDprev == 0:  # NO
                    # Stochastic process to determine if an operation cycle (use of water from case A) is starting
                    if aD < probyearD:  # process is starting
                        iD = iD + 1
                        fD = 1
                        tstartD.append(t)
                        bD.append(random.random())  # duration probability index
                        cD.append(max(0, np.random.normal(V_dot_D_m, V_dot_D_std)))  # pick-up flow
                        flowD = cD[iD]
                    else:  # Nothing is starting
                        fD = 0
                        flowD = 0

                    # Is appliance already functioning ?
                else:  # YES
                    # What is the duration of the on-going cycle ?
                    DELTAtonD = max(0, (t - tstartD[iD]))
                    if bD[iD] < cumprobdurD[DELTAtonD * 60]:
                        # Cycle stops
                        jD = jD + 1
                        tendD.append(t)
                        fD = 0
                        flowD = 0
                    else:  # cycle continue
                        fD = 1
                        flowD = cD[iD]

                fDprev = fD

            # flow (m3/s to liters/s)
            flowA = flowA * 1000
            flowB = flowB * 1000
            flowC = flowC * 1000
            flowD = flowD * 1000
            flowtot = flowA + flowB + flowC + flowD

            # min*week matrix
            flowAmat[minweek[t], weeki] = flowA
            flowBmat[minweek[t], weeki] = flowB
            flowCmat[minweek[t], weeki] = flowC
            flowDmat[minweek[t], weeki] = flowD

            V_dot_sinkA[t] = flowA  # l/s
            V_dot_sinkB[t] = flowB
            V_dot_shwr[t] = flowC
            V_dot_bath[t] = flowD
            V_dot_tot[t] = flowtot
            V_dot_tot_lmin[t] = flowtot * 60

            # DHW use
            fsinkA[t] = fA
            fsinkB[t] = fB
            fshwr[t] = fC
            fbath[t] = fD

            fAmat[minweek[t], weeki] = fA
            fBmat[minweek[t], weeki] = fB
            fCmat[minweek[t], weeki] = fC
            fDmat[minweek[t], weeki] = fD

            # Daily data
            # Events occurence
            if dayi == 0:
                n_sinkA[dayi] = iA
                n_sinkB[dayi] = iB
                n_shwr[dayi] = iC
                n_bath[dayi] = iD
            else:
                n_sinkA[dayi] = iA - sum(n_sinkA[:dayi])
                n_sinkB[dayi] = iB - sum(n_sinkB[:dayi])
                n_shwr[dayi] = iC - sum(n_shwr[:dayi])
                n_bath[dayi] = iD - sum(n_bath[:dayi])

            # Events duration in minutes
            t_sinkA[dayi] = t_sinkA[dayi] + fA
            t_sinkB[dayi] = t_sinkB[dayi] + fB
            t_shwr[dayi] = t_shwr[dayi] + fC
            t_bath[dayi] = t_bath[dayi] + fD

            # Events volume in liters
            V_sinkA_lit[dayi] = V_sinkA_lit[dayi] + flowA * DELTAtmin * 60
            V_sinkB_lit[dayi] = V_sinkB_lit[dayi] + flowB * DELTAtmin * 60
            V_shwr_lit[dayi] = V_shwr_lit[dayi] + flowC * DELTAtmin * 60
            V_bath_lit[dayi] = V_bath_lit[dayi] + flowD * DELTAtmin * 60
            self.V_tot_lit[dayi] = V_sinkA_lit[dayi] + V_sinkB_lit[dayi] + V_shwr_lit[dayi] + V_bath_lit[dayi]

            # Annual volume in liters
            V_tot_lit_yr[yri] = V_tot_lit_yr[yri] + flowtot * DELTAtmin * 60

        # Average daily usage

        # Total daily volume
        self.V_tot_davg_lit = np.mean(self.V_tot_lit)
        V_sinkA_davg_lit = np.mean(V_sinkA_lit)
        V_sinkB_davg_lit = np.mean(V_sinkB_lit)
        V_shwr_davg_lit = np.mean(V_shwr_lit)
        V_bath_davg_lit = np.mean(V_bath_lit)

        # Daily time of use (minutes)
        t_sinkA_davg = np.mean(t_sinkA)
        t_sinkB_davg = np.mean(t_sinkB)
        t_shwr_davg = np.mean(t_shwr)
        t_bath_davg = np.mean(t_bath)

        # Daily number of occurrences
        n_sinkA_davg = np.mean(n_sinkA)
        n_sinkB_davg = np.mean(n_sinkB)
        n_shwr_davg = np.mean(n_shwr)
        n_bath_davg = np.mean(n_bath)

        V_tot_litsecond = V_dot_tot

        return xr.Dataset({
            "sinkA": (["time"], V_dot_sinkA),
            "sinkB": (["time"], V_dot_sinkB),
            "shower": (["time"], V_dot_shwr),
            "bath": (["time"], V_dot_bath),
            "total": (["time"], V_dot_tot)
        }, coords={"time": time})



class StochasticDHWProfileRobust:
    """
    Robust stochastic DHW profile generator combining:
      • Region‐ and season‐aware base probabilities (hourly)
      • Seasonal scaling (±12−13% for BE; extendable per region)
      • Weekly modulation (e.g., higher bath probability on weekends)
      • Holiday modulation (lower usage during major holiday weeks)
      • Multiple fixtures: sinkA, sinkB, shower, bath, dishwasher
      • Correct units → instantaneous L/min at each minute
      • Calibration to target daily volume per person (default 50 L/p·d)
      • Optional energy output in kWh (ΔT = 50 K)

    Usage:
        gen = StochasticDHWProfileRobust(
            nday=7,
            start_date="2024-01-01",
            n_users=2,
            demographics={"has_children": True},
            region="BE",
            season="winter",
            target_L_per_person=50,
            seed=42
        )
        raw_ds, scaled_ds = gen.generate(return_in_kwh=False)
    """

    # 1. Per‐region, per‐season, per‐hour base probabilities for each fixture
    WATER_USE_PROFILES = {
        "BE": {
            "winter": {
                "shower": [0.013, 0.006, 0.004, 0.004, 0.009, 0.020, 0.050, 0.062,
                           0.070, 0.065, 0.050, 0.040, 0.035, 0.030, 0.028, 0.028,
                           0.032, 0.045, 0.050, 0.050, 0.045, 0.033, 0.025, 0.018],
                "bath": [0.002, 0.000, 0.000, 0.000, 0.001, 0.005, 0.010, 0.015,
                         0.020, 0.018, 0.015, 0.010, 0.008, 0.005, 0.005, 0.005,
                         0.012, 0.020, 0.025, 0.030, 0.030, 0.025, 0.020, 0.010],
                "sink": [0.015, 0.008, 0.006, 0.006, 0.008, 0.020, 0.045, 0.060,
                         0.065, 0.060, 0.055, 0.050, 0.049, 0.047, 0.044, 0.045,
                         0.050, 0.070, 0.075, 0.070, 0.060, 0.052, 0.042, 0.030],
                "dishwasher": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                               0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                               0.000, 0.010, 0.020, 0.030, 0.020, 0.010, 0.000, 0.000],
            },
            "spring": {
                "shower": [0.011, 0.005, 0.003, 0.003, 0.008, 0.018, 0.045, 0.055,
                           0.060, 0.055, 0.045, 0.035, 0.030, 0.025, 0.023, 0.023,
                           0.028, 0.040, 0.045, 0.045, 0.040, 0.030, 0.022, 0.015],
                "bath": [0.002, 0.000, 0.000, 0.000, 0.001, 0.005, 0.008, 0.012,
                         0.015, 0.013, 0.012, 0.008, 0.007, 0.005, 0.005, 0.005,
                         0.010, 0.018, 0.020, 0.025, 0.025, 0.020, 0.015, 0.008],
                "sink": [0.014, 0.007, 0.005, 0.005, 0.007, 0.018, 0.044, 0.058,
                         0.060, 0.058, 0.052, 0.048, 0.047, 0.045, 0.042, 0.043,
                         0.046, 0.064, 0.070, 0.065, 0.055, 0.048, 0.040, 0.028],
                "dishwasher": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                               0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                               0.000, 0.008, 0.015, 0.020, 0.015, 0.008, 0.000, 0.000],
            },
            "summer": {
                "shower": [0.010, 0.004, 0.003, 0.003, 0.008, 0.017, 0.042, 0.052,
                           0.058, 0.052, 0.042, 0.032, 0.028, 0.023, 0.020, 0.022,
                           0.025, 0.035, 0.040, 0.040, 0.040, 0.035, 0.025, 0.017],
                "bath": [0.001, 0.000, 0.000, 0.000, 0.001, 0.004, 0.007, 0.010,
                         0.012, 0.010, 0.009, 0.006, 0.005, 0.004, 0.004, 0.004,
                         0.008, 0.015, 0.018, 0.020, 0.020, 0.018, 0.015, 0.008],
                "sink": [0.013, 0.007, 0.005, 0.005, 0.006, 0.018, 0.042, 0.056,
                         0.058, 0.056, 0.048, 0.046, 0.045, 0.042, 0.040, 0.042,
                         0.044, 0.060, 0.068, 0.064, 0.052, 0.046, 0.038, 0.026],
                "dishwasher": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                               0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                               0.000, 0.006, 0.010, 0.015, 0.010, 0.006, 0.000, 0.000],
            },
            "autumn": {
                "shower": [0.012, 0.006, 0.004, 0.004, 0.009, 0.019, 0.048, 0.060,
                           0.065, 0.060, 0.050, 0.040, 0.036, 0.030, 0.028, 0.028,
                           0.030, 0.043, 0.048, 0.048, 0.044, 0.032, 0.024, 0.017],
                "bath": [0.002, 0.000, 0.000, 0.000, 0.001, 0.005, 0.009, 0.014,
                         0.018, 0.016, 0.014, 0.010, 0.009, 0.006, 0.006, 0.006,
                         0.014, 0.022, 0.028, 0.032, 0.032, 0.028, 0.022, 0.010],
                "sink": [0.015, 0.008, 0.006, 0.006, 0.008, 0.020, 0.045, 0.058,
                         0.062, 0.058, 0.053, 0.049, 0.048, 0.046, 0.044, 0.045,
                         0.048, 0.068, 0.072, 0.068, 0.058, 0.050, 0.042, 0.030],
                "dishwasher": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                               0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                               0.000, 0.009, 0.015, 0.025, 0.015, 0.009, 0.000, 0.000],
            },
        },
        # Placeholder for Netherlands; can be filled with actual data later
        "NL": {
            "winter": {"shower": [0.012] * 24, "bath": [0.008] * 24, "sink": [0.014] * 24, "dishwasher": [0.000] * 24},
            "spring": {"shower": [0.010] * 24, "bath": [0.006] * 24, "sink": [0.012] * 24, "dishwasher": [0.000] * 24},
            "summer": {"shower": [0.009] * 24, "bath": [0.005] * 24, "sink": [0.011] * 24, "dishwasher": [0.000] * 24},
            "autumn": {"shower": [0.011] * 24, "bath": [0.007] * 24, "sink": [0.013] * 24, "dishwasher": [0.000] * 24},
        },
    }

    # 2. Seasonal scaling factors (± around annual mean)
    SEASONAL_SCALING = {
        "winter": 1.12,  # +12% in winter
        "spring": 1.00,  # baseline
        "summer": 0.87,  # –13% in summer
        "autumn": 1.00,  # baseline
    }

    def __init__(
            self,
            nday=1,
            start_date="2024-01-01",
            n_users=1,
            demographics=None,
            seed=None,
            fixture_params=None,
            region="BE",
            season="winter",
            target_L_per_person=50,
    ):
        """
        Parameters
        ----------
        nday : int
            Number of days to simulate.
        start_date : str (YYYY-MM-DD)
            Calendar start date of simulation.
        n_users : int
            Number of occupants in the household.
        demographics : dict, optional
            e.g. {"has_children": True, "elderly": False}
        seed : int, optional
            Random seed for reproducibility.
        fixture_params : dict, optional
            Override any base fixture parameters, e.g. {"shower": {"flow_mean":...}}
        region : str
            Region code to select base hourly probabilities ("BE", "NL", etc.).
        season : str
            Season name ("winter", "spring", "summer", "autumn").
        target_L_per_person : float
            Calibration target: liters per person per day. Default 50 L/p·d.
        """
        self.nday = nday
        self.start_date = datetime.fromisoformat(start_date)
        self.n_users = n_users
        self.demographics = demographics or {}
        self.fixture_params = fixture_params or {}
        self.region = region
        self.season = season
        self.target_L_per_person = target_L_per_person

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Load base hourly probabilities and apply seasonal scaling
        self._load_time_of_day_probs()

        # Define fixture parameters & apply overrides
        self._set_fixtures()

    def _load_time_of_day_probs(self):
        """Lookup 24-hour probability lists for each fixture, then scale by season."""
        region_block = self.WATER_USE_PROFILES.get(self.region)
        if region_block is None:
            raise KeyError(f"No water-use profile for region '{self.region}'.")
        season_block = region_block.get(self.season)
        if season_block is None:
            raise KeyError(f"No water-use profile for season '{self.season}' in region '{self.region}'.")

        # Extract per-hour lists
        self.hourly_shower_probs = np.array(season_block["shower"], dtype=float)
        self.hourly_bath_probs = np.array(season_block["bath"], dtype=float)
        self.hourly_sink_probs = np.array(season_block["sink"], dtype=float)
        self.hourly_dishwasher_probs = np.array(season_block["dishwasher"], dtype=float)

        # Check length = 24
        for name, arr in [
            ("shower", self.hourly_shower_probs),
            ("bath", self.hourly_bath_probs),
            ("sink", self.hourly_sink_probs),
            ("dishwasher", self.hourly_dishwasher_probs),
        ]:
            if arr.shape[0] != 24:
                raise ValueError(f"Expected 24 hourly values for '{name}', got {arr.shape[0]}.")

        # Apply seasonal scaling
        scale = self.SEASONAL_SCALING.get(self.season, 1.0)
        self.hourly_shower_probs *= scale
        self.hourly_bath_probs *= scale
        self.hourly_sink_probs *= scale
        self.hourly_dishwasher_probs *= scale

    def _set_fixtures(self):
        """
        Define base parameters for each fixture:
          - flow_mean (m³/s), flow_std (m³/s)
          - dur_mean (sec), dur_std (sec)
        Then apply any overrides and demographic adjustments.
        """
        self.fixtures = {
            "sinkA": {"flow_mean": 1 / 60 / 1000, "flow_std": 1 / 60 / 1000, "dur_mean": 60, "dur_std": 30},
            "sinkB": {"flow_mean": 2 / 60 / 1000, "flow_std": 1 / 60 / 1000, "dur_mean": 60, "dur_std": 30},
            "shower": {"flow_mean": 8 / 60 / 1000, "flow_std": 2 / 60 / 1000, "dur_mean": 300, "dur_std": 150},
            "bath": {"flow_mean": 10 / 60 / 1000, "flow_std": 1 / 60 / 1000, "dur_mean": 300, "dur_std": 120},
            "dishwasher": {"flow_mean": 0.2 / 1000, "flow_std": 0.05 / 1000, "dur_mean": 3600, "dur_std": 900},
        }

        # Override base parameters if provided
        for key, override in self.fixture_params.items():
            if key in self.fixtures:
                self.fixtures[key].update(override)

        # Demographic adjustments
        if self.demographics.get("has_children"):
            self.fixtures["sinkA"]["flow_mean"] *= 1.2
            self.fixtures["bath"]["dur_mean"] *= 1.2
        if self.demographics.get("elderly"):
            for f in self.fixtures.values():
                f["flow_mean"] *= 0.8
                f["dur_mean"] *= 0.8

    def _minute_probabilities(self, hourly_probs):
        """Repeat a length-24 array into a length-(24*60) array."""
        return np.repeat(hourly_probs, 60)

    def _weekday_modifiers(self):
        """
        Create a per-minute “weekly” modifier for bath probability:
         • Mon–Thu: 1.0
         • Fri: 1.2
         • Sat: 1.5
         • Sun: 1.3
        Return an array of length nday*1440.
        """
        nstep = self.nday * 24 * 60
        modifiers = np.ones(nstep)
        for minute in range(nstep):
            dt = self.start_date + timedelta(minutes=minute)
            wd = dt.weekday()  # Mon=0 … Sun=6
            if wd <= 3:
                modifiers[minute] = 1.0
            elif wd == 4:
                modifiers[minute] = 1.2
            elif wd == 5:
                modifiers[minute] = 1.5
            else:  # Sunday
                modifiers[minute] = 1.3
        return modifiers

    def _holiday_modifiers(self):
        """
        Create a per-minute “holiday” modifier for all fixtures:
         • If date is Dec 24–Jan 2 or Apr 1–Apr 10, scale down to 0.7
         • Otherwise 1.0
        """
        nstep = self.nday * 24 * 60
        modifiers = np.ones(nstep)
        for minute in range(nstep):
            dt = self.start_date + timedelta(minutes=minute)
            doy = dt.timetuple().tm_yday
            # Dec 24 (≈359) → Jan 2 (≈2, wrap-around)
            if (doy >= 358) or (doy <= 2):
                modifiers[minute] = 0.7
            # Easter window (≈Apr 1–10, doy≈91–100)
            elif 91 <= doy <= 100:
                modifiers[minute] = 0.7
            else:
                modifiers[minute] = 1.0
        return modifiers

    def generate(self, return_in_kwh=False):
        """
        Simulate minute-by-minute fixture flows (L/min) for nday days.
        Returns two xarray Datasets: (raw_ds, scaled_ds).
        """
        # 1) Build per-minute base probabilities for one day, then tile
        prob_shower_1day = self._minute_probabilities(self.hourly_shower_probs)
        prob_bath_1day = self._minute_probabilities(self.hourly_bath_probs)
        prob_sink_1day = self._minute_probabilities(self.hourly_sink_probs)
        prob_dish_1day = self._minute_probabilities(self.hourly_dishwasher_probs)

        prob_shower_min = np.tile(prob_shower_1day, self.nday)
        prob_bath_min = np.tile(prob_bath_1day, self.nday)
        prob_sink_min = np.tile(prob_sink_1day, self.nday)
        prob_dishwasher_min = np.tile(prob_dish_1day, self.nday)

        # 2) Weekly (bath-only) and holiday (all) modifiers
        week_mod_bath = self._weekday_modifiers()  # length = nday*1440
        hol_mod_all = self._holiday_modifiers()  # length = nday*1440

        nstep = self.nday * 24 * 60
        flows = {f: np.zeros(nstep) for f in self.fixtures}

        # 3) Simulate each fixture minute by minute
        for fixture, params in self.fixtures.items():
            if fixture == "shower":
                base_probs = prob_shower_min
            elif fixture == "bath":
                base_probs = prob_bath_min * week_mod_bath
            elif fixture == "dishwasher":
                base_probs = prob_dishwasher_min
            else:  # sinkA or sinkB
                base_probs = prob_sink_min

            for t in range(nstep):
                # Base probability × holiday modifier
                p_base = base_probs[t] * hol_mod_all[t]
                # Scale by number of users / 10
                p_use = p_base * (self.n_users / 10.0)

                if random.random() < p_use:
                    dur_sec = max(1, int(np.random.normal(params["dur_mean"], params["dur_std"])))
                    flow_m3ps = max(0.0, np.random.normal(params["flow_mean"], params["flow_std"]))
                    end = min(nstep, t + dur_sec)

                    # Convert m³/s → L/min
                    flow_L_per_min = flow_m3ps * 1000 * 60
                    flows[fixture][t:end] += flow_L_per_min

        # 4) Build “raw” xarray Dataset
        time = np.arange(nstep)  # minutes since start_date
        raw_ds = xr.Dataset(
            {
                "sinkA": (["time"], flows["sinkA"]),
                "sinkB": (["time"], flows["sinkB"]),
                "shower": (["time"], flows["shower"]),
                "bath": (["time"], flows["bath"]),
                "dishwasher": (["time"], flows["dishwasher"]),
            },
            coords={"time": time},
        )
        raw_ds["total_L_per_min"] = (
                raw_ds.sinkA + raw_ds.sinkB + raw_ds.shower + raw_ds.bath + raw_ds.dishwasher
        )
        raw_total = raw_ds.total_L_per_min.sum().item()

        # 5) Calibrate to target_L_per_person × n_users
        desired_total = self.target_L_per_person * self.n_users * self.nday
        scale_factor = (desired_total / raw_total) if raw_total > 0 else 0.0

        scaled_ds = raw_ds.copy(deep=True)
        for var in ["sinkA", "sinkB", "shower", "bath", "dishwasher", "total_L_per_min"]:
            scaled_ds[var] = scaled_ds[var] * scale_factor

        # 6) Optionally convert liters to kWh (ΔT = 50 K)
        if return_in_kwh:
            kwh_per_L = 4.186 * 50 / 3600.0  # ≈ 0.05814 kWh/L
            raw_ds["total_kwh_per_min"] = raw_ds.total_L_per_min * kwh_per_L
            scaled_ds["total_kwh_per_min"] = scaled_ds.total_L_per_min * kwh_per_L

        return raw_ds, scaled_ds


# ------------------------------------------------------------------------------------
# 7) Example usage and visualization
# ------------------------------------------------------------------------------------
if __name__ == "__main__":

    BELGIAN_HOUSEHOLD_TYPES = [
        ("one_person", 0.361404, 1.0, {}),  # 36.1404%
        ("married_no_children", 0.182180, 2.0, {}),  # 18.2180%
        ("married_with_children", 0.187653, 4.5, {"has_children": True}),  # 18.7653%
        ("cohabiting_no_children", 0.065921, 2.0, {}),  # 6.5921%
        ("cohabiting_with_children", 0.082202, 3.5, {"has_children": True}),  # 8.2202%
        ("lone_parent", 0.098329, 2.0, {"has_children": True}),  # 9.8329%
        ("other", 0.022311, 3.0, {}),  # 2.2311%
    ]


    def sample_belgian_household():
        """
        Returns (household_type, n_users, demographics_dict),
        where types are sampled to match 2024 Belgian stats.
        """
        types, probs, sizes, demos = zip(*BELGIAN_HOUSEHOLD_TYPES)
        probs = np.array(probs)
        probs /= probs.sum()  # Normalize, though they already sum to 1.0
        idx = np.random.choice(len(types), p=probs)
        return types[idx], sizes[idx], demos[idx]


    # N = 1_000_000
    # total_size = 0
    # counts = {name: 0 for name, _, _, _ in BELGIAN_HOUSEHOLD_TYPES}
    # for _ in range(N):
    #     name, size, demo = sample_belgian_household()
    #     counts[name] += 1
    #     total_size += size
    #
    # print({k: v / N for k, v in counts.items()})
    # print("Simulated avg size ≈", total_size / N)

    summed_L_per_min = np.zeros(1 * 24 * 60)
    for _ in range(1000): # 5_163_139  # Total number of households in Belgium in 2024
        # Sample a random household type, get its size & demo
        _, n_users, demographics = sample_belgian_household()
        gen = StochasticDHWProfileRobust(
            nday=1,
            start_date="2024-01-01",  # or vary start_date if you want seasonally
            n_users=n_users,
            demographics=demographics,
            region="BE",
            season="winter",  # or select season from date
            target_L_per_person=50,
            seed=None
        )
        raw_ds, scaled_ds = gen.generate(return_in_kwh=False)

        # Extract the 7 individual days (each is 1440 minutes)
        arr = scaled_ds.total_L_per_min.values.reshape((1, 24 * 60))

        # PLOT 1: All 7 days (gray), plus 7-day average (blue)
        plt.figure(figsize=(10, 4))
        for day_idx in range(1):
            plt.plot(np.arange(24 * 60), arr[day_idx], color="gray", alpha=0.4)
        avg_curve = arr.mean(axis=0)
        plt.plot(np.arange(24 * 60), avg_curve, color="tab:blue", linewidth=2, label="7-day average")
        plt.title("Simulated Total DHW Flow (L/min) – 7 Separate Winter Days")
        plt.xlabel("Minute of Day (0–1439)")
        plt.ylabel("Flow (L/min)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # PLOT 2: Compare that 7-day average (blue) to a simple “benchmark” from Statbel patterns
        # We take the same 24-hour probabilities, assign typical volumes, normalize to 100 L (=2×50),
        # then expand each hour to 60 minutes (constant within the hour).
        hour_probs = np.array(gen.WATER_USE_PROFILES["BE"]["winter"]["shower"]) * 40 \
                     + np.array(gen.WATER_USE_PROFILES["BE"]["winter"]["bath"]) * 50 \
                     + np.array(gen.WATER_USE_PROFILES["BE"]["winter"]["sink"]) * 20 \
                     + np.array(gen.WATER_USE_PROFILES["BE"]["winter"]["dishwasher"]) * 15
        hour_norm = hour_probs / hour_probs.sum()
        benchmark_hourly_L = hour_norm * (gen.n_users * gen.target_L_per_person)
        benchmark_minute = np.repeat(benchmark_hourly_L / 60, 60)
        avg_n_users = total_people_simulated / total_households_simulated

        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(24 * 60), avg_curve, color="tab:blue", linewidth=2, label="Simulated 7-day avg")
        plt.plot(np.arange(24 * 60), benchmark_minute, color="tab:orange", linestyle="--", linewidth=2,
                 label="Benchmark (literature‐based hourly)")
        plt.title(f"Simulated vs. Benchmark DHW Flow (L/min) – Belgian Winter, avg {avg_n_users:.1f} persons/HH")
        plt.xlabel("Minute of Day")
        plt.ylabel("Flow (L/min)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    #
    # # Generate 7 days for Belgium winter, 2 users with children
    # gen = StochasticDHWProfileRobust(
    #     nday=7,
    #     start_date="2024-01-01",
    #     n_users=2,
    #     demographics={"has_children": True},
    #     region="BE",
    #     season="spring",
    #     target_L_per_person=80,
    #     seed=42,
    # )
    #
    # raw_ds, scaled_ds = gen.generate(return_in_kwh=False)
    #
    # # Print raw vs. scaled total volumes
    # raw_vol = raw_ds.total_L_per_min.sum().item()
    # scaled_vol = scaled_ds.total_L_per_min.sum().item()
    # print(f"Raw total volume (L over {gen.nday} days):    {raw_vol:.1f}")
    # print(f"Scaled total volume (L over {gen.nday} days): {scaled_vol:.1f} "
    #       f"(target was {gen.nday * gen.n_users * gen.target_L_per_person:.1f})")
    #
    # # Plot the 7-day average “total” minute-by-minute
    # arr = scaled_ds.total_L_per_min.values.reshape((gen.nday, 24 * 60))
    # daily_avg = arr.mean(axis=0)
    # time_minutes = np.arange(24 * 60)
    #
    # plt.figure(figsize=(10, 3))
    # plt.plot(time_minutes, daily_avg, color="tab:orange", linewidth=1)
    # plt.title("Average Total DHW Flow (L/min) – BE Winter (2 users)")
    # plt.xlabel("Minute of Day")
    # plt.ylabel("Flow (L/min)")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # # Plot each fixture’s 7-day average
    # fig, axs = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    # fixtures = ["sinkA", "sinkB", "shower", "bath", "dishwasher"]
    # for ax, var in zip(axs.flatten(), fixtures + [None]):
    #     if var is not None:
    #         arr = scaled_ds[var].values.reshape((gen.nday, 24 * 60))
    #         arr_avg = arr.mean(axis=0)
    #         ax.plot(time_minutes, arr_avg, color="tab:orange", linewidth=1)
    #         ax.set_title(f"{var} (avg L/min)")
    #         ax.grid(True)
    #     else:
    #         ax.axis("off")
    #
    # for ax in axs[-1]:
    #     ax.set_xlabel("Minute of Day")
    # plt.tight_layout()
    # plt.show()
