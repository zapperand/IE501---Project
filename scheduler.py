import pandas as pd
import pyomo.environ as pe
import pyomo.gdp as pyogdp
from itertools import product


class TheatreScheduler:

    def __init__(self, case_file_path, session_file_path):

        self.df_cases = pd.read_csv(case_file_path) 
        self.df_sessions = pd.read_csv(session_file_path)

        self.model = self.create_model()


    def _generate_case_durations(self):
        
        return pd.Series(self.df_cases["Expected Duration"].values, index=self.df_cases["CaseID"]).to_dict()


    def _generate_session_durations(self):
        
        return pd.Series(self.df_sessions["Duration"].values, index=self.df_sessions["SessionID"]).to_dict()


    def _generate_session_start_times(self):
        
        # Convert session start time from HH:MM:SS format into seconds elapsed since midnight
        self.df_sessions.loc[:, "Start"] = pd.to_timedelta(self.df_sessions["Start"])
        self.df_sessions["Start"]=pd.to_timedelta(self.df_sessions["Start"])
        self.df_sessions.loc[:, "Start"] = self.df_sessions["Start"].dt.total_seconds() / 60
        return pd.Series(self.df_sessions["Start"].values, index=self.df_sessions["SessionID"]).to_dict()


    def _get_ordinal_case_deadlines(self):
        
        self.df_cases.loc[:, "TargetDeadline"] = pd.to_datetime(self.df_cases["TargetDeadline"], format="%d/%m/%Y")
        self.df_cases.loc[:, "TargetDeadline"] = self.df_cases["TargetDeadline"].apply(lambda date: date.toordinal())
        return pd.Series(self.df_cases["TargetDeadline"].values, index=self.df_cases["CaseID"]).to_dict()


    def _get_ordinal_session_dates(self):
        
        self.df_sessions.loc[:, "Date"] = pd.to_datetime(self.df_sessions["Date"], format="%d/%m/%Y")
        self.df_sessions.loc[:, "Date"] = self.df_sessions["Date"].apply(lambda date: date.toordinal())
        return pd.Series(self.df_sessions["Date"].values, index=self.df_sessions["SessionID"]).to_dict()


    def _generate_disjunctions(self):
        
        cases = self.df_cases["CaseID"].to_list()
        sessions = self.df_sessions["SessionID"].to_list()
        disjunctions = []
        for (case1, case2, session) in product(cases, cases, sessions):
            if (case1 != case2) and (case2, case1, session) not in disjunctions:
                disjunctions.append((case1, case2, session))

        return disjunctions


    def create_model(self):
        # Define Model
        model = pe.ConcreteModel()

        # Import cases and sessions data into pyomo model

        # List of case IDs in surgical waiting list
        model.CASES = pe.Set(initialize=self.df_cases["CaseID"].tolist())
        # List of sessions IDs
        model.SESSIONS = pe.Set(initialize=self.df_sessions["SessionID"].tolist())
        # List of job shop tasks - all possible combinations of cases and sessions (caseID, sessionID)
        model.TASKS = pe.Set(initialize=model.CASES * model.SESSIONS, dimen=2)
        # The duration (expected case time) for each operation
        model.CASE_DURATION = pe.Param(model.CASES, initialize=self._generate_case_durations())
        # The duration of each theatre session
        model.SESSION_DURATION = pe.Param(model.SESSIONS, initialize=self._generate_session_durations())
        # The start time of each theatre session
        model.SESSION_START_TIME = pe.Param(model.SESSIONS, initialize=self._generate_session_start_times())
        # The deadline of each case
        model.CASE_DEADLINES = pe.Param(model.CASES, initialize=self._get_ordinal_case_deadlines())
        # The date of each theatre session
        model.SESSION_DATES = pe.Param(model.SESSIONS, initialize=self._get_ordinal_session_dates())


        model.DISJUNCTIONS = pe.Set(initialize=self._generate_disjunctions(), dimen=3)


        # Decision Variables

        # Upper bound (minutes in a day)
        ub = 1440  
        # Upper bound of session utilisation set to 85%
        max_util = 0.85
        model.M = pe.Param(initialize=1e3*ub)  # big M


        # Binary flag, 1 if case is assigned to session, 0 otherwise
        model.SESSION_ASSIGNED = pe.Var(model.TASKS, domain=pe.Binary)
        # Start time of a case
        model.CASE_START_TIME = pe.Var(model.TASKS, bounds=(0, ub), within=pe.PositiveReals)
        # Session utilisation
        model.UTILISATION = pe.Var(model.SESSIONS, bounds=(0, max_util), within=pe.PositiveReals)

        def objective_function(model):
            return pe.summation(model.UTILISATION)
  
        model.OBJECTIVE = pe.Objective(rule=objective_function, sense=pe.maximize)

        # Constraints

        # Constraint 1: Case start time must be after start time of assigned theatre session
        def case_start_time(model, case, session):
            return model.CASE_START_TIME[case, session] >= model.SESSION_START_TIME[session] - ((1 - model.SESSION_ASSIGNED[(case, session)])*model.M)
        model.CASE_START = pe.Constraint(model.TASKS, rule=case_start_time)

        # Constraint 2: Case end time must be before end time of assigned theatre session
        def case_end_time(model, case, session):
            return model.CASE_START_TIME[case, session] + model.CASE_DURATION[case] <= \
                model.SESSION_START_TIME[session] + model.SESSION_DURATION[session] + ((1 - model.SESSION_ASSIGNED[(case, session)]) * model.M)
        model.CASE_END_TIME = pe.Constraint(model.TASKS, rule=case_end_time)

        # Constraint 3: Cases can be assigned to a maximum of one session
        def session_assignment(model, case):
            return sum([model.SESSION_ASSIGNED[(case, session)] for session in model.SESSIONS]) <= 1
        model.SESSION_ASSIGNMENT = pe.Constraint(model.CASES, rule=session_assignment)

        # Constraint 4: Cases must be completed before their target deadline
        def set_deadline_condition(model, case, session):
            return model.SESSION_DATES[session] <= model.CASE_DEADLINES[case] + ((1 - model.SESSION_ASSIGNED[case, session])*model.M)
        model.APPLY_DEADLINE = pe.Constraint(model.TASKS, rule=set_deadline_condition)

        # Constraint 5: No two cases can overlap
        def no_case_overlap(model, case1, case2, session):
            return [model.CASE_START_TIME[case1, session] + model.CASE_DURATION[case1] <= model.CASE_START_TIME[case2, session] + \
                    ((2 - model.SESSION_ASSIGNED[case1, session] - model.SESSION_ASSIGNED[case2, session])*model.M),
                    model.CASE_START_TIME[case2, session] + model.CASE_DURATION[case2] <= model.CASE_START_TIME[case1, session] + \
                        ((2 - model.SESSION_ASSIGNED[case1, session] - model.SESSION_ASSIGNED[case2, session])*model.M)]
        model.DISJUNCTIONS_RULE = pyogdp.Disjunction(model.DISJUNCTIONS, rule=no_case_overlap)

        # Constraint 6: Utilisation is defined as fraction of a theatre session taken up by cases
        def theatre_util(model, session):
            return model.UTILISATION[session] == (1 / model.SESSION_DURATION[session]) * sum([model.SESSION_ASSIGNED[case, session]*model.CASE_DURATION[case] for case in model.CASES])
        model.THEATRE_UTIL = pe.Constraint(model.SESSIONS, rule=theatre_util)

        pe.TransformationFactory("gdp.bigm").apply_to(model)
        
        return model


    def solve(self, solver_name, options=None):
        
        solver = pe.SolverFactory(solver_name)

        if options is not None:
            for key, value in options.items():
                solver.options[key] = value

        solver.solve(self.model, tee=True)

        results = [{"Case": case,
                    "Session": session,
                    "Session Date": self.model.SESSION_DATES[session],
                    "Case Deadline": self.model.CASE_DEADLINES[case],
                    "Days before deadline": self.model.CASE_DEADLINES[case] - self.model.SESSION_DATES[session],
                    "Start": self.model.CASE_START_TIME[case, session](),
                    "Assignment": self.model.SESSION_ASSIGNED[case, session]()}
                   for (case, session) in self.model.TASKS]
       
        self.df_times = pd.DataFrame(results)

        all_cases = self.model.CASES.value_list
        print(self.df_times.to_string())
        cases_assigned = []
        for (case, session) in self.model.SESSION_ASSIGNED:
            if self.model.SESSION_ASSIGNED[case, session]() == 1.0:
                cases_assigned.append(case)
        
        cases_missed = list(set(all_cases).difference(cases_assigned))
        print("Number of cases assigned = {} out of {}:".format(len(cases_assigned), len(all_cases)))
        print("Cases assigned: ", cases_assigned)
        print("Number of cases missed = {} out of {}:".format(len(cases_missed), len(all_cases)))
        print("Cases missed: ", cases_missed)
        print(self.df_times[self.df_times["Assignment"] == 1].to_string())


if __name__ == "__main__":
    case_path = "C:\\Users\\Andreas\\Documents\\India\\Optimization Models\\Project\\cases.csv"
    session_path = "C:\\Users\\Andreas\\Documents\\India\\Optimization Models\\Project\\sessions.csv"
    
    options = {"tmlim": 300}
    scheduler = TheatreScheduler(case_file_path=case_path, session_file_path=session_path)
    scheduler.solve(solver_name="glpk", options=options)
