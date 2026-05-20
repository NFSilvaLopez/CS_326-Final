# astar_engine.py
# A* Search Engine for Heart Disease Diagnosis

import heapq
import copy


# -----------------------------
# State Class
# -----------------------------
class State:
    def __init__(self, rules_triggered=None, path=None, cost=0):
        self.rules_triggered = rules_triggered if rules_triggered else []
        self.path = path if path else ["Start"]
        self.cost = cost          # g(n)
        self.heuristic = 0        # h(n)
        self.score = 0           # f(n)

    def __lt__(self, other):
        return self.score < other.score


# -----------------------------
# Heuristic Function
# Uses Random Forest Probability
# -----------------------------
def heuristic(patient_data, model):
    risk = model.predict_proba(patient_data)[0][1] * 100
    return round(100 - risk, 2)


def state_priority(state):
    return (
        len(state.rules_triggered),
        state.cost,
        -state.heuristic
    )


def build_result(state, goal_reached):
    return {
        "path": state.path,
        "rules": state.rules_triggered,
        "cost": state.cost,
        "heuristic": state.heuristic,
        "score": state.score,
        "goal_reached": goal_reached
    }


# -----------------------------
# Goal Test
# -----------------------------
def goal_test(state, patient_data, model):
    risk = model.predict_proba(patient_data)[0][1] * 100

    major_rules = [
        "blocked_vessels",
        "asymptomatic_pain",
        "bad_slope",
        "exercise_angina"
    ]

    count = sum(1 for rule in state.rules_triggered if rule in major_rules)

    if count >= 3:
        return True

    if risk >= 80:
        return True

    return False


# -----------------------------
# Expand States
# -----------------------------
def expand_state(state, patient):
    children = []

    # Rule 1: blocked vessels
    if patient["ca"] >= 2 and "blocked_vessels" not in state.rules_triggered:
        child = copy.deepcopy(state)
        child.rules_triggered.append("blocked_vessels")
        child.path.append("Detected 2+ blocked vessels")
        children.append(child)

    # Rule 2: asymptomatic chest pain
    if patient["cp"] == 3 and "asymptomatic_pain" not in state.rules_triggered:
        child = copy.deepcopy(state)
        child.rules_triggered.append("asymptomatic_pain")
        child.path.append("Detected asymptomatic chest pain")
        children.append(child)

    # Rule 3: abnormal slope
    if patient["slope"] == 2 and "bad_slope" not in state.rules_triggered:
        child = copy.deepcopy(state)
        child.rules_triggered.append("bad_slope")
        child.path.append("Detected downsloping ECG response")
        children.append(child)

    # Rule 4: high cholesterol
    if patient["chol"] > 240 and "high_cholesterol" not in state.rules_triggered:
        child = copy.deepcopy(state)
        child.rules_triggered.append("high_cholesterol")
        child.path.append("Detected high cholesterol")
        children.append(child)

    # Rule 5: age risk
    if patient["age"] > 55 and "age_risk" not in state.rules_triggered:
        child = copy.deepcopy(state)
        child.rules_triggered.append("age_risk")
        child.path.append("Detected age-related cardiac risk")
        children.append(child)

    # Rule 6: exercise induced angina
    if patient["exang"] == 1 and "exercise_angina" not in state.rules_triggered:
        child = copy.deepcopy(state)
        child.rules_triggered.append("exercise_angina")
        child.path.append("Detected exercise-induced angina")
        children.append(child)

    return children


# -----------------------------
# A* Search
# -----------------------------
def astar_search(patient_data, model):

    patient = patient_data.iloc[0]

    open_list = []
    visited = set()

    start = State()
    start.heuristic = heuristic(patient_data, model)
    start.score = start.cost + start.heuristic
    best_state = copy.deepcopy(start)
    best_goal_state = None

    heapq.heappush(open_list, start)

    while open_list:

        current = heapq.heappop(open_list)

        state_key = tuple(sorted(current.rules_triggered))
        if state_key in visited:
            continue

        visited.add(state_key)

        if state_priority(current) > state_priority(best_state):
            best_state = copy.deepcopy(current)

        if goal_test(current, patient_data, model):
            if (
                best_goal_state is None or
                state_priority(current) > state_priority(best_goal_state)
            ):
                best_goal_state = copy.deepcopy(current)

        children = expand_state(current, patient)

        for child in children:
            child.cost = current.cost + 1
            child.heuristic = heuristic(patient_data, model)
            child.score = child.cost + child.heuristic

            heapq.heappush(open_list, child)

    if best_goal_state is not None:
        return build_result(best_goal_state, True)

    return build_result(best_state, False)
