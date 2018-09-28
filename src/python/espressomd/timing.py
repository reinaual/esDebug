
timing = {}

def init():
    global timing
    timing = {"Setup": 0, "Output": 0, "Integration": 0, "Splitting": 0, "Reducing": 0, "ReduceRemoving": 0, "AddingTotal": 0, "AddingUpdate": 0, "AddingNew": 0, "Updating": 0}
