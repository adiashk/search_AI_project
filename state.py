
class state:
    #keep track of states - the current path and its cost/value
    def __init__(self, path, value):
        self.path = path
        self.value = value