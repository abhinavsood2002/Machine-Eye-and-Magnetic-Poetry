class ExponentialWeightedMovingAverage:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_average = None

    def add_value(self, new_value):
        if self.previous_average is None:
            self.previous_average = new_value
        else:
            self.previous_average = self.alpha * new_value + (1 - self.alpha) * self.previous_average

    def get_average(self):
        return self.previous_average if self.previous_average is not None else 0
