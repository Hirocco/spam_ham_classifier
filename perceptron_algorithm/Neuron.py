# from feature_extraction.Vocabulary import vectorized_emails
class Neuron:
    def __init__(self, id, weight, bias=0, output=None, mail_input=None, is_activated=None):
        # Pierwsza wagi będą inicjowane losowo, jeżeli jesteśmy jużw innej warstwie niż pierwszej to będą zmieniane
        self.weight = weight
        self.input = mail_input
        self.id = id
        # suma elementów i wagi przez długość listy - pierwszy sposób obliczania biasu
        self.bias = bias
        self.output = output
        self.is_activated = is_activated

    def activation_func(self, output):
        if output > 1:
            return True
        else:
            return False

    def weighted_sum(self):
        # suma wazona ustawiamy na 0
        weighted_sum = 0.0 + self.weight

        # iteruje po liczbach w mailu
        for x in self.input:
            if x == -1:
                break
            weighted_sum += x * self.weight

        self.output = (weighted_sum - self.bias)
        # kalkulujemy czy aktywujemy neuron
        self.is_activated = self.activation_func(output=self.output)
        if self.is_activated:
            return self.output
        else:
            return 0
