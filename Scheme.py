import abc


class Scheme:

    @abc.abstractmethod
    def insertion(self):
        pass

    @abc.abstractmethod
    def detection(self):
        pass
