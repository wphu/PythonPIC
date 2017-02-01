class Constants():
    """
    Physical constants
    """
    def __init__(self, c : float = 1, epsilon_0 : float = 1):
        """
        :param float c: speed of light
        :param float epsilon_0: :math:`\epsilon_0`
        """
        self.c = c
        self.epsilon_0 = epsilon_0


class TestConstants():
    def test1(self):
        c = Constants()
        assert c.c == 1
        assert c.epsilon_0 == 1
    def test2(self):
        c = Constants(5, -3.45)
        assert c.c == 5
        assert c.epsilon_0 == -3.45