from collections import namedtuple

Constants = namedtuple('Constants', ['c', 'epsilon_0'])

class TestConstants():
    def test1(self):
        c = Constants(1, 1)
        assert c.c == 1
        assert c.epsilon_0 == 1
    def test2(self):
        c = Constants(5, -3.45)
        assert c.c == 5
        assert c.epsilon_0 == -3.45