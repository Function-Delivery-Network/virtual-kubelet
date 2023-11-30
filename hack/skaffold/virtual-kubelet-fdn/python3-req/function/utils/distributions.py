""" Collection of different distributions for Early Exit"""
def pareto(ub, lb = 2, samples = 5):
  """ Generates indexes following a Pareto Distribution"""
  samples = [round((ub - lb)*(1-(0.8**(idx + 1)))) + lb for idx in range(samples)]
  return sorted(list(set(samples)))