import unittest

from scythe2.sql import *
from scythe2.synthesizer import *
import os
import json

from pprint import pprint

input_table = [{"Value":"A","variable":"alpha","value":2},
		{"Value":"B","variable":"alpha","value":2},
		{"Value":"C","variable":"alpha","value":3},
		{"Value":"D","variable":"alpha","value":3},
		{"Value":"E","variable":"alpha","value":4},
		{"Value":"A","variable":"beta","value":2},
		{"Value":"B","variable":"beta","value":3},
		{"Value":"C","variable":"beta","value":3},
		{"Value":"D","variable":"beta","value":4},
		{"Value":"E","variable":"beta","value":3},
		{"Value":"A","variable":"gamma","value":3},
		{"Value":"B","variable":"gamma","value":3},
		{"Value":"C","variable":"gamma","value":3},
		{"Value":"D","variable":"gamma","value":2},
		{"Value":"E","variable":"gamma","value":2}]

output_table = [{"variable_x":"alpha","max_value":4,"Value":"E"},
		  {"variable_x":"beta","max_value":4,"Value":"D"},
		  {"variable_x":"gamma","max_value":3,"Value":"A"},
		  {"variable_x":"gamma","max_value":3,"Value":"B"},
		  {"variable_x":"gamma","max_value":3,"Value":"C"}]

class TestSynthesizer(unittest.TestCase):
	
	def test1(self):
		synthesizer = Synthesizer()
		queries = synthesizer.enumerative_search([input_table], output_table, 2)

		print("----")
		print(f"number of programs: {len(queries)}")

		for q in queries:
			print(q.stmt_string())
			print(q.eval([input_table]))


if __name__ == '__main__':
	unittest.main()