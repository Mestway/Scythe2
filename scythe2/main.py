import json
import pandas as pd

from scythe2.sql import *


def main():
	test_data = [{"Totals":7,"Value":"A","variable":"alpha","value":2,"cumsum":2},
			 {"Totals":8,"Value":"B","variable":"alpha","value":2,"cumsum":2},
			 {"Totals":9,"Value":"C","variable":"alpha","value":3,"cumsum":3},
			 {"Totals":9,"Value":"D","variable":"alpha","value":3,"cumsum":3},
			 {"Totals":9,"Value":"E","variable":"alpha","value":4,"cumsum":4},
			 {"Totals":7,"Value":"A","variable":"beta","value":2,"cumsum":4},
			 {"Totals":8,"Value":"B","variable":"beta","value":3,"cumsum":5},
			 {"Totals":9,"Value":"C","variable":"beta","value":3,"cumsum":6},
			 {"Totals":9,"Value":"D","variable":"beta","value":4,"cumsum":7},
			 {"Totals":9,"Value":"E","variable":"beta","value":3,"cumsum":7},
			 {"Totals":7,"Value":"A","variable":"gamma","value":3,"cumsum":7},
			 {"Totals":8,"Value":"B","variable":"gamma","value":3,"cumsum":8},
			 {"Totals":9,"Value":"C","variable":"gamma","value":3,"cumsum":9},
			 {"Totals":9,"Value":"D","variable":"gamma","value":2,"cumsum":9},
			 {"Totals":9,"Value":"E","variable":"gamma","value":2,"cumsum":9}]

	df = pd.DataFrame(test_data)

	bv = [True, False, True, True, True, False, True, True, True, False, True, True, True, False, True]

	print(df)

	print(df[bv])

if __name__ == '__main__':
	main()