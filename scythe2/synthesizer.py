import sys
import traceback
import copy
from pprint import pprint
import pandas as pd
import time

from scythe2.sql import HOLE, Node, Table, Project, Filter, BVFilter, Join, Aggregate
from scythe2.synth_utils import *

abstract_combinators = {
	"project": lambda q: Project(q, cols=HOLE),
	"filter": lambda q: BVFilter(q, bv=HOLE),
	"join": lambda q1, q2: Join(q1, q2),
	"aggregate": lambda q: Aggregate(q, group_cols=HOLE, aggr_col=HOLE, aggr_func=HOLE),
	"filter_aggregate": lambda q: Aggregate(BVFilter(q, bv=HOLE), group_cols=HOLE, aggr_col=HOLE, aggr_func=HOLE),
}

def update_tree_value(node, path, new_val):
	"""from a given ast node, locate the refence to the arg,
	   and update the value"""
	for k in path:
		node = node["children"][k]
	node["value"] = new_val

def get_node(node, path):
	for k in path:
		node = node["children"][k]
	return node

class Synthesizer(object):

	def __init__(self, config=None):
		if config is None:
			self.config = {
				"operators": ["join", "filter_aggregate"],
				"filer_op": [">", "<", "=="],
				"constants": [],
				"aggr_func": ["max"]#["sum", "count", "max", "min"]
			}
		else:
			self.config = config

	def enum_sketches(self, inputs, output, size):
		"""enumerate program sketches up to the given size"""

		# check if output contains a new value 
		# (this decides if we should use ops that generates new vals)
		
		inp_val_set = set([v for t in inputs for r in t for k, v in r.items()] + [k for t in inputs for k in t[0]])
		out_val_set = set([v for r in output for k, v in r.items()])
		new_vals = out_val_set - inp_val_set
		
		candidates = {}
		for level in range(0, size + 3):
			candidates[level] = []

		for level in range(0, size + 1):
			
			if level == 0:
				candidates[level] += [Table(data_id=i) for i in range(len(inputs))]
			else:
				for op in abstract_combinators:
					#ignore operators that are not set
					if op not in self.config["operators"]:
						continue

					if op == "join":
						for q1 in candidates[level - 1]:
							for q2 in candidates[0]:
								q = abstract_combinators[op](copy.copy(q1), copy.copy(q2))
								candidates[level + 1].append(q)
							for q2 in candidates[1]:
								q = abstract_combinators[op](copy.copy(q1), copy.copy(q2))
								candidates[level + 2].append(q)


						# TODO: maybe we should support join multiple aggregation
					else:
						for q0 in candidates[level - 1]:
							q = abstract_combinators[op](copy.copy(q0))
							candidates[level].append(q)

		# for level in range(0, size + 1):
		# 	candidates[level] = [q for q in candidates[level] if not enum_strategies.disable_sketch(q, new_vals)]

		return candidates

	def pick_vars(self, ast, inputs):
		"""list paths to all holes in the given ast"""
		def get_paths_to_all_holes(node):
			results = []
			for i, child in enumerate(node["children"]):
				if child["type"] == "node":
					# try to find a variable to infer
					paths = get_paths_to_all_holes(child)
					for path in paths:
						results.append([i] + path)
				elif child["value"] == HOLE:
					# we find a variable to infer
					results.append([i])
			return results
		return get_paths_to_all_holes(ast)

	def infer_domain(self, ast, var_path, inputs):
		node = Node.load_from_dict(get_node(ast, var_path[:-1]))
		return node.infer_domain(arg_id=var_path[-1], inputs=inputs, config=self.config)

	def instantiate(self, ast, var_path, inputs):
		"""instantiate one hole in the program sketch"""
		domain = self.infer_domain(ast, var_path, inputs)
		candidates = []
		for val in domain:
			new_ast = copy.deepcopy(ast)
			update_tree_value(new_ast, var_path, val)
			candidates.append(new_ast)
		return candidates

	def instantiate_one_level(self, ast, inputs):
		"""generate program instantitated from the most recent level
			i.e., given an abstract program, it will enumerate all possible abstract programs that concretize
		"""
		var_paths = self.pick_vars(ast, inputs)

		# there is no variables to instantiate
		if var_paths == []:
			return [], []

		# find all variables at the innermost level
		innermost_level = max([len(p) for p in var_paths])
		target_vars = [p for p in var_paths if len(p) == innermost_level]

		recent_candidates = [ast]
		for var_path in target_vars:
			temp_candidates = []
			for partial_prog in recent_candidates:
				temp_candidates += self.instantiate(partial_prog, var_path, inputs)
			recent_candidates = temp_candidates

		# for c in recent_candidates:
		# 	nd = Node.load_from_dict(c)
		# 	print(f"{' | '}{nd.stmt_string()}")
		
		# this show how do we trace to the most recent program level
		concrete_program_level = innermost_level - 1

		return recent_candidates, concrete_program_level

	
	def try_instantiate_proj_and_filter(self, q, inputs, output_table):
		""" given a candidate program q, input tables inputs, and an output table, decide:
			(1) if otuput_table can be obtained by filtering and projection on q(inputs)
			(2) if yes, return candidate projections and filters
		"""

		constants = self.config["constants"]

		dtypes, table_boundaries = q.infer_output_info(inputs)
		df = q.eval(inputs)

		schema_alignments = align_table_schema(output_table, df.to_dict(orient="records"), find_all_alignments=True)

		if not schema_alignments:
			# there is no valid schema alignment exists
			return [] 

		bv_by_size_w_trace = filter_synth.enum_bv_predicates(df, dtypes, table_boundaries, constants, print_bvs=False)

		output_schema = [key for key in output_table[0]]
		proj_filter_pairs = []
		for schema_map in schema_alignments:
			projected_df = df[[schema_map[h] for h in output_schema]]
			bv_map = infer_filter_bv(projected_df.to_dict(orient="records"), output_table)
			#print(bv_map)

			successful_bvs = []
			for size in bv_by_size_w_trace:
				for bv in bv_by_size_w_trace[size]:

					# if the projection is 
					if sum(bv) != len(output_table):
						continue
					else:
						if all([any([bv[i] for i in bv_map[out_row_id]]) for out_row_id in bv_map]):
							#print(bv)
							successful_bvs.append(bv)

			for bv in successful_bvs:
				proj_filter_pairs.append((schema_map, bv))

			if len(proj_filter_pairs) > 0:
				break

		candidates = []
		if len(proj_filter_pairs) > 0:

			# for each projection and bv_filter, intantiate a set of queries 
			for schema_map, bv in proj_filter_pairs:
				# print("*******************")
				# print(schema_map)
				# print("".join(["|" if x else "." for x in bv]))
				proj_index = [list(df.columns).index(schema_map[c]) for c in schema_map]
				#print(proj_index)

				# instantiate concrete predicates from the bvfilter
				pred_lists = filter_synth.instantiate_predicate(df, bv_by_size_w_trace, bv, beam_size=10)


				for concrete_q in q.replace_bvfilter_with_filter(inputs, constants, max_out_cnt=10):
					for pred in pred_lists:
						#print(filter_synth.pred_to_str(df,pred))
						candidates.append(Project(Filter(copy.deepcopy(concrete_q), pred), proj_index))

			return candidates

		return []


	def iteratively_instantiate_and_print(self, p, inputs, level, print_programs=False):
		"""iteratively instantiate a program (for the purpose of debugging)"""
		if print_programs:
			print(f"{'  '.join(['' for _ in range(level)])}{p.stmt_string()}")

		results = []
		if p.is_abstract():
			ast = p.to_dict()
			var_path = self.pick_vars(ast, inputs)[0]
			#domain = self.infer_domain(ast, path, inputs)
			candidates = self.instantiate(ast, var_path, inputs)

			for c in candidates:
				nd = Node.load_from_dict(c)
				results += self.iteratively_instantiate_and_print(nd, inputs, level + 1, print_programs)
			return results
		else:
			return [p]

	def iteratively_instantiate(self, p, inputs, level, print_programs=False):
		"""iteratively instantiate a program (for the purpose of debugging)"""
		if print_programs:
			print(f"{'  '.join(['' for _ in range(level)])}{p.stmt_string()}")

		results = []
		if p.is_abstract():
			ast = p.to_dict()
			
			candidates, _ = self.instantiate_one_level(ast, inputs)
			for c in candidates:
				nd = Node.load_from_dict(c)
				results += self.iteratively_instantiate(nd, inputs, level + 1, print_programs)
			return results
		else:
			return [p]

	def enumerative_search(self, inputs, output, max_prog_size):
		"""Given inputs and output, enumerate all programs in the search space until 
			find a solution p such that output ⊆ subseteq p(inputs)  """
		all_sketches = self.enum_sketches(inputs, output, size=max_prog_size)

		candidates = []
		for level, sketches in all_sketches.items():
			for s in sketches:
				concrete_programs = self.iteratively_instantiate(s, inputs, 1, True)

				for p in concrete_programs:
					queries = self.try_instantiate_proj_and_filter(p, inputs, output)
					if queries:
						for q in queries:
							#print(q.stmt_string())
							#print(q.eval(inputs))
							candidates.append(q)

				if len(candidates) > 0:
					return candidates
						
		return candidates
