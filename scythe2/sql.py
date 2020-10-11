import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import copy
import itertools
from collections import namedtuple

from scythe2.filter_synth import *


# special symbol used in the language
HOLE = "_?_"

class Node(ABC):
	def __init__(self):
		super(AbstractExpression, self).__init__()

	@abstractmethod
	def eval(self, inputs):
		"""the inputs are dataframes,
			it returns a pandas dataframe representation"""
		pass

	@abstractmethod
	def to_dict(self):
		pass

	@abstractmethod
	def infer_domain(self, arg_id, config):
		pass

	@abstractmethod
	def infer_output_info(self, inputs):
		pass

	@abstractmethod
	def replace_bvfilter_with_filter(self, inputs, constants, max_out_cnt):
		pass

	@staticmethod
	def load_from_dict(ast):
		"""given a dictionary represented AST, load it in to a program form"""
		constructors = {
			"project": Project,
			"filter": Filter,
			"bv_filter": BVFilter,
			"aggregate": Aggregate,
			"join": Join
		}
		if ast["op"] == "table_ref":
			return Table(ast["children"][0]["value"])
		else:
			if ast["op"] == "join":
				node = constructors[ast["op"]](
							Node.load_from_dict(ast["children"][0]), 
							Node.load_from_dict(ast["children"][1]))
			else:
				node = constructors[ast["op"]](
							Node.load_from_dict(ast["children"][0]), 
							*[arg["value"] for arg in ast["children"][1:]])
			return node

	def to_stmt_dict(self):
		"""translate the expression into a  """
		def _recursive_translate(ast, used_vars):

			#print(ast)

			if ast["op"] == "table_ref":
				# create a variable to capture the return variable
				stmt_dict = copy.copy(ast)
				var = get_temp_var(used_vars)
				stmt_dict["return_as"] = var
				return [stmt_dict], used_vars + [var]
			else:
				stmt_dict = copy.copy(ast)

				# iterate over all possible subtrees
				sub_tree_stmts = []	
				for i, arg in enumerate(ast["children"]):
					# check if the argument is an ast 
					if isinstance(arg, (dict,)) and arg["type"] == "node":
						stmts, used_vars = _recursive_translate(ast["children"][i], used_vars)
						sub_tree_stmts += stmts
						# the subtree is replaced by a reference to the variable
						retvar = stmts[-1]["return_as"]
						stmt_dict["children"][i] = {"value": retvar, "type": "variable"}
				
				# use a temp variable to wrap the current statement, and add it to the coolection
				var = get_temp_var(used_vars)
				stmt_dict["return_as"] = var
				return sub_tree_stmts + [stmt_dict], used_vars + [var]

		stmts, _ = _recursive_translate(self.to_dict(), [])
		return stmts

	def is_abstract(self):
		"""Check if the subtree is abstract (contains any holes)"""
		def contains_hole(node):
			for i, arg in enumerate(node["children"]):
				if arg["type"] == "node":
					if contains_hole(arg):
						return True
				elif arg["value"] == HOLE:
					# we find a variable to infer
					return True
			return False
		return contains_hole(self.to_dict())
	
	def stmt_string(self):
		"""generate a string from stmts, for the purpose of pretty printing"""

		def val_to_str(x):
			if x["value"] == HOLE:
				return "?"
			if x["type"] == "bv_filter":
				return "".join(["|" if k else "." for k in x["value"][:min(5, len(x["value"]))]])
			# if x["type"] == "predicates":
			# 	return pred_to_str(x["value"], hide_type=True)
			else:
				return str(x["value"])

		stmts = self.to_stmt_dict()
		result = []
		for s in stmts:
			lhs = s['return_as']
			f = s['op']
			arg_str = ', '.join([val_to_str(x) for x in s["children"]])
			result.append(f"{lhs} <- {f}({arg_str})")
		return "; ".join(result)


class Table(Node):
	def __init__(self, data_id):
		self.data_id = data_id

	def infer_domain(self, arg_id, inputs, config):
		assert False, "Table has no args to infer domain."

	def infer_output_info(self, inputs):
		"""infer output schema """
		inp = inputs[self.data_id]
		if isinstance(inp, (list,)):
			columns = [key for key in inp[0]]
			df = pd.DataFrame.from_dict(inp)[list(inp[0].keys())]
		else:
			df = inp
		dtypes = extract_table_schema(df)

		# table_boundaries capture join boundaires
		table_boundaries = [0]

		return dtypes, table_boundaries

	def eval(self, inputs):
		inp = inputs[self.data_id]
		if isinstance(inp, (list,)):
			df = pd.DataFrame.from_dict(inp)[list(inp[0].keys())]
		else:
			df = inp
		return df

	def replace_bvfilter_with_filter(self, inputs, constants, max_out_cnt):
		return [Table(self.data_id)]

	def to_dict(self):
		return {
			"type": "node",
			"op": "table_ref",
			"children": [ value_to_dict(self.data_id, "table_id") ]
		}


class Project(Node):
	def __init__(self, q, cols):
		self.q = q
		self.cols = cols

	def infer_domain(self, arg_id, inputs, config):
		if arg_id == 1:
			input_schema, _ = self.q.infer_output_info(inputs)
			col_num = len(input_schema)
			col_list_candidates = []
			for size in range(1, col_num + 1):
				col_list_candidates += list(itertools.combinations(list(range(col_num)), size))
			return col_list_candidates
		else:
			assert False, "[Project] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		schema, table_boundaries = self.q.infer_output_info(inputs)
		new_table_boundaries = list(set([len([x for x in self.cols if x < i]) for i in table_boundaries]))
		return [s for i, s in enumerate(schema) if i in self.cols], new_table_boundaries

	def eval(self, inputs):
		df = self.q.eval(inputs)
		return df[[df.columns[i] for i in self.cols]]

	def backward_eval(self, output):
		# the input table should contain every value appear in the output table
		return [output]

	def replace_bvfilter_with_filter(self, inputs, constants, max_out_cnt):
		subq_list = self.q.replace_bvfilter_with_filter(inputs, constants, max_out_cnt)
		q_list = []
		for q in subq_list:
			q_list.append(Project(q, self.cols))
		return q_list

	def to_dict(self):
		return {
			"type": "node",
			"op": "project",
			"children": [self.q.to_dict(), value_to_dict(self.cols, "col_index_list")]
		}


class Join(Node):

	def __init__(self, q1, q2):
		self.q1 = q1
		self.q2 = q2

	def infer_domain(self, arg_id, inputs, config):
		return None

	def infer_output_info(self, inputs):
		schema_1, table_boundaries_1 = self.q1.infer_output_info(inputs)
		schema_2, table_boundaries_2 = self.q2.infer_output_info(inputs)
		return schema_1 + schema_2, table_boundaries_1 + [x + len(schema_1) for x in table_boundaries_2]

	def eval(self, inputs):
		df1 = self.q1.eval(inputs)
		df2 = self.q2.eval(inputs)
		return (df1.assign(temp_join_key=1)
				.merge(df2.assign(temp_join_key=1), on="temp_join_key")
				.drop("temp_join_key", axis=1))

	def replace_bvfilter_with_filter(self, inputs, constants, max_out_cnt):
		subq1_list = self.q1.replace_bvfilter_with_filter(inputs, constants, max_out_cnt)
		subq2_list = self.q2.replace_bvfilter_with_filter(inputs, constants, max_out_cnt)
		q_list = []
		for q1 in subq1_list:
			for q2 in subq2_list:
				q_list.append(Join(q1, q2))
				if len(q_list) >= max_out_cnt:
					return q_list
		return q_list

	def to_dict(self):
		return {
			"type": "node",
			"op": "join",
			"children": [
				self.q1.to_dict(),
				self.q2.to_dict(),
			]
		}

# two types of predicates, the first one performs 
#	 p.type == "columm": column to column comparison
#	 p.type == "value": column to value comparison
Predicate = namedtuple('Predicate', ["left", 'op', 'right', 'type'])

class BVFilter(Node):
	def __init__(self, q, bv):
		self.q = q
		self.bv = bv

	def infer_domain(self, arg_id, inputs, config):

		df = self.q.eval(inputs)
		dtypes, table_boundaries = self.q.infer_output_info(inputs)

		bv_by_size_w_trace = enum_bv_predicates(df, dtypes, table_boundaries, constants=config["constants"])

		all_bvs = []
		for size in bv_by_size_w_trace:
			for bv in bv_by_size_w_trace[size]:
				if any(bv):
					#ignore all negative filters
					all_bvs.append(bv)

		return all_bvs

	def infer_output_info(self, inputs):
		return self.q.infer_output_info(inputs)

	def eval(self, inputs):
		df = self.q.eval(inputs)
		return df[list(self.bv)]

	def replace_bvfilter_with_filter(self, inputs, constants, max_out_cnt):

		if all(list(self.bv)):
			# the filter is all true, so there is no need to do filtering
			return [copy.deepcopy(self.q)]
		
		df = self.q.eval(inputs)
		dtypes, table_boundaries = self.q.infer_output_info(inputs)
		bv_by_size_w_trace = enum_bv_predicates(df, dtypes, table_boundaries, constants)

		concrete_preds = instantiate_predicate(df, bv_by_size_w_trace, self.bv, beam_size=max_out_cnt)

		subq_list = self.q.replace_bvfilter_with_filter(inputs, constants, max_out_cnt)

		q_list = []
		for q in subq_list:
			for preds in concrete_preds:
				q_list.append(Filter(q, preds))
				if len(q_list) >= max_out_cnt:
					return q_list

		return q_list

	def to_dict(self):
		return {
			"type": "node",
			"op": "bv_filter",
			"children": [
				self.q.to_dict(),
				value_to_dict(self.bv, "bv_filter")
			]}


class Filter(Node):
	def __init__(self, q, predicates):
		self.q = q
		self.predicates = predicates

	def infer_domain(self, arg_id, inputs, config):
		if arg_id == 1:
			pass
		else:
			assert False, "[Filter] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		return self.q.infer_output_info(inputs)

	def eval(self, inputs):
		df = self.q.eval(inputs)

		for p in self.predicates:
			left, op, right, ty = p[0], p[1], p[2], p[3]

			if ty == "column":
				filter_exp = lambda x: op_to_lambda(op)(x[df.columns[left]], x[df.columns[right]])
			elif ty == "value":
				filter_exp = lambda x: op_to_lambda(op)(x[df.columns[left]], right)
			else:
				print("[Error] Filter predicate argument is wrong")
				sys.exit(-1)

			df = df[df.apply(filter_exp, axis=1)]
		return df

	def replace_bvfilter_with_filter(self, inputs, constants, max_out_cnt):
		subq_list = self.q.replace_bvfilter_with_filter(inputs, constants, max_out_cnt)
		q_list = []
		for q in subq_list:
			q_list.append(Filter(q, self.predicates))
		return q_list

	def to_dict(self):
		return {
			"type": "node",
			"op": "filter",
			"children": [
				self.q.to_dict(),
				{
					"type": "predicates",
					"value": [p for p in self.predicates]
				}
			]}

class Aggregate(Node):
	def __init__(self, q, group_cols, aggr_col, aggr_func):
		self.q = q
		self.group_cols = group_cols
		self.aggr_col = aggr_col
		self.aggr_func = aggr_func

	def infer_domain(self, arg_id, inputs, config):
		schema, _ = self.q.infer_output_info(inputs)
		if arg_id == 1:
			# approximation: only get fields with more than one values
			# for the purpose of avoiding empty fields
			try:
				df = self.q.eval(inputs)
			except Exception as e:
				print(f"[eval error in infer_domain] {e}")
				return []

			# use this list to store primitive table keys, 
			# use them to elimiate column combinations that contain no duplicates
			table_keys = []

			col_num = len(schema)
			col_list_candidates = []
			for size in range(1, col_num + 1 - 1):
				for gb_keys in itertools.combinations(list(range(col_num)), size):
					if any([set(banned).issubset(set(gb_keys)) for banned in table_keys]):
						# current key group is subsumbed by a table key, so all fields will be distinct
						continue
					gb_cols = df[[df.columns[k] for k in gb_keys]]
					if not gb_cols.duplicated().any():
						# a key group is valid for aggregation 
						#   if there exists at least a key appear more than once
						table_keys.append(gb_keys)
						continue
		
					col_list_candidates += [gb_keys]
			return col_list_candidates
		elif arg_id == 2:
			number_fields = [i for i,s in enumerate(schema) if s == "number"]
			if self.group_cols != HOLE:
				cols = [i for i in number_fields if i not in self.group_cols]
			else:
				cols = number_fields
			# the special column -1 is used for the purpose of "count", no other real intent
			cols += [-1]
			return cols
		elif arg_id == 3:
			if self.aggr_col != HOLE:
				if self.aggr_col == -1:
					return ["count"] if "count" in config["aggr_func"] else []
				else:
					return [f for f in config["aggr_func"] if f != "count"]
			else:
				return config["aggr_func"]
		else:
			assert False, "[Aggregation] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		input_schema, _ = self.q.infer_output_info(inputs)
		aggr_type = input_schema[self.aggr_col] if self.aggr_func != "count" else "number"

		#after aggregation, join boundaries is no longer obvious
		table_boundaries = [0]

		return [s for i, s in enumerate(input_schema) if i in self.group_cols] + [aggr_type], table_boundaries

	def eval(self, inputs):
		df = self.q.eval(inputs)
		group_keys = [df.columns[idx] for idx in self.group_cols]
		target = df.columns[self.aggr_col]
		res = df.groupby(group_keys).agg({target: self.aggr_func})
		if self.aggr_func == "mean":
			res[target] = res[target].round(2)
		res = res.rename(columns={target: f'{self.aggr_func}_{target}'}).reset_index()
		return res

	def replace_bvfilter_with_filter(self, inputs, constants, max_out_cnt):
		subq_list = self.q.replace_bvfilter_with_filter(inputs, constants, max_out_cnt)
		q_list = []
		for q in subq_list:
			q_list.append(Aggregate(q, self.group_cols, self.aggr_col, self.aggr_func))
		return q_list

	def to_dict(self):
		return {
			"type": "node",
			"op": "aggregate",
			"children": [
				self.q.to_dict(), 
				value_to_dict(self.group_cols, "col_index_list"),
				value_to_dict(self.aggr_col, "col_index"), 
				value_to_dict(self.aggr_func, "aggr_func")
			]}

#utility functions

def get_temp_var(used_vars):
	"""get a temp variable name """
	for i in range(0, 1000):
		var_name = "t{}".format(i)
		if var_name not in used_vars:
			return var_name

def value_to_dict(val, val_type):
	"""given the value and its type, dump it to a dict 
		the helper function to dump values into dict ast
	"""
	return {"type": val_type, "value": val}

def extract_table_schema(df):
	"""Given a dataframe, extract it's schema """
	def dtype_mapping(dtype):
		"""map pandas datatype to c """
		dtype = str(dtype)
		if dtype == "object" or dtype == "string":
			return "string"
		elif "int" in dtype or "float" in dtype:
			return "number"
		elif "bool" in dtype:
			return "bool"
		else:
			print(f"[unknown type] {dtype}")
			sys.exit(-1)

	schema = [dtype_mapping(s) for s in df.infer_objects().dtypes]
	return schema