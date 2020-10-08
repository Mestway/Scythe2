import pandas as pd

DEBUG_SYNTH = True

def op_to_lambda(op):
	if op == "==":
		return lambda x, y: x == y
	if op == ">=":
		return lambda x, y: x >= y
	if op == "<=":
		return lambda x, y: x <= y
	if op == ">":
		return lambda x, y: x > y
	if op == "<":
		return lambda x, y: x < y
	if op == "!=":
		return lambda x, y: x != y

def pred_to_str(df, pred):
	"""predicate to str"""
	primitive_pred_to_str = lambda p: f"{df.columns[p[0]]} {p[1]} {p[2] if p[3] == 'value' else df.columns[p[2]]} ({p[4]})"  
	if isinstance(pred, list):
		return " and ".join([primitive_pred_to_str(p) for p in pred])
	else:
		return primitive_pred_to_str(pred)

def filter_to_bv(df, primitive_preds):
	"""convert filter predicates to bit-vectors """

	bv_to_preds = {}
	for p in primitive_preds:
		if p[3] == "column":
			filter_exp = lambda x: op_to_lambda(p[1])(x[df.columns[p[0]]], x[df.columns[p[2]]])
		elif p[3] == "value":
			filter_exp = lambda x: op_to_lambda(p[1])(x[df.columns[p[0]]], p[2])
		else:
			print(f"[ERROR] unknown predicate type {p[3]}")

		bv = tuple(df.apply(filter_exp, axis=1))

		if bv not in bv_to_preds:
			bv_to_preds[bv] = []
		bv_to_preds[bv].append(p)

	return bv_to_preds


def enum_compound_preds(df, table_boundaries, bv_to_preds):
	"""enumerate compound predicates from primitives """

	n = len(table_boundaries)

	max_size = int(2 * ( n + (n * n - n) / 2 ))

	# used to store existing bv's, 
	# if we can already achieve a certain bv in simpler form, 
	# 	we don't need to try out more complex ones
	all_bvs = [bv for bv in bv_to_preds]

	# structure of this: {size : { bv: [tr, ...]}}
	#	{
	#       1: { bv1: [tr11, tr12, ...], bv2: [tr21, tr22, ...]},
	#       2: ...,
	# 	    ....
	#  }
	bv_by_size_w_trace = {1: bv_to_preds}

	for size in range(2, max_size + 1):
		new_bvs = {}

		for lastest_bv in bv_by_size_w_trace[size - 1]:
			trace = bv_by_size_w_trace[size - 1][lastest_bv]

			for bv in bv_to_preds:

				new_bv = tuple([ lastest_bv[i] and bv[i] for i in range(len(bv)) ])

				if new_bv in all_bvs:
					# this bv can already be achieved by simpler filter predicates
					continue

				if new_bv not in new_bvs:
					new_bvs[new_bv] = []

				new_bvs[new_bv].append({"left": trace, "right": bv_to_preds[bv]})

		for bv in new_bvs:
			all_bvs.append(bv)

		bv_by_size_w_trace[size] = new_bvs

	return bv_by_size_w_trace


def check_pred(plist):
	
	tags = []
	for p in plist:
		tag = p[4]
		if ":" in tag:
			for t in tag.split(":"):
				tags.append(t) 
		else:
			tags.append(tag) 

	tag_cnt = {}
	for tag in tags:
		if tag not in tag_cnt:
			tag_cnt[tag] = 0
		tag_cnt[tag] += 1
	
	# only allow each constant to be used once
	# only allow length 2 predicates
	return all([tag_cnt[t] <= (1 if "const" in t else 2) for t in tag_cnt])


def instantiate_predicate(df, bv_by_size_w_trace, bv, beam_size=10):
	"""Given stored bv predicates (with trace), and a target bv, retrieve it's source
	Args:
		df: the datatrame
		bv_by_size_w_trace: a dictionary stored all compound predicates 
			together with source (generated from enum_compound_preds)
			structure of this: {size : { bv: [tr, ...]}}
			{
			    1: { bv1: [tr11, tr12, ...], bv2: [tr21, tr22, ...]},
			    2: ...,
				....
			 }
		bv: the target bv we want to extract
		beam_size: the number of candidates we want to extract
	"""
	
	def recursive_extract(tr, beam_size):

		left = []
		for sub_tr in tr["left"]:
			if isinstance(sub_tr, tuple):
				# it is already a primitive predicate, we don't need to furthur expand
				left += [[sub_tr]]
			else:
				# it is composed from simpler traces, recursively decompose it
				left += recursive_extract(sub_tr, beam_size)

		right = tr["right"]

		# combine left handside predicates with right handside predicates
		conj_preds = []
		for plist in left:
			for p2 in right:
				p = plist + [p2]

				if check_pred(p):
					conj_preds.append(p)

				if len(conj_preds) > beam_size:
					return conj_preds

		return conj_preds

	# starting from the current trace, extract it
	pred_list = []
	for size in bv_by_size_w_trace:
		if bv in bv_by_size_w_trace[size]:
			for tr in bv_by_size_w_trace[size][bv]:
				pred_list += recursive_extract(tr, beam_size)
				if len(pred_list) >= beam_size:
					break

	return pred_list


def enum_primitive_preds(df, dtypes, table_boundaries, constants):

	str_constants = [x for x in constants if isinstance(x, str)]
	num_constants = [x for x in constants if not isinstance(x, str)]

	primitives = []
	join_primitives = []

	table_boundaries = table_boundaries + [len(dtypes)]

	# enumerate within boundary operators
	for i in range(len(table_boundaries) - 1):

		# which segment of the table schema corresponds to this table
		l = table_boundaries[i]
		r = table_boundaries[i + 1] 

		for col_index in range(l, r):
			if dtypes[col_index] == "number":
				for num_val in num_constants:
					for op in ["==", ">", "<", ">=", "<="]:
						primitives.append((col_index, op, num_val, "value", f"table_{l}:const_{num_val}"))
			if dtypes[col_index] == "string":
				for str_val in str_constants:
					primitives.append((col_index, "==", str_val, "value", f"table_{l}:const_{str_val}"))

		# enumerate column comparision within tables
		for c1 in range(l, r):
			for c2 in range(c1 + 1, r):
				if dtypes[c1] == dtypes[c2]:
					if dtypes[c1] == "number":
						for op in ["==", ">", "<", ">=", "<="]:
							primitives.append((c1, op, c2, "column", f"table_{l}"))
					if dtypes[c1] == "string":
						primitives.append((c1, "==", c2, "column", f"table_{l}"))

	# enumerate join predicates
	for i in range(len(table_boundaries) - 1):
		for j in range(i + 1, len(table_boundaries) - 1):
			for c1 in range(table_boundaries[i], table_boundaries[i + 1]):
				for c2 in range(table_boundaries[j], table_boundaries[j + 1]):
					if dtypes[c1] == dtypes[c2]:
						# temporarily only enable equi-join
						join_primitives.append((c1, "==", c2, "column", f"join_{table_boundaries[i]}_{table_boundaries[j]}"))

	return primitives + join_primitives


def enum_bv_predicates(df, dtypes, table_boundaries, constants, print_bvs=False):
	"""enumerate bv predicates for the data"""

	# enumerate primitive predicates
	primitive_preds = enum_primitive_preds(df, dtypes, table_boundaries, constants)

	# store them as bit-vectors of the original df
	bv_to_preds = filter_to_bv(df, primitive_preds)

	# enumerate compound predicates
	bv_by_size_w_trace = enum_compound_preds(df, table_boundaries, bv_to_preds)

	if print_bvs:
		# print bv's
		print("======= primitive bv's and their syntactical form =======")
		for bv in bv_to_preds:
			print("".join(["|" if x else "." for x in bv]))
			for p in bv_to_preds[bv]:
				print(f'  {pred_to_str(df, p)}')

		print("\n======= Compound predicates generated =======")
		for size in bv_by_size_w_trace:
			print(f"## size {size}")
			for bv in bv_by_size_w_trace[size]:
				trace = bv_by_size_w_trace[size][bv]
				print(f'  {"".join(["|" if x else "." for x in bv])}  {len(trace)}')

	return bv_by_size_w_trace


from scythe2.sql import *

if __name__ == '__main__':

	inp = [{"Totals":7,"Value":"A","variable":"alpha","value":2,"cumsum":2},
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

	q = Table(data_id=0)
	q1 = Aggregate(q, [1], -1, "max")
	q2 = Table(data_id=0)
	q =  Join(q1, q2)
	
	inputs = {0: inp}

	dtypes, table_boundary = q.infer_output_info(inputs)

	print(dtypes)
	print(table_boundary)

	df = q.eval(inputs=inputs)

	#print(df)

	bv_by_size_w_trace = enum_bv_predicates(df, dtypes, table_boundary, constants=["A"], print_bvs=True)

	# instantiate one predicate
	target = (False, False, False, False, False, True, False, False, False, False, False, False, False, 
			False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 
			False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 
			False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 
			False, False, False, False, False, False, False, False, False, False, False, False, False, False, False)
	pred_list = instantiate_predicate(df, bv_by_size_w_trace, target)

	print("")
	print(f'======= Predicate instantiated =======\n{"".join(["|" if x else "." for x in target])}')
	for p in pred_list:
		print(f"  {pred_to_str(df, p)}")

		