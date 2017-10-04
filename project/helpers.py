def get_data(filename):
	f = open(filename, 'r')
	a = []
	for l in f:
		line = eval(l)
		a.append((line["reviewText"], int(line["overall"])))
	return a
