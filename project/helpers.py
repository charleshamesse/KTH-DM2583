def get_data(filename):
	f = open(filename, 'r')
	a = []
	i = 0
	for l in f:
		try:
			line = eval(l)
			a.append((line["reviewText"], int(line["overall"])))
		except Exception as e:
			print(e, line, "\n#", i)
		i = i + 1
	return a

def split_text_rating(data):
	texts = [d[0] for d in data]
	ratings = [d[1] for d in data]

	return texts, ratings
