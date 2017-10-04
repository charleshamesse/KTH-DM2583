def get_data(filename):
	f = open(filename, 'r')
	a = []
	for l in f:
		line = eval(l)
		a.append((line["reviewText"], int(line["overall"])))
	return a

def split_text_rating(data):
	texts = [d[0] for d in data]
	ratings = [d[1] for d in data]

	return texts, ratings