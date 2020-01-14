test_label = open("test_label","w")
with open("answer0.txt",'r') as fp:
	fp.readline()
	for a in fp.readlines():
		import re
		m = re.findall(r"[0-9]*\s*([0-1])\s*[0-1]",a)
		print(m[0])
		test_label.write(m[0]+'\n')

test_label.close()