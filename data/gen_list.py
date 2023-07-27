import os

dataset = 'DomainNet'

if dataset == 'office':
	domains = ['amazon', 'dslr', 'webcam']
elif dataset == 'office-caltech':
	domains = ['amazon', 'dslr', 'webcam', 'caltech']
elif dataset == 'office-home':
	domains = ['Art', 'Clipart', 'Product', 'Real_World']
elif dataset =='DomainNet':
	domains=['clipart','infograph','painting','quickdraw','real','sketch']
else:
	print('No such dataset exists!')



for domain in domains:
	log = open(dataset+'/'+domain+'_list.txt','w')
	# office
	# directory = os.path.join(dataset, os.path.join(domain,'images'))
	# office-home,office-caltech
	directory = os.path.join(dataset, os.path.join(domain))

	classes = [x[0] for x in os.walk(directory)]
	classes = classes[1:]
	classes.sort()
	for idx,f in enumerate(classes):
		files = os.listdir(f)
		for file in files:
			s = os.path.abspath(os.path.join(f,file)) + ' ' + str(idx) + '\n'
			log.write(s)
	log.close()

# python gen_list.py
