import numpy as np
import wfdb
import json
import time
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, ComprehensiveFCParameters
import tsfresh

def load_df(filename):    
    f = open(filename, "r")
    f_list = f.readlines()
    return (f_list)

def write_file(filename, lst):
    f = open(filename,'w')
    for i in lst:
        f.write(str(i))
        f.write('\n')
    f.close()

if __name__ == '__main__':
	STARTTIME= time.time()
	print (STARTTIME)
	with open('wf_id_matched_wf.txt', 'r') as file:
		wfid_list= json.load(file)
	wfid_sortedkey= sorted(list(wfid_list.keys()))
	#print (len(wfid_list.keys()))
	#print (wfid_sortedkey[:10])
	N= 5960
	start= 0
	end=  1
	#fn_out= "wfdict"+str(start)+"_"+str(end)+".txt"

	#my_dict= {}
	#my_dict= my_dict.fromkeys(wfid_sortedkey[start:end], {})
	#print (my_dict)

	for ind in range(start, end):
		last_sid= 0
		cur_sid= wfid_sortedkey[ind]
		for each_rec in wfid_list[cur_sid]:
			record= each_rec[:-1].split("/")[2]
			pbdir= "/".join(("mimic3wdb/matched/"+ each_rec[:-1]).split("/")[:-1])	
			try:
				rec = wfdb.rdrecord(record_name= record, pb_dir= pbdir)
				print ("interim time: ", time.time())
				qwer= tsfresh.utilities.dataframe_functions.impute_dataframe_zero(df)
				extract_qwer= extract_features(qwer, column_id= "id", default_fc_parameters=ComprehensiveFCParameters())
				print ("done extracting qwer: ", time.time())

				last_sid= cur_sid
				#print ("went thru")
			except:
				#print ("???")
				print ("exception caught")
				pass

	fn_out= "output.txt"
	with open(fn_out, 'w') as file:
		file.write(json.dumps(extract_qwer))

	ENDTIME= time.time()
	print ("num of records: ", len(my_dict.keys()))
	print ("lapse time: ", ENDTIME-STARTTIME)