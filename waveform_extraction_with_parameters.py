import numpy as np
import wfdb
import json
import time
import sys
import argparse
import datetime
import tsfresh
import pandas as pd
from tsfresh.feature_extraction import extract_features

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

def cl_parse():
	parser= argparse.ArgumentParser()
	parser.add_argument("--start", default= 0, type= int)
	parser.add_argument("--end", default= 5960, type= int)

	args= parser.parse_args()
	start= args.start
	end= args.end
	if (start >end):
		print ("end must be greater than start")
		sys.exit(1)
	return (start, end)


def readin_wfid(fn_in):
	with open(fn_in, "r") as file:
		idlist= json.load(file)
	sortedkey= sorted(list(idlist.keys()))
	return (idlist, sortedkey)

#def build_

'''
build ICUSTAY dict
my_dict= {
	sid: { # key type: numpy.int64
		"hadm_id": []   #str: list of numpy.int64
		"icustay_id": [] # str: list of numpy.int64
	}
}
'''
def readin_icustay():
	f_in= pd.read_csv("ICUSTAYS.csv")
	my_dict= {}
	last_sid= 0
	for ind in range(f_in.shape[0]):
		cur_sid= f_in.iloc[ind, 1]
		if cur_sid != last_sid:
			my_dict[cur_sid]= {"hadm_id": [f_in.iloc[ind, 2],],  #numpy.int64
								"icustay_id": [f_in.iloc[ind, 3],]} #numpy.int64
		else:
			my_dict[cur_sid]["hadm_id"].append(f_in.iloc[ind, 2])
			my_dict[cur_sid]["icustay_id"].append(f_in.iloc[ind, 3])
		last_sid= cur_sid
	return (my_dict)

'''
build TFR_DICT dict
my_dict_6hr= {
	icustay_id: [timestamp, ] # numpy.int64: list of timestamps
}
'''
def readin_tfr_dict():
	vr= pd.read_excel("1108_VolumeResponse.xlsx")
	my_dict= {}
	last_id= 0
	for ind in range(vr.shape[0]):
		cur_ind= vr.iloc[ind, 0]
		if cur_ind != last_id:
			my_dict[cur_ind]= [vr.iloc[ind, 2],]
		else:
			my_dict[cur_ind].append(vr.iloc[ind, 2])
		last_id = cur_ind

	my_dict_6hr= my_dict.copy()
	for each_icu in sorted(list(my_dict_6hr.keys())):
		#print (each_icu)
		latest= datetime.datetime(2000, 1, 1)
		thelist= []
		for each_tfluid in my_dict_6hr[each_icu]:
			#print ("here")
			#print (latest)
			#print ((each_tfluid- latest).total_seconds())
			if (each_tfluid- latest).total_seconds() >21600:
				#print ("ohoh")
				#print (each_tfluid)
				thelist.append(each_tfluid)
				latest= each_tfluid

		my_dict_6hr[each_icu]= thelist
		latest= datetime.datetime(2000, 1, 1)
		thelist= []
	return (my_dict_6hr)

# find all tfr belongs to the subject 
def find_tfr(icu_list):
	res= []
	for each_icu in icu_list:
		if each_icu in TFR_DICT:
			res.extend(TFR_DICT[each_icu])
	return (res)

def findicustaylist(sid_in):
	sid_in= int(sid_in)
	return ICUSTAY[sid_in]["icustay_id"]

def get_potential_tfr(rec_in, tfr_in):
	res= []

	rec_time= rec_in[:-1].split("/")[2].split("-")
	_, rec_Y, rec_m, rec_d, rec_h, rec_min= rec_time
	rec_starttime= datetime.datetime(int(rec_Y), int(rec_m), int(rec_d), 
										int(rec_h), int(rec_min))
	for each_tfr in tfr_in:
		time_delta= (each_tfr- rec_starttime).total_seconds()
		if time_delta >0:
			res.append(each_tfr)
	return (rec_starttime, res)

# get tfr within [rec_start, rec_end] AND that tfr - rec_start > 2hr
def get_tfr_within_range(potential_tfr, rec_start, rec_len, fs_in):
	print ("get_tfr_within_range")
	print (rec_start)
	res= []
	for i in potential_tfr:
		#print (rec_start)
		time_delta= (i-rec_start).total_seconds()
		print (time_delta)
		if (time_delta <= (rec_len/fs_in)) and (time_delta > 7200):
			res.append(i)
	return (res)

def get_extraction_int(rec_start, tfr_in, fs):
# 	if fs>1:
# 		freq= fs
# 	else:
# 		freq= int(1/fs)
	print ("printing from get_extraction_int")
	print (rec_start)
	print (tfr_in)
	time_delta= (tfr_in-rec_start).total_seconds()
	res_end  = (time_delta-7200 )*fs
	ext_start= (time_delta-28800)*fs
	if ext_start > 0:
		res_start= ext_start
		flg= 0
		interval= 21600
	else: #rec_start> t-8hr
		res_start= rec_start
		flg= 1
		interval= time_delta
	print (flg)
	print (res_start)
	print (res_end)
	print (interval)

	return (flg, int(res_start), int(res_end), int(interval))

def get_abpfollowup():
	pass


def feature_extraction(rec_in, tfr_in, cur_sid, rec_start):
	festart= time.time()
	print ("start fe for: ", rec_in)
	print (festart)
	res= {}

	fs= int(rec_in.fs)
	if fs <1:
		fs= int(1/fs)
	df= pd.DataFrame(rec_in.p_signal)
	df_signame= pd.DataFrame(rec_in.sig_name)
	df.columns= [ col.strip(" ").strip("+") for col in df_signame[0]]
	df["id"]= cur_sid
	print ("finished adding columns")

	sixhr_extraction_flg, ext_start, ext_end, ext_int= get_extraction_int(rec_start, tfr_in, fs)
	print ("printing extraction params")
	print (ext_start)
	print (ext_end)


	df_in= df.iloc[ ext_start:ext_end, ]
	# df_in.columns=df_signame[0]
	df_in= df_in.dropna(axis= 1, thresh= len(df)//2).dropna()
	### clean up the sig_names please
	print ("looking at sig_names now")
	dfin_cols= list(df_in.columns)
	#print (dfin_cols)
	sig_name= [col.strip(" ").strip("+") for col in dfin_cols]
	#df_in.columns= sig_name
	print (sig_name)
	### done cleaning? or no
	res["sig_name"]=  sig_name

	test_fc_param= {
		"abs_energy": None,
		"absolute_sum_of_changes": None,
		"count_above_mean": None,
		"count_below_mean": None,
		"first_location_of_maximum": None,
		"first_location_of_minimum": None,
		"kurtosis": None,
		"length": None,
		"longest_strike_above_mean": None,
		"longest_strike_below_mean": None,
		"maximum": None,
		"mean": None,
		"median": None,
		"minimum": None,
		"skewness": None,
		"standard_deviation": None,
		"variance": None,
		
		"agg_autocorrelation": [{"f_agg": "mean", "maxlag": 40},
		{"f_agg": "median", "maxlag": 40},
		{"f_agg": "var", "maxlag": 40}],
		"ar_coefficient": [{"coeff": 0, "k": 10},{"coeff": 1, "k": 10},{"coeff": 2, "k": 10},{"coeff": 3, "k": 10},{"coeff": 4, "k": 10}],
		"linear_trend":[{"attr": "pvalue"},{"attr": "rvalue"},{"attr": "intercept"},{"attr": "slope"},{"attr": "stderr"}],
		"linear_trend_timewise":[{"attr": "pvalue"},{"attr": "rvalue"},{"attr": "intercept"},{"attr": "slope"},{"attr": "stderr"}],
		"max_langevin_fixed_point":[{"m": 3, "r": 30}],
		"quantile":[{"q": 0.1},{"q": 0.2},{"q": 0.3},{"q": 0.4},{"q": 0.6},{"q": 0.7},{"q": 0.8},{"q": 0.9}],

		"binned_entropy": [{"max_bins": 10}],
		"autocorrelation":[{"lag": 0}, {"lag": 1},{"lag": 2},{"lag": 3},{"lag": 4},{"lag": 5},{"lag": 6},{"lag": 7},{"lag": 8},{"lag": 9}],
		#"cwt_coefficients":[{"coeff": 0, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 0, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 0, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 0, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 1, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 1, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 1, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 1, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 2, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 2, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 2, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 2, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 3, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 3, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 3, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 3, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 4, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 4, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 4, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 4, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 5, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 5, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 5, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 5, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 6, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 6, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 6, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 6, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 7, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 7, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 7, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 7, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 8, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 8, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 8, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 8, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 9, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 9, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 9, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 9, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 10, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 10, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 10, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 10, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 11, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 11, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 11, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 11, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 12, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 12, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 12, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 12, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 13, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 13, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 13, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 13, "widths": [2, 5, 10, 20], "w": 20}, {"coeff": 14, "widths": [2, 5, 10, 20], "w": 2}, {"coeff": 14, "widths": [2, 5, 10, 20], "w": 5}, {"coeff": 14, "widths": [2, 5, 10, 20], "w": 10}, {"coeff": 14, "widths": [2, 5, 10, 20], "w": 20}],

		"fft_aggregated":[{"aggtype": "centroid"},{"aggtype": "variance"},{"aggtype": "skew"},{"aggtype": "kurtosis"}],
		"large_standard_deviation":[{"r": 0.05},
		{"r": 0.1},
		{"r": 0.15000000000000002},
		{"r": 0.2},
		{"r": 0.25},
		{"r": 0.30000000000000004},
		{"r": 0.35000000000000003},
		{"r": 0.4},
		{"r": 0.45},
		{"r": 0.5},
		{"r": 0.55},
		{"r": 0.6000000000000001},
		{"r": 0.65},
		{"r": 0.7000000000000001},
		{"r": 0.75},
		{"r": 0.8},
		{"r": 0.8500000000000001},
		{"r": 0.9},
		{"r": 0.9500000000000001}]

	}

	extracted= extract_features(df_in, column_id= "id", 
								default_fc_parameters= test_fc_param) #FC_PARAM
	print (extracted)
	abp_followup= []
	abp_followup_int_flg= 1
	if "ABP" in sig_name:
		abp_followup= df["ABP"].iloc[ext_end:]
		if abp_followup.shape[0]/fs > 21600:
			abp_followup_int_flg= 0


	res["abp_followup"]= abp_followup
	res["under_6hr_flag_followup"]= abp_followup_int_flg
	res["under_6hr_flag"]= sixhr_extraction_flg
	res["extracted_length"]= ext_int
	res["extraction_result"]= extracted.to_json(orient= "records")
	res["fs"]= fs

	feend= time.time()
	print("end fe")
	print (feend)
	print ("fe time:")
	print (feend-festart)
	return (res)

'''
# sid can have multiple records, each record can have 1+ tfr.
# pipeline() 's res is documented below:
res= {
	sid: [
		   {
			rec_name: "rec_name",
			num_tfr_within_range: int(num_tfr),
			except_flag: Bool, 
			tfr_dict: {
				tfr_1: {
					under_6hr_flag: Bool, # 1== no suffient for t-8 to t-2
					under_6hr_flag_followup: Bool, # 1== no abp for entire t~ t+6
					sig_name: [],
					extracted_length: int(), #if flag=0 ==> 21600 (6hr) , in secs
					extraction_result: [],
					abp_followup: [],
					fs: int(fs)
				},
			}
		   },
		  
	],

}
'''
def pipeline():
	res= {}
	for ind in range(START, END):
		last_sid= 0
		cur_sid= WFID_SORTEDKEY[ind] # str
		sid_icustay_list= findicustaylist(cur_sid)
		all_tfr= find_tfr(sid_icustay_list)

		res[cur_sid]= []
		for each_rec in WFID_LIST[cur_sid]:

			rec_ind= WFID_LIST[cur_sid].index(each_rec)
			# print (rec_ind)
			except_flag= 0
			rec_dict= {"rec_name": each_rec
					   }
			rec_starttime, potential_tfr= get_potential_tfr(each_rec, all_tfr)
			print ("printing within pipeline")
			print (cur_sid)
			print (all_tfr)
			print (potential_tfr)
			if len(potential_tfr) == 0:
				continue

			record= each_rec[:-1].split("/")[2]
			pbdir= "/".join(("mimic3wdb/matched/"+ each_rec[:-1]).split("/")[:-1])	
			try:
				tfr_dict= {}
				rec = wfdb.rdrecord(record_name= record, pb_dir= pbdir)
				fs= int(rec.fs)
				if fs <1:
					fs= int(1/fs)
				print ("done reading in")
				tfr_within_range= get_tfr_within_range(potential_tfr, rec_starttime, rec.sig_len, fs)
				print ("printing within try clause")
				print (rec.sig_len/fs)
				print (tfr_within_range)
				rec_dict["num_tfr_within_range"]= len(tfr_within_range)
				for each_pot_tfr in tfr_within_range:
					each_tfr_dict= feature_extraction(rec, each_pot_tfr, cur_sid,
						rec_starttime)
					tfr_dict[str(each_pot_tfr)]= each_tfr_dict

				rec_dict["tfr_dict"]= tfr_dict
				print ("rec_dict[tfr_dict] built")
				#print ("went thru")
			except:
				#print ("???")
				#print ("exception caught")
				except_flag= 1
				pass
			if except_flag:
				rec_dict["num_tfr_within_range"]= 0

			rec_dict["except_flag"]= except_flag	
			print("printing rec_dict")
			print (rec_dict)	
			res[cur_sid].append(rec_dict)	
			last_sid= cur_sid
	return (res)

def get_fc_param():
	res = {}
	with open("fc_param_dumps.txt", "r") as file:
		res = json.load(file)
	return (res)


if __name__ == '__main__':

	STARTTIME = time.time()

	FC_PARAM= get_fc_param()

	SIG_INTERESTED= ["II", "V", "ABP", "MCL1", "PLETH", 
					 "AVR", "III", "I", "RESP", "PAP"]

	WFID_FN = "wf_id_matched_wf.txt"
	WFID_LIST, WFID_SORTEDKEY= readin_wfid(WFID_FN)

	TFR_DICT = readin_tfr_dict()
	ICUSTAY= readin_icustay()

	N = 5960

	# output batch size = 10
	# start = 0
	# end = 19 --> change to 5960 for ultimate results 
	starti,endi = cl_parse()
	for i in range(starti, endi):
		START = i*10
		END = (i+1)*10
		
		FN_OUT = "wfdict"+str(START)+"_"+str(END)+".txt"
		res = pipeline()

		with open(FN_OUT, 'w') as file:
			file.write(json.dumps(res))

		ENDTIME = time.time()
		print ("num of sid: ", len(res.keys()))
		print ("lapse time: ", ENDTIME-STARTTIME)
	
	

	# my_dict= {}
	# my_dict= my_dict.fromkeys(wfid_sortedkey[start:end], {})
	#print (my_dict)

	# res = pipeline()

	# with open(FN_OUT, 'w') as file:
	# 	file.write(json.dumps(res))

	# ENDTIME = time.time()
	# print ("num of sid: ", len(res.keys()))
	# print ("lapse time: ", ENDTIME-STARTTIME)