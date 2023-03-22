from hdf5libs import HDF5RawDataFile
  
import daqdataformats
import detdataformats.wib2
from rawdatautils.unpack.wib2 import *
import detchannelmaps
import glob
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle

CHANNEL_MAP = "PD2HDChannelMap"
CHANNELS_PER_WIB = 256
FRAMES_PER_RECORD = 8192
runNo = 15155
PULSE_LEN = 2303

def get_channel_map(h5_file,channel_map=CHANNEL_MAP):
    offline_ch_num_dict = {}
    wib_loc_dict = {}
    ch_map = detchannelmaps.make_map(channel_map)
    rec = h5_file.get_all_record_ids()[0]
    for i,gid in enumerate(h5_file.get_geo_ids(rec,daqdataformats.GeoID.SystemType.kTPC)):
        if(offline_ch_num_dict.get(gid) is None):
            if channel_map is None:
                offline_ch_num_dict[gid] = range(256)
            else:
                frag = h5_file.get_frag(rec,gid)
                wh = detdataformats.wib2.WIB2Frame(frag.get_data()).get_header()
                offline_ch_num_dict[gid] = [ch_map.get_offline_channel_from_crate_slot_fiber_chan(wh.crate, wh.slot, wh.link, c) for c in range(256)]
                wib_loc_dict[gid] = (wh.crate,wh.slot,wh.link)
    return offline_ch_num_dict, wib_loc_dict

def extract_adcs(h5_file,record,frames=FRAMES_PER_RECORD):
    wib_geo_ids = h5_file.get_geo_ids(record,daqdataformats.GeoID.SystemType.kTPC)
    adcs = np.zeros((len(wib_geo_ids),FRAMES_PER_RECORD,CHANNELS_PER_WIB),dtype='int64')
    tmst = np.zeros((len(wib_geo_ids),FRAMES_PER_RECORD),dtype='int64')

    for i,gid in enumerate(wib_geo_ids):
        frag = h5_file.get_frag(record,gid)
        n_frames = (frag.get_size()-frag.get_header().sizeof())//detdataformats.wib2.WIB2Frame.sizeof()

        if(n_frames!=frames):
            print(f"ERROR! Number of frames found {n_frames} in {gid} is not as expected ({frames}).")
            return adcs
        adcs[i] = np_array_adc(frag)
        tmst[i] = np_array_timestamp(frag)

    return adcs,tmst

def ResFunc(x, par0, par1):
    A1 = 4.31054*par0
    A2 = 2.6202*par0
    A3 = 0.464924*par0
    A4 = 0.762456*par0
    A5 = 0.327684*par0

    E1 = np.exp(-2.94809*x/par1)
    E2 = np.exp(-2.82833*x/par1)
    E3 = np.exp(-2.40318*x/par1)

    lambda1 = 1.19361*x/par1
    lambda2 = 2.38722*x/par1
    lambda3 = 2.5928*x/par1
    lambda4 = 5.18561*x/par1

    return A1*E1-A2*E2*(np.cos(lambda1)+np.cos(lambda1)*np.cos(lambda2)+np.sin(lambda1)*np.sin(lambda2))+A3*E3*(np.cos(lambda3)+np.cos(lambda3)*np.cos(lambda4)+np.sin(lambda3)*np.sin(lambda4))+A4*E2*(np.sin(lambda1)-np.cos(lambda2)*np.sin(lambda1)+np.cos(lambda1)*np.sin(lambda2))-A5*E3*(np.sin(lambda3)-np.cos(lambda4)*np.sin(lambda3)+np.cos(lambda3)*np.sin(lambda4))

ch_map = detchannelmaps.make_map(CHANNEL_MAP)
written_hdf5_files = glob.glob("/nfs/rscratch/hanjie/*{}_*".format(runNo))
print(written_hdf5_files)

all_adcs=[]
all_tmst=[]
nfile = len(written_hdf5_files)

for i in range(nfile):
    h5_file = HDF5RawDataFile(written_hdf5_files[i])
    records = h5_file.get_all_record_ids()

    if i==0:
        offline_ch_num_dict, wib_loc_dict = get_channel_map(h5_file)
        key_list=list(offline_ch_num_dict.keys())
    
    nevent = len(records)
    if nevent>500:
        nevent=500
    for iev in range(nevent):
        all_adcs.append(extract_adcs(h5_file,records[iev])[0])
        all_tmst.append(extract_adcs(h5_file,records[iev])[1])

#samples=all_adcs[0][0,:,3]
#xx=range(2303)
#plt.plot(xx, samples[0:2303])
#plt.plot(xx, samples[2303:4606])
#plt.plot(xx, samples[4606:6909])
#plt.show()
avgpulse = []
plstd = []
A0_np = np.zeros((10,256))
tp_np = np.zeros((10,256))
#for ilink in range(10):
#    for ich in range(256):

for ilink in range(10):
    avgpulse.append([])
    plstd.append([])
    for ich in range(256):
        npulse=0
        totpls=np.zeros(PULSE_LEN)
        allpls = []
        for iev in range(len(all_adcs)):
            evtdata = all_adcs[iev][ilink,:,ich]

            if iev==0:
               peak1_pos = np.argmax(evtdata[0:PULSE_LEN])
               peak_val = evtdata[peak1_pos]

               if peak1_pos>(PULSE_LEN-300):
                  t0 = np.argmax(evtdata[peak1_pos+300:peak1_pos+300+PULSE_LEN])
                  t0 = t0-300
                  totpls = totpls + evtdata[t0:t0+PULSE_LEN]
                  t0 = all_tmst[0][ilink][t0]
                  allpls.append(evtdata[t0:t0+PULSE_LEN])
               else:
                  totpls = totpls + evtdata[:PULSE_LEN]
                  t0 = all_tmst[0][ilink][0]
                  allpls.append(evtdata[:PULSE_LEN])
               npulse=1

            start_t = (PULSE_LEN*32-(all_tmst[iev][ilink][0]-t0)%(PULSE_LEN*32))//32
            end_t = len(evtdata)-PULSE_LEN
            for tt in range(start_t, end_t, PULSE_LEN):
                totpls = totpls + evtdata[tt:tt+PULSE_LEN]
                npulse = npulse+1
                allpls.append(evtdata[tt:tt+PULSE_LEN])
            
        apulse = totpls/npulse

        #pmax = np.amax(apulse)
        #maxpos = np.argmax(apulse)
        #pbl = apulse[maxpos-5]
 
        pl_std=[]
        allpls = np.array(allpls)
        for ibin in range(PULSE_LEN):
            tmp_std = np.std(allpls[:,ibin])
            pl_std.append(tmp_std)
        plstd[ilink].append(pl_std)
        avgpulse[ilink].append(apulse)

#        a_xx = np.array(range(25))*0.5
#        popt, pcov = curve_fit(ResFunc, a_xx, apulse[maxpos-5:maxpos+20]-pbl)
#        A0_np[ilink][ich]=popt[0]
#        tp_np[ilink][ich]=popt[1]
#
#        plt.scatter(a_xx, apulse[maxpos-5:maxpos+20]-pbl,c='r')
#        xx = np.linspace(0,25,100)*0.5
#        plt.plot(xx, ResFunc(xx,popt[0],popt[1]))
#        plt.xlabel('us')
#        plt.ylabel('ADC')
#        plt.title('link%d chan%d'%(ilink,ich))
#        plt.text(8,pmax-1500,'A0=%.2f'%popt[0],fontsize = 15)
#        plt.text(8,pmax-2500,'tp=%.2f'%popt[1],fontsize = 15)
#        plt.savefig("plots/run%d_link%d_ch%d"%(runNo,ilink,ich))
#        plt.close()

outfp = "results_1/avg_pulse_{}.bin".format(runNo)
with open(outfp, 'wb') as fn:
     pickle.dump([avgpulse,plstd], fn)

#outfp = "results/A0_{}.bin".format(runNo)
#with open(outfp, 'wb') as fn:
#     pickle.dump(A0_np, fn)
#
#outfp = "results/tp_{}.bin".format(runNo)
#with open(outfp, 'wb') as fn:
#     pickle.dump(tp_np, fn)

