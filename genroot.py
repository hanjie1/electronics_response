from hdf5libs import HDF5RawDataFile

import daqdataformats
import detdataformats
from rawdatautils.unpack.wib2 import *
from rawdatautils.utilities.wib2 import *
import detchannelmaps

import numpy as np
import glob

RunNo=15151

#have channel numbers per geoid in here
channel_map='PD2HDChannelMap'
ch_map = None
if channel_map is not None:
    ch_map = detchannelmaps.make_map(channel_map)
offline_ch_num_dict = {}
offline_ch_plane_dict = {}

ped = []
nevent = 0
offlinemap=[]

for irun in range(1):

    find_hdf5_file = glob.glob("/nfs/rscratch/hanjie/*{}_000{:d}_*.hdf5".format(RunNo,irun))[0]
    #find_hdf5_file = glob.glob("/data2/*{}_00{:02d}_*.hdf5".format(RunNo,irun))[0]

    print(find_hdf5_file)
    #filename = "/nfs/rscratch/hanjie/np04_hd_run018231_0000_dataflow0_datawriter_0_20221208T191729.hdf5.copied"
    h5_file = HDF5RawDataFile(find_hdf5_file)
    
    records = h5_file.get_all_record_ids()
    for r in records:
        ped.append([])
        #print(f'Processing (Record Number,Sequence Number)=({r[0],r[1]})')
        wib_geo_ids = h5_file.get_geo_ids_for_subdetector(r,detdataformats.DetID.Subdetector.kHD_TPC)
        nlinks = len(wib_geo_ids)
        #print("Number of WIBs: ", nlinks/2)
    
        ped[nevent] = np.empty(nlinks*256, dtype=list)
        for gid in wib_geo_ids:
            hex_gid = format(gid, '016x')
            #print(f'\tProcessing geoid {hex_gid}')
    
            frag = h5_file.get_frag(r,gid)
    
            wf = detdataformats.wib2.WIB2Frame(frag.get_data())
    
            #fill channel map info if needed
            if(offline_ch_num_dict.get(gid) is None):
                if channel_map is None:
                    offline_ch_num_dict[gid] = np.arange(256)
                    offline_ch_plane_dict[gid] = np.full(256,9999)
                else:
                    wh = wf.get_header()
                    offline_ch_num_dict[gid] = np.array([ch_map.get_offline_channel_from_crate_slot_fiber_chan(wh.crate, wh.slot, wh.link, c) for c in range(256)])
                    offline_ch_plane_dict[gid] = np.array([ ch_map.get_plane_from_offline_channel(uc) for uc in offline_ch_num_dict[gid] ])
                    #for ch in range(256):
                    #    offlinemap.append([offline_ch_num_dict[gid][ch], wh.crate, wh.slot, wh.link,ch])
                
    
            #unpack adcs into a n_frames x 256 numpy array of uint16
            adcs = np_array_adc(frag)
            #adcs_rms = np.std(adcs,axis=0)
            #adcs_ped = np.mean(adcs,axis=0)
      
            for ch in range(256):
                offline_ch = offline_ch_num_dict[gid][ch]
                ch_adcs = adcs[:,ch]
                ped[nevent][offline_ch]=ch_adcs
                
                
        nevent=nevent+1
        print("event ", nevent)

ped = np.array(ped)
rms = []
for i in range(len(ped[0])):  # loop offline channel
    tmp = np.hstack(ped[:,i])
    tmp_rms = np.std(tmp)
    rms.append(tmp_rms)
    #if tmp_rms<18:
    #    print("open channel: ",i)

np.savetxt('genroot_run{}.csv'.format(RunNo), rms, fmt="%f", delimiter=",")
#np.savetxt('offlinemap.csv', offlinemap, fmt="%d", delimiter=",")

