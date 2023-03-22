import csv
import matplotlib.pyplot as plot
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import datetime
import numpy as np

channels = []
avgRMS = []
spreadRMS = []
run=18035
apaNum = []
with open("Run%d.csv" % run, 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=' ')
	next(reader)
	for row in reader:
		if np.isnan(float(row[1])) or float(row[1]) == 0:
			continue
		chan=int(row[0])
		if chan < 2560:
			apaNum.append(1)
		elif chan < 5120:
			apaNum.append(3)
		elif chan < 7680:
			apaNum.append(2)
		else:
			apaNum.append(4)
		channels.append(chan)
		avgRMS.append(float(row[1]))

#colors = plot.cm.tab20b(np.linspace(0.5,4.5,4))
plot.scatter(channels, avgRMS, cmap=plot.cm.tab20b,c=apaNum, marker='o', linestyle='None', s=3)
plot.xlabel("Channel")
plot.ylabel("RMS (ADC Counts)")
plot.title("ProtoDUNE-II-HD RMS, Inside NP04 Cryostat")
plot.axvline(x=2560, color='black')
plot.axvline(x=5120, color='black')
plot.axvline(x=7680, color='black')
for i in [0, 2560, 5120, 7680]:
	plot.axvline(x=800+i, color='cyan', linestyle='dashed')
	plot.axvline(x=1600+i, color='cyan', linestyle='dashed')
plot.legend(handles=[Line2D([0],[0],marker='o',color=plot.cm.tab20b(0.33),label='APA 1'), Line2D([0],[0],marker='o',color=plot.cm.tab20b(0),label='APA 2'),Line2D([0],[0],marker='o',color=plot.cm.tab20b(0.99),label='APA 3'),Line2D([0],[0],marker='o',color=plot.cm.tab20b(0.66),label='APA 4')], loc='upper right')
plot.savefig("Run%d_RMS.png" % run,dpi=300,bbox_inches='tight')
plot.xlim(0,2560*4)
plot.ylim(0,60)
plot.legend(handles=[Line2D([0],[0],marker='o',color=plot.cm.tab20b(0.33),label='APA 1'), Line2D([0],[0],marker='o',color=plot.cm.tab20b(0),label='APA 2'),Line2D([0],[0],marker='o',color=plot.cm.tab20b(0.99),label='APA 3'),Line2D([0],[0],marker='o',color=plot.cm.tab20b(0.66),label='APA 4')], loc='upper left')
plot.savefig("Run%d_RMS_Zoom.png" % run,dpi=300,bbox_inches='tight')
plot.clf()
plot.scatter(channels, [val*37 for val in avgRMS], cmap=plot.cm.tab20b,c=apaNum, marker='o', linestyle='None', s=3)
plot.xlabel("Channel")
plot.ylabel("RMS ($e^{-}$ ENC)")
plot.title("ProtoDUNE-II-HD Baseline RMS - Warm (~300 K)")
plot.axvline(x=2560, color='black')
plot.axvline(x=5120, color='black')
plot.axvline(x=7680, color='black')
for i in [0, 2560, 5120, 7680]:
	plot.axvline(x=800+i, color='cyan', linestyle='dashed')
	plot.axvline(x=1600+i, color='cyan', linestyle='dashed')
plot.xlim(0,2560*4)
plot.ylim(0,2000)
plot.legend(handles=[Line2D([0],[0],marker='o',color=plot.cm.tab20b(0.33),label='APA 1'), Line2D([0],[0],marker='o',color=plot.cm.tab20b(0),label='APA 2'),Line2D([0],[0],marker='o',color=plot.cm.tab20b(0.99),label='APA 3'),Line2D([0],[0],marker='o',color=plot.cm.tab20b(0.66),label='APA 4')], loc='upper right')
plot.savefig("Run%d_RMS_ENC.png" % run,dpi=300,bbox_inches='tight')
