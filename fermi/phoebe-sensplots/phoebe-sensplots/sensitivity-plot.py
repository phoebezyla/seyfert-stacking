import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

X=np.arange(2.625,5.5,0.25)
#Xup=np.arange(2.625,5.5,0.25)
#print(X)
E=[]
F=[]
Ferrlow=[]
Ferrup=[]
Eerrlow=[]
Eerrup=[]
XF=np.arange(-1.375,3.,0.25)
analysis="NN"
for x in X:
  E+=[np.power(10,x-3)]
  Eerrlow+=[np.power(10,x-3)-np.power(10,x-3-0.125)]
  Eerrup+=[np.power(10,x-3+0.125)-np.power(10,x-3)]
for x in XF:
  F+=[np.power(10,x-3)]
  Ferrlow+=[np.power(10,x-3)-np.power(10,x-3-0.125)]
  Ferrup+=[np.power(10,x-3+0.125)-np.power(10,x-3)]
Ferr=[Ferrlow,Ferrup]
Eerr=[Eerrlow[1:],Eerrup[1:]]
E19=E
S19 = np.loadtxt('./'+analysis+'/D19.txt')
#S19=S19[1:]
c19 = np.polyfit(np.log(E19),np.log(S19*np.power(E19,-0.63)),2)
p19 = np.poly1d(c19)
print(E19,np.exp(p19(np.log(E19))))

exit()

E_26=E[2:]
S_26 = np.loadtxt('./'+analysis+'/D-26.txt')
S_26=S_26[2:]
c_26 = np.polyfit(np.log(E_26),np.log(S_26*np.power(E_26,-0.63)),2)
p_26 = np.poly1d(c_26)

S_11 = np.loadtxt('./'+analysis+'/D-11.txt')
c_11 = np.polyfit(np.log(E),np.log(S_11*np.power(E,-0.63)),2)
p_11 = np.poly1d(c_11)

S4 = np.loadtxt('./'+analysis+'/D4.txt')
c4 = np.polyfit(np.log(E),np.log(S4*np.power(E,-0.63)),2)
p4 = np.poly1d(c4)

S19 = np.loadtxt('./'+analysis+'/D19.txt')
c19 = np.polyfit(np.log(E),np.log(S19*np.power(E,-0.63)),2)
p19 = np.poly1d(c19)

#S20 = np.loadtxt('./'+analysis+'/D-20.txt')
#c20 = np.polyfit(np.log(E),np.log(S20*np.power(E,-0.63)),2)
#p20 = np.poly1d(c20)

#S_10 = np.loadtxt('./'+analysis+'/D-10.txt')
#c_10 = np.polyfit(np.log(E),np.log(S_10*np.power(E,-0.63)),2)
#p_10 = np.poly1d(c_10)

E_l,S_l = np.loadtxt('../lhaaso.txt',unpack=True)
S_l=S_l*1.60218/1.2
E_c,S_c = np.loadtxt('../cta-north.txt',unpack=True)
c_c=np.polyfit(np.log(E_c),np.log(S_c*1.60218/1.2),2)
p_c=np.poly1d(c_c)

E_cs,S_cs = np.loadtxt('../cta-south.txt',unpack=True)
c_cs=np.polyfit(np.log(E_cs),np.log(S_cs*1.60218/1.2),2)
p_cs=np.poly1d(c_cs)

E_v,S_v = np.loadtxt('../veritas.txt',unpack=True)
c_v=np.polyfit(np.log(E_v),np.log(S_v*1.60218/1.2),2)
p_v=np.poly1d(c_v)

E_m,S_m = np.loadtxt('../magic.txt',unpack=True)
c_m=np.polyfit(np.log(E_m),np.log(S_m*1.60218/1.2),2)
p_m=np.poly1d(c_m)

E_f,S_f = np.loadtxt('../Fermi-00.txt',unpack=True)
#E_f=E_f*1e-6
E_f=F
c_f=np.polyfit(np.log(F),np.log(S_f*1.60218/1.2),2)
p_f=np.poly1d(c_f)

E_h,S_h = np.loadtxt('../hess.txt',unpack=True)
S_h=S_h*1.60218/1.2
#S10 = np.loadtxt('./'+analysis+'/D10.txt')
#c10 = np.polyfit(np.log(E),np.log(S10*np.power(E,-0.63)),2)
#p10 = np.poly1d(c10)

#S30 = np.loadtxt('./'+analysis+'/D-30.txt')
#c30 = np.polyfit(np.log(E),np.log(S30*np.power(E,-0.63)),2)
#p30 = np.poly1d(c30)

#S58 = np.loadtxt('./'+analysis+'/D-58.txt')
#c58 = np.polyfit(np.log(E),np.log(S58*np.power(E,-0.63)),2)
#p58 = np.poly1d(c58)

#S38 = np.loadtxt('./'+analysis+'/D-38.txt')
#c38 = np.polyfit(np.log(E),np.log(S38*np.power(E,-0.63)),2)
#p38 = np.poly1d(c38)

#S0 = np.loadtxt('./'+analysis+'/D-0.txt')
#c0 = np.polyfit(np.log(E),np.log(S0*np.power(E,-0.63)),2)
#p0 = np.poly1d(c0)

#S50 = np.loadtxt('./'+analysis+'/D-50.txt')
#c50 = np.polyfit(np.log(E),np.log(S50*np.power(E,-0.63)),2)
#p50 = np.poly1d(c50)

S_27 = np.loadtxt('./'+analysis+'/D-27.txt')
c_27 = np.polyfit(np.log(E),np.log(S_27*np.power(E,-0.63)),2)
p_27 = np.poly1d(c_27)


S_23 = np.loadtxt('./'+analysis+'/D-23.txt')
c_23 = np.polyfit(np.log(E),np.log(S_23*np.power(E,-0.63)),2)
p_23 = np.poly1d(c_23)


S_21 = np.loadtxt('./'+analysis+'/D-21.txt')
c_21 = np.polyfit(np.log(E),np.log(S_21*np.power(E,-0.63)),2)
p_21 = np.poly1d(c_21)


S_18 = np.loadtxt('./'+analysis+'/D-18.txt')
c_18 = np.polyfit(np.log(E),np.log(S_18*np.power(E,-0.63)),2)
p_18 = np.poly1d(c_18)


S_16 = np.loadtxt('./'+analysis+'/D-16.txt')
c_16 = np.polyfit(np.log(E),np.log(S_16*np.power(E,-0.63)),2)
p_16 = np.poly1d(c_16)


S_13 = np.loadtxt('./'+analysis+'/D-13.txt')
c_13 = np.polyfit(np.log(E),np.log(S_13*np.power(E,-0.63)),2)
p_13 = np.poly1d(c_13)

S_8 = np.loadtxt('./'+analysis+'/D-8.txt')
c_8 = np.polyfit(np.log(E),np.log(S_8*np.power(E,-0.63)),2)
p_8 = np.poly1d(c_8)

S_6 = np.loadtxt('./'+analysis+'/D-6.txt')
c_6 = np.polyfit(np.log(E),np.log(S_6*np.power(E,-0.63)),2)
p_6 = np.poly1d(c_6)

S_4 = np.loadtxt('./'+analysis+'/D-4.txt')
c_4 = np.polyfit(np.log(E),np.log(S_4*np.power(E,-0.63)),2)
p_4 = np.poly1d(c_4)

S_1 = np.loadtxt('./'+analysis+'/D-1.txt')
c_1 = np.polyfit(np.log(E),np.log(S_1*np.power(E,-0.63)),2)
p_1 = np.poly1d(c_1)

S2 = np.loadtxt('./'+analysis+'/D2.txt')
c2 = np.polyfit(np.log(E),np.log(S2*np.power(E,-0.63)),2)
p2 = np.poly1d(c2)

S7 = np.loadtxt('./'+analysis+'/D7.txt')
c7 = np.polyfit(np.log(E),np.log(S7*np.power(E,-0.63)),2)
p7 = np.poly1d(c7)

S9 = np.loadtxt('./'+analysis+'/D9.txt')
c9 = np.polyfit(np.log(E),np.log(S9*np.power(E,-0.63)),2)
p9 = np.poly1d(c9)

S12 = np.loadtxt('./'+analysis+'/D12.txt')
c12 = np.polyfit(np.log(E),np.log(S12*np.power(E,-0.63)),2)
p12 = np.poly1d(c12)

S14 = np.loadtxt('./'+analysis+'/D14.txt')
c14 = np.polyfit(np.log(E),np.log(S14*np.power(E,-0.63)),2)
p14 = np.poly1d(c14)

S24 = np.loadtxt('./'+analysis+'/D24.txt')
c24 = np.polyfit(np.log(E),np.log(S24*np.power(E,-0.63)),2)
p24 = np.poly1d(c24)


S26 = np.loadtxt('./'+analysis+'/D26.txt')
c26 = np.polyfit(np.log(E),np.log(S26*np.power(E,-0.63)),2)
p26 = np.poly1d(c26)

S29 = np.loadtxt('./'+analysis+'/D29.txt')
c29 = np.polyfit(np.log(E),np.log(S29*np.power(E,-0.63)),2)
p29 = np.poly1d(c29)

S34 = np.loadtxt('./'+analysis+'/D34.txt')
c34 = np.polyfit(np.log(E),np.log(S34*np.power(E,-0.63)),2)
p34 = np.poly1d(c34)

S39 = np.loadtxt('./'+analysis+'/D39.txt')
c39 = np.polyfit(np.log(E),np.log(S39*np.power(E,-0.63)),2)
p39 = np.poly1d(c39)

S41 = np.loadtxt('./'+analysis+'/D41.txt')
c41 = np.polyfit(np.log(E),np.log(S41*np.power(E,-0.63)),2)
p41 = np.poly1d(c41)

S44 = np.loadtxt('./'+analysis+'/D44.txt')
c44 = np.polyfit(np.log(E),np.log(S44*np.power(E,-0.63)),2)
p44 = np.poly1d(c44)

S49 = np.loadtxt('./'+analysis+'/D49.txt')
c49 = np.polyfit(np.log(E),np.log(S49*np.power(E,-0.63)),2)
p49 = np.poly1d(c49)

S52 = np.loadtxt('./'+analysis+'/D52.txt')
c52 = np.polyfit(np.log(E),np.log(S52*np.power(E,-0.63)),2)
p52 = np.poly1d(c52)

S54 = np.loadtxt('./'+analysis+'/D54.txt')
c54 = np.polyfit(np.log(E),np.log(S54*np.power(E,-0.63)),2)
p54 = np.poly1d(c54)

S57 = np.loadtxt('./'+analysis+'/D57.txt')
c57 = np.polyfit(np.log(E),np.log(S57*np.power(E,-0.63)),2)
p57 = np.poly1d(c57)

S59 = np.loadtxt('./'+analysis+'/D59.txt')
c59 = np.polyfit(np.log(E),np.log(S59*np.power(E,-0.63)),2)
p59 = np.poly1d(c59)

S61 = np.loadtxt('./'+analysis+'/D61.txt')
c61 = np.polyfit(np.log(E),np.log(S61*np.power(E,-0.63)),2)
p61 = np.poly1d(c61)

S64 = np.loadtxt('./'+analysis+'/D64.txt')
c64 = np.polyfit(np.log(E),np.log(S64*np.power(E,-0.63)),2)
p64 = np.poly1d(c64)

S66 = np.loadtxt('./'+analysis+'/D66.txt')
c66 = np.polyfit(np.log(E),np.log(S66*np.power(E,-0.63)),2)
p66 = np.poly1d(c66)

S67 = np.loadtxt('./'+analysis+'/D67.txt')
c67 = np.polyfit(np.log(E),np.log(S67*np.power(E,-0.63)),2)
p67 = np.poly1d(c67)

def plotValue(x,K,alpha,beta):
  return np.power(x,2)*K*1e9*((x/2.0)**(alpha-beta*np.log(x/2.0)))

Xcrab=np.linspace(0.3,150,60)
Ycrab=plotValue(Xcrab,6.67e-21,-2.582,0.09)
#plt.plot(Xcrab,Ycrab,color="lightgray",linestyle="--")
#plt.plot(Xcrab,Ycrab*0.1,linestyle='dotted',color="lightgray")
#plt.plot(Xcrab,Ycrab*0.01,"--",color="lightgray")
fig,ax=plt.subplots(1)
plt.ylim(4e-14,2e-10)
#plt.xlim(0.2,330)
#plt.xlim(0.01,330)
plt.plot(E_29,np.exp(p29(np.log(E_29))), color = '#FCBC66')
ax.errorbar(E_29,np.exp(p29(np.log(E_29))),xerr=Eerr,color='#FCBC66', linewidth=1, marker='v')
#plt.plot(E_26,np.exp(p26(np.log(E_26))), color = 'black',linestyle = ':',linewidth=3)
#plt.plot(X_20,Y_20,color = '#FCBC66',linestyle = ':',linewidth=3)
#plt.plot(E,np.exp(p_11(np.log(E))), color = '#8DC6BF',linestyle = '--',linewidth=3)
#plt.plot(X0,Y0,color = '#8DC6BF',linestyle = '--',linewidth=3)
#plt.plot(E,np.exp(p4(np.log(E))),color = '#584053',linestyle = '-', linewidth=3)
#plt.plot(E_l,S_l,color = 'black',linestyle = '--', linewidth=2)
#plt.plot(E_c,np.exp(p_c(np.log(E_c))),color = 'black',linestyle = ':', linewidth=2)
#plt.plot(E_cs,np.exp(p_cs(np.log(E_cs))),color = 'black',linestyle = 'dashdot', linewidth=2)
#plt.plot(E_v,np.exp(p_v(np.log(E_v))),color = 'lightgrey', linewidth=2)
#plt.plot(E_m,np.exp(p_m(np.log(E_m))),color = 'grey',linestyle = ':', linewidth=2)
#plt.plot(E_h,S_h,color = 'grey',linestyle = 'dashdot', linewidth=2)
plt.plot(F,S_f,color = 'lightseagreen', marker='o')
ax.errorbar(F,S_f,xerr=Ferr,color='lightseagreen',linewidth=1)

#plt.plot(E,np.exp(p19(np.log(E))), color = 'tomato',linewidth=3)
#plt.plot(E,np.exp(p19(np.log(E))),color = 'tomato',linestyle = 'dashdot', linewidth=3)
#plt.legend([ 'Crab',r'0.1$\times$Crab',r'dec = -29$^{\circ}$', r'dec = -11$^{\circ}$',r'dec = 4$^{\circ}$',r'dec = 19$^{\circ}$'])
#plt.legend(['LHAASO 1 yr','CTA Northern array 50 h','CTA Southern array 50 h','VERITAS 50 h','MAGIC 50 h','H.E.S.S. 50 h','Fermi LAT 10 yr', 'HAWC 10 yr']) 
plt.legend(['HAWC GC Declination', r'$\it{Fermi}$-LAT (l,b)=(0,0)'])
#plt.xlabel('True Gamma-ray Energy (TeV)')
plt.title('Joint sensitivity to the Galactic Center with 10 years of data')
plt.xlabel('Gamma-ray Energy (TeV)')
plt.ylabel('$E^{2}$ Flux (TeV/s cm$^{2}$)')
plt.yscale('log')
plt.xscale('log')
plt.grid()
#plt.title(r'5 years quartet decade HAWC differential sensitivity')

plt.savefig('sensitivity-compare.pdf', transparent = 'True')
# plt.savefig('sensitivity-proposal.pdf', transparent = 'True') 
# #plt.savefig('sensitivity-'+analysis+'.pdf', transparent = 'True')
# #plt.show()
# plt.close()

# exit()
# D19 = interp1d(E,np.exp(p19(np.log(E))))
# D4 = interp1d(E,np.exp(p4(np.log(E))))
# D_11 = interp1d(E,np.exp(p_11(np.log(E))))
# D_26 = interp1d(E,np.exp(p_26(np.log(E))))

# D_29 = interp1d(E,np.exp(p_29(np.log(E))))
# D_27 = interp1d(E,np.exp(p_27(np.log(E))))
# D_23 = interp1d(E,np.exp(p_23(np.log(E))))
# D_21 = interp1d(E,np.exp(p_21(np.log(E))))
# D_18 = interp1d(E,np.exp(p_18(np.log(E))))
# D_16 = interp1d(E,np.exp(p_16(np.log(E))))
# D_13 = interp1d(E,np.exp(p_13(np.log(E))))
# D_8 = interp1d(E,np.exp(p_8(np.log(E))))
# D_6 = interp1d(E,np.exp(p_6(np.log(E))))
# D_4 = interp1d(E,np.exp(p_4(np.log(E))))
# D_1 = interp1d(E,np.exp(p_1(np.log(E))))
# D2 = interp1d(E,np.exp(p2(np.log(E))))
# D7 = interp1d(E,np.exp(p7(np.log(E))))
# D9 = interp1d(E,np.exp(p9(np.log(E))))
# D12 = interp1d(E,np.exp(p12(np.log(E))))
# D14 = interp1d(E,np.exp(p14(np.log(E))))
# D24 = interp1d(E,np.exp(p24(np.log(E))))
# D26 = interp1d(E,np.exp(p26(np.log(E))))
# D29 = interp1d(E,np.exp(p29(np.log(E))))
# D34 = interp1d(E,np.exp(p34(np.log(E))))
# D39 = interp1d(E,np.exp(p39(np.log(E))))
# D41 = interp1d(E,np.exp(p41(np.log(E))))
# D44 = interp1d(E,np.exp(p44(np.log(E))))
# D49 = interp1d(E,np.exp(p49(np.log(E))))
# D52 = interp1d(E,np.exp(p52(np.log(E))))
# D54 = interp1d(E,np.exp(p54(np.log(E))))
# D57 = interp1d(E,np.exp(p57(np.log(E))))
# D59 = interp1d(E,np.exp(p59(np.log(E))))
# D61 = interp1d(E,np.exp(p61(np.log(E))))
# D64 = interp1d(E,np.exp(p64(np.log(E))))
# D66 = interp1d(E,np.exp(p66(np.log(E))))
# D67 = interp1d(E,np.exp(p67(np.log(E))))

# x = [2,10,50]
# xdec = [-29,-27,-26,-23,-21,-18,-16,-13,-11,-8,-6,-4,-1,2,4,7,9,12,14,19,24,26,29,34,39,41,44,49,52,54,57,59,61,64,66,67]

# dec1 = [D_29(x)[0],D_27(x)[0],D_26(x)[0],D_23(x)[0],D_21(x)[0],D_18(x)[0],D_16(x)[0],D_13(x)[0],D_11(x)[0],D_8(x)[0],D_6(x)[0],D_4(x)[0],D_1(x)[0],D2(x)[0],D4(x)[0],D7(x)[0],D9(x)[0],D12(x)[0],D14(x)[0],D19(x)[0],D24(x)[0],D26(x)[0],D29(x)[0],D34(x)[0],D39(x)[0],D41(x)[0],D44(x)[0],D49(x)[0],D52(x)[0],D54(x)[0],D57(x)[0],D59(x)[0],D61(x)[0],D64(x)[0],D66(x)[0],D67(x)[0]]

# dec10 = [D_29(x)[1],D_27(x)[1],D_26(x)[1],D_23(x)[1],D_21(x)[1],D_18(x)[1],D_16(x)[1],D_13(x)[1],D_11(x)[1],D_8(x)[1],D_6(x)[1],D_4(x)[1],D_1(x)[1],D2(x)[1],D4(x)[1],D7(x)[1],D9(x)[1],D12(x)[1],D14(x)[1],D19(x)[1],D24(x)[1],D26(x)[1],D29(x)[1],D34(x)[1],D39(x)[1],D41(x)[1],D44(x)[1],D49(x)[1],D52(x)[1],D54(x)[1],D57(x)[1],D59(x)[1],D61(x)[1],D64(x)[1],D66(x)[1],D67(x)[1]]

# dec50 = [np.exp(p_29(np.log(50))),np.exp(p_27(np.log(50))),np.exp(p_26(np.log(50))),np.exp(p_23(np.log(50))),np.exp(p_21(np.log(50))),np.exp(p_18(np.log(50))),np.exp(p_16(np.log(50))),np.exp(p_13(np.log(50))),np.exp(p_11(np.log(50))),np.exp(p_8(np.log(50))),np.exp(p_6(np.log(50))),np.exp(p_4(np.log(50))),np.exp(p_1(np.log(50))),np.exp(p2(np.log(50))),np.exp(p4(np.log(50))),np.exp(p7(np.log(50))),np.exp(p9(np.log(50))),np.exp(p12(np.log(50))),np.exp(p14(np.log(50))),np.exp(p19(np.log(50))),np.exp(p24(np.log(50))),np.exp(p26(np.log(50))),np.exp(p29(np.log(50))),np.exp(p34(np.log(50))),np.exp(p39(np.log(50))),np.exp(p41(np.log(50))),np.exp(p44(np.log(50))),np.exp(p49(np.log(50))),np.exp(p52(np.log(50))),np.exp(p54(np.log(50))),np.exp(p57(np.log(50))),np.exp(p59(np.log(50))),np.exp(p61(np.log(50))),np.exp(p64(np.log(50))),np.exp(p66(np.log(50))),np.exp(p67(np.log(50))) ]

# c_1 = np.polyfit(xdec,np.log(dec1),2)
# p_1 = np.poly1d(c_1)
# #plt.plot(X_26,np.exp(p_26(np.log(X_26))),color = '#CA3074')
# c_10 = np.polyfit(xdec,np.log(dec10),2)
# p_10 = np.poly1d(c_10)
# c_50 = np.polyfit(xdec,np.log(dec50),2)
# p_50 = np.poly1d(c_50)



# plt.xlabel(r'Declination ($^{\circ}$)')
# plt.ylabel('$E^{2}$ Flux (TeV/s cm$^{2}$)')
# plt.yscale('log')
# #plt.plot(xdec,dec1, color = '#584053',marker = 'o',linestyle = 'None')
# #plt.plot(xdec,dec10, color = '#8DC6BF',marker = 'o',linestyle = 'None')
# #plt.plot(xdec,dec50,color = '#CA3074', marker='o',linestyle = 'None')

# plt.plot(xdec,np.exp(p_1(xdec)),color = '#584053',linewidth=2.5,linestyle=':')
# plt.plot(xdec,np.exp(p_10(xdec)),color = '#8DC6BF',linewidth=2.5)
# plt.plot(xdec,np.exp(p_50(xdec)),color = '#CA3074',linewidth=2.5,linestyle='--')
# #plt.title(r'5 years quartet decade HAWC differential sensitivity')
# plt.legend(['2 TeV','10 TeV','50 TeV'])
# plt.grid()
# plt.savefig('sensitivity-vs-declination-nopoints-TeV.pdf')

# #plt.show()
