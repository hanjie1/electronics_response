double response(double *x, double *par){
double f = 4.31054*exp(-2.94809*x[0]/par[1])*par[0]-2.6202*exp(-2.82833*x[0]/par[1])*cos(1.19361*x[0]/par[1])*par[0]
              -2.6202*exp(-2.82833*x[0]/par[1])*cos(1.19361*x[0]/par[1])*cos(2.38722*x[0]/par[1])*par[0]
                    +0.464924*exp(-2.40318*x[0]/par[1])*cos(2.5928*x[0]/par[1])*par[0]
                          +0.464924*exp(-2.40318*x[0]/par[1])*cos(2.5928*x[0]/par[1])*cos(5.18561*x[0]/par[1])*par[0]
                                +0.762456*exp(-2.82833*x[0]/par[1])*sin(1.19361*x[0]/par[1])*par[0]
                                      -0.762456*exp(-2.82833*x[0]/par[1])*cos(2.38722*x[0]/par[1])*sin(1.19361*x[0]/par[1])*par[0]
                                            +0.762456*exp(-2.82833*x[0]/par[1])*cos(1.19361*x[0]/par[1])*sin(2.38722*x[0]/par[1])*par[0]
                                                  -2.6202*exp(-2.82833*x[0]/par[1])*sin(1.19361*x[0]/par[1])*sin(2.38722*x[0]/par[1])*par[0] 
                                                        -0.327684*exp(-2.40318*x[0]/par[1])*sin(2.5928*x[0]/par[1])*par[0] + 
                                                              +0.327684*exp(-2.40318*x[0]/par[1])*cos(5.18561*x[0]/par[1])*sin(2.5928*x[0]/par[1])*par[0]
                                                                    -0.327684*exp(-2.40318*x[0]/par[1])*cos(2.5928*x[0]/par[1])*sin(5.18561*x[0]/par[1])*par[0]
                                                                          +0.464924*exp(-2.40318*x[0]/par[1])*sin(2.5928*x[0]/par[1])*sin(5.18561*x[0]/par[1])*par[0];

 if (x[0] >0&&x[0] < 20){
    return f;
 }else{
   return 0;
 }
}



void test_shift(double t1 = 0.5, double t2 = 3.0){
  TF1 *f1 = new TF1("func1",response,0,20,2);
  TF1 *f2 = new TF1("func2",response,0,20,2);
  
  f1->SetParameters(47.*1.012,t1);
  f2->SetParameters(47.*1.012,t2);


  // construct two sets of histograms with time shifted electronics response
  double t0 = 2.137; // us??

  TH1F *h1 = new TH1F("h1","h1",100,0,50);
  TH1F *h2 = new TH1F("h2","h2",100,0,50);
  TH1F *h3 = new TH1F("h3","h3",100,0,50);
  TH1F *h4 = new TH1F("h4","h4",100,0,50);
  for (Int_t i=0;i!=100;i++){
    double t = h1->GetBinCenter(i+1);
    h1->SetBinContent(i+1,f1->Eval(t));
    h2->SetBinContent(i+1,f1->Eval(t-t0));
    
    h3->SetBinContent(i+1,f2->Eval(t));
    h4->SetBinContent(i+1,f2->Eval(t-t0));
  }


  // do FFT on them ...
  TH1 *h1m = h1->FFT(0,"MAG");
  TH1 *h1p = h1->FFT(0,"PH");
  TH1 *h2m = h2->FFT(0,"MAG");
  TH1 *h2p = h2->FFT(0,"PH");
  
  TH1 *h3m = h3->FFT(0,"MAG");
  TH1 *h3p = h3->FFT(0,"PH");
  TH1 *h4m = h4->FFT(0,"MAG");
  TH1 *h4p = h4->FFT(0,"PH");

  // h1m->Divide(h2m);
  // h1p->Add(h2p,-1);
  // //  h1->Draw();
  // //h2->Draw("same");
  // h3m->Divide(h4m);
  // h3p->Add(h4p,-1);


  // validation plots ...
  TCanvas *c1 = new TCanvas("c1","c1",1200,800);
  c1->Divide(2,2);
  c1->cd(1);
  h1->Draw();
  h1->SetXTitle("Time (us)");
  h1->SetYTitle("mV/fC");
  h1->SetLineColor(1);
  h2->Draw("same");
  h2->SetLineColor(2);

  c1->cd(2);
  h3->Draw();
  h3->SetXTitle("Time (us)");
  h3->SetYTitle("mV/fC");
  h3->SetLineColor(1);
  h4->Draw("same");
  h4->SetLineColor(2);
  
  c1->cd(3);
  h1m->Draw();
  h1m->SetLineColor(1);
  h3m->Draw("same");
  h3m->SetLineColor(2);
  
  c1->cd(4);
  h1p->Draw();
  h1p->SetLineColor(1);
  h3p->Draw("same");
  h3p->SetLineColor(2);


  // do a shift of f1 according to f2 shift
  // if f1 = 0.5 us, and f2 = 3.0 us, we have some oscillation
  // if f1 = 2.0 us, and f2 = 3.0 us, we do not have oscillatins
  // in practice, we can use the corresponding f using ideal electronics response to do the shift
  TCanvas *c2 = new TCanvas("c2","c2",800,600);
  Double_t value_re[100];
  Double_t value_im[100];

  
  for (Int_t i=0;i!=100;i++){
    Double_t rho = h2m->GetBinContent(i+1);
    Double_t phi = h2p->GetBinContent(i+1);

    Double_t rho4 = h4m->GetBinContent(i+1);
    Double_t rho3 = h3m->GetBinContent(i+1);
    Double_t phi4 = h4p->GetBinContent(i+1);
    Double_t phi3 = h3p->GetBinContent(i+1);

    phi += phi3-phi4;
    rho *= rho3/rho4;
    
    value_re[i] = rho*cos(phi)/100.;
    value_im[i] = rho*sin(phi)/100.;
  }
  Int_t  n  =100;
  TVirtualFFT *ifft = TVirtualFFT::FFT(1,&n,"C2R M K");
  ifft->SetPointsComplex(value_re,value_im);
  ifft->Transform();
  TH1 *fb = 0;
  fb = TH1::TransformHisto(ifft,fb,"Re");
  fb->Draw();
  fb->SetTitle("Inverse FFT");


  
}
